"""
Resumes threeDSparseTransDiffusion_periodic_eval.py training from a saved
checkpoint, up to an absolute --target-epochs count. Meant for cluster jobs
with a wall-clock limit shorter than the full training run (see diffcont.sh):
each job trains as far as it can, checkpoints, and the next job picks up
where it left off.

threeDSparseTransDiffusion_periodic_eval.py's own train() always builds a
fresh optimizer/scheduler and 0-indexes its epoch loop internally, so it
can't resume correctly on its own (checkpoint filenames would restart from
_epoch_1, Adam's momentum would reset, and the cosine schedule would restart
at max LR) - this script reuses everything else from that module (model
class, dataset loading, TrajectoryDataset, the train/val split, the periodic
single-map eval, get_loss/evaluate) but replaces train()'s loop with a
resumable one:
  - the optimizer is restored from checkpoint["optimizer_state_dict"] (Adam's
    per-parameter momentum/variance carries over rather than restarting cold)
    unless --reset-optimizer is passed
  - CosineAnnealingLR is rebuilt fresh with T_max=target_epochs and
    last_epoch=start_epoch, rather than loading the checkpoint's own
    scheduler_state_dict wholesale - loading it directly would silently
    re-import whatever T_max an earlier run used even if --target-epochs now
    asks for a different (e.g. extended) one, which is the wrong behavior
    whenever this job's target differs from a previous stage's
  - checkpoint filenames and the "epoch" field use true absolute epoch
    numbers throughout
  - the periodic-eval CSV (periodic_eval_map_<id>.csv), if one already
    exists from the original run or an earlier continuation job, is loaded
    and appended to rather than overwritten, so the RMSE/variance-drop-vs-
    epoch history stays continuous across job restarts

Usage (matches diffcont.sh):
    python continue_sparse_trans_training.py \
        --checkpoint /path/to/sparse_trans_waypoints_epoch_1800.pth \
        --target-epochs 3000
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Importing this runs its module-level setup (dataset load + preprocessing,
# plus one throwaway NoisePredictor/AdamW instance) but NOT its __main__
# block (guarded), so none of the original from-scratch training runs here.
import threeDSparseTransDiffusion_periodic_eval as trainmod


def resolve_checkpoint_path(path_text):
    path = Path(path_text)
    if path.is_file():
        return path
    in_checkpoint_dir = trainmod.CHECKPOINT_DIR / path_text
    if in_checkpoint_dir.is_file():
        return in_checkpoint_dir
    in_script_dir = SCRIPT_DIR / path_text
    if in_script_dir.is_file():
        return in_script_dir
    raise FileNotFoundError(f"Could not find checkpoint: {path_text}")


def make_dataloader(dataset, batch_size, shuffle, num_workers):
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if num_workers <= 0:
            num_workers = 2 if torch.cuda.is_available() else 0
    num_workers = max(0, num_workers) if torch.cuda.is_available() else 0
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def resolve_start_epoch(checkpoint, checkpoint_path):
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        return int(checkpoint["epoch"])
    match = re.search(r"epoch_(\d+)", checkpoint_path.stem)
    if match:
        print(f"Checkpoint dict had no 'epoch' key; inferred start epoch {match.group(1)} from filename.", flush=True)
        return int(match.group(1))
    raise SystemExit(
        f"Could not determine the checkpoint's epoch from either its contents or filename: {checkpoint_path}"
    )


def load_periodic_eval_history(eval_metrics_path):
    if not eval_metrics_path.exists():
        return [], [], [], []
    prior = np.loadtxt(eval_metrics_path, delimiter=",", skiprows=1)
    if prior.size == 0:
        return [], [], [], []
    if prior.ndim == 1:
        prior = prior.reshape(1, -1)
    print(f"Loaded {prior.shape[0]} prior periodic-eval rows from {eval_metrics_path}", flush=True)
    return (
        prior[:, 0].astype(int).tolist(),
        prior[:, 1].tolist(),
        prior[:, 2].tolist(),
        prior[:, 3].tolist(),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                         help="Checkpoint path: absolute, relative to this script, or a filename in checkpoints/.")
    parser.add_argument("--target-epochs", type=int, required=True,
                         help="Absolute epoch count to train up to (not additional epochs on top of the checkpoint).")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to trainmod.BATCH_SIZE.")
    parser.add_argument("--lr", type=float, default=None, help="Defaults to trainmod.LR.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Defaults to trainmod.WEIGHT_DECAY.")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=trainmod.EVAL_EVERY)
    parser.add_argument("--num-workers", type=int, default=None,
                         help="DataLoader worker count. Defaults to SLURM_CPUS_PER_TASK when set, else 2 on CUDA / 0 on CPU.")
    parser.add_argument("--reset-optimizer", action="store_true",
                         help="Load model weights from the checkpoint but start Adam's momentum/variance state fresh.")
    return parser.parse_args()


def main():
    args = parse_args()
    lr = args.lr if args.lr is not None else trainmod.LR
    weight_decay = args.weight_decay if args.weight_decay is not None else trainmod.WEIGHT_DECAY
    batch_size = args.batch_size if args.batch_size is not None else trainmod.BATCH_SIZE

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=trainmod.device, weights_only=False)
    start_epoch = resolve_start_epoch(checkpoint, checkpoint_path)
    if start_epoch >= args.target_epochs:
        raise SystemExit(
            f"Checkpoint is already at epoch {start_epoch}, which is >= --target-epochs {args.target_epochs}; nothing to do."
        )
    print(f"Resuming from epoch {start_epoch}, training to epoch {args.target_epochs}", flush=True)

    model = trainmod.NoisePredictor().to(trainmod.device)
    raw_state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    raw_state_dict = trainmod.remap_legacy_state_dict_keys(raw_state_dict)
    trainmod.load_model_state_dict_compatible(model, raw_state_dict)
    trainmod.model = model  # sample_plot_traj() at the bottom reads this module global directly

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if not args.reset_optimizer and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = weight_decay
            print("Restored optimizer state (Adam momentum/variance buffers).", flush=True)
        except ValueError as exc:
            # Happens when the checkpoint predates a model architecture change
            # (e.g. an added conditioning branch) - load_model_state_dict_compatible
            # already tolerated that for the model weights via strict=False, but the
            # optimizer's param groups are still sized to the OLD parameter count, so
            # they can't be loaded into the current (larger) optimizer at all. Falling
            # back to fresh Adam state rather than crashing; the mismatch is real (not
            # a bug to silently paper over further), so it's still reported clearly.
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            print(
                f"Could not restore optimizer state ({exc}); this checkpoint likely predates "
                "a model architecture change. Starting Adam state fresh instead.",
                flush=True,
            )
    else:
        if args.reset_optimizer:
            print("--reset-optimizer set; starting Adam state fresh (model weights still loaded from checkpoint).", flush=True)
        else:
            print("No optimizer_state_dict in checkpoint - starting Adam state fresh.", flush=True)

    # CosineAnnealingLR requires 'initial_lr' on each param group before it can be
    # constructed with a non-default last_epoch; set explicitly rather than relying
    # on whatever the restored optimizer state happened to carry. T_max is always
    # rebuilt from THIS run's --target-epochs (see module docstring for why loading
    # the checkpoint's own scheduler_state_dict wholesale would be wrong whenever
    # target_epochs differs from an earlier stage's).
    for group in optimizer.param_groups:
        group["initial_lr"] = lr
    scheduler = CosineAnnealingLR(optimizer, T_max=args.target_epochs, eta_min=trainmod.MIN_LR, last_epoch=start_epoch)
    print(f"Scheduler resumed at epoch {start_epoch} (T_max={args.target_epochs}): lr={optimizer.param_groups[0]['lr']:.6e}", flush=True)

    train_mask, val_mask, val_map_ids = trainmod.build_map_id_split(val_count=2)
    print(f"Validation map_ids: {val_map_ids.tolist()}", flush=True)
    print(f"Training samples: {int(train_mask.sum().item())}", flush=True)
    print(f"Validation samples: {int(val_mask.sum().item())}", flush=True)

    train_dataset = trainmod.TrajectoryDataset(
        trainmod.trajectories[train_mask],
        trainmod.weights[train_mask],
        meanvarmarkermaps=trainmod.meanvarmarkermaps[train_mask],
        conditions=trainmod.conditions[train_mask],
        initial_heading_velocities=trainmod.initial_heading_velocities[train_mask],
        total_variance_conditions=trainmod.total_variance_conditions[train_mask],
    )
    val_dataset = trainmod.TrajectoryDataset(
        trainmod.trajectories[val_mask],
        trainmod.weights[val_mask],
        meanvarmarkermaps=trainmod.meanvarmarkermaps[val_mask],
        conditions=trainmod.conditions[val_mask],
        initial_heading_velocities=trainmod.initial_heading_velocities[val_mask],
        total_variance_conditions=trainmod.total_variance_conditions[val_mask],
    )
    dataloader = make_dataloader(train_dataset, batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = make_dataloader(val_dataset, batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"DataLoader workers: {dataloader.num_workers}", flush=True)

    eval_map = int(trainmod.EVAL_MAP_OVERRIDE) if trainmod.EVAL_MAP_OVERRIDE else int(val_map_ids[0].item())
    print(f"Periodic single-map eval: map={eval_map}, maptype={trainmod.EVAL_MAPTYPE}, every {args.eval_every} epochs", flush=True)
    eval_env = trainmod.prepare_eval_environment(eval_map, trainmod.EVAL_MAPTYPE)

    eval_metrics_path = trainmod.PLOT_DIR / f"periodic_eval_map_{eval_map}.csv"
    eval_epochs, eval_global_rmse_drops, eval_occupied_rmse_drops, eval_variance_drops = load_periodic_eval_history(eval_metrics_path)

    model.train()
    loss_vals, stepcount = [], []
    epoch_train_loss_vals, epoch_steps = [], []
    val_loss_vals, val_steps = [], []
    last_loss = None
    last_epoch_train_loss = None

    print("Starting resumed training loop", flush=True)
    for epoch in range(start_epoch + 1, args.target_epochs + 1):
        print(f"Epoch {epoch}/{args.target_epochs}", flush=True)
        epoch_loss_sum = 0.0
        epoch_weight_sum = 0.0
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            stepcount.append((epoch - 1) * len(dataloader) + step)
            if len(batch) == 6:
                (
                    traj,
                    current_position,
                    meanvarmarker_map,
                    initial_heading_velocity,
                    total_variance_condition,
                    batch_weights,
                ) = batch
            elif len(batch) == 5:
                traj, current_position, meanvarmarker_map, initial_heading_velocity, batch_weights = batch
                total_variance_condition = None
            else:
                traj, current_position, meanvarmarker_map, batch_weights = batch
                initial_heading_velocity = None
                total_variance_condition = None

            model_device = next(model.parameters()).device
            traj = traj.to(model_device, non_blocking=True)
            current_position = current_position.to(model_device, non_blocking=True)
            meanvarmarker_map = meanvarmarker_map.to(model_device, non_blocking=True)
            if initial_heading_velocity is not None:
                initial_heading_velocity = initial_heading_velocity.to(model_device, non_blocking=True)
            if total_variance_condition is not None:
                total_variance_condition = total_variance_condition.to(model_device, non_blocking=True)
            batch_weights = batch_weights.to(model_device, non_blocking=True)

            batch_size_actual = traj.shape[0]
            t = torch.randint(0, trainmod.T, (batch_size_actual,), device=traj.device).long()
            loss = trainmod.get_loss(
                model,
                traj,
                t,
                meanvarmarker_map,
                current_position,
                initial_heading_velocity,
                total_variance_condition,
                weights=batch_weights,
            )
            last_loss = loss.item()
            loss_vals.append(last_loss)
            batch_weight_sum = batch_weights.sum().item()
            epoch_loss_sum += last_loss * batch_weight_sum
            epoch_weight_sum += batch_weight_sum

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainmod.GRAD_CLIP_NORM)
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {last_loss:.4f}", flush=True)

        last_epoch_train_loss = epoch_loss_sum / max(epoch_weight_sum, 1e-6)
        epoch_train_loss_vals.append(last_epoch_train_loss)
        epoch_steps.append(epoch * len(dataloader))

        checkpoint_due = (epoch % args.save_every == 0) or (epoch == args.target_epochs)
        eval_due = eval_env is not None and (epoch % args.eval_every == 0)

        if checkpoint_due or eval_due:
            out_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": last_loss,
                "epoch_train_loss": last_epoch_train_loss,
            }
            out_path = trainmod.CHECKPOINT_DIR / f"sparse_trans_waypoints_epoch_{epoch}.pth"
            torch.save(out_checkpoint, out_path)
            print(f"Checkpoint saved: {out_path}", flush=True)

        if eval_due:
            eval_start_time = time.time()
            eval_metrics = trainmod.run_single_map_eval(model, eval_env)
            eval_elapsed = time.time() - eval_start_time
            eval_epochs.append(epoch)
            eval_global_rmse_drops.append(eval_metrics["global_rmse_drop"])
            eval_occupied_rmse_drops.append(eval_metrics["occupied_rmse_drop"])
            eval_variance_drops.append(eval_metrics["variance_drop"])
            print(
                f"Epoch {epoch} single-map eval (map {eval_env['selected_map']}, "
                f"{eval_elapsed:.1f}s): global_rmse_drop={eval_metrics['global_rmse_drop']:.4f}, "
                f"occupied_rmse_drop={eval_metrics['occupied_rmse_drop']:.4f}, "
                f"variance_drop={eval_metrics['variance_drop']:.4f}",
                flush=True,
            )
            np.savetxt(
                eval_metrics_path,
                np.column_stack(
                    [eval_epochs, eval_global_rmse_drops, eval_occupied_rmse_drops, eval_variance_drops]
                ),
                delimiter=",",
                header="epoch,global_rmse_drop,occupied_rmse_drop,variance_drop",
                comments="",
            )
            model.train()

        val_loss = trainmod.evaluate(model, val_dataloader)
        val_loss_vals.append(val_loss)
        val_steps.append(epoch * len(dataloader))
        print(f"Epoch {epoch} held-out validation loss: {val_loss:.6f}", flush=True)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} complete, train_loss={last_epoch_train_loss:.6f}, lr={current_lr:.6f}", flush=True)

    print("Resumed training loop complete", flush=True)

    # Distinct filenames for the raw loss-curve plots (this job's segment
    # only - the original run's own step-level loss history isn't persisted
    # anywhere this script can recover, so a merged plot isn't possible here).
    suffix = f"_resumed_{start_epoch}_to_{args.target_epochs}"

    if loss_vals:
        window = min(200, len(loss_vals))
        plt.figure()
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / window
            moving_avg = np.convolve(np.asarray(loss_vals, dtype=np.float32), kernel, mode="valid")
            moving_avg_steps = stepcount[window - 1:]
            plt.plot(stepcount, loss_vals, alpha=0.18, label="Raw Training Loss")
            plt.plot(moving_avg_steps, moving_avg, label=f"{window}-Step Moving Average")
        else:
            plt.plot(stepcount, loss_vals, label="Training Loss")
        plt.xlabel("Step (this job's segment)")
        plt.ylabel("Loss")
        plt.title(f"SparseTrans Diffusion Training Loss (epochs {start_epoch}-{args.target_epochs})")
        plt.legend()
        loss_plot_path = trainmod.PLOT_DIR / f"sparse_trans_training_loss{suffix}.png"
        plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved training loss plot to {loss_plot_path}", flush=True)

    if val_loss_vals:
        plt.figure()
        plt.plot(val_steps, val_loss_vals, marker="o", label="Held-Out Validation Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"SparseTrans Diffusion Held-Out Validation Loss (epochs {start_epoch}-{args.target_epochs})")
        plt.legend()
        val_loss_plot_path = trainmod.PLOT_DIR / f"sparse_trans_validation_loss{suffix}.png"
        plt.savefig(val_loss_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved validation loss plot to {val_loss_plot_path}", flush=True)

    # Periodic-eval plots DO use the full merged (prior + this job's) history,
    # so overwriting these fixed filenames each time is correct.
    if eval_epochs:
        plt.figure()
        plt.plot(eval_epochs, eval_global_rmse_drops, marker="o", label="Global RMSE drop")
        plt.plot(eval_epochs, eval_occupied_rmse_drops, marker="o", label="Occupied RMSE drop")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE drop (initial - final)")
        plt.title(f"Periodic single-map eval - RMSE drop (map {eval_map})")
        plt.legend()
        rmse_drop_plot_path = trainmod.PLOT_DIR / "periodic_eval_rmse_drop.png"
        plt.savefig(rmse_drop_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved periodic eval RMSE drop plot to {rmse_drop_plot_path}", flush=True)

        plt.figure()
        plt.plot(eval_epochs, eval_variance_drops, marker="o", color="tab:green", label="Important-area variance drop")
        plt.xlabel("Epoch")
        plt.ylabel(f"Variance drop in cells <= {trainmod.EVAL_UTILITY_THRESHOLD} (initial - final)")
        plt.title(f"Periodic single-map eval - Important-area variance drop (map {eval_map})")
        plt.legend()
        variance_drop_plot_path = trainmod.PLOT_DIR / "periodic_eval_variance_drop.png"
        plt.savefig(variance_drop_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved periodic eval variance drop plot to {variance_drop_plot_path}", flush=True)

    trainmod.sample_plot_traj(trainmod.PLOT_DIR / f"sparse_trans_training_sample{suffix}.png")

    print("Done. Final checkpoint:", trainmod.CHECKPOINT_DIR / f"sparse_trans_waypoints_epoch_{args.target_epochs}.pth", flush=True)


if __name__ == "__main__":
    main()
