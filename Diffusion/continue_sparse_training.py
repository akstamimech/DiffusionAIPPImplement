import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import SparseDiffusion as diffusion


SCRIPT_DIR = Path(__file__).resolve().parent


def make_dataloader(dataset, batch_size, shuffle=True):
    requested_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if requested_workers <= 0:
        requested_workers = 2 if torch.cuda.is_available() else 0
    num_workers = min(requested_workers, 4) if torch.cuda.is_available() else 0

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)


def resolve_checkpoint_path(path_text):
    path = Path(path_text)
    if path.is_file():
        return path

    checkpoint_path = diffusion.CHECKPOINT_DIR / path_text
    if checkpoint_path.is_file():
        return checkpoint_path

    script_path = SCRIPT_DIR / path_text
    if script_path.is_file():
        return script_path

    raise FileNotFoundError(f"Could not find checkpoint: {path_text}")


def configure_optimizer_param_groups(optimizer, lr, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        param_group["weight_decay"] = weight_decay


def load_checkpoint(model, optimizer, checkpoint_path, reset_optimizer=False):
    checkpoint = torch.load(checkpoint_path, map_location=diffusion.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(diffusion.remap_legacy_state_dict_keys(checkpoint["model_state_dict"]))
        if "optimizer_state_dict" in checkpoint and not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif reset_optimizer:
            print("Resetting optimizer state; only model weights were loaded.", flush=True)
        start_epoch = int(checkpoint.get("epoch", 0))
        previous_loss = checkpoint.get("loss")
    else:
        model.load_state_dict(diffusion.remap_legacy_state_dict_keys(checkpoint))
        start_epoch = 0
        previous_loss = None
        print("Loaded model weights only; optimizer state and epoch were not present.")

    return start_epoch, previous_loss


def train_from_checkpoint(
    checkpoint_path,
    target_epochs,
    batch_size,
    lr,
    weight_decay,
    save_every,
    reset_optimizer,
):
    model = diffusion.NoisePredictor().to(diffusion.device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch, previous_loss = load_checkpoint(
        model,
        optimizer,
        checkpoint_path,
        reset_optimizer=reset_optimizer,
    )
    configure_optimizer_param_groups(optimizer, lr=lr, weight_decay=weight_decay)
    diffusion.model = model

    if target_epochs <= start_epoch:
        raise ValueError(
            f"target_epochs={target_epochs} must be greater than checkpoint epoch {start_epoch}"
        )

    train_mask, val_mask, val_map_ids = diffusion.build_map_id_split(val_count=2)
    print(f"Validation map_ids: {val_map_ids.tolist()}", flush=True)
    print(f"Training samples: {int(train_mask.sum().item())}", flush=True)
    print(f"Validation samples: {int(val_mask.sum().item())}", flush=True)

    train_dataset = diffusion.TrajectoryDataset(
        diffusion.trajectories[train_mask],
        diffusion.weights[train_mask],
        meanvarmarkermaps=diffusion.meanvarmarkermaps[train_mask],
        conditions=diffusion.conditions[train_mask],
    )
    val_dataset = diffusion.TrajectoryDataset(
        diffusion.trajectories[val_mask],
        diffusion.weights[val_mask],
        meanvarmarkermaps=diffusion.meanvarmarkermaps[val_mask],
        conditions=diffusion.conditions[val_mask],
    )
    dataloader = make_dataloader(train_dataset, batch_size, shuffle=True)
    val_dataloader = make_dataloader(val_dataset, batch_size, shuffle=False)

    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"Checkpoint epoch: {start_epoch}", flush=True)
    if previous_loss is not None:
        print(f"Checkpoint loss: {previous_loss}", flush=True)
    print(f"Continuing to epoch: {target_epochs}", flush=True)
    print(f"Using device: {diffusion.device}", flush=True)
    print(f"Using lr: {lr}", flush=True)
    print(f"Using weight_decay: {weight_decay}", flush=True)
    print(f"Reset optimizer: {reset_optimizer}", flush=True)

    model.train()
    loss_vals = []
    stepcount = []
    val_loss_vals = []
    val_steps = []

    for epoch in range(start_epoch, target_epochs):
        print(f"Epoch {epoch + 1}/{target_epochs}", flush=True)
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            stepcount.append(epoch * len(dataloader) + step)
            traj, current_position, meanvarmarker_map, batch_weights = batch
            model_device = next(model.parameters()).device
            traj = traj.to(model_device, non_blocking=True)
            current_position = current_position.to(model_device, non_blocking=True)
            meanvarmarker_map = meanvarmarker_map.to(model_device, non_blocking=True)
            batch_weights = batch_weights.to(model_device, non_blocking=True)

            t = torch.randint(0, diffusion.T, (traj.shape[0],), device=traj.device).long()
            loss = diffusion.get_loss(
                model,
                traj,
                t,
                meanvarmarker_map,
                current_position,
                weights=batch_weights,
            )
            loss_vals.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}", flush=True)

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            output_path = diffusion.CHECKPOINT_DIR / f"sparse_waypoints_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, output_path)
            print(f"Checkpoint saved: {output_path}", flush=True)

        val_loss = diffusion.evaluate(model, val_dataloader)
        val_loss_vals.append(val_loss)
        val_steps.append((epoch + 1) * len(dataloader))
        print(f"Epoch {epoch + 1} held-out validation loss: {val_loss:.6f}", flush=True)

        print(f"Epoch {epoch + 1} complete", flush=True)

    if loss_vals:
        loss_plot_path = diffusion.PLOT_DIR / f"sparse_continue_loss_epoch_{target_epochs}.png"
        window = min(200, len(loss_vals))
        plt.figure()
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / window
            moving_avg = np.convolve(np.asarray(loss_vals, dtype=np.float32), kernel, mode="valid")
            plt.plot(stepcount, loss_vals, alpha=0.18, label="Raw continued training loss")
            plt.plot(stepcount[window - 1:], moving_avg, label=f"{window}-step moving average")
        else:
            plt.plot(stepcount, loss_vals, label="Continued training loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Sparse diffusion continued training loss")
        plt.legend()
        plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved continued loss plot to {loss_plot_path}", flush=True)

    if val_loss_vals:
        val_plot_path = diffusion.PLOT_DIR / f"sparse_continue_validation_loss_epoch_{target_epochs}.png"
        plt.figure()
        plt.plot(val_steps, val_loss_vals, marker="o", label="Held-out validation loss")
        plt.xlabel("Training Step")
        plt.ylabel("Validation Loss")
        plt.title("Sparse diffusion continued validation loss")
        plt.legend()
        plt.savefig(val_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved continued validation loss plot to {val_plot_path}", flush=True)

    final_sample_path = diffusion.PLOT_DIR / f"sparse_continue_sample_epoch_{target_epochs}.png"
    diffusion.sample_plot_traj(final_sample_path)
    print(f"Saved continued sample plot to {final_sample_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Continue SparseDiffusion training from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path. Can be absolute, relative to this script, or a filename in checkpoints/.",
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=diffusion.EPOCHS,
        help="Final epoch number to train to, not additional epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=diffusion.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=diffusion.WEIGHT_DECAY)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model weights from the checkpoint but start a fresh AdamW optimizer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_from_checkpoint(
        checkpoint_path=resolve_checkpoint_path(args.checkpoint),
        target_epochs=args.target_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        reset_optimizer=args.reset_optimizer,
    )
