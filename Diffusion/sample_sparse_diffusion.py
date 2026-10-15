import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch

import SparseDiffusion as diffusion
from SparseDiffusion import pytorch_cubic_spline

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "sparse_waypoints_epoch_2400.pth"
)
CONDITION_INDEX = 77
TRUTH_INDEX = 77
SEED = None
PLOT_DIR = Path(__file__).resolve().parent / "plots"
OUTPUT_PATH = PLOT_DIR / "sparse_sample.png"
NUM_STEPS = None
CLIP_X0 = False


def load_model(checkpoint_path, device):
    model = diffusion.NoisePredictor().to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    state_dict = diffusion.remap_legacy_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion.model = model
    return model


@torch.no_grad()
def sample_sparse(model, condition_index=0, seed=None, num_steps=None, clip_x0=False):
    if seed is not None:
        torch.manual_seed(seed)

    device = diffusion.device
    initial_noise = torch.randn((1, *diffusion.TARGET_SHAPE), device=device)
    meanvarmarker_map = diffusion.meanvarmarkermaps[condition_index:condition_index + 1].to(device)
    # SparseDiffusion.conditions is already normalized the same way as waypoints.
    current_position = diffusion.conditions[condition_index:condition_index + 1].to(device)
    return diffusion.ddim_sample(
        initial_noise,
        meanvarmarker_map,
        current_position,
        num_steps=num_steps,
        clip_x0=clip_x0,
    )


def plot_sample(sample, truth_index, condition_index, output_path=None):
    import matplotlib.pyplot as plt

    sampled_sparse = diffusion.extract_control_waypoints(sample[0].cpu())
    truth_sparse = diffusion.extract_control_waypoints(diffusion.trajectories[truth_index].cpu())
    mean_map = (
        diffusion.means[condition_index].detach().cpu() * diffusion.mean_scale.cpu()
        + diffusion.mean_center.cpu()
    )
    current_position = (
        (diffusion.conditions[condition_index].detach().cpu() + 1.0)
        / 2.0
        * diffusion.SCALE_FACTOR
    )
    sampled_sparse = pytorch_cubic_spline(sampled_sparse, current_position=current_position)[0]
    truth_sparse = pytorch_cubic_spline(truth_sparse, current_position=current_position)[0]

    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.imshow(
        mean_map,
        extent=(0, diffusion.SCALE_FACTOR, 0, diffusion.SCALE_FACTOR),
        origin="lower",
        cmap="viridis",
        alpha=0.65,
    )
    plt.colorbar(label="Mean")
    plt.grid(True, color="white", alpha=0.25)
    plt.plot(
        sampled_sparse[0],
        sampled_sparse[1],
        marker="o",
        linewidth=2,
        label="Generated control waypoints",
    )
    plt.plot(
        truth_sparse[0],
        truth_sparse[1],
        marker="x",
        linewidth=2,
        label="Ground truth control waypoints",
    )
    plt.scatter(truth_sparse[0, -1], truth_sparse[1, -1], s=70, marker="s", label="GT last waypoint")
    plt.xlim(0, diffusion.SCALE_FACTOR)
    plt.ylim(0, diffusion.SCALE_FACTOR)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"SparseDiffusion DDIM sample vs ground truth ({truth_index})")
    plt.legend()

    if output_path is None:
        output_path = OUTPUT_PATH

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--condition-index", type=int, default=CONDITION_INDEX)
    parser.add_argument("--truth-index", type=int, default=TRUTH_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--clip-x0", action="store_true", default=CLIP_X0)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Sampling with conditions from trajectory {args.condition_index}")
    model = load_model(args.checkpoint, diffusion.device)
    sample = sample_sparse(
        model,
        condition_index=args.condition_index,
        seed=args.seed,
        num_steps=args.num_steps,
        clip_x0=args.clip_x0,
    )
    plot_sample(
        sample,
        truth_index=args.truth_index,
        condition_index=args.condition_index,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
