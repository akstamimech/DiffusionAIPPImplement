import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

#max condition index + truth index = 5157
#validation row indices: 4809–5156
#5003 is nice


#PURELY DETERMINISTIC: ETA = 0
ETA = 1.0 


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_MODULE_CANDIDATES = (
    SCRIPT_DIR / "threeDSparseTransDiffusion.py",
)
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR
    / "checkpoints"
    / "sparse_trans_waypoints_epoch_1900_multimodal_3d.pth"
)
CONDITION_INDEX = 4822
TRUTH_INDEX = 4822
SEED = None
PLOT_DIR = SCRIPT_DIR / "plots"
OUTPUT_PATH = PLOT_DIR / "sparse_trans_3d_sample.png"
NUM_STEPS = None
CLIP_X0 = True


def load_training_module():
    training_module_path = next(
        (path for path in TRAINING_MODULE_CANDIDATES if path.is_file()),
        None,
    )
    if training_module_path is None:
        searched = ", ".join(str(path) for path in TRAINING_MODULE_CANDIDATES)
        raise FileNotFoundError(f"Could not find a 3D training module. Searched: {searched}")
    spec = importlib.util.spec_from_file_location(
        "sparse_trans_diffusion_3d",
        training_module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module: {training_module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diffusion = load_training_module()


def load_model(checkpoint_path, device):
    model = diffusion.NoisePredictor().to(device)
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    state_dict = diffusion.remap_legacy_state_dict_keys(state_dict)
    diffusion.load_model_state_dict_compatible(model, state_dict)
    model.eval()
    diffusion.model = model
    return model


@torch.no_grad()
def sample_sparse(model, condition_index=0, seed=None, num_steps=None, clip_x0=True, eta1 = ETA):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    diffusion.model = model
    device = next(model.parameters()).device
    initial_noise = torch.randn((1, *diffusion.TARGET_SHAPE), device=device)
    meanvarmarker_map = diffusion.meanvarmarkermaps[
        condition_index : condition_index + 1
    ].to(device)
    current_position = diffusion.conditions[
        condition_index : condition_index + 1
    ].to(device)
    initial_heading_velocity = diffusion.initial_heading_velocities[
        condition_index : condition_index + 1
    ].to(device)
    total_variance_condition = diffusion.total_variance_conditions[
        condition_index : condition_index + 1
    ].to(device)
    xprev, mean_theta, stateposteriorvariance = diffusion.ddim_sample(
        initial_noise,
        meanvarmarker_map,
        current_position,
        initial_heading_velocity,
        total_variance_condition,
        num_steps=num_steps,
        clip_x0=clip_x0,
        eta = eta1
    )

    
    return xprev


def _world_heading(condition_index):
    normalized = diffusion.initial_heading_velocities[condition_index].detach().cpu()
    scale = torch.tensor(
        [
            diffusion.XY_SCALE / 2.0,
            diffusion.XY_SCALE / 2.0,
            (diffusion.Z_MAX - diffusion.Z_MIN) / 2.0,
        ],
        dtype=normalized.dtype,
    )
    return normalized * scale


def build_sample_figure(sample, truth_index, condition_index):
    sampled_controls = diffusion.extract_control_waypoints(sample[0].detach().cpu())
    truth_controls = diffusion.extract_control_waypoints(
        diffusion.trajectories[truth_index].detach().cpu()
    )
    current_position = diffusion.denormalize_xyz(
        diffusion.conditions[condition_index].detach().cpu()
    )
    heading_velocity = _world_heading(condition_index)

    sampled_dense = diffusion.pytorch_cubic_spline(
        sampled_controls,
        current_position=current_position,
    )[0]
    truth_dense = diffusion.pytorch_cubic_spline(
        truth_controls,
        current_position=current_position,
    )[0]
    mean_map = (
        diffusion.means[condition_index].detach().cpu()
        * diffusion.mean_scale.detach().cpu()
        + diffusion.mean_center.detach().cpu()
    )

    figure = plt.figure(figsize=(17, 6), constrained_layout=True)
    axis_3d = figure.add_subplot(1, 3, 1, projection="3d")
    axis_xy = figure.add_subplot(1, 3, 2)
    axis_z = figure.add_subplot(1, 3, 3)

    axis_3d.plot(
        sampled_dense[0],
        sampled_dense[1],
        sampled_dense[2],
        linewidth=2,
        label="Generated",
    )
    axis_3d.plot(
        truth_dense[0],
        truth_dense[1],
        truth_dense[2],
        linewidth=2,
        label="Ground truth",
    )
    axis_3d.scatter(
        current_position[0],
        current_position[1],
        current_position[2],
        marker="*",
        s=110,
        color="black",
        label="Current position",
    )
    axis_3d.quiver(
        current_position[0],
        current_position[1],
        current_position[2],
        heading_velocity[0],
        heading_velocity[1],
        heading_velocity[2],
        color="black",
        arrow_length_ratio=0.2,
    )
    axis_3d.set_xlim(0.0, diffusion.XY_SCALE)
    axis_3d.set_ylim(0.0, diffusion.XY_SCALE)
    axis_3d.set_zlim(diffusion.Z_MIN, diffusion.Z_MAX)
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Altitude")
    axis_3d.set_title("3D trajectory")
    axis_3d.legend()

    image = axis_xy.imshow(
        mean_map,
        extent=(0.0, diffusion.XY_SCALE, 0.0, diffusion.XY_SCALE),
        origin="lower",
        cmap="viridis",
        alpha=0.65,
    )
    axis_xy.plot(
        sampled_dense[0],
        sampled_dense[1],
        linewidth=2,
        label="Generated",
    )
    axis_xy.plot(
        truth_dense[0],
        truth_dense[1],
        linewidth=2,
        label="Ground truth",
    )
    axis_xy.scatter(
        current_position[0],
        current_position[1],
        marker="*",
        s=110,
        color="black",
        label="Current position",
    )
    axis_xy.arrow(
        current_position[0].item(),
        current_position[1].item(),
        heading_velocity[0].item(),
        heading_velocity[1].item(),
        color="white",
        width=0.35,
        length_includes_head=True,
    )
    axis_xy.set_xlim(0.0, diffusion.XY_SCALE)
    axis_xy.set_ylim(0.0, diffusion.XY_SCALE)
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.set_xlabel("X")
    axis_xy.set_ylabel("Y")
    axis_xy.set_title("Top-down belief-map view")
    axis_xy.grid(True, color="white", alpha=0.25)
    axis_xy.legend()
    figure.colorbar(image, ax=axis_xy, label="Mean")

    axis_z.plot(sampled_dense[2], linewidth=2, label="Generated")
    axis_z.plot(truth_dense[2], linewidth=2, label="Ground truth")
    axis_z.axhline(
        current_position[2].item(),
        color="black",
        linestyle="--",
        label="Current altitude",
    )
    axis_z.set_ylim(diffusion.Z_MIN, diffusion.Z_MAX)
    axis_z.set_xlabel("Spline sample")
    axis_z.set_ylabel("Altitude")
    axis_z.set_title("Altitude profile")
    axis_z.grid(alpha=0.25)
    axis_z.legend()

    figure.suptitle(
        "3D SparseTransDiffusion DDIM sample vs ground truth "
        f"(condition={condition_index}, truth={truth_index})"
    )
    return figure


def plot_sample(sample, truth_index, condition_index, output_path=None):
    figure = build_sample_figure(
        sample,
        truth_index=truth_index,
        condition_index=condition_index,
    )
    output_path = Path(output_path or OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved sample plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--condition-index", type=int, default=CONDITION_INDEX)
    parser.add_argument("--truth-index", type=int, default=TRUTH_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--clip-x0", action=argparse.BooleanOptionalAction, default=CLIP_X0)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Sampling with conditions from trajectory {args.condition_index}")
    print(
        "Normalized initial heading: "
        f"{diffusion.initial_heading_velocities[args.condition_index].tolist()}"
    )
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
