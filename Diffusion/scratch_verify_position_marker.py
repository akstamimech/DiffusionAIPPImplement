"""
Verifies threeDSparseTransDiffusion.py's make_position_marker_maps: for a
handful of known (x, y) test positions spanning the domain, does the one-hot
marker light up the grid cell that actually corresponds to that real-world
position? Copies the function body directly (not imported) since the module
runs dataset-loading code at import time.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

XY_SCALE = 100.0
GRID_SIZE = 51
STEP = 2.0  # matches gaussianprocesstraining.py's step; xs/ys = arange(0, 100+eps, 2) -> 51 points


def make_position_marker_maps(positions, grid_size=51, marker_radius=2):
    # Exact copy of threeDSparseTransDiffusion.py's implementation.
    B = positions.shape[0]
    marker = torch.zeros((B, 1, grid_size, grid_size), dtype=torch.float32, device=positions.device)

    grid_pos = (positions / XY_SCALE) * (grid_size - 1)
    grid_pos = grid_pos.round().long().clamp(0, grid_size - 1)

    x_idx = grid_pos[:, 0]
    y_idx = grid_pos[:, 1]

    marker[torch.arange(B), 0, y_idx, x_idx] = 1.0
    return marker


test_positions = [
    ("origin (0, 0)", 0.0, 0.0),
    ("top-right corner (100, 100)", 100.0, 100.0),
    ("center (50, 50)", 50.0, 50.0),
    ("bottom-right (100, 0)", 100.0, 0.0),
    ("top-left (0, 100)", 0.0, 100.0),
    ("off-grid real value (38.0, 68.0)", 38.0, 68.0),
]

positions_tensor = torch.tensor([[x, y] for _, x, y in test_positions], dtype=torch.float32)
markers = make_position_marker_maps(positions_tensor, grid_size=GRID_SIZE)

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.ravel()

for i, (label, x, y) in enumerate(test_positions):
    ax = axes[i]
    marker_np = markers[i, 0].numpy()
    lit_rows, lit_cols = np.nonzero(marker_np)
    lit_row, lit_col = int(lit_rows[0]), int(lit_cols[0])
    # grid index -> real-world coordinate it represents, for the printed check
    recovered_x = lit_col * STEP
    recovered_y = lit_row * STEP

    im = ax.imshow(
        marker_np, origin="lower", extent=[0, XY_SCALE, 0, XY_SCALE],
        cmap="viridis", vmin=0, vmax=1, aspect="equal",
    )
    ax.scatter([x], [y], s=140, facecolors="none", edgecolors="red", linewidths=2, label="true (x, y)")
    ax.set_title(f"{label}\nlit cell -> ({recovered_x:.0f}, {recovered_y:.0f})", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)
    match = np.isclose(recovered_x, x, atol=STEP / 2 + 1e-6) and np.isclose(recovered_y, y, atol=STEP / 2 + 1e-6)
    print(f"{label}: input=({x}, {y}) -> lit grid cell (row={lit_row}, col={lit_col}) "
          f"-> recovered coord=({recovered_x}, {recovered_y}) -> {'OK' if match else 'MISMATCH'}")

fig.suptitle("make_position_marker_maps: red circle = true input (x,y), heatmap = one-hot marker output", fontsize=12)
fig.tight_layout()
out_path = "scratch_position_marker_check.png"
fig.savefig(out_path, dpi=150)
print(f"\nWrote {out_path}")
