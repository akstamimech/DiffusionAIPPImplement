import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"

GRF_MAP_ID = 181
NAIP_MAP_ID = 55


def load_grid(map_id, maptype):
    path = f"{CSV_DIR}\\map_{map_id}_{maptype}_grid_counts.csv"
    df = pd.read_csv(path)
    pivot = df.pivot(index="y", columns="x", values="count")
    return pivot.columns.values, pivot.index.values, pivot.values


grf_x, grf_y, grf_grid = load_grid(GRF_MAP_ID, "grf")
naip_x, naip_y, naip_grid = load_grid(NAIP_MAP_ID, "NAIP")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

im1 = ax1.imshow(
    grf_grid, origin="lower", cmap="YlGn",
    extent=[grf_x.min(), grf_x.max(), grf_y.min(), grf_y.max()],
    aspect="equal", vmin=0, vmax=1,
)
ax1.set_title(f"Synthetic GRF ground truth (map {GRF_MAP_ID})")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
fig.colorbar(im1, ax=ax1, label="field value", fraction=0.046, pad=0.04)

im2 = ax2.imshow(
    naip_grid, origin="lower", cmap="Greys",
    extent=[naip_x.min(), naip_x.max(), naip_y.min(), naip_y.max()],
    aspect="equal", vmin=0, vmax=1,
)
ax2.set_title(f"NAIP-derived ground truth (map {NAIP_MAP_ID})")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
fig.colorbar(im2, ax=ax2, label="field value", fraction=0.046, pad=0.04)

fig.suptitle("Ground-truth map comparison: synthetic GRF vs. NAIP-derived", fontsize=13)
fig.tight_layout()
out_path = "scratch_grf_naip_sidebyside.png"
fig.savefig(out_path, dpi=150)
print(f"GRF map id: {GRF_MAP_ID}, NAIP map id: {NAIP_MAP_ID}")
print(f"Wrote {out_path}")
