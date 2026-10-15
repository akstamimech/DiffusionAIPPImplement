"""
normalize_multiblob_maps.py

Rescale the raw multiblob grid-count maps to the [0, 1] range so they sit on a clean, bounded
value scale (raw multiblob counts span ~[0, 446], ~400-600x larger than NAIP). This keeps the
simulator's altitude noise model, the utility_threshold cutoff, and the diffusion model's
mean/var normalization operating on a sane scale instead of raw counts.

Normalization: a single GLOBAL min-max over the ENTIRE multiblob set,

    v_norm = (v_raw - global_min) / (global_max - global_min)   in [0, 1]

GLOBAL (one min/max for all maps), not per-map, on purpose: the utility threshold and the noise
model act on ABSOLUTE values, so a per-map [0,1] would make a given normalized value mean a
different physical level on every map and break that semantics. Global min-max maps the dataset's
overall [min, max] -> [0, 1] and keeps values comparable across maps while preserving each map's
relative amplitude. Only VALUES are rescaled; the spatial length-scale (17 vs NAIP's 4.79) is a
genuine, scale-invariant property and is deliberately left unchanged.

Reads  csv/map_*_multiblob_grid_counts.csv
Writes csv/map_*_multiblob_normalized_grid_counts.csv   (use as MAPTYPE="multiblob_normalized")
Originals are left untouched.
"""
import glob
import os

import numpy as np
import pandas as pd

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
VALUE_COL = "count"


def aggregate_stats(files):
    """Streaming aggregate min/max/mean/std of the value column over many files (memory-safe)."""
    n = 0
    total = 0.0
    total_sq = 0.0
    vmin = np.inf
    vmax = -np.inf
    for f in files:
        v = pd.read_csv(f, usecols=[VALUE_COL])[VALUE_COL].to_numpy(dtype=np.float64)
        n += v.size
        total += v.sum()
        total_sq += np.square(v).sum()
        vmin = min(vmin, float(v.min()))
        vmax = max(vmax, float(v.max()))
    mean = total / n
    std = np.sqrt(max(total_sq / n - mean * mean, 0.0))
    return {"min": vmin, "max": vmax, "mean": mean, "std": std, "frac_lt_0.3": None, "n": n}


def frac_below(files, thresh=0.3, mn=0.0, rng=1.0):
    """Fraction of (normalized) cells below `thresh`, computing the normalization on the fly."""
    n = 0
    below = 0
    for f in files:
        v = pd.read_csv(f, usecols=[VALUE_COL])[VALUE_COL].to_numpy(dtype=np.float64)
        vn = (v - mn) / rng
        n += vn.size
        below += int((vn < thresh).sum())
    return below / n


def main():
    naip_files = sorted(glob.glob(os.path.join(CSV_DIR, "map_*_NAIP_grid_counts.csv")))
    mb_files = sorted(glob.glob(os.path.join(CSV_DIR, "map_*_multiblob_grid_counts.csv")))
    mb_files = [f for f in mb_files if "normalized" not in os.path.basename(f)]  # defensive

    if not mb_files:
        raise SystemExit(f"No multiblob maps found in {CSV_DIR}")

    print(f"multiblob maps to normalize: {len(mb_files)}   (NAIP maps for reference: {len(naip_files)})")

    mb = aggregate_stats(mb_files)
    mn, mx = mb["min"], mb["max"]
    rng = mx - mn
    print(f"  multiblob RAW : min={mn:.3f} max={mx:.3f} mean={mb['mean']:.3f} std={mb['std']:.3f}")
    if naip_files:
        naip = aggregate_stats(naip_files)
        print(f"  NAIP (ref)    : min={naip['min']:.3f} max={naip['max']:.3f} "
              f"mean={naip['mean']:.3f} std={naip['std']:.3f}")
    print(f"  transform     : v_norm = (v - {mn:.3f}) / {rng:.3f}   -> [0, 1]")

    for i, f in enumerate(mb_files):
        df = pd.read_csv(f)
        df[VALUE_COL] = (df[VALUE_COL].to_numpy(dtype=np.float64) - mn) / rng
        out = f.replace("_multiblob_grid_counts.csv", "_multiblob_normalized_grid_counts.csv")
        df.to_csv(out, index=False, float_format="%.6g")
        if (i + 1) % 200 == 0:
            print(f"  ... wrote {i + 1}/{len(mb_files)}")

    print(f"Wrote {len(mb_files)} normalized maps to {CSV_DIR}")

    # Verify on a sample of written files + report where the 0.3 threshold now lands.
    written = sorted(glob.glob(os.path.join(CSV_DIR, "map_*_multiblob_normalized_grid_counts.csv")))
    ver = aggregate_stats(written[:50])
    frac = frac_below(mb_files[:50], thresh=0.3, mn=mn, rng=rng)
    print(f"Verify (50 normalized): min={ver['min']:.3f} max={ver['max']:.3f} "
          f"mean={ver['mean']:.3f} std={ver['std']:.3f}")
    print(f"  fraction of cells < 0.3 (occupied-threshold) after normalization: {frac:.3f}")


if __name__ == "__main__":
    main()
