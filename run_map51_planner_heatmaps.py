import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussianprocesstraining import fov_lateral_radius, initialize_gp


SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "Vizualization"
CSV_DIR = SCRIPT_DIR / "csv"


PLANNERS = [
    {
        "key": "cmaes",
        "label": "CMA-ES expert",
        "script": SCRIPT_DIR / "CMAES_classic_singlemap.py",
        "output_prefix": "classic_map",
    },
    {
        "key": "diffusion",
        "label": "Diffusion planner",
        "script": SCRIPT_DIR / "Diffusionplanner_singlemap.py",
        "output_prefix": "diffusion_map",
        "extra_env": {
            "ETA": "1.0",
            "RUN_SEED": "17051",
            "DIFFUSION_DATASET_PATH": str(
                SCRIPT_DIR / "CMAES_beamsearch_dataset_3d_synthetic_final.pt"
            ),
        },
    },
    {
        "key": "imitation",
        "label": "ImitateTrans",
        "script": SCRIPT_DIR / "ImitateTrans_singlemap.py",
        "output_prefix": "imitate_trans_map",
    },
]


SUMMARY_KEYWORDS = (
    "Initial total variance",
    "Final total variance",
    "Variance reduction",
    "Task Completion",
    "Global RMSE",
    "Occupied RMSE",
    "flight ended early",
)


def output_dir_for(planner, maptype, map_id, run_tag):
    stem = f"{planner['output_prefix']}_{maptype}_{map_id}_viz"
    if run_tag:
        stem = f"{stem}_{run_tag}"
    return VIZ_DIR / stem


def run_planner(planner, args):
    output_dir = output_dir_for(planner, args.maptype, args.map_id, args.run_tag)
    env = os.environ.copy()
    env.update(
        {
            "SELECTED_MAP": str(args.map_id),
            "MAPTYPE": args.maptype,
            "TIMEALLOTED": str(args.timealloted),
            "WALLCLOCK_SECONDS": str(args.wallclock_seconds),
            "EXECUTION_CHUNK": str(args.execution_chunk),
            "UTILITY_THRESHOLD": str(args.utility_threshold),
            "SENSORNOISE_SEED": str(args.sensornoise_seed),
            "ENFORCE_MIN_STEP_TIME": "1",
            "SKIP_VIZ": "1",
            "RUN_OUTPUT_TAG": args.run_tag,
        }
    )
    env.update(planner.get("extra_env", {}))

    log_path = output_dir / f"map_{args.map_id}_{planner['key']}_stdout.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{planner['key']}] running {planner['script'].name} "
        f"for map {args.map_id}, {args.wallclock_seconds:g}s wall clock, "
        f"{args.timealloted} timestep cap"
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-u", str(planner["script"])],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        summary_lines = []
        for line in process.stdout:
            log_file.write(line)
            if any(keyword in line for keyword in SUMMARY_KEYWORDS):
                print(line.rstrip())
                summary_lines.append(line.rstrip())
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{planner['script'].name} failed with exit code {return_code}; see {log_path}"
        )

    return output_dir, summary_lines


def load_grid(map_id, maptype):
    map_path = CSV_DIR / f"map_{map_id}_{maptype}_grid_counts.csv"
    data = np.loadtxt(map_path, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    value = data[:, 2]
    xs = np.unique(x)
    ys = np.unique(y)
    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    for px, py, pv in zip(x, y, value):
        grid[yi[py], xi[px]] = pv
    return map_path, xs, ys, grid


def read_trace(output_dir, map_id):
    path = output_dir / f"map_{map_id}_executed_trajectory.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return path, data


def fov_heatmap(trace, xs, ys, angle_of_view=60.0):
    heat = np.zeros((len(ys), len(xs)), dtype=float)
    for row in trace:
        radius = fov_lateral_radius(float(row["z"]), angle_of_view)
        x_mask = (xs >= float(row["x"]) - radius) & (xs <= float(row["x"]) + radius)
        y_mask = (ys >= float(row["y"]) - radius) & (ys <= float(row["y"]) + radius)
        heat[np.ix_(y_mask, x_mask)] += 1.0
    return heat


def read_metrics_trace(output_dir, map_id):
    path = output_dir / f"map_{map_id}_rmse_over_time.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return path, data


def save_heatmap_csv(path, xs, ys, heat):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "fov_time_spent_steps"])
        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                writer.writerow([x, y, int(heat[row_idx, col_idx])])


def save_value_exposure_csv(path, planners, heatmaps, ground_truth):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner", "environment_magnitude", "fov_time_spent_steps"])
        values = ground_truth.ravel()
        for planner in planners:
            heat = heatmaps[planner["key"]].ravel()
            for value, count in zip(values, heat):
                writer.writerow([planner["key"], float(value), int(count)])


def binned_exposure(values, counts, bins):
    bin_ids = np.digitize(values, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.full(len(centers), np.nan)
    totals = np.zeros(len(centers), dtype=float)
    cell_counts = np.zeros(len(centers), dtype=float)
    for idx in range(len(centers)):
        mask = bin_ids == idx
        if np.any(mask):
            cell_counts[idx] = float(np.count_nonzero(mask))
            means[idx] = float(np.mean(counts[mask]))
            totals[idx] = float(np.sum(counts[mask]))
    return centers, means, totals, cell_counts


def make_plot(args, run_outputs):
    map_path, xs, ys, ground_truth = load_grid(args.map_id, args.maptype)
    _gp, _x_test, _mean, cov, _xs_gp, _ys_gp, _x_grid, _y_grid, *_bounds = initialize_gp()
    initial_total_variance = float(np.sum(np.diag(cov)))
    comparison_dir = VIZ_DIR / "map51_planner_heatmaps"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    heatmaps = {}
    metric_traces = {}
    rows = []
    max_count = 1.0
    for planner, output_dir, summary_lines in run_outputs:
        trajectory_path, trace = read_trace(output_dir, args.map_id)
        metrics_path, metrics_trace = read_metrics_trace(output_dir, args.map_id)
        heat = fov_heatmap(trace, xs, ys)
        heatmaps[planner["key"]] = heat
        metric_traces[planner["key"]] = metrics_trace
        max_count = max(max_count, float(np.max(heat)))
        heat_csv = comparison_dir / f"map_{args.map_id}_{planner['key']}_fov_time_spent_heatmap.csv"
        save_heatmap_csv(heat_csv, xs, ys, heat)
        final_variance = float(metrics_trace["global_variance"][-1])
        rows.append(
            {
                "planner": planner["key"],
                "label": planner["label"],
                "steps_recorded": int(len(trace)),
                "final_timestep": int(trace["timestep"][-1]),
                "final_wall_time_seconds": float(trace["wall_time_seconds"][-1]),
                "initial_global_variance": initial_total_variance,
                "final_global_variance": final_variance,
                "variance_reduction": initial_total_variance - final_variance,
                "unique_fov_cells_seen": int(np.count_nonzero(heat)),
                "total_fov_cell_visits": int(np.sum(heat)),
                "max_fov_time_spent_at_one_cell_steps": int(np.max(heat)),
                "trajectory_csv": str(trajectory_path),
                "metrics_csv": str(metrics_path),
                "heatmap_csv": str(heat_csv),
                "output_dir": str(output_dir),
            }
        )

    summary_csv = comparison_dir / f"map_{args.map_id}_planner_heatmap_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    exposure_csv = comparison_dir / f"map_{args.map_id}_environment_value_vs_fov_time.csv"
    save_value_exposure_csv(
        exposure_csv,
        [planner for planner, _output_dir, _summary_lines in run_outputs],
        heatmaps,
        ground_truth,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 11.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    map_min = float(np.nanmin(ground_truth))
    map_max = float(np.nanmax(ground_truth))
    important_mask = np.ma.masked_where(ground_truth <= args.important_threshold, ground_truth)
    axes_flat[0].imshow(
        ground_truth,
        origin="lower",
        extent=extent,
        cmap="Greys",
        vmin=map_min,
        vmax=map_max,
        aspect="equal",
    )
    axes_flat[0].imshow(
        important_mask,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=args.important_threshold,
        vmax=map_max,
        alpha=0.65,
        aspect="equal",
    )
    axes_flat[0].contour(
        xs,
        ys,
        ground_truth,
        levels=[args.important_threshold],
        colors=["tab:red"],
        linewidths=1.2,
    )
    axes_flat[0].set_title(f"Ground truth > {args.important_threshold:g}")
    axes_flat[0].set_xlabel("x")
    axes_flat[0].set_ylabel("y")
    axes_flat[0].grid(False)

    im = None
    for ax, (planner, _output_dir, _summary_lines) in zip(axes_flat[1:], run_outputs):
        ax.imshow(
            ground_truth,
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=map_min,
            vmax=map_max,
            alpha=0.55,
            aspect="equal",
        )
        heat = np.ma.masked_where(heatmaps[planner["key"]] <= 0, heatmaps[planner["key"]])
        im = ax.imshow(
            heat,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=1,
            vmax=max_count,
            alpha=0.82,
            aspect="equal",
        )
        ax.set_title(planner["label"])
        ax.set_xlabel("x")
        ax.grid(False)
    axes_flat[2].set_ylabel("y")
    assert im is not None
    cbar = fig.colorbar(im, ax=axes_flat[1:], fraction=0.035, pad=0.035)
    cbar.set_label("FOV time spent at cell (simulation steps)")
    fig.suptitle(
        (
            f"Map {args.map_id} GRF sensor-FOV time-spent heatmaps "
            f"({args.wallclock_seconds:g}s wall clock, {args.timealloted} timestep cap)"
        ),
        fontsize=14,
    )
    plot_path = comparison_dir / f"map_{args.map_id}_planner_fov_time_spent_heatmaps.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    metrics_plot_path = comparison_dir / f"map_{args.map_id}_planner_fov_value_metrics.png"
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    flat_values = ground_truth.ravel()
    value_min = float(np.nanmin(flat_values))
    value_max = float(np.nanmax(flat_values))
    bins = np.linspace(value_min, value_max, 16)
    colors = {
        "cmaes": "tab:blue",
        "diffusion": "tab:orange",
        "imitation": "tab:green",
    }
    for planner, _output_dir, _summary_lines in run_outputs:
        heat = heatmaps[planner["key"]].ravel()
        centers, means, _totals, cell_counts = binned_exposure(flat_values, heat, bins)
        density = np.divide(
            means,
            cell_counts,
            out=np.full_like(means, np.nan),
            where=cell_counts > 0,
        )
        axes[0].plot(
            centers,
            density,
            marker="o",
            linewidth=2,
            label=planner["label"],
            color=colors.get(planner["key"]),
        )
    axes[0].set_xlabel("Environment magnitude")
    axes[0].set_ylabel("Mean FOV time / cells in bin")
    axes[0].set_title("FOV exposure density by map magnitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    labels = [row["label"] for row in rows]
    final_variances = [row["final_global_variance"] for row in rows]
    step_counts = [row["final_timestep"] for row in rows]
    unique_fov_cells = [row["unique_fov_cells_seen"] for row in rows]
    x = np.arange(len(rows))
    axes[1].bar(x, final_variances, color=[colors.get(row["planner"]) for row in rows])
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Final total variance")
    axes[1].set_title("Final GP uncertainty")
    axes[1].grid(True, axis="y", alpha=0.25)

    width = 0.38
    axes[2].bar(x - width / 2, step_counts, width=width, label="Executed steps")
    axes[2].bar(x + width / 2, unique_fov_cells, width=width, label="Unique FOV cells")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Steps and FOV coverage")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(loc="best")
    fig.suptitle(f"Map {args.map_id} GRF FOV exposure metrics", fontsize=14)
    fig.savefig(metrics_plot_path, dpi=180)
    plt.close(fig)

    print(f"Read map: {map_path}")
    print(f"Wrote heatmap plot: {plot_path}")
    print(f"Wrote metrics plot: {metrics_plot_path}")
    print(f"Wrote heatmap summary: {summary_csv}")
    print(f"Wrote value exposure CSV: {exposure_csv}")
    return plot_path, summary_csv, rows, heatmaps


def run_or_reuse_map(args, map_id):
    run_outputs = []
    for planner in PLANNERS:
        output_dir = output_dir_for(planner, args.maptype, map_id, args.run_tag)
        trajectory_path = output_dir / f"map_{map_id}_executed_trajectory.csv"
        metrics_path = output_dir / f"map_{map_id}_rmse_over_time.csv"
        if args.skip_run or (
            args.reuse_existing and trajectory_path.exists() and metrics_path.exists()
        ):
            if args.reuse_existing and trajectory_path.exists() and metrics_path.exists():
                print(f"[map {map_id} {planner['key']}] reusing existing outputs")
            run_outputs.append((planner, output_dir, []))
        else:
            original_map_id = args.map_id
            args.map_id = map_id
            try:
                output_dir, summary_lines = run_planner(planner, args)
            finally:
                args.map_id = original_map_id
            run_outputs.append((planner, output_dir, summary_lines))

    original_map_id = args.map_id
    args.map_id = map_id
    try:
        _plot_path, _summary_csv, rows, heatmaps = make_plot(args, run_outputs)
    finally:
        args.map_id = original_map_id
    return rows, heatmaps


def make_density_records(args, map_id, heatmaps):
    _map_path, _xs, _ys, ground_truth = load_grid(map_id, args.maptype)
    flat_values = ground_truth.ravel()
    bins = np.linspace(float(np.nanmin(flat_values)), float(np.nanmax(flat_values)), 16)
    records = []
    for planner in PLANNERS:
        heat = heatmaps[planner["key"]].ravel()
        centers, means, _totals, cell_counts = binned_exposure(flat_values, heat, bins)
        density = np.divide(
            means,
            cell_counts,
            out=np.full_like(means, np.nan),
            where=cell_counts > 0,
        )
        records.append(
            {
                "map_id": map_id,
                "planner": planner["key"],
                "centers": centers,
                "density": density,
            }
        )
    return records


def make_aggregate_plots(args, all_rows, density_records):
    comparison_dir = VIZ_DIR / "map51_planner_heatmaps"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    planners = [planner["key"] for planner in PLANNERS]
    labels = {planner["key"]: planner["label"] for planner in PLANNERS}
    colors = {
        "cmaes": "tab:blue",
        "diffusion": "tab:orange",
        "imitation": "tab:green",
    }

    aggregate_csv = (
        comparison_dir
        / f"maps_{min(args.maps)}_{max(args.maps)}_planner_fov_summary.csv"
    )
    with aggregate_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    for planner_key in planners:
        planner_records = [r for r in density_records if r["planner"] == planner_key]
        centers = planner_records[0]["centers"]
        densities = np.vstack([r["density"] for r in planner_records])
        mean_density = np.nanmean(densities, axis=0)
        lower = np.nanpercentile(densities, 25, axis=0)
        upper = np.nanpercentile(densities, 75, axis=0)
        axes[0].plot(
            centers,
            mean_density,
            marker="o",
            linewidth=2,
            label=labels[planner_key],
            color=colors.get(planner_key),
        )
        axes[0].fill_between(
            centers,
            lower,
            upper,
            color=colors.get(planner_key),
            alpha=0.15,
            linewidth=0,
        )
    axes[0].set_xlabel("Environment magnitude")
    axes[0].set_ylabel("Mean FOV time / cells in bin")
    axes[0].set_title("Mean FOV exposure density")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    x = np.arange(len(planners))
    mean_rows = []
    final_variance_means = []
    final_variance_stds = []
    step_means = []
    step_stds = []
    fov_cell_means = []
    fov_cell_stds = []
    for planner_key in planners:
        rows = [row for row in all_rows if row["planner"] == planner_key]
        final_variances = np.array([float(row["final_global_variance"]) for row in rows])
        variance_reductions = np.array([float(row["variance_reduction"]) for row in rows])
        steps = np.array([float(row["final_timestep"]) for row in rows])
        fov_cells = np.array([float(row["unique_fov_cells_seen"]) for row in rows])
        total_fov_visits = np.array([float(row["total_fov_cell_visits"]) for row in rows])
        max_revisits = np.array(
            [float(row["max_fov_time_spent_at_one_cell_steps"]) for row in rows]
        )
        final_variance_means.append(float(np.mean(final_variances)))
        final_variance_stds.append(float(np.std(final_variances)))
        step_means.append(float(np.mean(steps)))
        step_stds.append(float(np.std(steps)))
        fov_cell_means.append(float(np.mean(fov_cells)))
        fov_cell_stds.append(float(np.std(fov_cells)))
        mean_rows.append(
            {
                "planner": planner_key,
                "maps": len(rows),
                "mean_final_global_variance": float(np.mean(final_variances)),
                "std_final_global_variance": float(np.std(final_variances)),
                "mean_variance_reduction": float(np.mean(variance_reductions)),
                "mean_final_timestep": float(np.mean(steps)),
                "std_final_timestep": float(np.std(steps)),
                "mean_unique_fov_cells_seen": float(np.mean(fov_cells)),
                "std_unique_fov_cells_seen": float(np.std(fov_cells)),
                "mean_total_fov_cell_visits": float(np.mean(total_fov_visits)),
                "mean_max_fov_time_spent_at_one_cell_steps": float(np.mean(max_revisits)),
            }
        )

    axes[1].bar(
        x,
        final_variance_means,
        yerr=final_variance_stds,
        capsize=4,
        color=[colors.get(key) for key in planners],
    )
    axes[1].set_xticks(x, [labels[key] for key in planners], rotation=20, ha="right")
    axes[1].set_ylabel("Final total variance")
    axes[1].set_title("Final GP uncertainty")
    axes[1].grid(True, axis="y", alpha=0.25)

    width = 0.38
    axes[2].bar(
        x - width / 2,
        step_means,
        yerr=step_stds,
        capsize=4,
        width=width,
        label="Executed steps",
    )
    axes[2].bar(
        x + width / 2,
        fov_cell_means,
        yerr=fov_cell_stds,
        capsize=4,
        width=width,
        label="Unique FOV cells",
    )
    axes[2].set_xticks(x, [labels[key] for key in planners], rotation=20, ha="right")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Steps and FOV coverage")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(loc="best")

    fig.suptitle(
        (
            f"GRF maps {min(args.maps)}-{max(args.maps)} "
            f"FOV exposure metrics averaged over {len(args.maps)} maps"
        ),
        fontsize=14,
    )
    aggregate_plot = (
        comparison_dir
        / f"maps_{min(args.maps)}_{max(args.maps)}_planner_fov_value_metrics_mean.png"
    )
    fig.savefig(aggregate_plot, dpi=180)
    plt.close(fig)

    aggregate_means_csv = (
        comparison_dir
        / f"maps_{min(args.maps)}_{max(args.maps)}_planner_fov_metric_means.csv"
    )
    with aggregate_means_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mean_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mean_rows)

    print(f"Wrote aggregate plot: {aggregate_plot}")
    print(f"Wrote aggregate rows: {aggregate_csv}")
    print(f"Wrote aggregate means: {aggregate_means_csv}")
    return aggregate_plot, aggregate_csv, aggregate_means_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", type=int, default=51)
    parser.add_argument("--maps", type=int, nargs="+")
    parser.add_argument("--maptype", default="NAIP")
    parser.add_argument("--timealloted", type=int, default=3000)
    parser.add_argument("--wallclock-seconds", type=float, default=300.0)
    parser.add_argument("--execution-chunk", type=int, default=20)
    parser.add_argument("--utility-threshold", type=float, default=0.3)
    parser.add_argument("--sensornoise-seed", type=int, default=123)
    parser.add_argument("--run-tag", default="mission300_steps3000")
    parser.add_argument("--important-threshold", type=float, default=0.3)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Reuse existing trajectory CSVs and regenerate only the heatmap outputs.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse any existing trajectory CSVs and run only missing planners.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.maps:
        all_rows = []
        density_records = []
        for map_id in args.maps:
            rows, heatmaps = run_or_reuse_map(args, map_id)
            all_rows.extend(rows)
            density_records.extend(make_density_records(args, map_id, heatmaps))
        make_aggregate_plots(args, all_rows, density_records)
    else:
        run_or_reuse_map(args, args.map_id)


if __name__ == "__main__":
    main()
