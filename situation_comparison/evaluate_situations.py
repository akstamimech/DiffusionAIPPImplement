"""
Run all three planners on one or more candidate situations and print scores.
Writes preview PNGs to tmp_previews/ (gitignored) so you can eyeball a
candidate before deciding whether it's worth keeping - nothing here touches
the curated plots/ folder or results_index.csv. That only happens via
save_situation.py, deliberately, per situation.

Usage: python evaluate_situations.py <map_id> <row1> [row2] [row3] ...
"""
import sys

from situation_lib import reconstruct_full_state_at_row, run_three_planners, plot_situation, THIS_DIR

PREVIEW_DIR = THIS_DIR / "tmp_previews"


def main():
    selected_map = int(sys.argv[1])
    rows = [int(x) for x in sys.argv[2:]]

    results = []
    for row in rows:
        state = reconstruct_full_state_at_row(selected_map, row)
        print(f"\n=== map{selected_map} row={row} wall_time={state['wall_time']:.1f}s pose={tuple(state['pose'])} ===")
        result = run_three_planners(state)
        diff_ratio = result["u_diff"] / result["u_cmaes"]
        imit_ratio = result["u_imit"] / result["u_cmaes"]
        print(f"  avg utility along path: CMA-ES={result['u_cmaes']:.5f} Diffusion={result['u_diff']:.5f} ImitateTrans={result['u_imit']:.5f}")
        print(f"  ratio-to-CMAES: Diffusion={diff_ratio:.2f} ImitateTrans={imit_ratio:.2f}")

        out_png = PREVIEW_DIR / f"map{selected_map}_row{row}_preview.png"
        plot_situation(selected_map, state, result, out_png)
        print(f"  preview saved (NOT curated): {out_png}")

        results.append(dict(row=row, wall_time=state["wall_time"], diff_ratio=diff_ratio, imit_ratio=imit_ratio, png=str(out_png)))

    print("\n\n=== SUMMARY (sorted by diff_ratio - imit_ratio, i.e. Diffusion-fine-ImitateTrans-suffers first) ===")
    for r in sorted(results, key=lambda r: r["diff_ratio"] - r["imit_ratio"], reverse=True):
        print(
            f"  row={r['row']:>3} wall={r['wall_time']:>6.1f}s  "
            f"diff/cmaes={r['diff_ratio']:.2f}  imit/cmaes={r['imit_ratio']:.2f}  {r['png']}"
        )
    print(
        "\nIf any of these are worth keeping, run: "
        f"python save_situation.py {selected_map} <row>"
    )


if __name__ == "__main__":
    main()
