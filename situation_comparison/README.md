# Same-situation planner comparison

Freezes one real belief state `(mu, P, pose)` from an actual CMA-ES flight and
feeds it identically to CMA-ES, Diffusion, and ImitateTrans's existing
one-shot "situation in, plan out" functions, to compare what each proposes
when they can't blame different starting conditions. See `situation_lib.py`'s
module docstring for the full methodology and the reproducibility caveat
(Diffusion's own sampling noise is unseeded, so its exact numbers vary run to
run; the qualitative pattern doesn't).

## Workflow

1. **Get a real CMA-ES flight** on the target map, if you don't already have
   one: run `planners/cmaes/run.py` with `SELECTED_MAP=<n> SKIP_VIZ=1
   WALLCLOCK_SECONDS=150 RUN_OUTPUT_TAG=situation_search`. This writes
   `Vizualization/classic_map_<MAPTYPE>_<n>_viz_situation_search/`, which
   `situation_lib.py` reads from. `situation_lib.py`'s `MAPTYPE` /
   `UTILITY_THRESHOLD` / checkpoint defaults track this project's live
   settings as of 2026-08-17 (GRF, threshold 0.5) via
   `SITUATION_MAPTYPE`/`SITUATION_UTILITY_THRESHOLD`/etc. env vars if you
   need to override - **but re-verify against the live scripts first**, this
   project has flipped between NAIP and GRF settings multiple times across
   sessions. Results already in `results/` (map 55/56, the earlier NAIP
   batch) were generated under NAIP/LCB=True settings that were live at the
   time; results for other maps may use different settings - see each row's
   `note` and the map id itself to tell which regime a saved result used.
2. **Search for candidate situations**: `python search_clusters.py <map_id>`.
   Finds timesteps where the utility mask splits into >=2 spatially
   far-apart clusters (i.e. genuinely requires cross-map travel), using a
   cached, diagonal-covariance-only replay (`belief_cache/`, regenerable,
   gitignored).
3. **Evaluate candidates**: `python evaluate_situations.py <map_id> <row1>
   <row2> ...`. Runs all three planners for real, prints each one's average
   utility sampled along its proposed path (as a fraction of CMA-ES's own),
   and writes preview PNGs to `tmp_previews/` (gitignored). Nothing here is
   curated or committed.
4. **Save the ones worth keeping**: `python save_situation.py <map_id> <row>
   "optional note"`. Only this step writes to the durable, git-tracked tier:
   `results/map<M>_row<R>.npz` (utility grid + all three full paths +
   waypoints, so it can be re-plotted later without re-running any model),
   `plots/map<M>_row<R>.png`, and one row appended to `results_index.csv`.

## Why two tiers

Candidate situations get scored by the dozen while hunting for a specific
pattern; most don't show anything interesting. Only `save_situation.py`
promotes a situation to the curated tier, deliberately, one at a time - the
search/evaluate scripts never write there. `results_index.csv` is therefore a
log of situations a human judged worth keeping, not a log of everything tried.

## Score interpretation

`u_cmaes` / `u_diff` / `u_imit` = each planner's proposed dense path's
average utility (masked posterior variance, i.e. `importance_filter` output),
sampled along the path. `diff_ratio` / `imit_ratio` = that value divided by
CMA-ES's own, so 1.0 means "matches CMA-ES's target quality," well below 1.0
means "wandered through much lower-value territory than CMA-ES would have."
This is a cheap proxy for "did the plan actually aim at the same worthwhile
territory," not a full trajectory-similarity metric.
