"""
Staged full-grid search for the four metdep params.

Four stages, each a full 4D cross product (every combination of all four
params, not a coordinate-descent search), each gated on confirming the
previous stage's best fit is a genuine local minimum before moving on:

  Stage 1 (coarse, step=0.1): alpha_cc/alpha_Ia/g_cc in [0.1, 0.9],
  g_ratio in [1.8, 3.6] (shifted up from the original [1.0, 2.8] range,
  since the best fit kept landing on 2.8, the old range's edge).
  7,290 models.

  Stage 2 (medium, step=0.04): alpha_cc/alpha_Ia/g_cc +/-0.2 around
  Stage 1's best (11 points each). g_ratio's window is lopsided: just
  one grid step below Stage 1's best, +1.8 above it (47 points) - it's
  harder to constrain than the other three and consistently wants
  larger values. 11^3 * 47 = 62,557 models. This stage exists
  specifically to narrow g_ratio's wide search window down cheaply,
  before refining everything at step=0.02/0.01 - without it, keeping
  that wide a g_ratio window at step=0.02 is ~800k models (what this
  stage replaces).

  Stage 3 (fine, step=0.02): alpha_cc/alpha_Ia/g_cc +/-0.08 around
  Stage 2's best (9 points each). g_ratio again lopsided: one step
  below, +0.14 above Stage 2's best (9 points). 9^4 = 6,561 models.

  Stage 4 (finest, step=0.01): alpha_cc/alpha_Ia/g_cc +/-0.04 around
  Stage 3's best (9 points each). g_ratio lopsided: one step below,
  +0.07 above Stage 3's best (9 points). 9^4 = 6,561 models.

Convergence check (between stages): for each param, using that stage's own
already-computed grid, take the 1D slice through the best fit (other three
params held at their best-fit values) and confirm the best value's chi^2 is
<= both its immediate neighbors' - and that the best value isn't sitting at
either edge of the range that was searched (if it is, we can't tell whether
it's really a minimum or whether the true optimum lies outside the window).
If every other param is fine and only g_ratio is sitting on an edge, the
script pushes that edge out (further in whichever direction it was
reaching) and retries the stage automatically, reusing the rows it already
computed, up to 3 times - this used to require noticing the failure in the
log and manually rerunning with a wider range. If some other param fails
to converge, or g_ratio is still at an edge after 3 expansions, the stage
is NOT converged, the script prints why, and stops - it does not move on
to the next stage.

This is expensive by design (full grids at every stage, not a cheap
coordinate-descent search) - meant to run on a supercomputer, not
interactively.

Upsilon and yields_ref (see chemevo.evolution_V1's yields presets, e.g.
"W2024,moreFe10") are command-line options, so you can run this multiple
times with different yield assumptions without the outputs overwriting
each other:

    python grid_search_pipeline.py --yields-ref "W2024,moreFe10" --upsilon 1.0
    python grid_search_pipeline.py --yields-ref "W2024,moreFe30" --upsilon 1.2658 --output-dir my_run_name

If --output-dir isn't given, it's derived from --yields-ref/--upsilon, so
two different-settings runs land in different folders automatically.

By default, each run starts from a clean output directory (wiping
whatever was there before). If a run gets killed partway through (e.g. a
supercomputer walltime limit) and you want to continue it rather than
start over, rerun with the same arguments plus --resume: this skips the
directory wipe and picks up each stage's results.csv from wherever it
left off (each model's result is saved to disk as soon as it's computed,
not just at the end of a stage, so nothing finished is ever lost).

    python grid_search_pipeline.py --yields-ref "W2024,moreFe10" --upsilon 1.0 --resume
"""
import argparse
import contextlib
import csv
import itertools
import os
import shutil
import sys

import numpy as np
import pandas as pd

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines, compute_reduced_chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PARAMS = ["alpha_cc", "alpha_Ia", "g_cc", "g_ratio"]

# Stage 1: coarse, fixed absolute range (not centered on anything - this is
# the starting sweep over the full plausible range). Same as grid_search.py.
STAGE1_RANGES = {
    "alpha_cc": np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2),
    "alpha_Ia": np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2),
    "g_cc": np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2),
    "g_ratio": np.round(np.arange(1.8, 3.6 + 0.001, 0.2), 2),
}

# Stage 2: medium, centered on Stage 1's best. Exists to narrow down
# g_ratio's wide window cheaply (at step=0.04) before refining everything
# further at step=0.02/0.01.
STAGE2_STEP = 0.04
STAGE2_NUM_POINTS = 11  # +/-0.2, for alpha_cc/alpha_Ia/g_cc

# Stage 3: fine, centered on Stage 2's best, +/-0.08 for alpha_cc/alpha_Ia/g_cc.
STAGE3_STEP = 0.02
STAGE3_NUM_POINTS = 9

# Stage 4: finest, centered on Stage 3's best, +/-0.04 for alpha_cc/alpha_Ia/g_cc.
STAGE4_STEP = 0.01
STAGE4_NUM_POINTS = 9

# g_ratio is harder to constrain than the other three and consistently
# wants larger values. Stage 1 keeps its fixed absolute range (above),
# but at every later stage its window is lopsided around the previous
# stage's best: just one grid step below center, several steps above
# it. If the best fit still lands on an edge of that window (Stage 1
# included), the edge it landed on is pushed out by G_RATIO_*_EXPAND and
# the stage is retried - reusing already-computed rows instead of
# recomputing them (same mechanism as --resume) - up to
# MAX_G_RATIO_EXPANSIONS times. This is what used to require rerunning
# the whole stage by hand with a wider range.
MAX_G_RATIO_EXPANSIONS = 3

G_RATIO_STAGE1_EXPAND = 0.6  # STAGE1_RANGES's own g_ratio step is 0.2

G_RATIO_STAGE2_BELOW = STAGE2_STEP
G_RATIO_STAGE2_ABOVE = 1.8
G_RATIO_STAGE2_EXPAND = 1.8

G_RATIO_STAGE3_BELOW = STAGE3_STEP
G_RATIO_STAGE3_ABOVE = 0.14
G_RATIO_STAGE3_EXPAND = 0.14

G_RATIO_STAGE4_BELOW = STAGE4_STEP
G_RATIO_STAGE4_ABOVE = 0.07
G_RATIO_STAGE4_EXPAND = 0.07


@contextlib.contextmanager
def tee_stdout(log_path, mode="w"):
    """
    While this context is active, everything printed (stage progress,
    convergence results, the final best fit) goes to both the terminal and
    log_path - so it's still readable later, after the terminal/job log is
    gone. Restores normal printing when the context exits, even on error.

    mode="a" (used by --resume) appends to an existing log instead of
    overwriting it, so a resumed run's log picks up where the killed run's
    left off.
    """
    log_file = open(log_path, mode)
    original_stdout = sys.stdout

    class Tee:
        def write(self, text):
            original_stdout.write(text)
            log_file.write(text)

        def flush(self):
            original_stdout.flush()
            log_file.flush()

    sys.stdout = Tee()
    try:
        yield
    finally:
        sys.stdout = original_stdout
        log_file.close()


def get_already_done_combos(output_path):
    """
    If output_path already holds results from an earlier, interrupted
    attempt at this exact stage, return the set of (alpha_cc, alpha_Ia,
    g_cc, g_ratio) combinations it already has - rounded, so float noise
    doesn't cause a false mismatch - so run_full_grid can skip
    recomputing them. Returns an empty set if there's no file yet.
    """
    if not os.path.exists(output_path):
        return set()

    existing_df = pd.read_csv(output_path)
    done = set()
    for _, row in existing_df.iterrows():
        key = tuple(round(float(row[param]), 6) for param in PARAMS)
        done.add(key)
    return done


def run_full_grid(param_values, lines_df, output_path, Upsilon, yields_ref, resume=False):
    """
    Build + measure every combination in param_values (a dict mapping each
    param name to an array of candidate values), using the given Upsilon
    and yields_ref for every model.

    Each model's result is appended to output_path as soon as it's
    computed, instead of being held in memory until the whole stage
    finishes - so if the job is killed partway through (e.g. a
    supercomputer walltime limit), everything computed so far is already
    on disk.

    If resume=True and output_path already has results from an earlier
    attempt at this stage, those combinations are skipped instead of
    recomputed, and the new results are added to them.

    Returns the full results DataFrame (old + newly computed rows),
    sorted best-first - and leaves output_path holding that same sorted
    DataFrame.
    """
    value_lists = [param_values[param] for param in PARAMS]
    grid = list(itertools.product(*value_lists))
    n_total = len(grid)

    already_done = get_already_done_combos(output_path) if resume else set()
    if already_done:
        print(f"Resuming: {len(already_done)}/{n_total} models already done in "
              f"{output_path}, skipping those.", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = PARAMS + ["reduced_chi2", "Upsilon", "yields_ref"]
    append_mode = len(already_done) > 0
    output_file = open(output_path, "a" if append_mode else "w", newline="")
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    if not append_mode:
        writer.writeheader()

    n_to_run = n_total - len(already_done)
    print(f"Running {n_to_run} models (Upsilon={Upsilon}, yields_ref={yields_ref})...", flush=True)

    n_done = 0
    for combo_values in grid:
        combo = dict(zip(PARAMS, combo_values))
        key = tuple(round(float(combo[param]), 6) for param in PARAMS)
        if key in already_done:
            continue

        model_df = iterative_1D_method.build_model_df(
            alpha_cc=combo["alpha_cc"], alpha_Ia=combo["alpha_Ia"],
            g_cc=combo["g_cc"], g_ratio=combo["g_ratio"],
            Upsilon=Upsilon, yields_ref=yields_ref,
        )
        combo["reduced_chi2"] = compute_reduced_chi2(model_df, lines_df)
        combo["Upsilon"] = Upsilon
        combo["yields_ref"] = yields_ref

        writer.writerow(combo)
        output_file.flush()

        n_done += 1
        if n_done % 1000 == 0:
            print(f"{n_done}/{n_to_run} done this run ({len(already_done) + n_done}/{n_total} total)", flush=True)

    output_file.close()

    results_df = pd.read_csv(output_path)
    results_df = results_df.sort_values(by="reduced_chi2", ascending=True).reset_index(drop=True)
    results_df.to_csv(output_path, index=False)
    print(f"Wrote {len(results_df)} total results to {output_path}")

    return results_df


def get_best_values(results_df):
    best_row = results_df.iloc[0]
    return {param: best_row[param] for param in PARAMS}


def slice_through_best(df, param, best_values):
    """
    Rows of `df` where every param except `param` equals its best-fit
    value - i.e. a 1D slice through the grid, along `param`'s axis,
    passing through the best fit. Sorted by `param`.
    """
    other_params = [p for p in PARAMS if p != param]

    keep = np.ones(len(df), dtype=bool)
    for other in other_params:
        keep = keep & np.isclose(df[other], best_values[other])

    subset = df[keep].sort_values(by=param)
    return subset[[param, "reduced_chi2"]].reset_index(drop=True)


def check_converged(grid_df, param_values, best_values):
    """
    For each param, confirm the best-fit value is a local minimum along
    that axis, within this stage's own grid: not at either edge of the
    range that was searched, and its chi^2 (holding the other three
    params at their best-fit values) is <= both its immediate neighbors'.

    Prints one line per param explaining the result. Returns a dict
    mapping each param to one of:
      "ok"         - confirmed local minimum.
      "edge_low"   - best value is at the low edge of this stage's grid.
      "edge_high"  - best value is at the high edge of this stage's grid.
      "not_minimum" - not at an edge, but chi^2 isn't <= both neighbors.
    """
    status = {}

    for param in PARAMS:
        values = np.sort(param_values[param])
        best_value = best_values[param]
        best_index = np.where(np.isclose(values, best_value))[0][0]

        if best_index == 0:
            print(f"  {param}: best value {best_value:.4g} is at the low edge of "
                  f"this stage's grid - can't confirm it's a minimum")
            status[param] = "edge_low"
            continue
        if best_index == len(values) - 1:
            print(f"  {param}: best value {best_value:.4g} is at the high edge of "
                  f"this stage's grid - can't confirm it's a minimum")
            status[param] = "edge_high"
            continue

        slice_df = slice_through_best(grid_df, param, best_values)
        chi2_by_value = dict(zip(np.round(slice_df[param], 6), slice_df["reduced_chi2"]))

        chi2_best = chi2_by_value[round(float(values[best_index]), 6)]
        chi2_left = chi2_by_value[round(float(values[best_index - 1]), 6)]
        chi2_right = chi2_by_value[round(float(values[best_index + 1]), 6)]

        if chi2_best <= chi2_left and chi2_best <= chi2_right:
            print(f"  {param}: OK, local minimum at {best_value:.4g} "
                  f"(chi2={chi2_best:.6g}, neighbors={chi2_left:.6g}/{chi2_right:.6g})")
            status[param] = "ok"
        else:
            print(f"  {param}: NOT a local minimum at {best_value:.4g} "
                  f"(chi2={chi2_best:.6g}, neighbors={chi2_left:.6g}/{chi2_right:.6g})")
            status[param] = "not_minimum"

    return status


def g_ratio_initial_range(center, step, below_width, above_width):
    """
    g_ratio's starting window for a stage: from (center - below_width) to
    (center + above_width), in steps of `step`. Lopsided on purpose - see
    the G_RATIO_* constants above.
    """
    return np.round(np.arange(center - below_width, center + above_width + step / 2, step), 2)


def g_ratio_expanded_range(current_values, edge, step, expand_width):
    """
    Push one edge of g_ratio's current grid out by expand_width (same
    units as the values), keeping the other edge fixed. `edge` is "low"
    or "high" - whichever edge the best fit landed on.
    """
    values = np.sort(current_values)
    if edge == "low":
        new_low = values[0] - expand_width
        return np.round(np.arange(new_low, values[-1] + step / 2, step), 2)
    else:
        new_high = values[-1] + expand_width
        return np.round(np.arange(values[0], new_high + step / 2, step), 2)


def run_stage(stage_label, ranges, lines_df, stage_path, args, g_ratio_step, g_ratio_expand_width):
    """
    Run one stage's full grid and check convergence. If g_ratio alone is
    the problem - its best fit landed on an edge of this stage's window,
    and every other param is fine - push that edge out by
    g_ratio_expand_width and retry, up to MAX_G_RATIO_EXPANSIONS times,
    reusing already-computed rows instead of recomputing them (the same
    mechanism --resume uses). If anything else fails to converge, or
    g_ratio is still at an edge after all the retries, gives up.

    Returns (results_df, best_values, converged).
    """
    ranges = dict(ranges)

    for attempt in range(MAX_G_RATIO_EXPANSIONS + 1):
        resume = args.resume or attempt > 0
        results_df = run_full_grid(ranges, lines_df, stage_path, args.upsilon, args.yields_ref, resume=resume)
        best_values = get_best_values(results_df)
        print(f"{stage_label} best:", best_values)

        print("Checking convergence...")
        status = check_converged(results_df, ranges, best_values)

        if all(param_status == "ok" for param_status in status.values()):
            print("Converged.")
            return results_df, best_values, True

        failing_params = [param for param, param_status in status.items() if param_status != "ok"]
        g_ratio_only_at_edge = failing_params == ["g_ratio"] and status["g_ratio"] in ("edge_low", "edge_high")
        if not g_ratio_only_at_edge:
            return results_df, best_values, False

        if attempt == MAX_G_RATIO_EXPANSIONS:
            print(f"g_ratio is still at the edge after {MAX_G_RATIO_EXPANSIONS} expansions - giving up.")
            return results_df, best_values, False

        edge = "low" if status["g_ratio"] == "edge_low" else "high"
        print(f"g_ratio is at the {edge} edge of this stage's window - expanding it and retrying...")
        ranges["g_ratio"] = g_ratio_expanded_range(ranges["g_ratio"], edge, g_ratio_step, g_ratio_expand_width)


def sanitize_for_path(text):
    """Make a string safe to use as part of a directory name."""
    return text.replace(",", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Staged full-grid search for the four metdep params.")
    parser.add_argument(
        "--yields-ref", default="W2024,moreFe10",
        help="chemevo yields_ref preset (see evolution_V1.py's yields table). Default: W2024,moreFe",
    )
    parser.add_argument(
        "--upsilon", type=float, default=1.0,
        help="Upsilon (overall yield scale) to use. Default: 1.0",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write output. Default: grid_search_pipeline_output_<yields_ref>_U<upsilon>, "
             "next to this script, so different --yields-ref/--upsilon runs don't overwrite each other.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue a run that was killed partway through (e.g. hit a walltime limit), "
             "instead of starting over. Picks up each stage's results.csv where it left off "
             "and skips the output-directory-wipe that a fresh run does. Default: off (fresh start).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        dir_name = f"grid_search_pipeline_output_{sanitize_for_path(args.yields_ref)}_U{args.upsilon:g}"
        output_root = os.path.join(SCRIPT_DIR, dir_name)
    if not args.resume:
        # Start from a clean output directory every fresh run. Without
        # this, a run that only reaches Stage 1 this time (e.g. after
        # changing a grid range) would leave an older run's Stage 2/3
        # results sitting there unchanged - stale, but still picked up by
        # anything that reads this directory later. --resume skips this,
        # since the whole point is to keep what's already there.
        shutil.rmtree(output_root, ignore_errors=True)
    os.makedirs(output_root, exist_ok=True)

    log_path = os.path.join(output_root, "log.txt")
    log_mode = "a" if args.resume else "w"
    with tee_stdout(log_path, mode=log_mode):
        print(f"Upsilon={args.upsilon}, yields_ref={args.yields_ref!r}, "
              f"output_dir={output_root}, resume={args.resume}")

        lines_df = load_mn_fe_lines()

        # --- Stage 1: coarse (step=0.1) ---
        print("=== Stage 1: coarse grid (step=0.1) ===")
        stage1_path = os.path.join(output_root, "stage1_coarse", "results.csv")
        stage1_df, stage1_best, stage1_converged = run_stage(
            "Stage 1", STAGE1_RANGES, lines_df, stage1_path, args,
            g_ratio_step=0.2, g_ratio_expand_width=G_RATIO_STAGE1_EXPAND,
        )
        if not stage1_converged:
            print("NOT CONVERGED with grid size 0.1")
            return

        # --- Stage 2: medium (step=0.04, lopsided g_ratio window) ---
        print("\n=== Stage 2: medium grid (step=0.04) ===")
        stage2_ranges = {}
        for param in ["alpha_cc", "alpha_Ia", "g_cc"]:
            stage2_ranges[param] = iterative_1D_method.centered_array(
                stage1_best[param], STAGE2_STEP, STAGE2_NUM_POINTS
            )
        stage2_ranges["g_ratio"] = g_ratio_initial_range(
            stage1_best["g_ratio"], STAGE2_STEP, G_RATIO_STAGE2_BELOW, G_RATIO_STAGE2_ABOVE
        )

        stage2_path = os.path.join(output_root, "stage2_medium", "results.csv")
        stage2_df, stage2_best, stage2_converged = run_stage(
            "Stage 2", stage2_ranges, lines_df, stage2_path, args,
            g_ratio_step=STAGE2_STEP, g_ratio_expand_width=G_RATIO_STAGE2_EXPAND,
        )
        if not stage2_converged:
            print("NOT CONVERGED with grid size 0.04")
            return

        # --- Stage 3: fine (step=0.02, lopsided g_ratio window) ---
        print("\n=== Stage 3: fine grid (step=0.02) ===")
        stage3_ranges = {}
        for param in ["alpha_cc", "alpha_Ia", "g_cc"]:
            stage3_ranges[param] = iterative_1D_method.centered_array(
                stage2_best[param], STAGE3_STEP, STAGE3_NUM_POINTS
            )
        stage3_ranges["g_ratio"] = g_ratio_initial_range(
            stage2_best["g_ratio"], STAGE3_STEP, G_RATIO_STAGE3_BELOW, G_RATIO_STAGE3_ABOVE
        )

        stage3_path = os.path.join(output_root, "stage3_fine", "results.csv")
        stage3_df, stage3_best, stage3_converged = run_stage(
            "Stage 3", stage3_ranges, lines_df, stage3_path, args,
            g_ratio_step=STAGE3_STEP, g_ratio_expand_width=G_RATIO_STAGE3_EXPAND,
        )
        if not stage3_converged:
            print("NOT CONVERGED with grid size 0.02")
            return

        # --- Stage 4: finest (step=0.01, lopsided g_ratio window) ---
        print("\n=== Stage 4: finest grid (step=0.01) ===")
        stage4_ranges = {}
        for param in ["alpha_cc", "alpha_Ia", "g_cc"]:
            stage4_ranges[param] = iterative_1D_method.centered_array(
                stage3_best[param], STAGE4_STEP, STAGE4_NUM_POINTS
            )
        stage4_ranges["g_ratio"] = g_ratio_initial_range(
            stage3_best["g_ratio"], STAGE4_STEP, G_RATIO_STAGE4_BELOW, G_RATIO_STAGE4_ABOVE
        )

        stage4_path = os.path.join(output_root, "stage4_finest", "results.csv")
        stage4_df, stage4_best, stage4_converged = run_stage(
            "Stage 4", stage4_ranges, lines_df, stage4_path, args,
            g_ratio_step=STAGE4_STEP, g_ratio_expand_width=G_RATIO_STAGE4_EXPAND,
        )
        if stage4_converged:
            print("Converged.")
        else:
            print("NOT CONVERGED with grid size 0.01")

        print("\nFinal best fit:", stage4_best)


if __name__ == "__main__":
    main()
