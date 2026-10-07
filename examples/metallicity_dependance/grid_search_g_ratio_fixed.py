"""
Same staged full-grid search as grid_search_pipeline.py, except g_ratio is
held FIXED (not searched) - only alpha_cc, alpha_Ia, g_cc are optimized.
Answers: if g_ratio is pinned at some value, how do the other three params
want to settle?

Three stages, each a full 3D cross product over alpha_cc/alpha_Ia/g_cc,
each gated on confirming the previous stage's best fit is a genuine local
minimum before moving on (same convergence check as grid_search_pipeline.py,
just over three params instead of four - g_ratio isn't being optimized, so
there's nothing to check for it):

  Stage 1 (coarse, step=0.1): alpha_cc/alpha_Ia/g_cc in [0.1, 0.9].
  9^3 = 729 models.

  Stage 2 (fine, step=0.02): all three +/-0.2 around Stage 1's best
  (21 points each). 21^3 = 9,261 models.

  Stage 3 (finest, step=0.01): all three +/-0.04 around Stage 2's best
  (9 points each). 9^3 = 729 models.

g_ratio, Upsilon, and yields_ref are all command-line options:

    python grid_search_g_ratio_fixed.py --g-ratio 2.0 --yields-ref "W2024,moreFe10" --upsilon 1.0

If --output-dir isn't given, it's derived from --g-ratio/--yields-ref/--upsilon,
so different-settings runs don't overwrite each other.
"""
import argparse
import itertools
import os
import shutil

import numpy as np
import pandas as pd

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines, compute_reduced_chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PARAMS = ["alpha_cc", "alpha_Ia", "g_cc"]

# Stage 1: coarse, fixed absolute range (not centered on anything - this is
# the starting sweep over the full plausible range). Same as
# grid_search_pipeline.py's Stage 1, minus g_ratio.
STAGE1_RANGES = {
    "alpha_cc": np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2),
    "alpha_Ia": np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2),
    "g_cc": np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2),
}

# Stage 2: fine, centered on Stage 1's best, +/-0.2 for every param.
STAGE2_STEP = 0.02
STAGE2_NUM_POINTS = 21

# Stage 3: finest, centered on Stage 2's best, +/-0.04 for every param.
STAGE3_STEP = 0.01
STAGE3_NUM_POINTS = 9


def run_full_grid(param_values, lines_df, output_path, g_ratio, Upsilon, yields_ref):
    """
    Build + measure every combination in param_values (a dict mapping
    alpha_cc/alpha_Ia/g_cc to an array of candidate values), with g_ratio
    held fixed at the given value for every model, using the given
    Upsilon and yields_ref. Writes the results (sorted best-first) to
    output_path, and returns that DataFrame.
    """
    value_lists = [param_values[param] for param in PARAMS]
    grid = list(itertools.product(*value_lists))
    n_total = len(grid)
    print(f"Running {n_total} models (g_ratio fixed at {g_ratio}, "
          f"Upsilon={Upsilon}, yields_ref={yields_ref})...", flush=True)

    records = []
    for i, combo_values in enumerate(grid):
        combo = dict(zip(PARAMS, combo_values))

        model_df = iterative_1D_method.build_model_df(
            alpha_cc=combo["alpha_cc"], alpha_Ia=combo["alpha_Ia"],
            g_cc=combo["g_cc"], g_ratio=g_ratio,
            Upsilon=Upsilon, yields_ref=yields_ref,
        )
        combo["g_ratio"] = g_ratio
        combo["reduced_chi2"] = compute_reduced_chi2(model_df, lines_df)
        combo["Upsilon"] = Upsilon
        combo["yields_ref"] = yields_ref
        records.append(combo)

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{n_total} done", flush=True)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(by="reduced_chi2", ascending=True).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Wrote {n_total} results to {output_path}")

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
    For each free param (alpha_cc/alpha_Ia/g_cc), confirm the best-fit
    value is a local minimum along that axis, within this stage's own
    grid: not at either edge of the range that was searched, and its
    chi^2 (holding the other params at their best-fit values) is <= both
    its immediate neighbors'.

    Prints one line per param explaining the result. Returns True only if
    every param passes.
    """
    all_converged = True

    for param in PARAMS:
        values = np.sort(param_values[param])
        best_value = best_values[param]
        best_index = np.where(np.isclose(values, best_value))[0][0]

        if best_index == 0 or best_index == len(values) - 1:
            print(f"  {param}: best value {best_value:.4g} is at the edge of "
                  f"this stage's grid - can't confirm it's a minimum")
            all_converged = False
            continue

        slice_df = slice_through_best(grid_df, param, best_values)
        chi2_by_value = dict(zip(np.round(slice_df[param], 6), slice_df["reduced_chi2"]))

        chi2_best = chi2_by_value[round(float(values[best_index]), 6)]
        chi2_left = chi2_by_value[round(float(values[best_index - 1]), 6)]
        chi2_right = chi2_by_value[round(float(values[best_index + 1]), 6)]

        if chi2_best <= chi2_left and chi2_best <= chi2_right:
            print(f"  {param}: OK, local minimum at {best_value:.4g} "
                  f"(chi2={chi2_best:.6g}, neighbors={chi2_left:.6g}/{chi2_right:.6g})")
        else:
            print(f"  {param}: NOT a local minimum at {best_value:.4g} "
                  f"(chi2={chi2_best:.6g}, neighbors={chi2_left:.6g}/{chi2_right:.6g})")
            all_converged = False

    return all_converged


def sanitize_for_path(text):
    """Make a string safe to use as part of a directory name."""
    return text.replace(",", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Staged full-grid search over alpha_cc/alpha_Ia/g_cc with g_ratio held fixed."
    )
    parser.add_argument(
        "--g-ratio", type=float, default=2.0,
        help="g_ratio value to hold fixed (not searched). Default: 2.0",
    )
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
        help="Where to write output. Default: grid_search_g_ratio_fixed_output_g<g_ratio>_<yields_ref>_U<upsilon>, "
             "next to this script, so different settings don't overwrite each other.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        dir_name = (
            f"grid_search_g_ratio_fixed_output_g{args.g_ratio:g}"
            f"_{sanitize_for_path(args.yields_ref)}_U{args.upsilon:g}"
        )
        output_root = os.path.join(SCRIPT_DIR, dir_name)
    print(f"g_ratio fixed at {args.g_ratio}, Upsilon={args.upsilon}, "
          f"yields_ref={args.yields_ref!r}, output_dir={output_root}")

    # Start from a clean output directory every run. Without this, a run
    # that only reaches Stage 1 this time (e.g. after changing a grid
    # range) would leave an older run's Stage 2/3 results sitting there
    # unchanged - stale, but still picked up by anything that reads this
    # directory later.
    shutil.rmtree(output_root, ignore_errors=True)

    lines_df = load_mn_fe_lines()

    # --- Stage 1: coarse (step=0.1) ---
    print("=== Stage 1: coarse grid (step=0.1) ===")
    stage1_path = os.path.join(output_root, "stage1_coarse", "results.csv")
    stage1_df = run_full_grid(STAGE1_RANGES, lines_df, stage1_path, args.g_ratio, args.upsilon, args.yields_ref)
    stage1_best = get_best_values(stage1_df)
    print("Stage 1 best:", stage1_best)

    print("Checking convergence...")
    if not check_converged(stage1_df, STAGE1_RANGES, stage1_best):
        print("NOT CONVERGED with grid size 0.1")
        return
    print("Converged.")

    # --- Stage 2: fine (step=0.02) ---
    print("\n=== Stage 2: fine grid (step=0.02) ===")
    stage2_ranges = {}
    for param in PARAMS:
        stage2_ranges[param] = iterative_1D_method.centered_array(
            stage1_best[param], STAGE2_STEP, STAGE2_NUM_POINTS
        )

    stage2_path = os.path.join(output_root, "stage2_fine", "results.csv")
    stage2_df = run_full_grid(stage2_ranges, lines_df, stage2_path, args.g_ratio, args.upsilon, args.yields_ref)
    stage2_best = get_best_values(stage2_df)
    print("Stage 2 best:", stage2_best)

    print("Checking convergence...")
    if not check_converged(stage2_df, stage2_ranges, stage2_best):
        print("NOT CONVERGED with grid size 0.02")
        return
    print("Converged.")

    # --- Stage 3: finest (step=0.01) ---
    print("\n=== Stage 3: finest grid (step=0.01) ===")
    stage3_ranges = {}
    for param in PARAMS:
        stage3_ranges[param] = iterative_1D_method.centered_array(
            stage2_best[param], STAGE3_STEP, STAGE3_NUM_POINTS
        )

    stage3_path = os.path.join(output_root, "stage3_finest", "results.csv")
    stage3_df = run_full_grid(stage3_ranges, lines_df, stage3_path, args.g_ratio, args.upsilon, args.yields_ref)
    stage3_best = get_best_values(stage3_df)
    print("Stage 3 best:", stage3_best)

    print("Checking convergence...")
    if check_converged(stage3_df, stage3_ranges, stage3_best):
        print("Converged.")
    else:
        print("NOT CONVERGED with grid size 0.01")

    print(f"\nFinal best fit (g_ratio fixed at {args.g_ratio}):", stage3_best)


if __name__ == "__main__":
    main()
