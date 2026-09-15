"""
Fine-grained full-grid search around the best fit from grid_search.py.

Same style as grid_search.py - every combination of all four params is
built and measured - but zoomed into a local neighborhood: starting from
the single best (alpha_cc, alpha_Ia, g_cc, g_ratio) combination found by
the exhaustive grid_search.py run, each param is crossed with the other
three over +/-0.2 around its best-fit value (21 points, 0.02 apart).
21^4 = 194,481 models total - about 27x the coarse grid_search.py run.
"""
import itertools
import os

import pandas as pd

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines, compute_reduced_chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COARSE_RESULTS_PATH = os.path.join(SCRIPT_DIR, "grid_search_output", "grid_search_results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "grid_search_fine_output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "grid_search_fine_results.csv")

PARAMS = ["alpha_cc", "alpha_Ia", "g_cc", "g_ratio"]
STEP = 0.02

# 21 points, 0.02 apart, centered on the best-fit value: that's
# best-0.2, best-0.18, ..., best, ..., best+0.18, best+0.2.
NUM_POINTS = 21


def get_best_fit():
    """Read grid_search.py's results and return its single best row."""
    coarse_df = pd.read_csv(COARSE_RESULTS_PATH)
    coarse_df = coarse_df.sort_values(by="reduced_chi2", ascending=True)
    return coarse_df.iloc[0]


def get_param_grids(best_values):
    """For each param, its 21 candidate values (+/-0.2 around best, 0.02 apart)."""
    grids = {}
    for param in PARAMS:
        center = best_values[param]
        grids[param] = iterative_1D_method.centered_array(center, STEP, NUM_POINTS)
    return grids


def run_one_model(alpha_cc, alpha_Ia, g_cc, g_ratio, lines_df):
    """Build one model and return its reduced chi^2 against the data."""
    model_df = iterative_1D_method.build_model_df(alpha_cc, alpha_Ia, g_cc, g_ratio)
    return compute_reduced_chi2(model_df, lines_df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines_df = load_mn_fe_lines()

    best_row = get_best_fit()
    best_values = {
        "alpha_cc": best_row["alpha_cc"],
        "alpha_Ia": best_row["alpha_Ia"],
        "g_cc": best_row["g_cc"],
        "g_ratio": best_row["g_ratio"],
    }
    print(f"Best fit from grid_search.py: {best_values}")

    param_grids = get_param_grids(best_values)
    grid = list(itertools.product(
        param_grids["alpha_cc"], param_grids["alpha_Ia"],
        param_grids["g_cc"], param_grids["g_ratio"],
    ))
    n_total = len(grid)
    print(f"Running {n_total} models...", flush=True)

    records = []
    for i, (alpha_cc, alpha_Ia, g_cc, g_ratio) in enumerate(grid):
        reduced_chi2 = run_one_model(alpha_cc, alpha_Ia, g_cc, g_ratio, lines_df)

        records.append({
            "alpha_cc": alpha_cc,
            "alpha_Ia": alpha_Ia,
            "g_cc": g_cc,
            "g_ratio": g_ratio,
            "reduced_chi2": reduced_chi2,
        })

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{n_total} done", flush=True)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(by="reduced_chi2", ascending=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(results_df)} results to {OUTPUT_PATH}")
    print("Best 10:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
