"""
Fine-grained per-param sweep around the best fit from grid_search.py.

Starting from the single best (alpha_cc, alpha_Ia, g_cc, g_ratio)
combination found by the exhaustive grid_search.py run, this sweeps each of
the four params +/-0.2 around its best-fit value (21 points, 0.02 apart)
while holding the other three fixed - 84 models total. This is NOT a full
4D cross product: at 0.02 steps that would already be 21^4 = 194,481
models, and the resolution originally asked for (0.01 steps) would be
41^4 = 2,825,761 - far too many for a quick local check.
"""
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


def sweep_one_param(param, best_values, lines_df):
    """
    Build models for `param` swept +/-0.2 (9 points, 0.05 apart) around
    its best-fit value, holding the other three params fixed at their
    best-fit values. Returns one result dict per point.
    """
    center = best_values[param]
    candidate_values = iterative_1D_method.centered_array(center, STEP, NUM_POINTS)

    records = []
    for value in candidate_values:
        values = dict(best_values)
        values[param] = value

        model_df = iterative_1D_method.build_model_df(
            alpha_cc=values["alpha_cc"], alpha_Ia=values["alpha_Ia"],
            g_cc=values["g_cc"], g_ratio=values["g_ratio"],
        )
        reduced_chi2 = compute_reduced_chi2(model_df, lines_df)

        records.append({
            "swept_param": param,
            "alpha_cc": values["alpha_cc"],
            "alpha_Ia": values["alpha_Ia"],
            "g_cc": values["g_cc"],
            "g_ratio": values["g_ratio"],
            "reduced_chi2": reduced_chi2,
        })

    return records


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

    n_total = len(PARAMS) * NUM_POINTS
    print(f"Running {n_total} models ({NUM_POINTS} per param, {len(PARAMS)} params)...", flush=True)

    records = []
    for param in PARAMS:
        records.extend(sweep_one_param(param, best_values, lines_df))
        print(f"  finished sweeping {param}", flush=True)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(by="reduced_chi2", ascending=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(results_df)} results to {OUTPUT_PATH}")
    print("Best 10:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
