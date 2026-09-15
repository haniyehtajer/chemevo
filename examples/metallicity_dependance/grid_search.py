"""
Exhaustive grid search over the four metdep params.

Unlike iterative_1D_method.py's coordinate-descent fit (which only ever
explores a small neighborhood around its current best guess, one param at a
time), this builds every single combination in a fixed grid and measures
each one's reduced chi^2 against the data - so the "best" result doesn't
depend on a starting point or search path.

This is a lot of models (see the grid below) - it's meant to be run on
another machine, not started from here.
"""
import itertools
import os

import numpy as np
import pandas as pd

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines, compute_reduced_chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "grid_search_output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "grid_search_results.csv")

# alpha_cc, alpha_Ia, g_cc: 0.1 to 0.9, both ends included, step 0.1 -> 9
# values each. g_ratio: 1.0 to 2.8, step 0.2 -> 10 values (3.0 excluded) -
# this matches 9*9*9*10 = 7290 total models.
ALPHA_CC_VALUES = np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2)
ALPHA_IA_VALUES = np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2)
G_CC_VALUES = np.round(np.arange(0.1, 0.9 + 0.001, 0.1), 2)
G_RATIO_VALUES = np.round(np.arange(1.0, 3.0, 0.2), 2)


def run_one_model(alpha_cc, alpha_Ia, g_cc, g_ratio, lines_df):
    """Build one model and return its reduced chi^2 against the data."""
    model_df = iterative_1D_method.build_model_df(alpha_cc, alpha_Ia, g_cc, g_ratio)
    return compute_reduced_chi2(model_df, lines_df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines_df = load_mn_fe_lines()

    grid = list(itertools.product(ALPHA_CC_VALUES, ALPHA_IA_VALUES, G_CC_VALUES, G_RATIO_VALUES))
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

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{n_total} done", flush=True)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(by="reduced_chi2", ascending=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(results_df)} results to {OUTPUT_PATH}")
    print("Best 10:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
