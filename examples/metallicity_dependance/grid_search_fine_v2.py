"""
Fine-grained full-grid search around the coarse best fit, take 2.

Same as grid_search_fine.py (every combination of all four params, zoomed
into a local neighborhood around grid_search.py's best fit), except
g_ratio's window is widened and shifted to an explicit [2.5, 3.2] range
instead of +/-0.2 around the coarse best (2.6). grid_search_fine.py's best
result landed right at the edge of its g_ratio window (2.8) - this makes
sure the true optimum isn't just outside what got searched.

alpha_cc, alpha_Ia, g_cc keep the same +/-0.2 window around the coarse
best fit (0.3, 0.3, 0.3) that grid_search_fine.py used.

21 (alpha_cc) x 21 (alpha_Ia) x 21 (g_cc) x 36 (g_ratio) = 333,396 models
total - about 1.7x the original grid_search_fine.py run.
"""
import itertools
import os

import numpy as np
import pandas as pd

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines, compute_reduced_chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COARSE_RESULTS_PATH = os.path.join(SCRIPT_DIR, "grid_search_output", "grid_search_results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "grid_search_fine_v2_output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "grid_search_fine_v2_results.csv")

STEP = 0.02

# alpha_cc, alpha_Ia, g_cc: 21 points, 0.02 apart, centered on the coarse
# best-fit value (best-0.2 ... best ... best+0.2).
NUM_POINTS = 21

# g_ratio: explicit range instead of +/-0.2 around the coarse best (2.6),
# since grid_search_fine.py's best result (2.8) landed right at the edge
# of that window.
G_RATIO_MIN = 2.5
G_RATIO_MAX = 3.2


def get_coarse_best_fit():
    """Read grid_search.py's results and return its single best row."""
    coarse_df = pd.read_csv(COARSE_RESULTS_PATH)
    coarse_df = coarse_df.sort_values(by="reduced_chi2", ascending=True)
    return coarse_df.iloc[0]


def get_param_grids(best_values):
    """
    alpha_cc/alpha_Ia/g_cc: 21 candidate values (+/-0.2 around the coarse
    best, 0.02 apart). g_ratio: the explicit [G_RATIO_MIN, G_RATIO_MAX]
    range, 0.02 apart.
    """
    grids = {}
    for param in ["alpha_cc", "alpha_Ia", "g_cc"]:
        center = best_values[param]
        grids[param] = iterative_1D_method.centered_array(center, STEP, NUM_POINTS)

    grids["g_ratio"] = np.round(np.arange(G_RATIO_MIN, G_RATIO_MAX + 0.001, STEP), 2)
    print(grids)

    return grids


def run_one_model(alpha_cc, alpha_Ia, g_cc, g_ratio, lines_df):
    """Build one model and return its reduced chi^2 against the data."""
    model_df = iterative_1D_method.build_model_df(alpha_cc, alpha_Ia, g_cc, g_ratio)
    return compute_reduced_chi2(model_df, lines_df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines_df = load_mn_fe_lines()

    best_row = get_coarse_best_fit()
    best_values = {
        "alpha_cc": best_row["alpha_cc"],
        "alpha_Ia": best_row["alpha_Ia"],
        "g_cc": best_row["g_cc"],
    }
    print(f"Coarse best fit from grid_search.py: {dict(best_row)}")

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
