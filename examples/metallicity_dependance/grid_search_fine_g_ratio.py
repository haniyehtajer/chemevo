"""
Extend the g_ratio search beyond grid_search_fine.py's +/-0.2 window.

grid_search_fine.py centered its search on the coarse best fit
(g_ratio=2.6, +/-0.2 -> explored [2.4, 2.8]), and its own best result
landed right at the edge of that window (g_ratio=2.8) - a sign the true
optimum for g_ratio might lie further out, past what was actually searched.

This sweeps g_ratio alone from 2.0 to 3.0 (0.02 steps, 51 models), holding
alpha_cc, alpha_Ia, g_cc fixed at grid_search_fine.py's best values.

Note this is NOT a full joint re-exploration - it only checks whether
g_ratio wants to move further while the other three params stay exactly
where grid_search_fine.py left them. A full 4D re-exploration at this
wider g_ratio range would mean crossing the same 21x21x21=9261
alpha_cc/alpha_Ia/g_cc combinations with the ~30 new g_ratio values not
already covered - about 278,000 more models, bigger than the original
5-hour run. If alpha_cc/alpha_Ia/g_cc might also want to shift once
g_ratio moves further, this sweep alone won't reveal that.
"""
import os

import numpy as np
import pandas as pd

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines, compute_reduced_chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINE_RESULTS_PATH = os.path.join(SCRIPT_DIR, "grid_search_fine_output", "grid_search_fine_results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "grid_search_fine_g_ratio_output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "grid_search_fine_g_ratio_results.csv")

G_RATIO_MIN = 2.0
G_RATIO_MAX = 3.0
G_RATIO_STEP = 0.02


def get_best_fit():
    """Read grid_search_fine.py's results and return its single best row."""
    fine_df = pd.read_csv(FINE_RESULTS_PATH)
    fine_df = fine_df.sort_values(by="reduced_chi2", ascending=True)
    return fine_df.iloc[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines_df = load_mn_fe_lines()

    best_row = get_best_fit()
    alpha_cc = best_row["alpha_cc"]
    alpha_Ia = best_row["alpha_Ia"]
    g_cc = best_row["g_cc"]
    print(f"Holding alpha_cc={alpha_cc}, alpha_Ia={alpha_Ia}, g_cc={g_cc} fixed; sweeping g_ratio.")

    g_ratio_values = np.round(np.arange(G_RATIO_MIN, G_RATIO_MAX + 0.001, G_RATIO_STEP), 2)
    n_total = len(g_ratio_values)
    print(f"Running {n_total} models (g_ratio from {G_RATIO_MIN} to {G_RATIO_MAX}, step {G_RATIO_STEP})...", flush=True)

    records = []
    for g_ratio in g_ratio_values:
        model_df = iterative_1D_method.build_model_df(alpha_cc, alpha_Ia, g_cc, g_ratio)
        reduced_chi2 = compute_reduced_chi2(model_df, lines_df)

        records.append({
            "alpha_cc": alpha_cc,
            "alpha_Ia": alpha_Ia,
            "g_cc": g_cc,
            "g_ratio": g_ratio,
            "reduced_chi2": reduced_chi2,
        })

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(by="reduced_chi2", ascending=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(results_df)} results to {OUTPUT_PATH}")
    print("Best 10:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
