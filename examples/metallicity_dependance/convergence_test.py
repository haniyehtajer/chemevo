"""
Convergence test for the 1D iterative metallicity-dependence fit.

There are four metallicity-dependence ("metdep") parameters used in the Mn
yields: alpha_cc, alpha_Ia, g_cc, g_ratio. This script checks whether
chemevo.iterative_1D_method converges to the same optimized values
regardless of which initial value it starts from.

For each metdep param in turn, it sweeps a list of initial values for that
param (holding the other three at their default values) and runs the full
coordinate-descent fit (iterative_1D_method.iterative_1D_loop). Each run's
per-iteration model/chi^2 CSVs land in their own subfolder under
convergence_test_output/. At the end, a single summary CSV
(convergence_summary.csv) collects, for every run: the initial values used,
the final optimized values, and the reduced chi^2 of the converged fit.
"""
import os

import numpy as np
import pandas as pd

from chemevo import iterative_1D_method

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data_lines_with_vals.csv")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "convergence_test_output")

N_ITERS = 10
GRID_SIZE = 0.1
GRID_LEN = 5

# iterative_1D_loop fits the params in this order, each iteration; the last
# one fit in the last iteration is what determines the run's final chi^2.
PARAM_FIT_ORDER = ["alpha_cc", "alpha_Ia", "g_cc", "g_ratio"]

DEFAULTS = {
    "alpha_cc": 0.4,
    "alpha_Ia": 0.3,
    "g_cc": 0.4,
    "g_ratio": 1.8,
}

# Initial values to sweep per param (same as convergence_test.ipynb)
INIT_VALUES = {
    "alpha_cc": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "alpha_Ia": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "g_cc": list(np.linspace(0.1, 1, 10)),
    "g_ratio": [1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9],
}


def run_one_sweep(swept_param, init_value):
    """
    Run the full iterative fit with `swept_param` starting at `init_value`
    and the other three metdep params at their defaults.

    Returns a dict with the initial values, the final optimized values, and
    the reduced chi^2 of the converged (final iteration, final param) fit.
    """
    inits = dict(DEFAULTS)
    inits[swept_param] = init_value

    output_dir = os.path.join(OUTPUT_ROOT, f"{swept_param}_init_{init_value}")
    os.makedirs(output_dir, exist_ok=True)

    # iterative_1D_loop builds output paths via string concatenation
    # internally, so output_path must end with a path separator.
    iterative_1D_method.iterative_1D_loop(
        data_path=DATA_PATH,
        output_path=output_dir + os.sep,
        n_iters=N_ITERS, grid_size=GRID_SIZE, grid_len=GRID_LEN,
        alpha_cc_init=inits["alpha_cc"],
        alpha_Ia_init=inits["alpha_Ia"],
        g_cc_init=inits["g_cc"],
        g_ratio_init=inits["g_ratio"],
    )

    best_params_df = pd.read_csv(os.path.join(output_dir, "best_parameters_per_iteration.csv"))
    final_row = best_params_df.iloc[-1]

    # The reduced chi^2 of the fully-converged fit: the min-chi2_sum row of
    # the last param fit in the last iteration (find_best_model picks its
    # "best" row by min chi_2_sum, not min reduced_chi2 - matching that here
    # keeps this consistent with what actually drove the final best_g_ratio).
    last_param = PARAM_FIT_ORDER[-1]
    chi2_path = os.path.join(output_dir, f"chi_2_results_{last_param}_iter_{N_ITERS - 1}.csv")
    chi2_df = pd.read_csv(chi2_path)
    best_chi2_row = chi2_df.loc[chi2_df["chi_2_sum"].idxmin()]

    return {
        "swept_param": swept_param,
        "alpha_cc_init": inits["alpha_cc"],
        "alpha_Ia_init": inits["alpha_Ia"],
        "g_cc_init": inits["g_cc"],
        "g_ratio_init": inits["g_ratio"],
        "best_alpha_cc": final_row["best_alpha_cc"],
        "best_alpha_Ia": final_row["best_alpha_Ia"],
        "best_g_cc": final_row["best_g_cc"],
        "best_g_ratio": final_row["best_g_ratio"],
        "reduced_chi2": best_chi2_row["reduced_chi2"],
    }


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    records = []
    for swept_param in PARAM_FIT_ORDER:
        for init_value in INIT_VALUES[swept_param]:
            print(f"Running {swept_param}_init_{init_value} ...", flush=True)
            records.append(run_one_sweep(swept_param, init_value))

    summary_df = pd.DataFrame(records)
    summary_path = os.path.join(OUTPUT_ROOT, "convergence_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary of {len(summary_df)} runs to {summary_path}")


if __name__ == "__main__":
    main()
