"""
Random-initialization convergence test for the 1D iterative metallicity-
dependence fit.

Unlike convergence_test.py (which sweeps one metdep param at a time while
holding the other three at their defaults), this draws all four metdep
params (alpha_cc, alpha_Ia, g_cc, g_ratio) simultaneously from independent
uniform distributions and re-runs the full fit from each random starting
point, to check whether the fit converges to the same optimum regardless of
where in the 4D parameter space it starts.
"""
import os

import numpy as np
import pandas as pd

from chemevo import iterative_1D_method

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data_lines_with_vals.csv")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "convergence_test_random_output")

N_ITERS = 10
GRID_SIZE = 0.1
GRID_LEN = 5

N_DRAWS = 100
RANDOM_SEED = 42

# Uniform sampling ranges (low, high) for each metdep param's random initial value
SAMPLE_RANGES = {
    "alpha_cc": (0.0, 1.0),
    "alpha_Ia": (0.0, 1.0),
    "g_cc": (0.1, 1.0),
    "g_ratio": (1.0, 3.0),
}

# iterative_1D_loop fits the params in this order, each iteration; the last
# one fit in the last iteration is what determines the run's final chi^2.
PARAM_FIT_ORDER = ["alpha_cc", "alpha_Ia", "g_cc", "g_ratio"]


def sample_initial_values(rng):
    return {
        param: rng.uniform(low, high)
        for param, (low, high) in SAMPLE_RANGES.items()
    }


def run_one_draw(draw_idx, inits):
    """
    Run the full iterative fit starting from the random initial values in
    `inits`. Returns a dict with the initial values, the final optimized
    values, and the reduced chi^2 of the converged (final iteration, final
    param) fit.
    """
    output_dir = os.path.join(OUTPUT_ROOT, f"draw_{draw_idx:02d}")
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
        "draw": draw_idx,
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
    rng = np.random.default_rng(RANDOM_SEED)

    records = []
    for draw_idx in range(N_DRAWS):
        inits = sample_initial_values(rng)
        print(f"Running draw {draw_idx}: {inits}", flush=True)
        records.append(run_one_draw(draw_idx, inits))

    summary_df = pd.DataFrame(records)
    summary_path = os.path.join(OUTPUT_ROOT, "convergence_summary_random.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary of {len(summary_df)} runs to {summary_path}")


if __name__ == "__main__":
    main()
