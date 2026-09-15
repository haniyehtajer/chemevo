"""
Higher-resolution refinement of the coarse-grid convergence test's best-fit
results.

Takes the first N_STARTING_POINTS unique (alpha_cc, alpha_Ia, g_cc, g_ratio)
combinations found by the coarse-grid sweep (convergence_test.py's
convergence_test_output/convergence_summary.csv, sorted by reduced_chi2 -
not the random-draw version), and re-runs the coordinate-descent fit from
each of those starting points at a much finer grid resolution: instead of
grid_size=0.1 with grid_len=5 (5 points, +/-0.2 around the current value),
this uses grid_size=0.01 with grid_len=41 (41 points, 0.01 apart, still
+/-0.2 around the current value).
"""
import os

import pandas as pd

from chemevo import iterative_1D_method

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data_lines_with_vals.csv")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "iterative_1D_method_higher_res_output")

COARSE_SUMMARY_PATH = os.path.join(SCRIPT_DIR, "convergence_test_output", "convergence_summary.csv")
N_STARTING_POINTS = 5

N_ITERS = 10
GRID_SIZE = 0.01

# 41 points, 0.01 apart, centered on the starting value, spans the starting
# value +/- 0.2 - the same total window as the coarse grid's
# grid_size=0.1, grid_len=5, just resolved 10x finer.
GRID_LEN = 41

# iterative_1D_loop always fits params in this order each iteration; the
# last one fit in the last iteration is what a run's final reduced_chi2
# comes from.
LAST_PARAM = "g_ratio"


def get_starting_points(n=N_STARTING_POINTS):
    """
    Read the coarse-grid convergence test's summary, and return the first
    `n` unique (alpha_cc, alpha_Ia, g_cc, g_ratio) best-fit combinations,
    ordered from lowest to highest reduced_chi2.
    """
    coarse_df = pd.read_csv(COARSE_SUMMARY_PATH)
    coarse_df = coarse_df.sort_values(by="reduced_chi2", ascending=True)

    best_value_columns = ["best_alpha_cc", "best_alpha_Ia", "best_g_cc", "best_g_ratio"]

    # Two rows can hold the "same" best-fit values but differ by tiny
    # floating-point noise (e.g. 0.6 vs 0.6000000000000001), which would
    # make drop_duplicates() treat them as different rows. Round to far
    # more precision than the coarse grid actually has (grid_size=0.1)
    # before comparing, so only genuinely different values count as unique.
    rounded_best_values = coarse_df[best_value_columns].round(6)
    is_first_occurrence = ~rounded_best_values.duplicated()
    unique_df = coarse_df[is_first_occurrence]

    return unique_df.head(n)


def get_final_reduced_chi2(output_dir):
    """
    The reduced_chi2 of a finished run's converged fit: the min-chi2_sum
    candidate in the last param's last-iteration chi_2_results file.
    """
    chi2_path = os.path.join(output_dir, f"chi_2_results_{LAST_PARAM}_iter_{N_ITERS - 1}.csv")
    chi2_df = pd.read_csv(chi2_path)
    best_row = chi2_df.loc[chi2_df["chi_2_sum"].idxmin()]
    return best_row["reduced_chi2"]


def run_one_refinement(run_index, alpha_cc_init, alpha_Ia_init, g_cc_init, g_ratio_init):
    """
    Run the higher-resolution fit starting from one (alpha_cc, alpha_Ia,
    g_cc, g_ratio) point. Returns a dict with the starting values, the
    refined final values, and the refined fit's reduced_chi2.
    """
    output_dir = os.path.join(OUTPUT_ROOT, f"refine_{run_index}")
    os.makedirs(output_dir, exist_ok=True)

    # iterative_1D_loop builds output paths via string concatenation
    # internally, so output_path must end with a path separator.
    iterative_1D_method.iterative_1D_loop(
        data_path=DATA_PATH,
        output_path=output_dir + os.sep,
        n_iters=N_ITERS,
        grid_size=GRID_SIZE,
        grid_len=GRID_LEN,
        alpha_cc_init=alpha_cc_init,
        alpha_Ia_init=alpha_Ia_init,
        g_cc_init=g_cc_init,
        g_ratio_init=g_ratio_init,
    )

    best_params_df = pd.read_csv(os.path.join(output_dir, "best_parameters_per_iteration.csv"))
    final_row = best_params_df.iloc[-1]

    return {
        "run_index": run_index,
        "alpha_cc_init": alpha_cc_init,
        "alpha_Ia_init": alpha_Ia_init,
        "g_cc_init": g_cc_init,
        "g_ratio_init": g_ratio_init,
        "best_alpha_cc": final_row["best_alpha_cc"],
        "best_alpha_Ia": final_row["best_alpha_Ia"],
        "best_g_cc": final_row["best_g_cc"],
        "best_g_ratio": final_row["best_g_ratio"],
        "reduced_chi2": get_final_reduced_chi2(output_dir),
    }


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    starting_points = get_starting_points()

    records = []
    for run_index, row in enumerate(starting_points.itertuples()):
        print(
            f"Refining starting point {run_index}: "
            f"alpha_cc={row.best_alpha_cc}, alpha_Ia={row.best_alpha_Ia}, "
            f"g_cc={row.best_g_cc}, g_ratio={row.best_g_ratio}",
            flush=True,
        )

        result = run_one_refinement(
            run_index,
            alpha_cc_init=row.best_alpha_cc,
            alpha_Ia_init=row.best_alpha_Ia,
            g_cc_init=row.best_g_cc,
            g_ratio_init=row.best_g_ratio,
        )
        records.append(result)

    summary_df = pd.DataFrame(records)
    summary_path = os.path.join(OUTPUT_ROOT, "higher_res_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary of {len(summary_df)} refinements to {summary_path}")


if __name__ == "__main__":
    main()
