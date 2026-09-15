# %% [markdown]
# # Plot a grid_search.py result
#
# Reads the metdep params from one row of grid_search_output/grid_search_results.csv,
# builds that model, and makes the same [Mn/Fe] vs [Fe/Mg] 5-panel plot as
# plot_convergence_test.ipynb.

# %%
import pandas as pd
import matplotlib.pyplot as plt

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines
from chemevo.plotting import plot_mn_fe_vs_fe_mg
from chemevo import plotstyle

plotstyle.use()

# %%
RESULTS_PATH = "grid_search_output/grid_search_results.csv"

results_df = pd.read_csv(RESULTS_PATH)
results_df = results_df.sort_values(by="reduced_chi2", ascending=True).reset_index(drop=True)
results_df

# %%
# Which row of the sorted results to plot - 0 is the best fit, 1 is the
# second best, and so on. Change this and re-run the cells below to look
# at a different row.
ROW_INDEX = 0

row = results_df.iloc[ROW_INDEX]
row

# %%
model_df = iterative_1D_method.build_model_df(
    alpha_cc=row["alpha_cc"],
    alpha_Ia=row["alpha_Ia"],
    g_cc=row["g_cc"],
    g_ratio=row["g_ratio"],
)
model_df

# %%
lines_df = load_mn_fe_lines()

label = (
    f"alpha_cc={row['alpha_cc']:.2f}, alpha_Ia={row['alpha_Ia']:.2f}, "
    f"g_cc={row['g_cc']:.2f}, g_ratio={row['g_ratio']:.2f} "
    f"(reduced_chi2={row['reduced_chi2']:.4f})"
)

fig, axes = plot_mn_fe_vs_fe_mg({label: model_df}, lines_df)
plt.show()

# %%
