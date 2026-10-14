# %% [markdown]
# # Plot a grid_search_pipeline.py / grid_search_g_ratio_fixed.py run
#
# Give it the output directory (e.g.
# grid_search_g_ratio_fixed_output_g2_W2024_moreFe20_U1.27) and it finds
# every stage's results.csv inside (stage1_coarse, stage2_fine,
# stage3_finest - however many actually completed), builds each stage's
# best-fit model, and overlays all of them in one 5-panel plot, so you can
# see how the fit moved from coarse to fine to finest.

# %%
import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

from chemevo import iterative_1D_method
from chemevo.mn_fe_lines import load_mn_fe_lines
from chemevo.plotting import plot_mn_fe_vs_fe_mg
from chemevo.plotting import plot_mn_mg_vs_fe_mg
from chemevo import plotstyle

plotstyle.use()

# %%
RESULTS_DIR = "/Users/honeyeah/Codes/chemevo/examples/metallicity_dependance/grid_search_g_ratio_fixed_output_g2_W2024_moreFe20_U1.27/stage3_finest"


def desanitize_yields_ref(sanitized):
    """
    Reverse grid_search_pipeline.py's sanitize_for_path() on a yields_ref
    preset name: every preset is either plain ("W2017", "sanders-test") or
    "W2024,<suffix>", so the first underscore (if any) is put back as a
    comma.
    """
    if "_" not in sanitized:
        return sanitized
    first, rest = sanitized.split("_", 1)
    return f"{first},{rest}"


def parse_settings_from_dirname(results_dir):
    """
    Best-effort recovery of (upsilon, yields_ref) from a
    grid_search_pipeline.py / grid_search_g_ratio_fixed.py output
    directory's auto-generated name - needed for runs made before those
    scripts started also saving Upsilon/yields_ref as CSV columns.
    Returns (None, None) if the name doesn't match either pattern.
    """
    name = os.path.basename(os.path.normpath(results_dir))

    patterns = [
        r"^grid_search_pipeline_output_(?P<yields>.+)_U(?P<upsilon>[\d.]+)$",
        r"^grid_search_g_ratio_fixed_output_g[\d.]+_(?P<yields>.+)_U(?P<upsilon>[\d.]+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            yields_ref = desanitize_yields_ref(match.group("yields"))
            upsilon = float(match.group("upsilon"))
            return upsilon, yields_ref

    return None, None


def get_settings_for_stage(stage_df, results_dir):
    """
    Upsilon/yields_ref to build a stage's best-fit model with: read from
    the results.csv's own columns if present (newer runs save these),
    else parse from the output directory's name, else fall back to
    chemevo's defaults and say so.
    """
    if "Upsilon" in stage_df.columns and "yields_ref" in stage_df.columns:
        row = stage_df.iloc[0]
        return row["Upsilon"], row["yields_ref"]

    upsilon, yields_ref = parse_settings_from_dirname(results_dir)
    if upsilon is not None:
        return upsilon, yields_ref

    print(f"Could not determine Upsilon/yields_ref for {results_dir} - "
          f"falling back to chemevo defaults (Upsilon=1, yields_ref='W2024,moreFe10').")
    return 1.0, "W2024,moreFe10"


# %%
stage_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "results.csv"), recursive=True))

print(f"Found {len(stage_paths)} stage(s):")
for path in stage_paths:
    print(" ", path)

# %%
lines_df = load_mn_fe_lines()

models = {}
for path in stage_paths:
    stage_name = os.path.basename(os.path.dirname(path))

    stage_df = pd.read_csv(path).sort_values(by="reduced_chi2", ascending=True).reset_index(drop=True)
    best_row = stage_df.iloc[0]

    upsilon, yields_ref = get_settings_for_stage(stage_df, RESULTS_DIR)

    model_df = iterative_1D_method.build_model_df(
        alpha_cc=best_row["alpha_cc"], alpha_Ia=best_row["alpha_Ia"],
        g_cc=best_row["g_cc"], g_ratio=best_row["g_ratio"],
        Upsilon=upsilon, yields_ref=yields_ref,
    )

    label = (
        f"Upsilon = {upsilon}, yield = {yields_ref}: alpha_cc={best_row['alpha_cc']:.2f}, alpha_Ia={best_row['alpha_Ia']:.2f}, "
        f"g_cc={best_row['g_cc']:.2f}, g_ratio={best_row['g_ratio']:.2f} "
        f"(reduced_chi2={best_row['reduced_chi2']:.4f})"
    )
    models[label] = model_df

fig, axes = plot_mn_fe_vs_fe_mg(models, lines_df, show_data=True, legend_on=0, color = 'hotpink')
plt.show()

# %%

fig, axes = plot_mn_mg_vs_fe_mg(models, lines_df, show_data=True, legend_on=0, color = 'hotpink')
plt.show()

# %%
