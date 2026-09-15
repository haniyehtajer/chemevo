import numpy as np
import pandas as pd
import random
from chemevo import evolution_V3 as evo
import itertools
import make_models_script
import glob

import matplotlib.pyplot as plt
import argparse

import os # Make sure to add this at the top

alpha_cc_init = 0.5
alpha_Ia_init = 0.3
g_cc_init = 0.4
g_ratio_init = 1.9
save_dir = "models"

# Create the directory immediately so we don't get errors later
os.makedirs(save_dir, exist_ok=True)

mg_h_big_bin_edges = np.arange(-0.7, 0.50, 0.20) 
mg_h_big_bin_centers = 0.5 * (mg_h_big_bin_edges[:-1] + mg_h_big_bin_edges[1:])

fe_mg_bin_edges = np.arange(-0.35, 0.15, 0.1)
fe_mg_centers = 0.5 * (fe_mg_bin_edges[:-1] + fe_mg_bin_edges[1:])


# A few SFRS
t_array = np.linspace(0.0001,14,int(1e3))

def rise_fall(t, tau_1, tau_2):
    return 1 * (1 - np.exp(-t/tau_1)) * np.exp(-t/tau_2)


m_g_array_const = np.ones(len(t_array))
tau_sfh = 6 #Gyrs
m_g_array_exp = 1 * np.exp(-t_array/tau_sfh)
m_g_array_exp /= np.trapezoid(m_g_array_exp, t_array)

m_g_array_rise_fall_1 = rise_fall(t_array, 2, 8)
m_g_array_rise_fall_1 /= np.trapezoid(m_g_array_rise_fall_1, t_array)

m_g_array_rise_fall_2 = rise_fall(t_array, 4, 30)
m_g_array_rise_fall_2 /= np.trapezoid(m_g_array_rise_fall_2, t_array)

m_g_array_rise_fall_3 = rise_fall(t_array, 6, 12)
m_g_array_rise_fall_3 /= np.trapezoid(m_g_array_rise_fall_3, t_array)


sfrs = [m_g_array_const, m_g_array_exp, m_g_array_rise_fall_1, m_g_array_rise_fall_2, m_g_array_rise_fall_3]

summary_df = pd.read_csv("data_lines_with_vals.csv")

def compute_mn_fe_model(row, fe_centers):
    return (row['slope_mn_fe'] * fe_centers) + row['intercept_mn_fe']

summary_df['mn_fe_model_vals'] = summary_df.apply(
    lambda r: compute_mn_fe_model(r, fe_mg_centers).tolist(), axis=1
)

# 3. Main Optimization & Plotting Function
def optimize_for_param(param_name, file_pattern, save_dir):
    #print(f"\n--- Optimizing {param_name} ---")
    
    # --- A. Chi-Square Calculation ---
    model_files = glob.glob(file_pattern)
    chi2_results = []

    for file in model_files:
        model_df = pd.read_csv(file)
        model_df['mg_h_bin_round'] = model_df['mg_h_bin'].round(3)
        
        merged_df = pd.merge(model_df, summary_df, on='mg_h_bin_round', how='inner')

        merged_df.to_csv(f"{save_dir}/merged_df_example.csv")

        expected_mn_fe = (merged_df['slope_mn_fe'] * merged_df['fe_mg']) + merged_df['intercept_mn_fe']
        residuals_sq = (merged_df['mn_fe'] - expected_mn_fe)**2

        n_points = len(residuals_sq)
        chi2_total = residuals_sq.sum()

        chi2_results.append({
            'file': file,
            'alpha_cc': merged_df['alpha_cc'].iloc[0],
            'alpha_Ia': merged_df['alpha_Ia'].iloc[0],
            'g_cc': merged_df['g_cc'].iloc[0],
            'g_ratio': merged_df['g_ratio'].iloc[0],
            'Upsilon': merged_df['Upsilon'].iloc[0],
            'chi2_total': chi2_total,
            'reduced_chi2': chi2_total / n_points,
            'n_points': n_points
        })

    results_df = pd.DataFrame(chi2_results).sort_values(by='reduced_chi2').reset_index(drop=True)
    #print("Top 3 Best Fitting Global Models:")
    #print(results_df.head(3))

    best_model = results_df.iloc[0]
    '''print(f"\nThe best fit is {best_model['file']} with alpha_cc = {best_model['alpha_cc']}, "
          f"alpha_Ia = {best_model['alpha_Ia']}, g_cc = {best_model['g_cc']}, "
          f"g_ratio = {best_model['g_ratio']}, Upsilon = {best_model['Upsilon']}")
    '''

    best_val = best_model[param_name] # <--- ADD THIS LINE
    #print(f"\nThe best fit is {best_model['file']}...")

    return best_val

    


num_iters = 1 # Increased so we can see the grid size change at iteration 3 and 6

# 1. Store your initial parameters in a dictionary
current_params = {
    "alpha_cc": alpha_cc_init,
    "alpha_Ia": alpha_Ia_init,
    "g_cc": g_cc_init,
    "g_ratio": g_ratio_init
}
grid_size_this_iter = 0.1
params_to_optimize = ["alpha_cc", "alpha_Ia", "g_cc", "g_ratio"]

# Initialize a list to track the history
optimization_history = []

for i in range(num_iters):
    for param in params_to_optimize:
        
        # 2. Feed the dictionary values into make_models
        # It automatically uses the 'latest find' because current_params is updated below
        make_models_script.make_models(
            param_to_optimize=param, 
            t_array=t_array,
            sfrs=sfrs, 
            alpha_cc_init=current_params["alpha_cc"],
            alpha_Ia_init=current_params["alpha_Ia"],
            g_cc_init=current_params["g_cc"],
            g_ratio_init=current_params["g_ratio"],
            etas=np.logspace(-2, 1, 10),
            tau_stars=np.linspace(0.5, 6, 10),
            grid_size=grid_size_this_iter,
            grid_len=9,
            iter=i,
            save_dir=save_dir
        )
        
        # 3. Get the newly optimized value for THIS specific parameter
        best_val = optimize_for_param(param_name=param, file_pattern=f"{save_dir}/{param}_iter{i}_*.csv", save_dir=save_dir)
        print(f"best {param} in iter {i} = {best_val}")
        
        # 4. IMMEDIATELY update the dictionary so the next loop uses it
        current_params[param] = best_val
        print(current_params)
        
        # Record the current state at this step
        optimization_history.append({
            "iteration": i,
            "optimized_param": param,
            "best_val": best_val,
            "alpha_cc": current_params["alpha_cc"],
            "alpha_Ia": current_params["alpha_Ia"],
            "g_cc": current_params["g_cc"],
            "g_ratio": current_params["g_ratio"],
            "grid_size": grid_size_this_iter
        })

    print(f"iter {i} with grid len {grid_size_this_iter} is done.")
    
    # Stay in one grid size for 3 iterations, then shrink it
    if (i + 1) % 3 == 0:
        grid_size_this_iter = grid_size_this_iter / 10.0
        print(f"--> 3 iterations completed. Grid size updated to {grid_size_this_iter} for the next iterations.")

# Convert the history to a DataFrame and save to CSV at the very end
history_df = pd.DataFrame(optimization_history)
history_df.to_csv(f"{save_dir}/optimization_history.csv", index=False)
print(f"Optimization history saved to {save_dir}/optimization_history.csv")