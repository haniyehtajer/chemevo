import numpy as np
import pandas as pd
import random
from chemevo import evolution_V1 as evo
import itertools
import os
import matplotlib.pyplot as plt

mg_h_big_bin_edges = np.arange(-0.7, 0.50, 0.20)
mg_h_big_bin_centers = 0.5 * (mg_h_big_bin_edges[:-1] + mg_h_big_bin_edges[1:])

fe_mg_bin_edges = np.arange(-0.35, 0.15, 0.1)
fe_mg_centers = 0.5 * (fe_mg_bin_edges[:-1] + fe_mg_bin_edges[1:])


# A few SFRS
t_array = np.linspace(0.0001,14,int(1e2))
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

#sfrs = [m_g_array_const, m_g_array_exp, m_g_array_rise_fall_1, m_g_array_rise_fall_2, m_g_array_rise_fall_3]
#etas=np.logspace(-2, 1, 10)
#tau_stars=np.linspace(0.5, 6, 10)

sfrs = [m_g_array_const, m_g_array_exp, m_g_array_rise_fall_1]
etas =np.logspace(-2, 0.3, 10)
tau_stars =np.linspace(1, 6, 10)


def centered_array(center, step, num):
    shift_steps = (num - 1) / 2
    start = center - (shift_steps * step)
    stop = center + (shift_steps * step)
    
    arr = np.linspace(start, stop, num)
    
    # Only keep strictly positive values
    positive_arr = []
    for val in arr:
        if val > 0:
            positive_arr.append(val)
            
    return np.array(positive_arr)

def get_sampled_gals(t_array, params, alpha_cc, alpha_Ia, g_cc, g_ratio):
        gals_list = []
        for eta, tau_star, sfr in params:
            gals_list.append(evo.Galaxy(
                t_array=t_array, m_g_array=sfr, tau_star=tau_star,
                eta=eta, Upsilon=1, yields_ref="W2024,moreFe",
                g_cc_Mn=g_cc, g_ratio_Mn=g_ratio,
                alpha_cc_Mn=alpha_cc, alpha_Ia_Mn=alpha_Ia
            ))
        return gals_list

def find_endpoints(galaxies, bin_centers):
    endpoints_gals = []
    for bin_cent in bin_centers:
        endpoints = []
        for gal in galaxies:
            mg_h = gal.Mg_H
            crossed = False
            for indx in range(len(mg_h)):
                mg_h_at_time = mg_h[indx]
                if mg_h_at_time > bin_cent:
                    endpoints.append(indx)
                    crossed = True
                    break
            if not crossed:
                endpoints.append(-1) # Critical fix to maintain 1:1 mapping with galaxies
        endpoints_gals.append(np.array(endpoints))
    return endpoints_gals


def make_models(param_to_optimize, t_array, sfrs, alpha_cc_init, alpha_Ia_init, g_cc_init, g_ratio_init, etas, tau_stars,
                grid_size, grid_len, iter, save_dir="models", save="on"):

    alpha_ccs = centered_array(center = alpha_cc_init, step = grid_size, num = grid_len)
    alpha_Ias = centered_array(center = alpha_Ia_init, step = grid_size, num = grid_len)
    g_ccs = centered_array(center=g_cc_init, step = grid_size, num = grid_len)
    g_ratios = centered_array(center = g_ratio_init, step = grid_size, num = grid_len)

    if param_to_optimize == "alpha_cc":
        met_dep_param_to_fit = alpha_ccs
        print("alpha_ccs = ", met_dep_param_to_fit)
    elif param_to_optimize == "alpha_Ia":
        met_dep_param_to_fit = alpha_Ias
        print("alpha_Ias = ", met_dep_param_to_fit)
    elif param_to_optimize == "g_cc":
        met_dep_param_to_fit = g_ccs
        print("g_ccs = ", met_dep_param_to_fit)
    elif param_to_optimize == "g_ratio":
        met_dep_param_to_fit = g_ratios
        print("g_ratios = ", met_dep_param_to_fit)

    n_gals = len(etas) * len(tau_stars) * len(sfrs)
    print(f"n_gals in each {param_to_optimize} = ", n_gals)
    print("n_gals total = ", n_gals * len(alpha_ccs))

    all_params = list(itertools.product(etas, tau_stars, sfrs))
    random.seed(42)
    # In case we want to sample a few and not use all of them
    #sampled_params = random.sample(all_params, 100)

    # 2. Iterate and build models
    for current_val in met_dep_param_to_fit:
        # Set all to initial defaults first
        a_cc = alpha_cc_init
        a_Ia = alpha_Ia_init
        g_cc = g_cc_init
        g_rat = g_ratio_init

        # Override only the parameter currently being optimized
        if param_to_optimize == "alpha_cc":
            a_cc = current_val
        elif param_to_optimize == "alpha_Ia":
            a_Ia = current_val
        elif param_to_optimize == "g_cc":
            g_cc = current_val
        elif param_to_optimize == "g_ratio":
            g_rat = current_val

        # Call the function once per loop iteration
        gals_list = get_sampled_gals(t_array=t_array, params=all_params, 
            alpha_cc=a_cc, alpha_Ia=a_Ia, g_cc=g_cc, g_ratio=g_rat)

        gals_endpoints = find_endpoints(gals_list, mg_h_big_bin_centers)


        # 2. Extract data
        extracted_data = []

        for i in range(len(mg_h_big_bin_centers)):
            idxs = gals_endpoints[i]
            current_mg_h = mg_h_big_bin_centers[i]

            for gal, idx in zip(gals_list, idxs):
                sfr = gal.m_g_array
                if np.allclose(sfr, m_g_array_const):
                    sfr_label = "const SFR"
                elif np.allclose(sfr, m_g_array_exp):
                    sfr_label = "exp SFR"
                elif np.allclose(sfr, m_g_array_rise_fall_1):
                    sfr_label = "rise+fall  1"
                elif np.allclose(sfr, m_g_array_rise_fall_2):
                    sfr_label = "rise+fall  2"
                elif np.allclose(sfr, m_g_array_rise_fall_3):
                    sfr_label = "rise+fall  3"
                else:
                    sfr_label = "unknown SFR"

                if idx != -1:  
                    extracted_data.append({
                        'mn_fe': gal.Mn_Fe[idx],
                        'mn_mg': gal.Mn_Mg[idx],
                        'fe_mg': gal.Fe_Mg[idx],
                        'fe_h': gal.Fe_H[idx],
                        'mg_h': gal.Mg_H[idx],
                        'mn_h': gal.Mn_H[idx],
                        't': gal.t[idx],
                        'tau_star': gal.tau_star_arr[idx],
                        'eta': gal.eta_arr[idx],
                        'sfr': sfr_label,
                        'alpha_cc': gal.alpha_cc_Mn,
                        'alpha_Ia': gal.alpha_Ia_Mn,
                        'g_cc': gal.g_cc_Mn,
                        'g_ratio': gal.g_ratio_Mn,
                        'Upsilon': gal.Upsilon,
                        'mg_h_bin': current_mg_h
                    })
        print(f"Done generating models for {param_to_optimize}, iter = {iter}, grid size = {grid_size}, grid_len = {grid_len}")
        df_endpoints = pd.DataFrame(extracted_data)
        if save == "on":
            os.makedirs(save_dir, exist_ok=True)
            df_endpoints.to_csv(os.path.join(save_dir, f"{param_to_optimize}_iter{iter}_{current_val:.4f}.csv"), index=False)



def find_best_model(filepath_list, param, iter, output_path, data):
    chi2_results = []
    for file in filepath_list:
        model_df = pd.read_csv(file)
        chi2_sum_per_bin = []
        for mg_h_bin in model_df['mg_h_bin'].unique():
            model_subset = model_df[np.isclose(model_df['mg_h_bin'], mg_h_bin)]
            model_subset = model_subset[model_subset['fe_mg'] >= 0.35]
            data_subset = data[np.isclose(data['mg_h_bin_center'], mg_h_bin)]
            expected_mn_fe = (model_subset['fe_mg'] * data_subset['slope_mn_fe'].iloc[0]) + data_subset['intercept_mn_fe'].iloc[0]
            model_mn_fe = model_subset['mn_fe']
            subset_res_sq = (model_mn_fe - expected_mn_fe)**2
            n_points = len(subset_res_sq)
            chi2_sum_per_bin.append(subset_res_sq.sum())

        chi_2_sum = sum(chi2_sum_per_bin)
        chi2_results.append({'file': file,
                            'alpha_cc': model_df['alpha_cc'].iloc[0],
                            'alpha_Ia': model_df['alpha_Ia'].iloc[0],
                            'g_cc': model_df['g_cc'].iloc[0],
                            'g_ratio': model_df['g_ratio'].iloc[0],
                            'Upsilon': model_df['Upsilon'].iloc[0],
                            'chi_2_sum': chi_2_sum,
                            'reduced_chi2': chi_2_sum/n_points,
                            'n_points': n_points 
                            })
            
    results_df = pd.DataFrame(chi2_results).reset_index(drop=True)
    results_df.to_csv(output_path+f"chi_2_results_{param}_iter_{iter}.csv")
    min_chi_2 = min(results_df['chi_2_sum'])
    best_result = results_df[results_df['chi_2_sum'] == min_chi_2]
    return (best_result)



def iterative_1D_loop(data_path, output_path, n_iters, grid_size, grid_len,
                      alpha_cc_init, alpha_Ia_init, g_cc_init, g_ratio_init, 
                      sfrs = sfrs, etas = etas, tau_stars = tau_stars):
    data = pd.read_csv(data_path)
    parameters_to_fit = ["alpha_cc", "alpha_Ia", "g_cc", "g_ratio"]
    iteration_results = []

    current_alpha_cc = alpha_cc_init
    current_alpha_Ia = alpha_Ia_init
    current_g_cc = g_cc_init
    current_g_ratio = g_ratio_init
    
    for i in range(n_iters):
        
        for param in parameters_to_fit:
            make_models(param, t_array, sfrs, 
                        alpha_cc_init=current_alpha_cc, 
                        alpha_Ia_init=current_alpha_Ia,
                        g_cc_init=current_g_cc, 
                        g_ratio_init=current_g_ratio, 
                        etas=etas, tau_stars=tau_stars,
                        grid_size=grid_size, grid_len=grid_len, 
                        iter=i, save_dir=output_path, save="on")
            
            filepath_list = []
            for filename in os.listdir(output_path):
                file_prefix = param + "_iter" + str(i)
                if filename.startswith(file_prefix) and filename.endswith(".csv"):
                    filepath_list.append(output_path + filename)
                    
            best_model = find_best_model(filepath_list=filepath_list, param=param, iter=i, output_path=output_path, data=data)
            print(f"iter = {i}")
            print(f"best {param} =", best_model[param].iloc[0])

            if param == "alpha_cc":
                current_alpha_cc = best_model['alpha_cc'].iloc[0]
            if param == "alpha_Ia":
                current_alpha_Ia = best_model['alpha_Ia'].iloc[0]
            if param == "g_cc":
                current_g_cc = best_model['g_cc'].iloc[0]
            if param == "g_ratio":
                current_g_ratio = best_model['g_ratio'].iloc[0]

            iteration_results.append({'iteration': i,
                                    'best_alpha_cc': current_alpha_cc,
                                    'best_alpha_Ia': current_alpha_Ia,
                                    'best_g_cc': current_g_cc,
                                    'best_g_ratio': current_g_ratio})

    iteration_results_df = pd.DataFrame(iteration_results)
    iteration_results_df.to_csv(output_path + "best_parameters_per_iteration.csv", index=False)