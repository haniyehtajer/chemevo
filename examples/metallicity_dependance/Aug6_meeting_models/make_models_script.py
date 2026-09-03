import numpy as np
import pandas as pd
import random
from chemevo import evolution_V1 as evo
import itertools
import os

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

'''
def centered_array(center, step, num):
    # Calculate how many steps we need to go in either direction from the center
    shift_steps = (num - 1) / 2
    
    # Calculate the exact start and end values
    start = center - (shift_steps * step)
    stop = center + (shift_steps * step)
    
    # Generate the array
    return np.linspace(start, stop, num)
'''

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

def compute_mn_fe_model(row, fe_centers):
    return (row['slope_mn_fe'] * fe_centers) + row['intercept_mn_fe']

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

# Get endpoint indexes
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
    """
    Generates GCE models with various metallicity dependance parameters.

    Parameters
    ----------
    param_to_optimize: string
        metallicity dependance parameter to be optimized. choices are:
        "alpha_cc", "alpha_Ia", "g_cc", "g_ratio"
    t_array : array-like
        Array of time steps (e.g., in Gyr).
    sfrs : list of array-like
        List of Star Formation Rate (SFR) arrays evaluated at `t_array`.
    alpha_cc_init : float
        Initial alpha_cc
    alpha_Ia_init : float
        Initial alpha_Ia
    g_cc_init : foat
        Initial g_cc
    g_ratio_init : float
        Initial g_ratio
    etas : array-like
        Array of ourflow efficencies.
    tau_stars : array-like
        Array of star formation efficiencies.
    grid_size : float
        when making a met dep param grid, the difference between two sample values
        e.g. alpha_ccs = [0.1, 0.2, 0.3, 0.4], then grid size = 0.1
    grid_len : int
        number of parameter values in the grid
        e.g. alpha_ccs = [0.1, 0.2, 0.3, 0.4], then grid_len = 3

    Returns
    -------
    models : pd DataFrame
        A dataframe with the columns:
        'mn_fe'
        'mn_mg'
        'fe_mg'
        'fe_h'
        'mg_h'
        'mn_h'
        't'
        'tau_star'
        'eta'
        'sfr'
        'alpha_cc'
        'alpha_Ia'
        'g_cc'
        'g_ratio'
        'Upsilon'
        'mg_h_bin'
    """
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

        df_endpoints = pd.DataFrame(extracted_data)
        df_endpoints = pd.DataFrame(extracted_data)
        if save == "on":
            os.makedirs(save_dir, exist_ok=True)
            df_endpoints.to_csv(os.path.join(save_dir, f"{param_to_optimize}_iter{iter}_{current_val:.4f}.csv"), index=False)








    