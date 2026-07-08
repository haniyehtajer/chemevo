import numpy as np
import pandas as pd
import random
from chemevo import evolution_V1 as evo

# Star formation histories
t_array = np.linspace(0.0001,14,int(1e3))
dt = t_array[1] - t_array[0]

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


# Default met dep params
gcc_Mn_def = 0.40
gcc_ratio_def = 1.9
alpha_cc_def = 0.50
alpha_ia_def = 0.40


# Etas and Tau*s 
etas = np.logspace(-2, 1, 10)
tau_stars = np.linspace(0.5, 6, 10)


alpha_ccs = np.arange(0.4, 0.6, 0.02)
#alpha_ccs = [0.1, 0.3]
print(alpha_ccs)

n_gals = len(etas) * len(tau_stars) * len(sfrs)
print("n_gals in each alpha cc = ", n_gals)
print("n_gals total = ", n_gals * len(alpha_ccs))


import itertools
import random

all_params = list(itertools.product(etas, tau_stars, sfrs))

random.seed(42)
sampled_params = random.sample(all_params, 100)


# 3. Define a function that only iterates over the 100 sampled parameters
def get_sampled_gals(params, alpha_cc=alpha_cc_def, alpha_Ia=alpha_ia_def, g_ratio=gcc_ratio_def, gcc=gcc_Mn_def):
    gals_list = []
    for eta, tau_star, sfr in params:
        gals_list.append(evo.Galaxy(
            t_array=t_array, m_g_array=sfr, tau_star=tau_star,
            eta=eta, Upsilon=1, yields_ref="W2024,moreFe",
            g_cc_Mn=gcc, g_ratio_Mn=g_ratio,
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

mg_h_big_bin_edges = np.arange(-0.7, 0.50, 0.20) 
mg_h_big_bin_centers = 0.5 * (mg_h_big_bin_edges[:-1] + mg_h_big_bin_edges[1:])


for alpha_cc_count in range(len(alpha_ccs)):
    alpha_cc_this_model = alpha_ccs[alpha_cc_count]
    gals_list = []
    # Change all params to sampled params if you only want a few random models
    gals_list = get_sampled_gals(all_params, alpha_cc=alpha_cc_this_model)
    gals_endpoints = find_endpoints(gals_list, mg_h_big_bin_centers)

    # 2. Extract data
    extracted_data = []

    for i in range(len(mg_h_big_bin_centers)):
        idxs = gals_endpoints[i]
        current_mg_h = mg_h_big_bin_centers[i]

        for gal, idx in zip(gals_list, idxs):
            sfr = gal.m_g_array
            if np.array_equal(sfr, m_g_array_const):
                sfr_label = "const SFR"
            elif np.array_equal(sfr, m_g_array_exp):
                sfr_label = "exp SFR"
            elif np.array_equal(sfr, m_g_array_rise_fall_1):
                sfr_label = "rise+fall  1"
            elif np.array_equal(sfr, m_g_array_rise_fall_2):
                sfr_label = "rise+fall  2"
            elif np.array_equal(sfr, m_g_array_rise_fall_3):
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
                    'gcc': gal.g_cc_Mn,
                    'gIa/gcc': gal.g_ratio_Mn,
                    'Upsilon': gal.Upsilon,
                    'mg_h_bin': current_mg_h
                })

    # 3. Create DataFrame and save
    df_endpoints = pd.DataFrame(extracted_data)
    df_endpoints.to_csv(f"models/alpha_cc_iter2_{alpha_cc_count}.csv", index=False)

        


