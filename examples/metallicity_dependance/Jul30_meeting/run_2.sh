#!/usr/bin/env bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=1d_iter_fit_2
#SBATCH --output=1d_iter_input_2.out

# COMMANDS TO RUN
cd /home/tajer.1/chemevo/examples/metallicity_dependance/Jul16_meeting/input_init_vals

# Call the Conda environment's Python directly
/home/tajer.1/envs/chem_ev/bin/python 1D_iter_fit_full_script_with_dir.py --alpha_cc 0.50 --alpha_Ia 0.80 --g_cc 0.40 --g_ratio 1.90 --save_dir "unity_models_2"