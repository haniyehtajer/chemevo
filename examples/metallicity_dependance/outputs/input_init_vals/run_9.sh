#!/usr/bin/env bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=1d_iter_fit
#SBATCH --output=1d_iter_input_9.out
#SBATCH --partition=ast

# COMMANDS TO RUN

# COMMANDS TO RUN
cd /home/tajer.1/chemevo/examples/metallicity_dependance/Jul16_meeting/input_init_vals
source /home/tajer.1/envs/chem_ev/bin/activate

python 1D_iter_fit_full_script_with_dir.py --alpha_cc 0.80 --alpha_Ia 0.30 --g_cc 0.60 --g_ratio 1.50 --save_dir "unity_models_9"