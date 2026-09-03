#!/usr/bin/env bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=1d_iter_fit
#SBATCH --output=1d_iter_fit_%j.out
#SBATCH --partition=ast

# COMMANDS TO RUN

# 1. Navigate to the directory containing your script
cd /home/tajer.1/chemevo/examples/metallicity_dependance/Jul16_meeting/

# 2. Activate your virtual environment (adjust if you use conda instead of venv)
source /home/tajer.1/envs/chem_ev/bin/activate

# 3. Run the python script
python 1D_iter_fit_full_script.py