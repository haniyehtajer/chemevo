#!/usr/bin/env bash

#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=g_cc
#SBATCH --output=g_ratio.out

cd /home/tajer.1/chemevo
conda activate chem_ev
pip install -e .
cd /home/tajer.1/chemevo/examples/metallicity_dependance/May28_meeting
python save_models_script_g_ratio.py

