#!/bin/bash
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -p long-24core
#SBATCH --mail-user sebastian.dick@stonybrook.edu
#SBATCH --mail-type ALL

python g2_spin.py 

