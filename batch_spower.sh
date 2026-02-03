#!/bin/bash

#SBATCH --job-name=sp_model
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16Gb
#SBATCH --time=02:00:00
#SBATCH --account=bckiedro0
#SBATCH --export=ALL
#SBATCH --output=sp_mod.out

module load ffmpeg
#srun --cpu-bind=cores python LiPetrasso_spower.py
#srun --cpu-bind=cores python fourier_source.py
#srun --cpu-bind=cores python bc_source.py
srun --cpu-bind=cores python3 run_sp_mod.py
