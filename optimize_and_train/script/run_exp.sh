#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

python3 run_exp.py --adsorbate $1 --root_dir "test-data/${1}/root_dir"