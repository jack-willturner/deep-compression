#!/bin/sh
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=logs/results.out
#SBATCH --job-name=result_extractor
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"

cd ..

source activate bertie

cd checkpoints 
python print_results.py

