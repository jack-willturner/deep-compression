#!/bin/sh
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=logs/baselines.out
#SBATCH --job-name=baselines
#SBATCH --gres=gpu:3
#SBATCH --mem=42000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"
export DATA_LOC="../datasets/cifar10"

cd ..

source activate bertie
echo 'bertie activated'
nvidia-smi

for seed in 1 2 3
do
    python train.py --model='wrn_40_2' --data_loc="../datasets/cifar10" --seed=$seed --n_gpus=1 &
    python train.py --model='wrn_16_2' --data_loc="../datasets/cifar10" --seed=$seed --n_gpus=1 &
    python train.py --model='wrn_40_1' --data_loc="../datasets/cifar10" --seed=$seed --n_gpus=1
done
