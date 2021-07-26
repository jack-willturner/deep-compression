#!/bin/sh
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=logs/baselines.out
#SBATCH --job-name=baselines
#SBATCH --gres=pu:1
#SBATCH --mem=14000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"

cd ..

source activate bertie
echo 'bertie activated'
nvidia-smi

for seed in 3
do
    python train.py --model='resnet18' --checkpoint='resnet18' --seed=$seed 
    python train.py --model='resnet34' --checkpoint='resnet34' --seed=$seed
    python train.py --model='resnet50' --checkpoint='resnet50' --seed=$seed

    python train.py --model='wrn_40_2' --checkpoint='wrn_40_2' --seed=$seed
    python train.py --model='wrn_16_2' --checkpoint='wrn_16_2' --seed=$seed
    python train.py --model='wrn_40_1' --checkpoint='wrn_40_1' --seed=$seed
done
