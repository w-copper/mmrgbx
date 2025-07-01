#!/bin/bash
#SBATCH -A xdzhang
#SBATCH -p priv_v100x4
#SBATCH -M priv
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

source activate rsmae
module load nvidia/cuda/11.6

CONFIG=$1
PY_ARGS=${@:2}

srun python tools/datasets/generate_mask_for_rgb.py  --dataset $CONFIG