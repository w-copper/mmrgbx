#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -p <your_partition, gpu>
#SBATCH -M <your_machine>
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -o slurm/slurm-%j.out

CONFIG=$1
PY_ARGS=${@:2} 
echo "" > ${CONFIG##*/}.log

srun python tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS} >> ${CONFIG##*/}.log 2>&1 \
        --cfg-options train_dataloader.batch_size=16 train_dataloader.num_workers=4