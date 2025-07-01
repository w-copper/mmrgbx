#!/bin/bash
#SBATCH -A xdzhang
#SBATCH -p priv_v100x4
#SBATCH -M priv
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -o slurm/slurm-%j.out

source activate rsmae
module load nvidia/cuda/11.6

CONFIG=$1
PY_ARGS=${@:2} 
echo "" > ${CONFIG##*/}.log
# sbatch ./slurm_train_4g16c.sh configs/ksfa/res18-p16-a5r1-potsdam.py --cfg-options train_dataloader.batch_size=16 train_dataloader.num_workers=4
/usr/bin/srun python tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS} >> ${CONFIG##*/}.log 2>&1