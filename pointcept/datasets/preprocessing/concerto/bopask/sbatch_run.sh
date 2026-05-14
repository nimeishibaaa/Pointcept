#!/bin/bash
#SBATCH -J bopask_prep
#SBATCH -p gpua800
#SBATCH -q 16a800
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 02:00:00

bash /gpfs/work/mec/yiwenwang18/Pointcept/pointcept/datasets/preprocessing/concerto/bopask/preprocess_bopask.sh