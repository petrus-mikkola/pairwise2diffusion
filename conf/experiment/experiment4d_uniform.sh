#!/bin/bash
#SBATCH --job-name=experiment4d_uniform
#SBATCH --account=project_2006300
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=50000
#SBATCH --time=48:00:00

SEEDS=$1  # e.g. "1,2,3,4,5"

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#srun python main.py --config-name=experiment4d_uniform --multirun exp.target=mixturegaussians,onegaussian exp.seed=1,2,3,4,5,6,7,8,9,10
srun python main.py --config-name=experiment4d_uniform --multirun exp.target=mixturegaussians,onegaussian exp.seed=$SEEDS
