#!/bin/bash
#SBATCH --job-name=ablation2d
#SBATCH --account=project_2006300
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=14000
#SBATCH --time=48:00:00

SEEDS=$1  # e.g. "1"

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#srun python main.py --config-name=ablation_RUM_2d --multirun exp.target=onemoon,twomoons,ring exp.rum_noise_dist=gumbel,exponential,normal exp.s_true=0.1,5.0,1.0,0.7797 exp.seed=1,2,3,4,5,6,7,8,9,10 
srun python main.py --config-name=ablation_RUM_2d --multirun exp.target=onemoon,twomoons,ring exp.rum_noise_dist=gumbel,exponential,normal exp.s_true=0.1,5.0,1.0,0.7797 exp.seed=$SEEDS