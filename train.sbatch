#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --account=pr_130_tandon_advanced 


module purge

singularity exec --nv \
	    --overlay /scratch/yl9727/neuro_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python main.py"