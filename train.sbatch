#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=50GB
#SBATCH --account=pr_130_tandon_advanced 


module purge

singularity exec --nv \
	    --overlay /scratch/yl9727/neuro_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python main.py --pid 5246af08-0730-40f7-83de-29b5d62b9b6d --ephys_path ephys_data --out_path output --model_type glm --behavior wheel_velocity"


		