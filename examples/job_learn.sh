#!/usr/local_rwth/bin/zsh
 
### Job name

#SBATCH --job-name=SbSORL_PPO_continous

#SBATCH --mail-type=ALL
#SBATCH --mail-user=example@rwth-aachen.de
 
### Output path for stdout and stderr
### %J is the job ID
#SBATCH --output output_%J.txt
 
### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=72:00:00
 
### Request the amount of memory you need for your job.
### You can specify this in either MB (1024M) or GB (4G).
#SBATCH --mem-per-cpu=2000M
 
### Request the number of parallel threads for your job
#SBATCH --ntasks=4
 
### activate your console environment (should also activate conda)
cd ~/
source ~/.zshrc
# spack load splinelib
 


## Activate the correct conda environemnt (PLEASE check if the environemnt name is correct)
conda activate sbsovrl

# PLASE change the following lone to the base directory of yout experiments
cd ~/seminararbeit/sa072021-rl-shape-opti-framework/05-Code/SORL

# PLEASE change the following line so that the path points to the json file describing the current experiment.