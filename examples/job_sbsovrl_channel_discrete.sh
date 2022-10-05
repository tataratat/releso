#!/usr/local_rwth/bin/zsh
 
### Job name

#SBATCH --job-name=SbSOvRL_PPO_discrete_channel

#SBATCH --mail-type=ALL

# Please enter your email address here...

#SBATCH --mail-user=clemens.fricke@rwth-aachen.de
#SBATCH -p c18m
#SBATCH --nodes 1
 
### Output path for stdout and stderr
### %J is the job ID
#SBATCH --output output_%J.txt
 
### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=48:00:00
 
### Request the amount of memory you need for your job.
### You can specify this in either MB (1024M) or GB (4G).
#SBATCH --mem-per-cpu=400M
 
### Request the number of parallel threads for your job
#SBATCH --ntasks=4
 
### activate spack, splinelib
cd ~/
source ~/.zshrc

conda activate sbsovrl

# adapt this path to your specific location
cd ~/git/Examples/channel
python -m SbSOvRL -i sbsovrl_configurations/sbsovrl_channel_discrete.json
