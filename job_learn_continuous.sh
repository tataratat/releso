#!/usr/local_rwth/bin/zsh
 
### Job name

#SBATCH --job-name=SbSORL_PPO_continous

#SBATCH --mail-type=ALL
#SBATCH --mail-user=clemens.fricke@rwth-aachen.de
 
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
 
### activate spack, splinelib
cd ~/
source ~/.zshrc
# spack load splinelib
 
### Load python module
# module load python/3.8.7
# cd /home/na103664/Schreibtisch/HeatPipe_venv/bin
# source activate

conda activate stable_baselines

cd ~/seminararbeit/sa072021-rl-shape-opti-framework/05-Code/SORL
python -m SbSOvRL -i ../../04-Data/examples/input_c_multi_continuous.json

### If you only want to use a CPU, the CUDA and cuDNN modules are
### unnecessary. Make sure you have 'tensorflow' installed, because
### using 'tensorflow-gpu' without those modules will crash your
### program.
 
### Execute your application
# cd /home/na103664/Schreibtisch/shape-opti-rl/uebungen/stable_baselines/FreeFormDeformation_learn
# python freeFormv1.py
# cd /home/na103664/Schreibtisch/HeatPipe_venv/bin
# deactivate
