#!/bin/sh -f
#
###############################
# Specify your SLURM directives
###############################
# User's Mail:
##SBATCH --mail-user=matthias.mayr@unibw.de
# When to send mail?:
##SBATCH --mail-type=BEGIN,END,FAIL
#
# Job name:
#SBATCH --job-name "releso"
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Define if the job should only be run on certain nodes.
##{is_feature}SBATCH --constraint={self.feature}
#
# Standard case: specify only number of cpus
# #SBATCH --ntasks=24
#
# If you want to specify a certain number of nodes
# and exactly 'ntasks-per-node' cpus on each node.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#
# For hybrid mpi: e.g. 2 mpi processes each with
# 4 openmp threads
# #SBATCH --ntasks=2
# #SBATCH --cpus-per-task=4
#
# Allocate full node and block for other jobs
#SBATCH --exclusive
#
# Walltime:
#SBATCH --time=48:00:00
###########################################

# Store calling directroy
CWD=$(pwd)
echo $CWD

# Setup shell environment
echo $HOME
cd $HOME
#source /etc/profile.d/modules.sh
#source /home/opt/cluster_tools/core/load_baci_environment.sh
source /home/a11bivst/load_python.sh

# Go back to calling directroy
cd  $CWD

python3 -m SbSOvRL -i examples/bw_beispiel_incremental.json

# END ################## DO NOT TOUCH ME #########################
echo
echo "Job finished with exit code $? at: `date`"
