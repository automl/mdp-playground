#!/bin/bash
#MOAB -N mdpp
#MOAB -t 0-9 # specifies array job indices
#MOAB -l nodes=1:ppn=1
#MOAB -l walltime=0:00:30:00
#MOAB -l pmem=4GB # Seems like it is memory per CPU core
#MOAB -d /work/ws/nemo/fr_rr1034-ws_mdpp-0 # initial working dir.
##MOAB -V # export env. variables from launch env. I think
##MOAB -o output_filename
##MOAB


# This is a workaround for a know bug.
# Arrayjobs need to be given the output directory
#mkdir -p mdpp_${MOAB_JOBID}
#cd mdpp_${MOAB_JOBID}

echo -e '\033[32m'
echo "Started at $(date)";
# Output general info, timing info
echo "TMPDIR: " $TMPDIR

printenv

export EXP_NAME='rainbow_hydra' # Ideally contains Area of research + algorithm + dataset # Could just pass this as job name?

echo -e '\033[32m'
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with Job ID: $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
#echo SLURM_CONF location: ${SLURM_CONF}

#cat /proc/cpuinfo
#cat /proc/meminfo
#df -h
/bin/hostname -f

python3 -V

#export PATH="/home/rajanr/anaconda2/bin:$PATH"
echo Paths: $PATH
echo Parent program $0
echo Shell used is $SHELL
# type -a source

# source activate /home/rajanr/anaconda2/envs/py36
# source activate /home/rajanr/anaconda3/envs/py36_toy_rl
. /home/fr/fr_fr/fr_rr1034/anaconda3/etc/profile.d/conda.sh # for anaconda3
conda activate /home/fr/fr_fr/fr_rr1034/anaconda3/envs/old_py36_toy_rl # should be conda activate and not source when using anaconda3?
echo $?
echo Paths: $PATH
#/home/rajanr/anaconda3/bin/conda activate /home/rajanr/anaconda2/envs/py36
which python

python -V
which python3
python3 -V
ping google.com -c 3

echo "Line common to all tasks with MOAB_JOBID: ${MOAB_JOBID}, MOAB_JOBID: ${MOAB_JOBID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo -e '\033[0m'

echo -e "Script file start:\n====================="
cat $0
echo -e "\n======================\nScript file end!"
#pip install hyperopt lightgbm


# ================================================== #
# Begin actual Code
JOB_ID=`echo ${MOAB_JOBID} | cut -d'[' -f 1`
mkdir -p mdpp_${JOB_ID}
cd mdpp_${JOB_ID}
# cd /home/rajanr/mdpp
echo ${MOAB_JOBID} ${MOAB_JOBARRAYINDEX} ${MOAB_JOBNAME}
\time -v /home/fr/fr_fr/fr_rr1034/anaconda3/envs/py36_toy_rl/bin/python3 /home/fr/fr_fr/fr_rr1034/mdp-playground/run_experiments.py --exp-name ${EXP_NAME} --config-file /home/fr/fr_fr/fr_rr1034/mdp-playground/experiments/${EXP_NAME} --agent-config-num ${MOAB_JOBARRAYINDEX}

#python output_argv_1.py
#\time -v rllib train --env=BreakoutDeterministic-v4 --run=DQN --config '{"num_workers": 0, "monitor": true}'
#rllib train -f $HOME/atari-dqn_mod_breakout.yaml

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
