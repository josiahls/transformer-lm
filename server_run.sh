#! /bin/bash
# exec 1>$PBS_O_WORKDIR/out 2>$PBS_O_WORKDIR/err
#
# ===== PBS OPTIONS =====
#
#PBS -N "robot_chef_jlaivins"
#PBS -q copperhead
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=1:gpus=1,mem=16GB
#PBS -V
#
# ===== END PBS OPTIONS =====
#
# ==== Main ======
cd $PBS_O_WORKDIR
mkdir log
{
module load pytorch/1.0.1-anaconda3-cuda10.0
conda activate master36
export DISPLAY=:0.0

sh tests/test_shakespeare.sh

} > log/output_"$PBS_JOBNAME"_$PBS_JOBID 2>log/errorLog_"$PBS_JOBNAME"_$PBS_JOBID
