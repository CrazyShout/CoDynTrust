#!/bin/bash 
#SBATCH --job-name=xyj_gpu_job
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1

# 进入你的工作目录
cd /public/home/lilingzhi/xyj/CoAlign/
source /public/home/lilingzhi/xyj/CoAlign/env_vars.sh
eval "$(conda shell.bash hook)"
conda activate coalign

python opencood/tools/train.py -y opencood/hypes_yaml/opv2v/lidar_only_with_noise/coalign/pointpillar_uncertainty.yaml -f no_w_uncertainty
