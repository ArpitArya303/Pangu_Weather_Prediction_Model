#!/bin/bash
#SBATCH --job-name=Pangu       
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=8            
#SBATCH --partition=gpu_prio  
#SBATCH --gres=gpu:1
#SBATCH --time=00:09:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_train.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_train.log

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Current date: $(date)"

# Load Conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate Pangu

module load cuda-12.4
module load cudnn-8.2

echo "CUDA devices:"
nvidia-smi

echo "Job started at: $(date)"

python -u test.py --model_name epochs/pangu_lite_epoch_10.pth --lead_time 24

echo "Job ended at: $(date)"