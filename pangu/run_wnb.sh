#!/bin/bash
#SBATCH --job-name=Pangu       
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64            
#SBATCH --partition=gpu_prio  
#SBATCH --gres=gpu:1
#SBATCH --time=7-23:59:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_train_wnb.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_train_wnb.log

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

export PYTHONPATH="/storage/arpit:${PYTHONPATH}"

echo "Job started at: $(date)"

python -u pangu_train_wnb.py \
    --config /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/config.yml \
    --run_name run_500epoch_30patience_wnb \

echo "Job ended at: $(date)"
