#!/bin/bash
#SBATCH --job-name=Pangu_plot 
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=8           
#SBATCH --partition=GPU-AI  
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_testing.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_testing.log

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Current date: $(date)"

# Load Conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate Pangu

module load cuda-12.9
module load cudnn-8.2

echo "CUDA devices:"
nvidia-smi

export PYTHONPATH="/storage/arpit:${PYTHONPATH}"

# Run the test script
echo "check data_utils.py"

python ../data_utils.py 

echo "Done checking data_utils.py"

