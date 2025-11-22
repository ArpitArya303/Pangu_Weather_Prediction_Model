#!/bin/bash
#SBATCH --job-name=Pangu_predict
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64            
#SBATCH --partition=iiser  
#SBATCH --time=1-00:09:59
#SBATCH --gres=gpu:1      
#SBATCH --output=/storage/arpit/Pangu/Output/output_predict.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_predict.log

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

python -u ../climatology.py \
    --zarr_path /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --output_path /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/climatology/exp_19var     
echo "Job ended at: $(date)"