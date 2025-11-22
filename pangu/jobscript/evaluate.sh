#!/bin/bash
#SBATCH --job-name=Pangu_test       
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64            
#SBATCH --partition=iiser 
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:09:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_prediction.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_prediction.log

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

python -u ../evaluate.py \
    --zarr_path /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --climatology_path /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/climatology/exp_19var/clim_6hourly.nc \
    --prediction_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/prediction/exp_19var_autoregressive \
    --output_path /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/evaluate/exp_19var/300epoch_64b

echo "Job ended at: $(date)"