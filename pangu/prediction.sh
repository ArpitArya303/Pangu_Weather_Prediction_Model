#!/bin/bash
#SBATCH --job-name=Pangu_3 
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64            
#SBATCH --partition=gpu_prio  
#SBATCH --gres=gpu:1
#SBATCH --time=7-23:59:59      
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

python prediction.py \
    --data_path /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --model_path /storage/arpit/Pangu/Logs/exp/run_200epoch_no_stop/best_model.pth \
    --transform_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data \
    --output_path /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/prediction/together \
    --surface_variables 2m_temperature mean_sea_level_pressure \
    --upper_air_variables geopotential temperature \
    --pressure_levels 100 200 300 \
    --static_variables land_sea_mask soil_type \
    --test_years 2021 2023 \
    --num_workers 8 \
    --num_samples 5 \
    --lead_time 6 \
    --batch_size 128 

echo "Job ended at: $(date)"
