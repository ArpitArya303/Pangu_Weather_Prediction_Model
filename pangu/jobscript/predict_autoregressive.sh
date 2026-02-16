#!/bin/bash
#SBATCH --job-name=Pangu_predict       
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=16            
#SBATCH --partition=GPU-AI  
#SBATCH --time=07:23:59      
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
conda activate pangu

echo "CUDA devices:"
nvidia-smi

export PYTHONPATH="/storage/arpit:${PYTHONPATH}"

echo "Job started at: $(date)"

python -u ../predict_autoregressive.py \
    --data_path /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
    --model_path /storage/arpit/Pangu/Logs/exp_20var/run_200epoch_64b_2gpu/best_model.pth \
    --transform_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data/20var \
    --surface_variables 2m_temperature mean_sea_level_pressure total_column_water_vapour 10m_u_component_of_wind 10m_v_component_of_wind \
    --upper_air_variables geopotential specific_humidity temperature u_component_of_wind v_component_of_wind \
    --pressure_levels 250 500 850 \
    --output_path /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/prediction/exp_20var_autoregressive \
    --batch_size 64 \
    --num_samples 20 \
    --lead_time 120 \
    --num_samples 1 \
    --test_years 2021 2022 2023
echo "Job ended at: $(date)"