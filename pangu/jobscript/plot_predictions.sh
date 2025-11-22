#!/bin/bash
#SBATCH --job-name=Pangu_test       
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64            
#SBATCH --partition=iiser_gpu  
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

echo "Job started at: $(date)"

python -u plot_predictions.py \
    --zarr_path /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --prediction_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/prediction/exp_19var/100epoch_64b \
    --output_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/visualizations/exp_19var/100epoch_64b/predictions \
    --surface_variables 2m_temperature mean_sea_level_pressure 10m_u_component_of_wind 10m_v_component_of_wind \
    --upper_air_variables geopotential specific_humidity temperature u_component_of_wind v_component_of_wind \
    --pressure_levels 250 500 850 \
    --cmap viridis \
    --sample_limit 2 \
    --test_years 2021 2023 \
    --lead_time_hours 6

echo "Job ended at: $(date)"