#!/bin/bash
#SBATCH --job-name=Pangu       
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64            
#SBATCH --partition=iiser_gpu  
#SBATCH --gres=gpu:1
#SBATCH --time=07:23:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_mean.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_mean.log

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

python -u cal_mean_std.py --zarr_path /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --surface_variables 2m_temperature mean_sea_level_pressure 10m_u_component_of_wind 10m_v_component_of_wind \
    --upper_air_variables geopotential specific_humidity temperature u_component_of_wind v_component_of_wind \
    --plevels 250 500 850 \
    --output_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data/test_19var
echo "Job ended at: $(date)"