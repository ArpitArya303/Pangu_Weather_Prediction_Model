#!/bin/bash
#SBATCH --job-name=Pangu 
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=16         
#SBATCH --partition=iiser_gpu  
#SBATCH --gres=gpu:1
#SBATCH --time=7-23:59:59      
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

export PYTHONPATH="/storage/arpit:${PYTHONPATH}"

echo "Job started at: $(date)"

python -u pangu_train.py \
    --data /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --surface_variables 2m_temperature mean_sea_level_pressure 10m_u_component_of_wind 10m_v_component_of_wind \
    --upper_air_variables geopotential specific_humidity temperature u_component_of_wind v_component_of_wind \
    --pLevels 250 500 850 \
    --static_variables soil_type land_sea_mask \
    --batch_size 64 \
    --num_epochs 200 \
    --log_dir /storage/arpit/Pangu/Logs/exp_19var/run_200epoch_64b1 \
    --transform_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data/test_19var \
    --accumulation_steps 1 
echo "Job ended at: $(date)"
