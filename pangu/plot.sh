#!/bin/bash
#SBATCH --job-name=Pangu_plot 
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64           
#SBATCH --partition=iiser_gpu  
#SBATCH --gres=gpu:1
#SBATCH --time=7-23:59:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_plotting.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_plotting.log

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

# Create comparison plots for SPECIFIC pressure levels only
echo "Creating comparison plots for SPECIFIC pressure levels (250, 500, 850 hPa)..."
python plot.py \
    --mode compare \
    --data_path /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --prediction_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/prediction/exp_8var/original \
    --output_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/visualizations/exp_8var/original_together \
    --lead_time 6 \
    --surface_variables 2m_temperature mean_sea_level_pressure  \
    --upper_air_variables geopotential  temperature \
    --pressure_levels 100 200 300 \
    --num_samples 2 \

echo "Done! Check /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/visualizations/exp_8var/original_together for the comparison plots at specific levels."

