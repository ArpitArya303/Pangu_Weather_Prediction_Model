#!/bin/bash
#SBATCH --job-name=Pangu_6hr
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1          # Single task, accelerate handles processes
#SBATCH --cpus-per-task=16           # CPUs for data loading
#SBATCH --partition=gpu  
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --time=7-23:59:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_train1.log
#SBATCH --error=/storage/arpit/Pangu/Output/error_train1.log

# Print job information
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Current date: $(date)"
echo "======================================"

# Load Conda
source /home/arpit/miniconda3/etc/profile.d/conda.sh
conda activate Pangu

# Verify conda environment
echo "Conda environment: $CONDA_DEFAULT_ENV"
which python
python --version

# Load CUDA modules
module load cuda-12.9
module load cudnn-8.2

# Accelerate config expects gpu_ids: 0,1 and num_processes: 2
export CUDA_VISIBLE_DEVICES=0,1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Checking GPUs:"
nvidia-smi

# Verify PyTorch can see GPUs
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

export PYTHONPATH="/storage/arpit:${PYTHONPATH}"

echo "======================================"
echo "Job started at: $(date)"
echo "======================================"

# Launch using saved accelerate config
accelerate launch pangu_train.py \
    --data /home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
    --surface_variables 2m_temperature mean_sea_level_pressure 10m_u_component_of_wind 10m_v_component_of_wind \
    --upper_air_variables geopotential specific_humidity temperature u_component_of_wind v_component_of_wind \
    --pLevels 250 500 850 \
    --static_variables soil_type land_sea_mask \
    --batch_size 64 \
    --num_epochs 200 \
    --log_dir /storage/arpit/Pangu/Logs/exp_19var/run_200epoch_64b_2gpu \
    --transform_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data/test_19var \
    --accumulation_steps 1 \
    --num_workers 8 \
    --patience 10 \
    --seed 42

echo "======================================"
echo "Job ended at: $(date)"
echo "======================================"