#!/bin/bash
#SBATCH --job-name=Pangu_plot_metrics       
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   		
#SBATCH --cpus-per-task=64            
#SBATCH --partition=GPU-AI  
#SBATCH --time=1-00:09:59      
#SBATCH --output=/storage/arpit/Pangu/Output/output_plot_metrics.log  # Save logs here
#SBATCH --error=/storage/arpit/Pangu/Output/error_plot_metrics.log

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

python -u ../plot_metrics.py \
    --csv_file /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/evaluate/exp_20var/200epoch_64b_2gpu/acc_lead_time.csv \
    --output_dir /storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/visualizations/exp_20var/200epoch_64b_2gpu/ACC \
    --metric rmse 
echo "Job ended at: $(date)"