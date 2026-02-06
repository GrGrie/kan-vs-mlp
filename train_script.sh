#!/bin/bash
#SBATCH --time=02:00:00                      
#SBATCH -J KAN_REG_Weight_0.01_lr0.01_seed42
#SBATCH --mem=24G
#SBATCH --ntasks=1					# User 1 tasks
#SBATCH --cpus-per-task=3           # Use 1 thread per task        
#SBATCH -N 1						# Request slots on 1 node
#SBATCH --gres=gpu:1                  
#SBATCH --partition=informatik-mind    
#SBATCH -o ./slurm_output/%x.%j.out # Output: %j expands to jobid
#SBATCH -e ./slurm_output/%x.%j.err # Error: %j expands to jobid

export TORCH_CUDNN_V8_API_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge
module load miniforge3/latest
module load gcc/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate grgrie_pyold

cd /home/xar68reb/kan-vs-mlp/

# ——— hyperparameters ———
HEAD="kan"
EPOCHS=200
KAN_REG_WEIGHT=0.01
BATCH_SIZE=64


echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Running training with arguments:
  --head       				$HEAD
  --epochs        			$EPOCHS
  --KAN_REG_WEIGHT			$KAN_REG_WEIGHT
  --batch_size				$BATCH_SIZE"

 
python train.py \
  --head "$HEAD" \
  --epochs "$EPOCHS" \
  --kan_reg_weight "$KAN_REG_WEIGHT" \
  --batch_size "$BATCH_SIZE"
 

conda deactivate

