#!/bin/bash
# SLURM job for the NTU GPU cluster (MGPU-TC2 partition).
# Usage (from the repo root):
#   sbatch slurm/run_experiment.sh --model openllama-3b --adapter lora
#   sbatch slurm/run_experiment.sh --model openllama-3b --adapter lora --data math_10k
#SBATCH --job-name=peft-math
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-jupyterenv}"

cd "$SLURM_SUBMIT_DIR"
python run_experiment.py "$@"
