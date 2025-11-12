#!/bin/bash
#SBATCH --job-name=diagnose_tokens
#SBATCH --output=logs/diagnose_tokens_%j.out
#SBATCH --error=logs/diagnose_tokens_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=chorus
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Print job information
echo "========================================"
echo "gLM2 Tokenization Diagnostic"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate glm2_env

echo "========================================"
echo "Running Diagnostic"
echo "========================================"
echo ""

# Run the diagnostic
python scripts/embeddings/diagnose_tokenization.py

echo ""
echo "========================================"
echo "Job Complete!"
echo "========================================"
