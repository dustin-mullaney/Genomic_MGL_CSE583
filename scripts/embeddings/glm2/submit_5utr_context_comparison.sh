#!/bin/bash
#SBATCH --job-name=5utr_context
#SBATCH --output=logs/5utr_context_comparison_%j.out
#SBATCH --error=logs/5utr_context_comparison_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Print job information
echo "========================================"
echo "5' UTR Context Effect Comparison"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate glm2_env

echo "========================================"
echo "Comparing 5' UTR Jacobians"
echo "========================================"
echo ""

# Run the analysis
python scripts/embeddings/compare_5utr_context_effect.py \
    --utr-jacobian-dir results/jacobian_5utr \
    --genomic-jacobian-dir results/jacobian_genomic \
    --output-dir results/jacobian_5utr_context_comparison \
    --dna-flank-length 500

echo ""
echo "========================================"
echo "Job Complete!"
echo "========================================"
