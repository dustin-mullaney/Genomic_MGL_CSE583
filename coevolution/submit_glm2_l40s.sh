#!/bin/bash
#SBATCH --job-name=glm2_embeddings
#SBATCH --partition=chorus
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/glm2_%A_%a.out
#SBATCH --error=logs/glm2_%A_%a.err

################################################################################
# gLM2 Embedding Generation on L40S GPUs
#
# Usage:
#   sbatch scripts/embeddings/submit_glm2_l40s.sh
#
# For array job (parallel processing):
#   sbatch --array=0-99 scripts/embeddings/submit_glm2_l40s.sh
################################################################################

echo "========================================"
echo "gLM2 Embedding Generation - L40S GPU"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
fi
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load CUDA module
module load CUDA/11.7.0

# Load conda/micromamba
source ~/.bashrc
micromamba activate glm2_env

# Verify GPU
echo "Checking GPU availability..."
nvidia-smi
echo ""
echo "CUDA module loaded:"
module list
echo ""

# Check if this is an array job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    MODE="array"
    TASK_ID=$SLURM_ARRAY_TASK_ID
else
    MODE="single"
    TASK_ID=0
fi

# Configuration
GENOME_DIR="data/refseq_genomes"
ANNOTATION_DIR="data/refseq_gene_annotations"
OUTPUT_DIR="embeddings/glm2"
MODEL_NAME="tattabio/gLM2_650M"

# Embedding mode: "genome" or "genes"
EMBEDDING_MODE="${EMBEDDING_MODE:-genome}"

# Batch size (L40S has 48GB memory - can handle larger batches)
BATCH_SIZE="${BATCH_SIZE:-16}"

# Chunk size for genome mode
CHUNK_SIZE="${CHUNK_SIZE:-3000}"

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Embedding mode: $EMBEDDING_MODE"
echo "  Genome dir: $GENOME_DIR"
echo "  Annotation dir: $ANNOTATION_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Chunk size: $CHUNK_SIZE"
echo ""

# Run embedding generation
if [ "$MODE" = "array" ]; then
    # Array job mode - process subset of genomes
    echo "Running array job (task $TASK_ID)..."

    # Calculate genome range for this task
    # Assuming 7,664 genomes total, split across array tasks
    TOTAL_GENOMES=$(ls -1 $GENOME_DIR/*.fna 2>/dev/null | wc -l)
    if [ $TOTAL_GENOMES -eq 0 ]; then
        TOTAL_GENOMES=$(ls -1 $GENOME_DIR/*.fasta 2>/dev/null | wc -l)
    fi

    # For 100 array tasks (0-99), each processes ~77 genomes
    GENOMES_PER_TASK=$(( (TOTAL_GENOMES + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
    START_IDX=$(( TASK_ID * GENOMES_PER_TASK ))

    echo "  Total genomes: $TOTAL_GENOMES"
    echo "  Genomes per task: $GENOMES_PER_TASK"
    echo "  Start index: $START_IDX"
    echo ""

    # Create temporary directory with subset of genomes
    TMP_GENOME_DIR="tmp/glm2_genomes_$SLURM_JOB_ID.$TASK_ID"
    mkdir -p $TMP_GENOME_DIR

    # Copy/link subset of genomes
    ls -1 $GENOME_DIR/*.fna 2>/dev/null | tail -n +$((START_IDX + 1)) | head -n $GENOMES_PER_TASK | \
        while read genome; do
            ln -s "$(realpath $genome)" "$TMP_GENOME_DIR/"
        done

    # If no .fna files, try .fasta
    if [ $(ls -1 $TMP_GENOME_DIR/*.fna 2>/dev/null | wc -l) -eq 0 ]; then
        ls -1 $GENOME_DIR/*.fasta 2>/dev/null | tail -n +$((START_IDX + 1)) | head -n $GENOMES_PER_TASK | \
            while read genome; do
                ln -s "$(realpath $genome)" "$TMP_GENOME_DIR/"
            done
    fi

    ACTUAL_GENOMES=$(ls -1 $TMP_GENOME_DIR/*.fna 2>/dev/null | wc -l)
    if [ $ACTUAL_GENOMES -eq 0 ]; then
        ACTUAL_GENOMES=$(ls -1 $TMP_GENOME_DIR/*.fasta 2>/dev/null | wc -l)
    fi

    echo "Processing $ACTUAL_GENOMES genomes in this task"

    # Run on subset
    python scripts/embeddings/get_glm2_embeddings.py \
        --genome-dir $TMP_GENOME_DIR \
        --annotation-dir $ANNOTATION_DIR \
        --output-dir $OUTPUT_DIR \
        --model-name $MODEL_NAME \
        --mode $EMBEDDING_MODE \
        --batch-size $BATCH_SIZE \
        --chunk-size $CHUNK_SIZE \
        --device cuda \
        --dtype bfloat16 \
        --save-format npz

    # Cleanup
    rm -rf $TMP_GENOME_DIR

else
    # Single job mode - process all or specified number
    echo "Running single job mode..."

    NUM_GENOMES="${NUM_GENOMES:-}"
    if [ -n "$NUM_GENOMES" ]; then
        echo "  Processing first $NUM_GENOMES genomes"
        NUM_GENOMES_ARG="--num-genomes $NUM_GENOMES"
    else
        echo "  Processing all genomes"
        NUM_GENOMES_ARG=""
    fi

    python scripts/embeddings/get_glm2_embeddings.py \
        --genome-dir $GENOME_DIR \
        --annotation-dir $ANNOTATION_DIR \
        --output-dir $OUTPUT_DIR \
        --model-name $MODEL_NAME \
        --mode $EMBEDDING_MODE \
        --batch-size $BATCH_SIZE \
        --chunk-size $CHUNK_SIZE \
        --device cuda \
        --dtype bfloat16 \
        --save-format hdf5 \
        $NUM_GENOMES_ARG
fi

echo ""
echo "========================================"
echo "Job completed!"
echo "End time: $(date)"
echo "========================================"
