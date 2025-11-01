# gLM2 Quick Start Guide - L40S GPUs

This guide will get you up and running with gLM2 embeddings on L40S GPUs.

## Step 1: Create Micromamba Environment

```bash
# Create the environment
micromamba env create -f environment_glm2.yml

# Activate it
micromamba activate glm2_env

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel; print('Transformers: OK')"
```

## Step 2: Test the Installation

```bash
# Run the test suite
python scripts/embeddings/test_glm2_embeddings.py
```

This will:
- ✓ Test imports
- ✓ Download and load gLM2_650M model (~2.5GB)
- ✓ Run inference on test sequences
- ✓ Create test genome and generate embeddings

## Step 3: Quick Test on Real Data

```bash
# Test on a single genome
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --output-dir tmp/glm2_test/ \
    --num-genomes 1 \
    --mode genome \
    --device cuda \
    --batch-size 16
```

## Step 4: Submit to SLURM (L40S GPU)

### Single Test Job

```bash
# Create logs directory
mkdir -p logs

# Test with 10 genomes
NUM_GENOMES=10 EMBEDDING_MODE=genome sbatch scripts/embeddings/submit_glm2_l40s.sh
```

### Full Array Job (Parallel Processing)

```bash
# Process all genomes across 100 L40S GPUs
sbatch --array=0-99 scripts/embeddings/submit_glm2_l40s.sh

# Check job status
squeue -u $USER

# Monitor specific job
tail -f logs/glm2_<JOB_ID>_0.out
```

## Step 5: Choose Your Mode

### Genome Mode (Recommended for Context Analysis)

```bash
EMBEDDING_MODE=genome sbatch scripts/embeddings/submit_glm2_l40s.sh
```

**Output:** Genome chunks with genomic context
- Best for: Genomic organization, regulatory regions, operons
- Chunks: ~3000 tokens each
- Embedding: (n_chunks, 1280)

### Genes Mode (Comparable to ESM-C)

```bash
EMBEDDING_MODE=genes sbatch scripts/embeddings/submit_glm2_l40s.sh
```

**Output:** Individual gene embeddings
- Best for: Gene-level analysis, comparison with ESM-C
- One embedding per gene
- Embedding: (n_genes, 1280)

## Performance on L40S

- **GPU Memory:** 48GB (can handle large batches)
- **Speed:** ~30-60 seconds per genome (genome mode)
- **Batch size:** 16-32 (recommended)
- **Expected time (7,664 genomes):**
  - Single GPU: ~10-15 hours
  - 100 GPUs (array): ~10-15 minutes

## Environment Variables

Customize the SLURM job with these variables:

```bash
# Embedding mode
EMBEDDING_MODE=genome  # or "genes"

# Batch size (adjust based on memory)
BATCH_SIZE=16          # up to 32 on L40S

# Chunk size (for genome mode)
CHUNK_SIZE=3000        # tokens per chunk

# Test with subset
NUM_GENOMES=10         # for testing only

# Example: Test with 10 genomes in genes mode
NUM_GENOMES=10 EMBEDDING_MODE=genes BATCH_SIZE=32 \
    sbatch scripts/embeddings/submit_glm2_l40s.sh
```

## Check Results

```python
import h5py
import numpy as np

# Load results
with h5py.File('embeddings/glm2/glm2_embeddings_genome.h5', 'r') as f:
    print(f"Genomes processed: {len(f.keys())}")

    # First genome
    genome_id = list(f.keys())[0]
    print(f"\nGenome: {genome_id}")
    print(f"  Chunks: {f[genome_id].attrs['n_items']}")
    print(f"  Embedding dim: {f[genome_id].attrs['embedding_dim']}")

    # Load embeddings
    embeddings = f[genome_id]['embeddings'][:]
    print(f"  Shape: {embeddings.shape}")
```

## Troubleshooting

### Issue: Environment creation fails

```bash
# Update micromamba
micromamba self-update

# Try with mamba instead
mamba env create -f environment_glm2.yml
```

### Issue: CUDA not available

```bash
# Check CUDA version
nvidia-smi

# Install matching pytorch-cuda version
micromamba install pytorch-cuda=12.1  # or 11.8
```

### Issue: Model download fails

```bash
# Pre-download the model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('tattabio/gLM2_650M')"

# Or set cache directory
export HF_HOME=/path/to/huggingface/cache
```

## Next Steps

1. **Compare with ESM-C:** Generate embeddings in both modes and compare
2. **Dimensionality Reduction:** Run UMAP on gLM2 embeddings
3. **Clustering:** Apply clustering algorithms
4. **Functional Analysis:** Merge with COG annotations

See [docs/README_glm2.md](../docs/README_glm2.md) for detailed documentation.
