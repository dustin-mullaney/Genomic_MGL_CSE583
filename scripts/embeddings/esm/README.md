# ESM-C Embedding Scripts

Scripts for generating ESM-C (Evolutionary Scale Modeling) embeddings for protein sequences.

## Overview

**ESM-C** is a 600M parameter protein language model trained on evolutionary data. It generates 1152-dimensional embeddings that capture:
- Protein structure
- Function
- Evolutionary relationships

## Scripts

### Core Embedding Generation

- `get_esm_embeddings.py` - Main script for generating ESM-C embeddings
- `compute_protein_embeddings.py` - Batch embedding computation
- `compute_gpu_embeddings.py` - GPU-optimized embedding generation

### Testing & Utilities

- `test_embeddings.py` - Test embedding generation pipeline
- `test_gpu.py` - GPU availability testing
- `inspect_embeddings.py` - Examine embedding files

### SLURM Submission

- `submit_protein_embeddings.sh` - Batch job submission
- `submit_gpu_array.sh` - GPU array job for multiple genomes
- `submit_gpu_test.sh` - Test GPU setup
- `fix_pytorch_gpu.sh` - Fix PyTorch GPU issues

## Usage

### Generate embeddings for a single genome
```bash
python get_esm_embeddings.py \
    --input data/refseq_proteins/GCF_000001.faa \
    --output data/refseq_esm_embeddings/GCF_000001_embeddings.npz
```

### Batch process all genomes
```bash
sbatch submit_protein_embeddings.sh
```

### GPU array job
```bash
sbatch submit_gpu_array.sh
```

## Output Format

Embeddings saved as `.npz` files:
```python
import numpy as np
data = np.load('GCF_000001_embeddings.npz')

embeddings = data['embeddings']  # (n_genes, 1152)
gene_ids = data['gene_ids']      # (n_genes,)
```

## Requirements

- PyTorch with CUDA support
- ESM model from Facebook Research
- GPU with ≥16GB VRAM recommended

## Model Details

- **Model**: ESM-C (esmc_600m)
- **Parameters**: 600M
- **Embedding dimension**: 1152
- **Input**: Protein sequences (FASTA)
- **Training**: Evolutionary sequences from UniRef

## Performance

- **Speed**: ~100-500 sequences/second (GPU)
- **Memory**: ~8-16GB VRAM
- **Batch size**: Typically 8-32 sequences

## Troubleshooting

### GPU not detected
```bash
bash fix_pytorch_gpu.sh
```

### Out of memory
- Reduce batch size
- Use CPU instead (slower)
- Process in smaller chunks

## See Also

- **Clustering**: `../../clustering/`
- **gLM2 embeddings**: `../glm2/`
- **Visualization**: `../../visualization/`
