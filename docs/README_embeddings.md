# ESM-C Embedding Generation for RefSeq Genomes

This pipeline generates ESM-C embeddings for all protein-coding genes in RefSeq genomes using the 600M parameter ESM-C model.

**Location**: `scripts/embeddings/`

## Setup

The required environment is already defined in `environment.yml`. Activate it with:

```bash
micromamba activate esm3_env
```

## Overview

- **Input**: Protein sequences from Prodigal annotations (`.faa` files) in `data/refseq_gene_annotations/`
- **Model**: ESM-C 600M (evolutionary scale modeling for proteins)
- **Output**: Per-gene embeddings saved in HDF5 or NPZ format

## Scripts

### 1. `test_embeddings.py` - Test Suite

Run this first to verify everything works:

```bash
python scripts/embeddings/test_embeddings.py
```

This will:
- Check that the ESM-C model loads correctly
- Test embedding generation on a single sequence
- Process 5 genes from a real protein file

### 2. `get_esm_embeddings.py` - Main Pipeline

Generate embeddings for all genomes:

```bash
# Test on 10 genomes first
python scripts/embeddings/get_esm_embeddings.py --num-genomes 10

# Run on all genomes
python scripts/embeddings/get_esm_embeddings.py

# With custom settings
python scripts/embeddings/get_esm_embeddings.py \
    --batch-size 64 \
    --max-length 2048 \
    --output-dir embeddings/esmc_600m \
    --save-format hdf5
```

### Options

- `--gene-dir`: Directory with `.faa` files (default: `data/refseq_gene_annotations`)
- `--output-dir`: Where to save embeddings (default: `embeddings`)
- `--model-name`: ESM-C model variant (default: `esmc_600m`)
- `--batch-size`: Sequences per batch (default: 32)
- `--max-length`: Max sequence length (default: 1024)
- `--device`: `cuda` or `cpu` (auto-detected)
- `--num-genomes`: Limit number of genomes for testing
- `--save-format`: `hdf5` or `npz`

## Output Format

### HDF5 Format (default)

Single file: `embeddings/esmc_embeddings.h5`

Structure:
```
genome_id_1/
  ├── embeddings    # (n_genes, embedding_dim) float32
  ├── gene_ids      # (n_genes,) string
  └── lengths       # (n_genes,) int
genome_id_2/
  ├── ...
```

Access in Python:
```python
import h5py
import numpy as np

with h5py.File("embeddings/esmc_embeddings.h5", "r") as f:
    genome_id = "GCF_000006985.1"
    embeddings = f[genome_id]["embeddings"][:]
    gene_ids = f[genome_id]["gene_ids"][:]
    lengths = f[genome_id]["lengths"][:]
```

### NPZ Format

One file per genome: `embeddings/{genome_id}_embeddings.npz`

Access in Python:
```python
import numpy as np

data = np.load("embeddings/GCF_000006985.1_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
gene_ids = data["gene_ids"]
lengths = data["lengths"]
```

### Metadata

`embeddings/embedding_metadata.csv` contains:
- `genome_id`: Genome identifier
- `num_genes`: Number of genes processed
- `embedding_dim`: Dimensionality of embeddings
- `avg_length`: Average protein sequence length

## Expected Performance

- **7,664 genomes** total
- **~32 genes/second** on A100 GPU (estimated)
- **Runtime**: ~several hours for all genomes
- **Storage**: ~1-5 GB depending on total genes

## Embedding Dimension

ESM-C 600M produces embeddings of dimension **960** (per residue), which are mean-pooled to create fixed-size sequence-level representations.

## Notes

- Sequences with stop codons (`*`) are cleaned
- Sequences longer than `--max-length` are filtered out
- Progress is tracked with tqdm
- Processing resumes if interrupted (when using NPZ format)
- GPU highly recommended for large-scale processing
