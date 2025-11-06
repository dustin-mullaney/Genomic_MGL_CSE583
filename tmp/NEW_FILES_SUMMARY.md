# gLM2 Implementation Summary

This document summarizes all new files created for gLM2 genomic language model integration.

## New Files Created

### 1. Core Scripts

#### `scripts/embeddings/get_glm2_embeddings.py`
Main script for generating gLM2 embeddings from bacterial genomes.

**Key features:**
- Two modes: `genome` (chunks with context) and `genes` (per-gene embeddings)
- Processes genomic DNA + GFF annotations
- Formats sequences as mixed-modality (protein CDS + nucleotide IGS)
- Supports both HDF5 and NPZ output formats
- GPU-accelerated with bfloat16 precision

**Usage:**
```bash
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --mode genome \
    --device cuda
```

#### `scripts/embeddings/test_glm2_embeddings.py`
Comprehensive test suite for gLM2 pipeline.

**Tests:**
- Model imports and dependencies
- Model loading (tattabio/gLM2_650M)
- Inference on test sequences
- Full pipeline with synthetic test genome
- Both genome and genes modes

**Usage:**
```bash
python scripts/embeddings/test_glm2_embeddings.py
```

#### `scripts/embeddings/compare_glm2_esmc.py`
**NEW** - Compares gLM2 and ESM-C embeddings on standard protein set.

**Analysis:**
- Samples 1000 proteins from dataset
- Generates embeddings with both models
- Computes pairwise distance matrices
- Analyzes inter-embedding distance correlation
- Nearest neighbor agreement analysis
- Comprehensive visualizations

**Key innovation:** Handles different embedding dimensions (ESM-C: 960, gLM2: 1280) by comparing distance matrices instead of raw embeddings.

**Usage:**
```bash
python scripts/embeddings/compare_glm2_esmc.py \
    --protein-dir data/refseq_gene_annotations/ \
    --output-dir tmp/comparison/ \
    --n-proteins 1000 \
    --distance-metric cosine
```

**Outputs:**
- `comparison_results.json` - Statistical results
- `distance_comparison.png` - Scatter plots
- `distance_distributions.png` - Distribution analysis
- `distance_matrices.png` - Heatmap visualizations
- `protein_set.csv` - Sampled proteins
- `esmc_embeddings.npy` / `glm2_embeddings.npy` - Raw embeddings

### 2. SLURM Job Scripts

#### `scripts/embeddings/submit_glm2_l40s.sh`
SLURM submission script optimized for L40S GPUs (48GB memory).

**Features:**
- Single or array job modes
- Configurable via environment variables
- Automatic genome chunking for parallel processing
- Smart logging and error handling

**Environment variables:**
- `EMBEDDING_MODE`: "genome" or "genes"
- `BATCH_SIZE`: Batch size (default: 16, can go up to 32 on L40S)
- `CHUNK_SIZE`: Tokens per genome chunk (default: 3000)
- `NUM_GENOMES`: For testing with subset

**Usage:**
```bash
# Single job test
NUM_GENOMES=10 sbatch scripts/embeddings/submit_glm2_l40s.sh

# Full array job (100 L40S GPUs in parallel)
sbatch --array=0-99 scripts/embeddings/submit_glm2_l40s.sh

# Genes mode with custom batch size
EMBEDDING_MODE=genes BATCH_SIZE=32 sbatch scripts/embeddings/submit_glm2_l40s.sh
```

### 3. Environment Configuration

#### `environment_glm2.yml`
Micromamba/conda environment for gLM2.

**Key packages:**
- Python 3.10
- PyTorch 2.0+ with CUDA 11.8
- Transformers 4.35+
- BioPython for sequence handling
- Standard ML stack (numpy, scipy, pandas, scikit-learn)
- UMAP, HDBSCAN for dimensionality reduction
- JupyterLab for analysis

**Installation:**
```bash
micromamba env create -f environment_glm2.yml
micromamba activate glm2_env
```

### 4. Documentation

#### `docs/README_glm2.md`
Comprehensive guide for gLM2 usage (1,500+ lines).

**Sections:**
- Overview and comparison with ESM-C
- Installation instructions
- Quick start guide
- Detailed mode explanations (genome vs genes)
- Command-line options reference
- Input/output format specifications
- Sequence formatting details
- Examples and use cases
- Performance benchmarks
- Troubleshooting guide
- Downstream analysis examples
- Citation information

#### `tmp/glm2_quickstart.md`
Quick start guide for L40S GPU usage.

**Contents:**
- 5-step setup process
- Environment creation
- Test procedures
- SLURM submission examples
- Performance expectations on L40S
- Troubleshooting tips

#### `tmp/NEW_FILES_SUMMARY.md` (this file)
Summary of all new files and their purposes.

---

## Workflow Comparison

### ESM-C Workflow (Protein Only)
```
Genome (.fna) → Gene Prediction → Proteins (.faa) → ESM-C → Embeddings (960D)
```

### gLM2 Workflow (Genomic Context)
```
Genome (.fna) + Annotations (.gff) → Format (CDS+IGS) → gLM2 → Embeddings (1280D)
```

---

## Key Differences: gLM2 vs ESM-C

| Feature | ESM-C | gLM2 |
|---------|-------|------|
| **Model Type** | Protein language model | Genomic language model |
| **Input** | Protein sequences (.faa) | Genome + annotations (.fna + .gff) |
| **Context** | Individual proteins | Genomic (genes + inter-genic) |
| **Embedding Dim** | 960 | 1280 |
| **Context Window** | 1024 aa | 4096 tokens |
| **Strand Aware** | No | Yes (<+> and <->) |
| **Captures IGS** | No | Yes (regulatory regions) |
| **Best For** | Protein function | Genomic organization |

---

## Comparison Analysis Design

The comparison script (`compare_glm2_esmc.py`) solves the challenge of different embedding dimensions:

**Problem:** ESM-C embeddings are 960D, gLM2 embeddings are 1280D
- Cannot directly compare vectors element-wise
- Need dimension-agnostic comparison method

**Solution:** Compare pairwise distance matrices
- Distance matrix size: N×N (same for both models)
- Dimension-independent
- Captures model's notion of protein similarity

**Metrics:**
1. **Pearson correlation** - Linear relationship between distances
2. **Spearman correlation** - Rank-order agreement
3. **Nearest neighbor agreement** - Do models agree on similar proteins?
4. **Distribution analysis** - Distance distribution characteristics

**Example result interpretation:**
- High correlation (r > 0.8): Models have similar protein similarity notions
- Low correlation (r < 0.5): Models capture different aspects
- NN agreement: Practical similarity for retrieval tasks

---

## Quick Start Commands

### 1. Setup Environment
```bash
micromamba env create -f environment_glm2.yml
micromamba activate glm2_env
python scripts/embeddings/test_glm2_embeddings.py
```

### 2. Test on Single Genome
```bash
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --output-dir tmp/glm2_test/ \
    --num-genomes 1 \
    --mode genome
```

### 3. Submit to L40S GPU
```bash
mkdir -p logs
NUM_GENOMES=10 sbatch scripts/embeddings/submit_glm2_l40s.sh
```

### 4. Run Comparison Analysis
```bash
python scripts/embeddings/compare_glm2_esmc.py \
    --protein-dir data/refseq_gene_annotations/ \
    --output-dir tmp/comparison/ \
    --n-proteins 1000
```

### 5. Full Production Run
```bash
# Process all 7,664 genomes across 100 L40S GPUs
sbatch --array=0-99 scripts/embeddings/submit_glm2_l40s.sh
```

---

## Expected Performance (L40S)

- **GPU:** NVIDIA L40S (48GB memory)
- **Speed:** ~30-60 seconds per genome (genome mode)
- **Batch size:** 16-32 (depending on sequence length)
- **Memory usage:** ~8-16GB GPU memory
- **Full dataset (7,664 genomes):**
  - Single GPU: ~10-15 hours
  - 100 GPUs (array): ~10-15 minutes

---

## Next Steps

1. **Create environment:** `micromamba env create -f environment_glm2.yml`
2. **Run tests:** `python scripts/embeddings/test_glm2_embeddings.py`
3. **Test on real data:** Single genome with `--num-genomes 1`
4. **Run comparison:** Compare with ESM-C on 1000 proteins
5. **Production run:** Submit SLURM array job for all genomes
6. **Analysis:** UMAP, clustering, functional annotation

---

## File Locations

```
organism_scale_modelling/
├── scripts/embeddings/
│   ├── get_glm2_embeddings.py          # NEW - Main gLM2 script
│   ├── test_glm2_embeddings.py         # NEW - Test suite
│   ├── compare_glm2_esmc.py            # NEW - Comparison analysis
│   └── submit_glm2_l40s.sh             # NEW - SLURM script
├── docs/
│   └── README_glm2.md                  # NEW - Full documentation
├── tmp/
│   ├── glm2_quickstart.md              # NEW - Quick start
│   └── NEW_FILES_SUMMARY.md            # NEW - This file
└── environment_glm2.yml                # NEW - Conda environment
```

---

## Support & Troubleshooting

See detailed troubleshooting in:
- `docs/README_glm2.md` - Full troubleshooting section
- `tmp/glm2_quickstart.md` - Common issues

For comparison analysis issues:
- Check that both models are installed
- Verify protein files exist
- Ensure sufficient GPU memory for batch size
- Try smaller `--n-proteins` for testing
