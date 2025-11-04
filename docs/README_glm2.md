# gLM2 Genomic Language Model Embeddings

This guide explains how to generate genomic embeddings using the **gLM2_650M** model from TattaBio, which processes both protein-coding sequences and inter-genic DNA regions in their genomic context.

## Overview

### What is gLM2?

gLM2 (Genomic Language Model 2) is a transformer-based model that processes **mixed-modality genomic sequences**:
- **Protein-coding regions (CDS)**: Tokenized as amino acids (uppercase)
- **Inter-genic sequences (IGS)**: Tokenized as nucleotides (lowercase)
- **Strand information**: `<+>` for forward strand, `<->` for reverse strand

### Key Differences from ESM-C

| Feature | ESM-C | gLM2 |
|---------|-------|------|
| **Input** | Protein sequences only (.faa) | Genomic DNA + annotations (.fna + .gff) |
| **Context** | Individual proteins | Genomic context (genes + inter-genic regions) |
| **Model Size** | 600M parameters | 650M parameters |
| **Embedding Dim** | 960 | 1280 |
| **Context Window** | 1024 amino acids | 4096 tokens |
| **Training Data** | Protein sequences | Mixed genomic sequences (OMG dataset) |

### When to Use gLM2 vs ESM-C

**Use gLM2 when:**
- You want to capture genomic context (regulatory regions, operons)
- Analyzing gene organization and synteny
- Studying non-coding regions
- Need strand-aware embeddings
- Working with complete genomes

**Use ESM-C when:**
- Focusing purely on protein function
- Analyzing individual proteins
- Need protein-specific features
- Working with protein sequences only

---

## Installation

### 1. Install Dependencies

```bash
# Using conda/micromamba
micromamba install -c conda-forge transformers torch biopython

# Or using pip
pip install transformers torch biopython
```

### 2. Test Installation

```bash
python scripts/embeddings/test_glm2_embeddings.py
```

This will:
- Test gLM2 imports
- Load the gLM2_650M model
- Run inference on test sequences
- Create a test genome and generate embeddings

---

## Quick Start

### Basic Usage

```bash
# Generate genome-level embeddings
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --output-dir embeddings/glm2/ \
    --mode genome

# Generate gene-level embeddings
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --output-dir embeddings/glm2/ \
    --mode genes
```

---

## Embedding Modes

gLM2 supports two embedding modes:

### 1. Genome Mode (Default)

Generates embeddings for **genomic chunks** that preserve context.

**Use cases:**
- Analyzing genomic organization
- Capturing long-range dependencies
- Studying regulatory regions
- Gene context analysis

**Example:**
```bash
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --mode genome \
    --chunk-size 3000
```

**Output:**
- Each genome divided into ~3000 token chunks
- Chunks include CDS (as amino acids) + IGS (as nucleotides)
- Embeddings shape: `(n_chunks, 1280)`

### 2. Genes Mode

Generates embeddings for **individual genes** (similar to ESM-C).

**Use cases:**
- Gene-level analysis
- Direct comparison with ESM-C
- Gene clustering and classification
- Functional annotation

**Example:**
```bash
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --mode genes \
    --batch-size 16
```

**Output:**
- One embedding per gene
- Each gene formatted as: `<+>PROTEIN_SEQUENCE` or `<->PROTEIN_SEQUENCE`
- Embeddings shape: `(n_genes, 1280)`

---

## Command-Line Options

### Required Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--genome-dir` | Directory with genome .fna files | `data/refseq_genomes` |
| `--annotation-dir` | Directory with .gff annotation files | `data/refseq_gene_annotations` |
| `--output-dir` | Output directory for embeddings | `embeddings/glm2` |

### Model Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-name` | HuggingFace model name | `tattabio/gLM2_650M` |
| `--device` | Device to use (cuda/cpu) | `cuda` if available |
| `--dtype` | Computation dtype | `bfloat16` |

### Processing Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Embedding mode (genome/genes) | `genome` |
| `--chunk-size` | Tokens per genome chunk | `3000` |
| `--batch-size` | Batch size for inference | `8` |
| `--max-length` | Max sequence length | `4096` |
| `--num-genomes` | Number of genomes to process | `None` (all) |

### Output Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--save-format` | Output format (hdf5/npz) | `hdf5` |

---

## Input Data Requirements

### 1. Genome Files (.fna or .fasta)

Bacterial genome sequences in FASTA format:

```
>GCF_000006985.1
ATGAAAGTTCTGTTTGCCGCGGTGATTGGTGGCGGCACC...
```

### 2. Annotation Files (.gff) - Optional but Recommended

GFF3 format gene annotations:

```
##gff-version 3
GCF_000006985.1  Prodigal  CDS  1  630  .  +  0  ID=gene1;Name=mdh
GCF_000006985.1  Prodigal  CDS  650  1141  .  +  0  ID=gene2;Name=atpA
```

**Note:** If no annotations are provided, the entire genome is treated as inter-genic sequence.

---

## Output Format

### HDF5 Format (Default)

Single file: `glm2_embeddings_{mode}.h5`

**Structure:**
```python
import h5py

with h5py.File('glm2_embeddings_genome.h5', 'r') as f:
    # Access genome
    genome = f['GCF_000006985.1']

    # Get embeddings
    embeddings = genome['embeddings'][:]  # (n_chunks, 1280)
    chunk_ids = genome['chunk_ids'][:]    # ['chunk_00000', 'chunk_00001', ...]

    # Get metadata
    mode = genome.attrs['mode']           # 'genome' or 'genes'
    n_items = genome.attrs['n_items']     # number of chunks/genes
    embed_dim = genome.attrs['embedding_dim']  # 1280
```

### NPZ Format

Separate file per genome: `{genome_id}_glm2_embeddings.npz`

**Structure:**
```python
import numpy as np

data = np.load('GCF_000006985.1_glm2_embeddings.npz', allow_pickle=True)

embeddings = data['embeddings']     # (n_items, 1280)
chunk_ids = data['chunk_ids']       # or 'gene_ids' in genes mode
mode = data['mode']                 # 'genome' or 'genes'
```

### Metadata CSV

Summary file: `glm2_embedding_metadata_{mode}.csv`

| Column | Description |
|--------|-------------|
| `genome_id` | Genome identifier |
| `n_items` | Number of chunks/genes |
| `embedding_dim` | Embedding dimension (1280) |
| `mode` | Embedding mode used |

---

## Sequence Formatting

### How gLM2 Processes Sequences

gLM2 expects sequences in this format:

```
<+>PROTEIN1<+>atcgatcg<->PROTEIN2<+>ggccttaa
```

Where:
- `<+>` = forward strand marker
- `<->` = reverse strand marker
- `UPPERCASE` = amino acids (protein coding)
- `lowercase` = nucleotides (inter-genic)

### Example Formatting

**Input genome:**
```
>genome1
ATGAAACTG...TGACCCGGG...ATGGCGATC...TGA
```

**Input GFF:**
```
genome1  Prodigal  CDS  1  630  .  +  0  ID=gene1
genome1  Prodigal  CDS  650  900  .  -  0  ID=gene2
```

**Formatted for gLM2:**
```
<+>MKLFAAVIGGGASGSIGAVDAHAAKAHAL<+>acccggg<->MADILHEIFRSIVPKGILTTEIANLTA
```

---

## Examples

### Example 1: Test on Single Genome

```bash
# Test with first genome only
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --output-dir tmp/glm2_test/ \
    --num-genomes 1 \
    --mode genome \
    --save-format npz
```

### Example 2: GPU Batch Processing

```bash
# Use GPU with larger batch size
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --output-dir embeddings/glm2/ \
    --mode genes \
    --device cuda \
    --batch-size 32 \
    --dtype bfloat16
```

### Example 3: Process Subset

```bash
# Process first 100 genomes
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --annotation-dir data/refseq_gene_annotations/ \
    --output-dir embeddings/glm2_subset/ \
    --num-genomes 100 \
    --mode genome
```

### Example 4: Without Annotations

```bash
# Process genomes without annotations (treats all as IGS)
python scripts/embeddings/get_glm2_embeddings.py \
    --genome-dir data/refseq_genomes/ \
    --output-dir embeddings/glm2_no_annot/ \
    --mode genome \
    --annotation-dir ""
```

---

## Performance Considerations

### Memory Usage

| Mode | Batch Size | Memory (GPU) | Notes |
|------|------------|--------------|-------|
| Genome | 8 | ~8 GB | Chunks ~3000 tokens |
| Genes | 16 | ~10 GB | Average gene ~500 tokens |
| Genome | 4 | ~4 GB | For smaller GPUs |

### Speed Benchmarks

On NVIDIA A100 GPU:
- **Genome mode**: ~30-60 seconds per genome
- **Genes mode**: ~1-2 genes per second
- **Full dataset (7,664 genomes)**: ~10-15 hours

### Optimization Tips

1. **Use bfloat16**: Faster and uses less memory than float32
2. **Increase batch size**: On larger GPUs (A100, H100)
3. **Use genome mode for context**: More efficient than per-gene
4. **Smaller chunks**: Reduce `--chunk-size` if OOM errors occur

---

## Troubleshooting

### Issue: Model fails to load

**Error:**
```
OSError: Unable to load weights from HuggingFace Hub
```

**Solution:**
```bash
# Install/update transformers
pip install --upgrade transformers

# Or login to HuggingFace (if model requires authentication)
huggingface-cli login
```

### Issue: CUDA out of memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
--batch-size 4

# Reduce chunk size
--chunk-size 2000

# Use float16 instead of bfloat16
--dtype float16

# Use CPU (slower)
--device cpu
```

### Issue: No annotations found

**Warning:**
```
No valid sequences for GCF_000006985.1
```

**Solutions:**
1. Check annotation directory path
2. Verify GFF file naming matches genome files
3. Run without annotations (genome treated as all IGS)
4. Generate annotations using Prodigal

### Issue: Genome files not found

**Error:**
```
ERROR: No genome files found!
```

**Solutions:**
```bash
# Check file extensions
ls data/refseq_genomes/*.fna
ls data/refseq_genomes/*.fasta

# Verify path
--genome-dir /correct/path/to/genomes/
```

---

## Downstream Analysis

### Load Embeddings

```python
import h5py
import numpy as np

# Load from HDF5
with h5py.File('embeddings/glm2/glm2_embeddings_genome.h5', 'r') as f:
    genome_ids = list(f.keys())

    # Get embeddings for first genome
    embeddings = f[genome_ids[0]]['embeddings'][:]
    chunk_ids = f[genome_ids[0]]['chunk_ids'][:]
```

### Compare with ESM-C

```python
import h5py
import numpy as np
from scipy.spatial.distance import cosine

# Load both embeddings
glm2_data = h5py.File('glm2_embeddings_genes.h5', 'r')
esmc_data = h5py.File('esmc_embeddings.h5', 'r')

genome_id = 'GCF_000006985.1'

# Get embeddings
glm2_emb = glm2_data[genome_id]['embeddings'][0]
esmc_emb = esmc_data[genome_id]['embeddings'][0]

# Compute similarity
similarity = 1 - cosine(glm2_emb, esmc_emb)
print(f"Embedding similarity: {similarity:.4f}")
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt

# Load embeddings
embeddings = load_all_embeddings()  # Your loading function

# PCA
pca = PCA(n_components=50)
pca_emb = pca.fit_transform(embeddings)

# UMAP
umap = UMAP(n_neighbors=15, min_dist=0.1)
umap_emb = umap.fit_transform(pca_emb)

# Plot
plt.scatter(umap_emb[:, 0], umap_emb[:, 1], s=1, alpha=0.5)
plt.savefig('glm2_umap.png', dpi=300)
```

---

## Citation

If you use gLM2 in your research, please cite:

```bibtex
@article{tattabio2024glm2,
  title={gLM2: Genomic Language Models with Mixed Modality},
  author={TattaBio Team},
  journal={HuggingFace Model Hub},
  year={2024},
  url={https://huggingface.co/tattabio/gLM2_650M}
}
```

---

## Additional Resources

- **gLM2 Model Card**: https://huggingface.co/tattabio/gLM2_650M
- **ESM-C Documentation**: [docs/README_embeddings.md](README_embeddings.md)
- **Gene Prediction**: [docs/README_gene_prediction.md](README_gene_prediction.md)
- **Comparison Guide**: [docs/README_glm2_vs_esmc.md](README_glm2_vs_esmc.md) (coming soon)

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section above
2. Run test suite: `python scripts/embeddings/test_glm2_embeddings.py`
3. Review gLM2 model documentation on HuggingFace
4. Open an issue in this repository
