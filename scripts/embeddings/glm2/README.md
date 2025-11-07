# gLM2 Embedding Scripts

Scripts for generating gLM2 (genomic Language Model 2) embeddings with genomic context.

## Overview

**gLM2** is a 10B parameter genomic foundation model that:
- Processes both DNA and protein sequences
- Captures regulatory context (5' UTR, promoters)
- Understands protein-DNA relationships
- Trained on diverse microbial genomes

**Key difference from ESM-C**: gLM2 uses genomic context, not just protein sequence.

## Scripts

### Core Embedding Generation

- `get_glm2_embeddings.py` - Generate gLM2 embeddings with context
- `sample_proteins_with_context.py` - Extract proteins with flanking regions

### Environment Setup

- `setup_glm2_env_l40s.sh` - Environment setup for L40S GPUs
- `fix_glm2_env.sh` - Fix environment issues
- `test_glm2_embeddings.py` - Test embedding generation
- `test_glm2_on_l40s.sh` - Test on L40S GPU nodes

### Context Effect Analysis

- `compare_protein_context_effect.py` - Compare protein-only vs protein+context
- `compare_5utr_context_effect.py` - Analyze 5' UTR context effects
- `compare_glm2_esmc.py` - Compare gLM2 vs ESM-C embeddings

### Jacobian Analysis

Analyzes how embeddings change with context:

- `calculate_protein_jacobian.py` - Protein sequence sensitivity
- `calculate_5utr_jacobian.py` - 5' UTR sensitivity
- `calculate_genomic_jacobian.py` - Full genomic context sensitivity
- `calculate_dna_vs_protein_jacobian.py` - DNA vs protein context comparison
- `analyze_jacobian_differences.py` - Compare Jacobian patterns

### Tokenization Diagnostics

- `diagnose_tokenization.py` - Debug gLM2 tokenization issues

### SLURM Submission

- `submit_glm2_l40s.sh` - Main gLM2 job submission
- `submit_context_comparison.sh` - Context effect jobs
- `submit_5utr_context_comparison.sh` - 5' UTR specific analysis
- `submit_jacobian_analysis.sh` - Jacobian computation jobs
- `submit_5utr_jacobian.sh` - 5' UTR Jacobian
- `submit_genomic_jacobian.sh` - Genomic Jacobian
- `submit_dna_vs_protein_jacobian.sh` - DNA vs protein comparison
- `submit_diagnose_tokenization.sh` - Tokenization debugging

## Usage

### Generate embeddings with context
```bash
python get_glm2_embeddings.py \
    --genome data/refseq_genomes/GCF_000001.fna \
    --proteins data/refseq_proteins/GCF_000001.faa \
    --output data/refseq_glm2_embeddings/GCF_000001_embeddings.npz \
    --context-size 100  # bases of flanking context
```

### Compare context effects
```bash
sbatch submit_context_comparison.sh
```

### Jacobian analysis
```bash
sbatch submit_jacobian_analysis.sh
```

## Key Findings

### 5' UTR Context Matters
- Regulatory regions benefit from protein context
- 5' UTR Jacobian shows regulatory sensitivity
- Genes with strong promoters show different embeddings

### DNA vs Protein Context
- DNA context captures regulatory information
- Protein context captures functional information
- Combined context best for regulatory genes

## Output Format

```python
import numpy as np
data = np.load('GCF_000001_glm2_embeddings.npz')

embeddings = data['embeddings']        # (n_genes, embedding_dim)
gene_ids = data['gene_ids']            # (n_genes,)
contexts = data['contexts']            # (n_genes,) genomic context strings
context_types = data['context_types']  # (n_genes,) e.g., '5utr+protein+3utr'
```

## Requirements

- PyTorch with CUDA support
- gLM2 model (10B parameters)
- GPU with ≥40GB VRAM (L40S recommended)
- Genomic sequences (not just proteins)

## Model Details

- **Model**: gLM2-10B
- **Parameters**: 10 billion
- **Context window**: Up to 8192 tokens
- **Input**: DNA + protein sequences
- **Training**: Diverse microbial genomes

## Performance

- **Speed**: ~10-50 sequences/second (depends on context size)
- **Memory**: ~40GB VRAM for 10B model
- **Recommended GPU**: L40S (48GB), A100 (80GB)

## Comparison with ESM-C

| Feature | ESM-C | gLM2 |
|---------|-------|------|
| Input | Protein only | DNA + protein |
| Context | No | Yes (5' UTR, 3' UTR) |
| Size | 600M params | 10B params |
| Speed | Fast | Slower |
| Best for | Protein structure/function | Regulatory context |

## Documentation

See comprehensive gLM2 documentation in:
- `../../../docs/glm2_tokenization_comprehensive.md`
- `../../../docs/glm2_jacobian_analysis.md`

## Troubleshooting

### Model loading issues
```bash
bash fix_glm2_env.sh
```

### Tokenization errors
```bash
python diagnose_tokenization.py --sequence "ATGCCC..."
```

### GPU memory issues
- Reduce context size
- Use gradient checkpointing
- Process in smaller batches

## See Also

- **ESM-C embeddings**: `../esm/`
- **Clustering**: `../../clustering/`
- **Analysis**: `../../analysis/`
