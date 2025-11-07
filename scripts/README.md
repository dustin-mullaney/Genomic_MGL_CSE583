# Scripts Directory

Organized collection of scripts for the RefSeq genome-scale protein analysis pipeline.

## Directory Structure

### `gene_prediction/`
Gene prediction and ORF extraction from genome sequences.
- `predict_genes.py` - Prodigal-based gene prediction
- `extract_simple_orfs.py` - Simple ORF extraction
- `test_gene_prediction.py` - Testing utilities

### `functional_annotation/`
Functional annotation of predicted genes using COG and eggNOG databases.
- `run_diamond_cog_refseq.sh` - DIAMOND search against COG database
- `run_eggnog_refseq.sh` - eggNOG-mapper annotation
- `parse_*.py` - Annotation parsing and processing
- `annotate_umap_with_cogs.py` - Add COG annotations to UMAP visualizations

### `embeddings/`
Protein sequence embedding generation using different models.

#### `embeddings/esm/`
ESM-C (Evolutionary Scale Modeling) embeddings for protein sequences.
- `get_esm_embeddings.py` - Generate ESM-C embeddings
- `compute_protein_embeddings.py` - Batch embedding computation
- `submit_protein_embeddings.sh` - SLURM submission scripts

#### `embeddings/glm2/`
gLM2 (genomic Language Model) embeddings with genomic context.
- `get_glm2_embeddings.py` - Generate gLM2 embeddings
- `calculate_*_jacobian.py` - Jacobian analysis for context effects
- `diagnose_tokenization.py` - Tokenization diagnostics
- `compare_*_context_effect.py` - Context effect analysis

### `clustering/`
Dimensionality reduction, clustering, and cluster evaluation.
- **UMAP**: `compute_umap_*.py`, `submit_umap_*.sh`
- **Leiden clustering**: `cluster_leiden_*.py`, `submit_leiden_*.sh`
- **Evaluation**: `evaluate_clustering_*.py`
- **Stability testing**: `evaluate_clustering_stability_efficient.py`
- **Full dataset assignment**: `assign_full_dataset_to_clusters.py`
- **Diversity sampling**: `diversity_subsample.py`

### `visualization/`
Plotting and analysis scripts for results visualization.
- `plot_leiden_clustering.py` - Visualize Leiden clustering results
- `plot_individual_cogs.py` - Per-COG category plots
- `visualize_clustering_stability.py` - Stability analysis plots
- `summarize_stability_results.py` - Stability summary tables
- `compare_clustering_metrics.py` - Compare clustering methods
- `analyze_umap_results.py` - UMAP analysis

### `analysis/`
Higher-level analysis scripts (currently empty, for future analyses).

---

## Pipeline Overview

```
1. Gene Prediction → 2. Functional Annotation → 3. Embeddings → 4. Clustering → 5. Visualization
```

1. **Gene Prediction**: Extract protein-coding genes from genomes
2. **Functional Annotation**: Annotate with COG/eggNOG
3. **Embeddings**: Generate vector representations (ESM-C or gLM2)
4. **Clustering**: Reduce dimensions (UMAP/PCA), cluster (Leiden), evaluate
5. **Visualization**: Generate plots and summary statistics

---

## Quick Links

- **Main project README**: `../README.md`
- **Documentation**: `../docs/`
- **Results**: `../results/`
- **Current status**: `../CURRENT_STATUS.md`
