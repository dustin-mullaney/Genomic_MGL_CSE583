# Masked Graph Learning for Bacterial Genomes

Project: Masked graph learning for bacterial genomics
Goals:
1. Define gene nodes (mostly done)
2. Represent genomes as graphs
3. Implement [GraphMAE](https://arxiv.org/abs/2205.10803)

![Proposed Graph Structure](graph_concept.png)
 
The data:
(https://www.ncbi.nlm.nih.gov/refseq/about/prokaryotes/)
 
For defining gene nodes:
[ESM Atlas](https://esmatlas.com/)
[ESM Cambrian](https://www.evolutionaryscale.ai/blog/esm-cambrian)

## Project Structure

```
.
├── data/
│   ├── refseq_genomes/          # Full bacterial genome sequences from refseq
│   └── refseq_gene_annotations/ # Annotations from refseq
├── scripts/
│   ├── gene_prediction/         # Tools for extracting genes from genomes
│   └── embeddings/              # Tools for generating ESM-C embeddings
├── docs/                        # Documentation
├── notebooks/                   # Jupyter notebooks
└── environment.yml             # Conda/micromamba environment
```
