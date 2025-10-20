# Bacterial Gene Prediction and Translation

Scripts to extract protein sequences from bacterial genome FASTA files.

**Location**: `scripts/gene_prediction/`

## Two Approaches

### 1. `predict_genes.py` - Proper Gene Prediction (Recommended)

Uses Prodigal, the standard tool for bacterial gene prediction.

**Advantages:**
- Identifies real genes (not just ORFs)
- Handles frameshifts and edge cases
- Provides gene coordinates, scores, and statistics
- Standard in bacterial genomics

**Installation:**

First, update your environment:

```bash
micromamba env update -f environment.yml -n esm3_env
micromamba activate esm3_env
```

This installs both `prodigal` (conda) and `pyrodigal` (Python wrapper).

**Usage:**

```bash
# Single genome
python scripts/gene_prediction/predict_genes.py --genome data/refseq_genomes/GCF_000006985.1_Chlorobaculum_tepidum_TLS.fasta --output-dir test_genes/

# All genomes in a directory
python scripts/gene_prediction/predict_genes.py --genome-dir data/refseq_genomes/ --output-dir all_genes/

# Test on first 10 genomes
python scripts/gene_prediction/predict_genes.py --genome-dir data/refseq_genomes/ --output-dir all_genes/ --num-genomes 10

# Use command-line Prodigal instead of pyrodigal
python scripts/gene_prediction/predict_genes.py --genome genome.fasta --method prodigal --output-dir genes/

# For complete genomes (closed circular)
python scripts/gene_prediction/predict_genes.py --genome genome.fasta --closed --output-dir genes/
```

**Output Files:**

For each genome `GCF_XXXXXX`:

```
output_dir/
├── GCF_XXXXXX_proteins.faa    # Amino acid sequences (for ESM-C)
├── GCF_XXXXXX_genes.fna       # DNA sequences of genes
├── GCF_XXXXXX_genes.gff       # Gene coordinates and annotations
├── GCF_XXXXXX_stats.txt       # Statistics
└── prediction_summary.csv     # Summary of all genomes
```

**Integration with ESM-C Pipeline:**

After gene prediction, you can run embeddings:

```bash
# Predict genes
python scripts/gene_prediction/predict_genes.py --genome-dir genomes/ --output-dir predicted_genes/

# Generate embeddings from predicted proteins
python scripts/embeddings/get_esm_embeddings.py --gene-dir predicted_genes/ --output-dir embeddings/
```

### 2. `extract_simple_orfs.py` - Simple ORF Extraction

Basic ORF finder that doesn't require external tools.

**Advantages:**
- No dependencies beyond Biopython
- Fast and simple
- Good for quick analysis

**Disadvantages:**
- Finds all possible ORFs (not just real genes)
- No gene quality scoring
- May include spurious ORFs

**Usage:**

```bash
# Basic usage
python scripts/gene_prediction/extract_simple_orfs.py genome.fasta --output proteins.faa

# Custom minimum length
python scripts/gene_prediction/extract_simple_orfs.py genome.fasta --output proteins.faa --min-length 50

# Custom start codons
python scripts/gene_prediction/extract_simple_orfs.py genome.fasta --output proteins.faa --start-codons "ATG,GTG"
```

## Comparison

| Feature | predict_genes.py (Prodigal) | extract_simple_orfs.py |
|---------|----------------------------|------------------------|
| Gene prediction quality | High (trained model) | Basic (all ORFs) |
| External dependencies | Prodigal or pyrodigal | None |
| Speed | Fast | Very fast |
| Output formats | FASTA, GFF, stats | FASTA only |
| Recommended for | Production use | Quick tests |

## Example Workflow

### Starting from Scratch

```bash
# 1. Download or place bacterial genomes
ls my_genomes/
# genome1.fasta
# genome2.fasta

# 2. Predict genes and get proteins
python scripts/gene_prediction/predict_genes.py \
    --genome-dir my_genomes/ \
    --output-dir predicted_genes/ \
    --method pyrodigal

# 3. Generate ESM-C embeddings
python scripts/embeddings/get_esm_embeddings.py \
    --gene-dir predicted_genes/ \
    --output-dir embeddings/ \
    --num-genomes 10  # test first

# 4. Inspect results
python scripts/embeddings/inspect_embeddings.py embeddings/esmc_embeddings.h5
```

### Using Existing Annotations

Your RefSeq data already has Prodigal predictions:

```bash
# Gene annotations are already here
ls data/refseq_gene_annotations/
# GCF_000006985.1_prodigal_proteins.faa  <- Use these!
# GCF_000006985.1_prodigal_genes.fna
# GCF_000006985.1_prodigal_genes.gff

# So you can directly run embeddings
python scripts/embeddings/get_esm_embeddings.py \
    --gene-dir data/refseq_gene_annotations/ \
    --output-dir embeddings/
```

## Prodigal Details

### Methods

1. **`pyrodigal`** (default, recommended)
   - Pure Python implementation
   - Easy to install: `pip install pyrodigal`
   - No external binary needed
   - Same algorithm as Prodigal

2. **`prodigal`** (classic)
   - Original C implementation
   - Slightly faster
   - Requires system installation
   - Install: `conda install -c bioconda prodigal`

### Genetic Codes

Prodigal automatically detects the translation table, but for reference:

- **Table 11**: Bacterial, Archaeal, Plant Plastid (default)
- **Table 4**: Mycoplasma, Spiroplasma
- **Table 1**: Universal (eukaryotic)

## File Formats

### Protein FASTA (.faa)

```
>GCF_000006985.1_1 # 2 # 1126 # -1 # partial=00
MKFNTTIKRLQEAVNKVILAVPAKSLDARFDNINLTLENGMLTMFATDGELSITTNCDVA...
>GCF_000006985.1_2 # 1398 # 2879 # -1 # partial=00
MSDTIQQEAPDNLQVTPTHGRSFAEKVWSACLGLIQENINTLAFKTWFLPIRPLSFSGSE...
```

Header format: `>{gene_id} # {start} # {end} # {strand} # {metadata}`

### Gene DNA FASTA (.fna)

Same format but with nucleotide sequences.

### GFF (Gene Feature Format)

```
##gff-version 3
NC_002932.3	Prodigal	CDS	2	1126	45.2	-	0	ID=GCF_000006985.1_1
NC_002932.3	Prodigal	CDS	1398	2879	89.6	-	0	ID=GCF_000006985.1_2
```

Columns: `seqid source type start end score strand phase attributes`

## Tips

1. **For new genomes**: Use `predict_genes.py` with `--method pyrodigal`
2. **For RefSeq genomes**: Use existing `.faa` files in `data/refseq_gene_annotations/`
3. **Complete genomes**: Add `--closed` flag
4. **Draft assemblies**: Use default (meta mode)
5. **Testing**: Always use `--num-genomes 10` first to verify pipeline

## Performance

- **Prodigal**: ~1-5 seconds per typical bacterial genome (~3-5 Mbp)
- **7,664 genomes**: ~2-10 hours total
- Can parallelize by running multiple instances on different genome subsets

## Troubleshooting

**"pyrodigal not found"**
```bash
pip install pyrodigal
```

**"prodigal not found"**
```bash
conda install -c bioconda prodigal
# or use --method pyrodigal instead
```

**"No genes found"**
- Check if genome file is valid FASTA
- Try lowering `--min-gene-length`
- Check sequence isn't all N's or ambiguous bases
