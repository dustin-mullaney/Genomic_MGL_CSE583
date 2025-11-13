#!/usr/bin/env python
"""
Generate gLM2 (Genomic Language Model) embeddings for bacterial genomes.

This script processes genomic sequences (FASTA) and generates embeddings using
the gLM2_650M model from TattaBio, which handles both protein-coding sequences
and inter-genic DNA regions in genomic context.

Key differences from ESM-C:
- Input: Genomic DNA sequences (.fna) instead of protein sequences (.faa)
- Model: gLM2_650M (genomic context) vs ESM-C (protein only)
- Processing: Requires gene annotation to properly format coding/non-coding regions
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
import warnings

# Suppress tokenizer warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate gLM2 embeddings for bacterial genomes"
    )
    parser.add_argument(
        "--genome-dir",
        type=str,
        default="data/refseq_genomes",
        help="Directory containing genome .fna files",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="data/refseq_gene_annotations",
        help="Directory containing gene annotation .gff files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="embeddings/glm2",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tattabio/gLM2_650M",
        help="gLM2 model to use (default: tattabio/gLM2_650M)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing sequences",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length (gLM2 context window is 4096 tokens)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3000,
        help="Chunk genomes into segments of this many tokens",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num-genomes",
        type=int,
        default=None,
        help="Number of genomes to process (for testing)",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="hdf5",
        choices=["hdf5", "npz"],
        help="Format to save embeddings",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for model computation (gLM2 trained with bfloat16)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="genome",
        choices=["genome", "genes"],
        help="Embedding mode: 'genome' for whole genome chunks, 'genes' for per-gene embeddings",
    )
    return parser.parse_args()


def get_genome_files(genome_dir: str) -> List[Path]:
    """Get list of all genome .fna files."""
    genome_dir = Path(genome_dir)
    if not genome_dir.exists():
        print(f"Warning: Genome directory {genome_dir} does not exist")
        return []

    genome_files = sorted(genome_dir.glob("*.fna"))
    if len(genome_files) == 0:
        genome_files = sorted(genome_dir.glob("*.fasta"))

    print(f"Found {len(genome_files)} genome files")
    return genome_files


def extract_genome_id(filename: str) -> str:
    """Extract genome ID from filename (e.g., GCF_000006985.1)."""
    # Handle various naming conventions
    name = Path(filename).stem
    # Remove common suffixes
    name = name.replace("_genomic", "").replace("_complete", "")
    return name


def load_gff_annotations(gff_file: Path) -> List[Dict]:
    """
    Load gene annotations from GFF file.

    Returns:
        List of dicts with 'start', 'end', 'strand', 'type', 'product'
    """
    annotations = []

    if not gff_file.exists():
        return annotations

    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            feature_type = fields[2]
            if feature_type != 'CDS':
                continue

            start = int(fields[3]) - 1  # GFF is 1-indexed
            end = int(fields[4])
            strand = fields[6]

            annotations.append({
                'start': start,
                'end': end,
                'strand': strand,
                'type': 'CDS',
            })

    return sorted(annotations, key=lambda x: x['start'])


def format_genome_for_glm2(
    genome_seq: str,
    annotations: List[Dict],
    chunk_size: int = 3000,
) -> List[str]:
    """
    Format genome sequence for gLM2 model.

    gLM2 expects mixed-modality format:
    - Protein coding regions (CDS): uppercase amino acids
    - Inter-genic regions: lowercase nucleotides
    - Strand markers: <+> for forward, <-> for reverse

    Args:
        genome_seq: Full genome sequence (DNA)
        annotations: List of gene annotations from GFF
        chunk_size: Size of genome chunks to create

    Returns:
        List of formatted genome chunks ready for gLM2
    """
    formatted_chunks = []

    # If no annotations, treat whole genome as inter-genic
    if len(annotations) == 0:
        # Chunk the genome
        for i in range(0, len(genome_seq), chunk_size):
            chunk = genome_seq[i:i+chunk_size].lower()
            formatted_chunks.append(f"<+>{chunk}")
        return formatted_chunks

    # Build formatted sequence with annotations
    formatted = []
    last_pos = 0

    for annot in annotations:
        start, end, strand = annot['start'], annot['end'], annot['strand']

        # Add inter-genic region before this gene
        if start > last_pos:
            igs = genome_seq[last_pos:start].lower()
            if igs:
                formatted.append(f"<+>{igs}")

        # Add coding sequence (translate to protein)
        cds_seq = genome_seq[start:end]

        if strand == '-':
            # Reverse complement for reverse strand
            cds_seq = str(Seq(cds_seq).reverse_complement())

        try:
            # Translate to amino acids
            protein_seq = str(Seq(cds_seq).translate(to_stop=True))
            # Remove stop codons
            protein_seq = protein_seq.replace('*', '')

            if protein_seq:
                strand_marker = '<+>' if strand == '+' else '<->'
                formatted.append(f"{strand_marker}{protein_seq.upper()}")
        except Exception as e:
            # If translation fails, skip this gene
            pass

        last_pos = end

    # Add remaining inter-genic region
    if last_pos < len(genome_seq):
        igs = genome_seq[last_pos:].lower()
        if igs:
            formatted.append(f"<+>{igs}")

    # Join and chunk
    full_formatted = ''.join(formatted)

    # Create chunks of appropriate size
    for i in range(0, len(full_formatted), chunk_size):
        chunk = full_formatted[i:i+chunk_size]
        if chunk:
            formatted_chunks.append(chunk)

    return formatted_chunks


def format_genes_for_glm2(
    genome_seq: str,
    annotations: List[Dict],
) -> Tuple[List[str], List[str]]:
    """
    Format individual genes for gLM2 model.

    Returns:
        Tuple of (formatted_sequences, gene_ids)
    """
    formatted_seqs = []
    gene_ids = []

    for idx, annot in enumerate(annotations):
        start, end, strand = annot['start'], annot['end'], annot['strand']

        # Extract coding sequence
        cds_seq = genome_seq[start:end]

        if strand == '-':
            cds_seq = str(Seq(cds_seq).reverse_complement())

        try:
            # Translate to amino acids
            protein_seq = str(Seq(cds_seq).translate(to_stop=True))
            protein_seq = protein_seq.replace('*', '')

            if protein_seq:
                strand_marker = '<+>' if strand == '+' else '<->'
                formatted_seqs.append(f"{strand_marker}{protein_seq.upper()}")
                gene_ids.append(f"gene_{idx+1:05d}")
        except Exception:
            pass

    return formatted_seqs, gene_ids


def get_embeddings_batch(
    model,
    tokenizer,
    sequences: List[str],
    device: str,
) -> np.ndarray:
    """
    Get embeddings for a batch of sequences using gLM2.

    Args:
        model: gLM2 model
        tokenizer: gLM2 tokenizer
        sequences: List of formatted sequences
        device: Device to use

    Returns:
        Array of embeddings (batch_size, embedding_dim)
    """
    # Tokenize sequences
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use last hidden state
        hidden_states = outputs.last_hidden_state

        # Mean pool over sequence length (excluding padding)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        embeddings = sum_hidden / sum_mask.clamp(min=1e-9)

        # Convert to numpy
        embeddings = embeddings.cpu().float().numpy()

    return embeddings


def process_genome_mode(
    genome_file: Path,
    annotation_file: Optional[Path],
    model,
    tokenizer,
    args,
) -> Tuple[np.ndarray, List[str]]:
    """
    Process genome in 'genome' mode - generate embeddings for genome chunks.

    Returns:
        embeddings: Array of shape (n_chunks, embedding_dim)
        chunk_ids: List of chunk identifiers
    """
    # Load genome sequence
    genome_record = next(SeqIO.parse(genome_file, "fasta"))
    genome_seq = str(genome_record.seq)

    # Load annotations if available
    annotations = []
    if annotation_file and annotation_file.exists():
        annotations = load_gff_annotations(annotation_file)

    # Format genome for gLM2
    formatted_chunks = format_genome_for_glm2(
        genome_seq,
        annotations,
        chunk_size=args.chunk_size,
    )

    if len(formatted_chunks) == 0:
        return None, None

    # Process in batches
    all_embeddings = []
    chunk_ids = []

    for i in range(0, len(formatted_chunks), args.batch_size):
        batch = formatted_chunks[i:i + args.batch_size]

        # Get embeddings
        embeddings = get_embeddings_batch(model, tokenizer, batch, args.device)
        all_embeddings.append(embeddings)

        # Generate chunk IDs
        for j in range(len(batch)):
            chunk_ids.append(f"chunk_{i+j:05d}")

    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings, chunk_ids


def process_genes_mode(
    genome_file: Path,
    annotation_file: Optional[Path],
    model,
    tokenizer,
    args,
) -> Tuple[np.ndarray, List[str]]:
    """
    Process genome in 'genes' mode - generate embeddings for individual genes.

    Returns:
        embeddings: Array of shape (n_genes, embedding_dim)
        gene_ids: List of gene identifiers
    """
    # Load genome sequence
    genome_record = next(SeqIO.parse(genome_file, "fasta"))
    genome_seq = str(genome_record.seq)

    # Load annotations
    annotations = []
    if annotation_file and annotation_file.exists():
        annotations = load_gff_annotations(annotation_file)

    if len(annotations) == 0:
        return None, None

    # Format genes for gLM2
    formatted_seqs, gene_ids = format_genes_for_glm2(genome_seq, annotations)

    if len(formatted_seqs) == 0:
        return None, None

    # Process in batches
    all_embeddings = []

    for i in range(0, len(formatted_seqs), args.batch_size):
        batch = formatted_seqs[i:i + args.batch_size]

        # Get embeddings
        embeddings = get_embeddings_batch(model, tokenizer, batch, args.device)
        all_embeddings.append(embeddings)

    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings, gene_ids


def save_embeddings_hdf5(
    output_file: Path,
    genome_id: str,
    embeddings: np.ndarray,
    ids: List[str],
    mode: str,
):
    """Save embeddings to HDF5 file."""
    with h5py.File(output_file, "a") as f:
        # Create group for this genome
        grp = f.create_group(genome_id)

        # Save embeddings
        grp.create_dataset("embeddings", data=embeddings, compression="gzip")

        # Save IDs as strings
        dt = h5py.string_dtype(encoding='utf-8')
        id_name = "chunk_ids" if mode == "genome" else "gene_ids"
        grp.create_dataset(id_name, data=ids, dtype=dt)

        # Save metadata
        grp.attrs['mode'] = mode
        grp.attrs['n_items'] = len(ids)
        grp.attrs['embedding_dim'] = embeddings.shape[1]


def save_embeddings_npz(
    output_dir: Path,
    genome_id: str,
    embeddings: np.ndarray,
    ids: List[str],
    mode: str,
):
    """Save embeddings to individual NPZ files per genome."""
    output_file = output_dir / f"{genome_id}_glm2_embeddings.npz"
    id_name = "chunk_ids" if mode == "genome" else "gene_ids"

    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        **{id_name: ids},
        mode=mode,
    )


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("gLM2 Genomic Language Model Embedding Generation")
    print("="*80)
    print(f"\nMode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Model: {args.model_name}")
    print(f"Chunk size: {args.chunk_size} tokens")
    print()

    # Load gLM2 model and tokenizer
    print("Loading gLM2 model...")
    from transformers import AutoModel, AutoTokenizer

    # Set dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    print(f"✓ Model loaded successfully")
    print(f"  Embedding dimension: {model.config.hidden_size}")
    print()

    # Get list of genome files
    genome_files = get_genome_files(args.genome_dir)

    if len(genome_files) == 0:
        print("ERROR: No genome files found!")
        print(f"  Looked in: {args.genome_dir}")
        print("  Please provide genome files (.fna or .fasta)")
        sys.exit(1)

    if args.num_genomes is not None:
        genome_files = genome_files[:args.num_genomes]
        print(f"Processing first {args.num_genomes} genomes")

    # Setup output file
    if args.save_format == "hdf5":
        output_file = output_dir / f"glm2_embeddings_{args.mode}.h5"
        print(f"Saving embeddings to: {output_file}")
    else:
        print(f"Saving embeddings to: {output_dir}")
    print()

    # Process each genome
    metadata = []
    annotation_dir = Path(args.annotation_dir) if args.annotation_dir else None

    for genome_file in tqdm(genome_files, desc="Processing genomes"):
        genome_id = extract_genome_id(genome_file.name)

        # Find corresponding annotation file
        annotation_file = None
        if annotation_dir and annotation_dir.exists():
            # Try common patterns
            patterns = [
                f"{genome_id}_prodigal.gff",
                f"{genome_id}.gff",
                f"{genome_id}_genes.gff",
            ]
            for pattern in patterns:
                candidate = annotation_dir / pattern
                if candidate.exists():
                    annotation_file = candidate
                    break

        try:
            # Process genome based on mode
            if args.mode == "genome":
                embeddings, ids = process_genome_mode(
                    genome_file, annotation_file, model, tokenizer, args
                )
            else:  # genes mode
                embeddings, ids = process_genes_mode(
                    genome_file, annotation_file, model, tokenizer, args
                )

            if embeddings is None:
                print(f"  No valid sequences for {genome_id}")
                continue

            # Save embeddings
            if args.save_format == "hdf5":
                save_embeddings_hdf5(
                    output_file, genome_id, embeddings, ids, args.mode
                )
            else:
                save_embeddings_npz(
                    output_dir, genome_id, embeddings, ids, args.mode
                )

            # Track metadata
            metadata.append({
                "genome_id": genome_id,
                "n_items": len(ids),
                "embedding_dim": embeddings.shape[1],
                "mode": args.mode,
            })

        except Exception as e:
            print(f"  Error processing {genome_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_file = output_dir / f"glm2_embedding_metadata_{args.mode}.csv"
    metadata_df.to_csv(metadata_file, index=False)

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)
    print(f"Processed {len(metadata)} genomes")
    print(f"Total items: {metadata_df['n_items'].sum()}")
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
