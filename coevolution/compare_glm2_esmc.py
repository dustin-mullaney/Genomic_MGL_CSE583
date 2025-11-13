#!/usr/bin/env python
"""
Compare gLM2 and ESM-C embeddings on a standard protein set.

This script:
1. Samples 1000 proteins from the dataset
2. Generates embeddings using both gLM2 and ESM-C
3. Compares inter-protein distances between the two embedding spaces
4. Analyzes correlation and agreement between the two models

Key challenge: Different embedding dimensions (gLM2: 1280, ESM-C: 960)
Solution: Compare pairwise distances (distance matrices are same size)
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import torch
import h5py


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare gLM2 and ESM-C embeddings"
    )
    parser.add_argument(
        "--protein-dir",
        type=str,
        default="data/refseq_gene_annotations",
        help="Directory containing protein .faa files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmp/comparison",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--n-proteins",
        type=int,
        default=1000,
        help="Number of proteins to sample for comparison",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for protein sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum protein length to include",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum protein length to include",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "correlation"],
        help="Distance metric for comparison",
    )
    return parser.parse_args()


def sample_proteins(
    protein_dir: Path,
    n_proteins: int,
    min_length: int,
    max_length: int,
    seed: int,
) -> List[Dict]:
    """
    Sample N proteins from the dataset.

    Returns:
        List of dicts with 'sequence', 'id', 'genome_id', 'length'
    """
    print(f"\nSampling {n_proteins} proteins...")
    print(f"  Length range: {min_length}-{max_length} aa")

    np.random.seed(seed)

    # Collect all proteins
    all_proteins = []
    protein_files = sorted(protein_dir.glob("*_prodigal_proteins.faa"))

    print(f"  Found {len(protein_files)} genome files")

    for faa_file in tqdm(protein_files[:100], desc="Scanning genomes"):  # Sample from first 100 genomes
        genome_id = faa_file.name.split("_prodigal_proteins")[0]

        for record in SeqIO.parse(faa_file, "fasta"):
            seq = str(record.seq).replace("*", "")  # Remove stop codons
            seq_len = len(seq)

            if min_length <= seq_len <= max_length:
                all_proteins.append({
                    "sequence": seq,
                    "id": record.id,
                    "genome_id": genome_id,
                    "length": seq_len,
                    "description": record.description,
                })

    print(f"  Total candidate proteins: {len(all_proteins)}")

    # Sample N proteins
    if len(all_proteins) > n_proteins:
        indices = np.random.choice(len(all_proteins), n_proteins, replace=False)
        sampled = [all_proteins[i] for i in indices]
    else:
        sampled = all_proteins

    print(f"  Sampled: {len(sampled)} proteins")
    print(f"  Avg length: {np.mean([p['length'] for p in sampled]):.1f} aa")

    return sampled


def get_esmc_embeddings(
    proteins: List[Dict],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Generate ESM-C embeddings for proteins.

    Returns:
        Array of shape (n_proteins, 960)
    """
    print("\nGenerating ESM-C embeddings...")

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein

    # Load model
    print("  Loading ESM-C 600M model...")
    model = ESMC.from_pretrained("esmc_600m").to(device)
    model.eval()

    sequences = [p["sequence"] for p in proteins]
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="  Processing"):
            batch = sequences[i:i+batch_size]
            batch_proteins = [ESMProtein(sequence=seq) for seq in batch]

            batch_embeddings = []
            for protein in batch_proteins:
                # Get embedding and mean pool
                emb = model.encode(protein)
                emb_pooled = emb.mean(axis=0).cpu().numpy()
                batch_embeddings.append(emb_pooled)

            all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings)
    print(f"  ESM-C embeddings shape: {embeddings.shape}")

    return embeddings


def get_glm2_embeddings(
    proteins: List[Dict],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Generate gLM2 embeddings for proteins.

    Returns:
        Array of shape (n_proteins, 1280)
    """
    print("\nGenerating gLM2 embeddings...")

    from transformers import AutoModel, AutoTokenizer

    # Load model
    print("  Loading gLM2 650M model...")
    model = AutoModel.from_pretrained(
        "tattabio/gLM2_650M",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "tattabio/gLM2_650M",
        trust_remote_code=True,
    )

    # Format sequences for gLM2 (protein mode: <+>PROTEIN)
    sequences = [f"<+>{p['sequence'].upper()}" for p in proteins]
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="  Processing"):
            batch = sequences[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)

            # Get embeddings and mean pool
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            embeddings = sum_hidden / sum_mask.clamp(min=1e-9)

            # Convert to numpy
            batch_embeddings = embeddings.cpu().float().numpy()
            all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)
    print(f"  gLM2 embeddings shape: {embeddings.shape}")

    return embeddings


def compute_distance_matrix(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Args:
        embeddings: (n_samples, n_dims)
        metric: distance metric

    Returns:
        Distance matrix (n_samples, n_samples)
    """
    # Compute pairwise distances
    distances = pdist(embeddings, metric=metric)

    # Convert to square matrix
    dist_matrix = squareform(distances)

    return dist_matrix


def compare_distance_matrices(
    dist_esmc: np.ndarray,
    dist_glm2: np.ndarray,
    protein_info: List[Dict],
    output_dir: Path,
    metric: str,
):
    """
    Compare distance matrices from ESM-C and gLM2.

    Analyzes:
    1. Correlation between distances
    2. Agreement on nearest neighbors
    3. Visualization of distance relationships
    """
    print("\n" + "="*80)
    print("Distance Matrix Comparison")
    print("="*80)

    n = len(protein_info)
    print(f"\nMatrix size: {n} x {n} = {n*(n-1)//2:,} unique pairs")
    print(f"Distance metric: {metric}")

    # Extract upper triangle (unique pairs)
    triu_indices = np.triu_indices(n, k=1)
    esmc_distances = dist_esmc[triu_indices]
    glm2_distances = dist_glm2[triu_indices]

    print(f"\nESM-C distances:")
    print(f"  Min: {esmc_distances.min():.4f}")
    print(f"  Max: {esmc_distances.max():.4f}")
    print(f"  Mean: {esmc_distances.mean():.4f}")
    print(f"  Median: {np.median(esmc_distances):.4f}")

    print(f"\ngLM2 distances:")
    print(f"  Min: {glm2_distances.min():.4f}")
    print(f"  Max: {glm2_distances.max():.4f}")
    print(f"  Mean: {glm2_distances.mean():.4f}")
    print(f"  Median: {np.median(glm2_distances):.4f}")

    # Correlation analysis
    print("\n" + "-"*80)
    print("Correlation Analysis")
    print("-"*80)

    pearson_r, pearson_p = pearsonr(esmc_distances, glm2_distances)
    spearman_r, spearman_p = spearmanr(esmc_distances, glm2_distances)

    print(f"\nPearson correlation:  r = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"Spearman correlation: ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

    # Nearest neighbor agreement
    print("\n" + "-"*80)
    print("Nearest Neighbor Agreement")
    print("-"*80)

    for k in [1, 5, 10, 20]:
        agreement = compute_nn_agreement(dist_esmc, dist_glm2, k)
        print(f"  Top-{k} neighbors: {agreement*100:.1f}% agreement")

    # Save results
    results = {
        "n_proteins": n,
        "n_pairs": len(esmc_distances),
        "metric": metric,
        "esmc_stats": {
            "min": float(esmc_distances.min()),
            "max": float(esmc_distances.max()),
            "mean": float(esmc_distances.mean()),
            "median": float(np.median(esmc_distances)),
            "std": float(esmc_distances.std()),
        },
        "glm2_stats": {
            "min": float(glm2_distances.min()),
            "max": float(glm2_distances.max()),
            "mean": float(glm2_distances.mean()),
            "median": float(np.median(glm2_distances)),
            "std": float(glm2_distances.std()),
        },
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }

    # Save to JSON
    import json
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(
        esmc_distances, glm2_distances,
        dist_esmc, dist_glm2,
        protein_info, output_dir, metric
    )

    return results


def compute_nn_agreement(
    dist_matrix1: np.ndarray,
    dist_matrix2: np.ndarray,
    k: int,
) -> float:
    """
    Compute nearest neighbor agreement between two distance matrices.

    Returns:
        Fraction of neighbors that agree (0-1)
    """
    n = dist_matrix1.shape[0]
    agreements = []

    for i in range(n):
        # Get top k neighbors (excluding self)
        neighbors1 = np.argsort(dist_matrix1[i])[1:k+1]
        neighbors2 = np.argsort(dist_matrix2[i])[1:k+1]

        # Compute overlap
        overlap = len(set(neighbors1) & set(neighbors2))
        agreements.append(overlap / k)

    return np.mean(agreements)


def create_visualizations(
    esmc_distances: np.ndarray,
    glm2_distances: np.ndarray,
    dist_esmc_matrix: np.ndarray,
    dist_glm2_matrix: np.ndarray,
    protein_info: List[Dict],
    output_dir: Path,
    metric: str,
):
    """Create comparison visualizations."""

    # 1. Scatter plot of distances
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subsample for visualization if too many points
    if len(esmc_distances) > 50000:
        indices = np.random.choice(len(esmc_distances), 50000, replace=False)
        esmc_sub = esmc_distances[indices]
        glm2_sub = glm2_distances[indices]
    else:
        esmc_sub = esmc_distances
        glm2_sub = glm2_distances

    # Hexbin plot
    hb = axes[0].hexbin(esmc_sub, glm2_sub, gridsize=50, cmap='viridis', mincnt=1)
    axes[0].set_xlabel(f"ESM-C Distance ({metric})")
    axes[0].set_ylabel(f"gLM2 Distance ({metric})")
    axes[0].set_title("Pairwise Distance Comparison")
    axes[0].plot([esmc_sub.min(), esmc_sub.max()],
                 [esmc_sub.min(), esmc_sub.max()],
                 'r--', alpha=0.5, label='y=x')
    axes[0].legend()
    plt.colorbar(hb, ax=axes[0], label='Count')

    # Density scatter
    axes[1].scatter(esmc_sub, glm2_sub, alpha=0.1, s=1)
    axes[1].set_xlabel(f"ESM-C Distance ({metric})")
    axes[1].set_ylabel(f"gLM2 Distance ({metric})")
    axes[1].set_title("Distance Correlation")

    # Add correlation text
    pearson_r, _ = pearsonr(esmc_distances, glm2_distances)
    spearman_r, _ = spearmanr(esmc_distances, glm2_distances)
    axes[1].text(0.05, 0.95,
                 f"Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}",
                 transform=axes[1].transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "distance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Distance distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(esmc_distances, bins=50, alpha=0.7, label='ESM-C', density=True)
    axes[0].hist(glm2_distances, bins=50, alpha=0.7, label='gLM2', density=True)
    axes[0].set_xlabel(f"Distance ({metric})")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distance Distributions")
    axes[0].legend()

    # Cumulative distribution
    axes[1].hist(esmc_distances, bins=100, alpha=0.7, label='ESM-C',
                 density=True, cumulative=True, histtype='step', linewidth=2)
    axes[1].hist(glm2_distances, bins=100, alpha=0.7, label='gLM2',
                 density=True, cumulative=True, histtype='step', linewidth=2)
    axes[1].set_xlabel(f"Distance ({metric})")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("Cumulative Distance Distributions")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "distance_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Nearest neighbor agreement heatmap (for subset)
    n_sample = min(100, len(protein_info))
    indices = np.random.choice(len(protein_info), n_sample, replace=False)
    indices = np.sort(indices)

    dist_esmc_sub = dist_esmc_matrix[np.ix_(indices, indices)]
    dist_glm2_sub = dist_glm2_matrix[np.ix_(indices, indices)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(dist_esmc_sub, cmap='viridis', aspect='auto')
    axes[0].set_title("ESM-C Distance Matrix")
    axes[0].set_xlabel("Protein Index")
    axes[0].set_ylabel("Protein Index")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(dist_glm2_sub, cmap='viridis', aspect='auto')
    axes[1].set_title("gLM2 Distance Matrix")
    axes[1].set_xlabel("Protein Index")
    axes[1].set_ylabel("Protein Index")
    plt.colorbar(im1, ax=axes[1])

    # Difference
    diff = dist_esmc_sub - dist_glm2_sub
    im2 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto',
                         vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title("Difference (ESM-C - gLM2)")
    axes[2].set_xlabel("Protein Index")
    axes[2].set_ylabel("Protein Index")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / "distance_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualizations to {output_dir}")


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("ESM-C vs gLM2 Embedding Comparison")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Proteins to sample: {args.n_proteins}")
    print(f"  Length range: {args.min_length}-{args.max_length} aa")
    print(f"  Distance metric: {args.distance_metric}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_dir}")

    # Step 1: Sample proteins
    protein_dir = Path(args.protein_dir)
    proteins = sample_proteins(
        protein_dir,
        args.n_proteins,
        args.min_length,
        args.max_length,
        args.seed,
    )

    # Save protein list
    protein_df = pd.DataFrame([{
        "id": p["id"],
        "genome_id": p["genome_id"],
        "length": p["length"],
        "sequence": p["sequence"],
    } for p in proteins])
    protein_df.to_csv(output_dir / "protein_set.csv", index=False)
    print(f"\nSaved protein set to {output_dir / 'protein_set.csv'}")

    # Step 2: Generate ESM-C embeddings
    esmc_embeddings = get_esmc_embeddings(proteins, args.device, args.batch_size)

    # Save embeddings
    np.save(output_dir / "esmc_embeddings.npy", esmc_embeddings)

    # Step 3: Generate gLM2 embeddings
    glm2_embeddings = get_glm2_embeddings(proteins, args.device, args.batch_size)

    # Save embeddings
    np.save(output_dir / "glm2_embeddings.npy", glm2_embeddings)

    # Step 4: Compute distance matrices
    print("\nComputing distance matrices...")
    dist_esmc = compute_distance_matrix(esmc_embeddings, args.distance_metric)
    dist_glm2 = compute_distance_matrix(glm2_embeddings, args.distance_metric)

    # Save distance matrices
    np.save(output_dir / "distance_matrix_esmc.npy", dist_esmc)
    np.save(output_dir / "distance_matrix_glm2.npy", dist_glm2)

    # Step 5: Compare distance matrices
    results = compare_distance_matrices(
        dist_esmc, dist_glm2, proteins, output_dir, args.distance_metric
    )

    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)
    print(f"\nKey Results:")
    print(f"  Pearson correlation:  {results['pearson_r']:.4f}")
    print(f"  Spearman correlation: {results['spearman_r']:.4f}")
    print(f"\nAll results saved to: {output_dir}")
    print("  - comparison_results.json")
    print("  - distance_comparison.png")
    print("  - distance_distributions.png")
    print("  - distance_matrices.png")
    print("  - esmc_embeddings.npy")
    print("  - glm2_embeddings.npy")


if __name__ == "__main__":
    main()
