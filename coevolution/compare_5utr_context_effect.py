#!/usr/bin/env python3
"""
Compare 5' UTR Jacobians computed in isolation vs with genomic context.

This script extracts the 5' UTR sub-matrix from the genomic Jacobian
and compares it with the 5' UTR-only Jacobian to see if downstream
protein context affects 5' UTR internal coupling patterns.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_jacobian(filepath: Path) -> np.ndarray:
    """Load a Jacobian matrix from a numpy file."""
    return np.load(filepath)


def extract_5utr_submatrix(genomic_jacobian: np.ndarray,
                           L_dna_5: int) -> np.ndarray:
    """
    Extract the 5' UTR sub-matrix from a genomic Jacobian.

    Args:
        genomic_jacobian: Full genomic Jacobian (DNA5 + Protein + DNA3)
        L_dna_5: Length of 5' DNA region

    Returns:
        5' UTR sub-matrix (L_dna_5 x L_dna_5)
    """
    return genomic_jacobian[:L_dna_5, :L_dna_5]


def plot_comparison(utr_only: np.ndarray,
                   utr_from_genomic: np.ndarray,
                   seq_id: str,
                   output_path: Path):
    """
    Plot side-by-side comparison of 5' UTR Jacobians.

    Args:
        utr_only: Jacobian from 5' UTR-only encoding
        utr_from_genomic: 5' UTR sub-matrix from genomic encoding
        seq_id: Sequence identifier
        output_path: Path to save the figure
    """
    L = utr_only.shape[0]

    # Apply log10 transform
    utr_only_log = np.log10(utr_only + 1e-10)
    utr_genomic_log = np.log10(utr_from_genomic + 1e-10)

    # Calculate difference
    difference = utr_from_genomic - utr_only
    difference_log = utr_genomic_log - utr_only_log

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Determine shared color scale for log-transformed heatmaps
    vmin = min(utr_only_log.min(), utr_genomic_log.min())
    vmax = max(utr_only_log.max(), utr_genomic_log.max())

    # Plot 1: 5' UTR in isolation
    ax1 = axes[0, 0]
    sns.heatmap(utr_only_log, cmap='viridis', ax=ax1,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'log₁₀(Coupling)'})
    ax1.set_title('5\' UTR in Isolation\n(gLM2: <|>5\'UTR)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position j (mutated)', fontsize=11)
    ax1.set_ylabel('Position i (measured)', fontsize=11)

    # Add statistics
    mean_val = utr_only.mean()
    max_val = utr_only.max()
    ax1.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nMax: {max_val:.2f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

    # Plot 2: 5' UTR from genomic context
    ax2 = axes[0, 1]
    sns.heatmap(utr_genomic_log, cmap='viridis', ax=ax2,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'log₁₀(Coupling)'})
    ax2.set_title('5\' UTR with Genomic Context\n(gLM2: <|>5\'UTR<+>Protein<|>3\'UTR)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Position j (mutated)', fontsize=11)
    ax2.set_ylabel('Position i (measured)', fontsize=11)

    # Add statistics
    mean_val = utr_from_genomic.mean()
    max_val = utr_from_genomic.max()
    ax2.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nMax: {max_val:.2f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

    # Plot 3: Absolute difference
    ax3 = axes[1, 0]
    max_abs = max(abs(difference.min()), abs(difference.max()))
    sns.heatmap(difference, cmap='RdBu_r', ax=ax3, center=0,
                vmin=-max_abs, vmax=max_abs,
                cbar_kws={'label': 'Difference (Context - Isolated)'})
    ax3.set_title('Absolute Difference\n(With Context - Isolated)',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Position j (mutated)', fontsize=11)
    ax3.set_ylabel('Position i (measured)', fontsize=11)

    # Add statistics
    mean_diff = difference.mean()
    ax3.text(0.02, 0.98, f'Mean Δ: {mean_diff:.2f}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

    # Plot 4: Log-ratio
    ax4 = axes[1, 1]
    max_abs_log = max(abs(difference_log.min()), abs(difference_log.max()))
    sns.heatmap(difference_log, cmap='RdBu_r', ax=ax4, center=0,
                vmin=-max_abs_log, vmax=max_abs_log,
                cbar_kws={'label': 'log₁₀(Context / Isolated)'})
    ax4.set_title('Log-Ratio\nlog₁₀(With Context / Isolated)',
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Position j (mutated)', fontsize=11)
    ax4.set_ylabel('Position i (measured)', fontsize=11)

    # Add statistics
    mean_log_ratio = difference_log.mean()
    ax4.text(0.02, 0.98, f'Mean log-ratio: {mean_log_ratio:.3f}',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

    # Main title
    fig.suptitle(f'5\' UTR Context Effect Comparison\n{seq_id}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    print(f"    Isolated         - Mean: {utr_only.mean():.4f}, Max: {utr_only.max():.4f}")
    print(f"    With context     - Mean: {utr_from_genomic.mean():.4f}, Max: {utr_from_genomic.max():.4f}")
    print(f"    Absolute diff    - Mean: {difference.mean():.4f}")
    print(f"    Relative change  - Mean: {(difference.mean() / utr_only.mean() * 100):.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare 5' UTR Jacobians with and without genomic context"
    )
    parser.add_argument(
        "--utr-jacobian-dir",
        type=str,
        default="results/jacobian_5utr",
        help="Directory containing 5' UTR-only Jacobians"
    )
    parser.add_argument(
        "--genomic-jacobian-dir",
        type=str,
        default="results/jacobian_genomic",
        help="Directory containing genomic Jacobians"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/jacobian_5utr_context_comparison",
        help="Output directory for comparison plots"
    )
    parser.add_argument(
        "--dna-flank-length",
        type=int,
        default=500,
        help="Length of 5' UTR region"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("5' UTR Jacobian Context Effect Analysis")
    print("Comparing 5' UTR coupling with and without downstream protein context")
    print("=" * 80)
    print(f"5' UTR-only directory: {args.utr_jacobian_dir}")
    print(f"Genomic directory: {args.genomic_jacobian_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"5' UTR length: {args.dna_flank_length} nt")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    utr_dir = Path(args.utr_jacobian_dir)
    genomic_dir = Path(args.genomic_jacobian_dir)

    # Find all 5' UTR Jacobian files
    utr_files = sorted(utr_dir.glob("*_5utr_jacobian.npy"))

    if not utr_files:
        print(f"ERROR: No 5' UTR Jacobian files found in {utr_dir}")
        return

    print(f"Found {len(utr_files)} 5' UTR sequences to compare")
    print()

    # Process each sequence
    for utr_file in utr_files:
        # Extract sequence ID from filename
        # Format: GCF_024496245.1_1_2691_5utr_jacobian.npy
        seq_id = utr_file.stem.replace("_5utr_jacobian", "")

        print(f"Processing: {seq_id}")

        # Load 5' UTR-only Jacobian
        utr_jacobian = load_jacobian(utr_file)
        L_utr = utr_jacobian.shape[0]
        print(f"  5' UTR length: {L_utr} nt")

        # Find corresponding genomic Jacobian
        genomic_file = genomic_dir / f"{seq_id}_glm2_genomic_jacobian.npy"

        if not genomic_file.exists():
            print(f"  WARNING: Genomic Jacobian not found: {genomic_file}")
            print()
            continue

        # Load genomic Jacobian
        genomic_jacobian = load_jacobian(genomic_file)
        print(f"  Genomic total length: {genomic_jacobian.shape[0]} (5'UTR + Protein + 3'UTR)")

        # Extract 5' UTR sub-matrix from genomic Jacobian
        utr_from_genomic = extract_5utr_submatrix(genomic_jacobian, L_utr)

        # Verify dimensions match
        if utr_jacobian.shape != utr_from_genomic.shape:
            print(f"  ERROR: Shape mismatch!")
            print(f"    Isolated: {utr_jacobian.shape}")
            print(f"    From genomic: {utr_from_genomic.shape}")
            print()
            continue

        # Create comparison plot
        output_path = output_dir / f"{seq_id}_5utr_context_comparison.png"
        plot_comparison(utr_jacobian, utr_from_genomic, seq_id, output_path)

    print("=" * 80)
    print("✓ Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
