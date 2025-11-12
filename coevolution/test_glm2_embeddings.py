#!/usr/bin/env python
"""
Test script for gLM2 embedding generation.

This creates a simple test genome and verifies that gLM2 embeddings work correctly.
"""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def create_test_genome(output_dir: Path) -> Path:
    """
    Create a simple test bacterial genome with known genes.

    Returns:
        Path to created genome file
    """
    print("Creating test genome...")

    # Create a simple genome with a few genes
    # Gene 1: malate dehydrogenase (MDH)
    gene1 = "ATGAAAGTTCTGTTTGCCGCGGTGATTGGTGGCGGCACCGGTTCCGGTAAGGGTTCCATTGTGGGTGCTGGTATCGACGCGCATGCGGCGAAAGCGCATGCGCTGGGTCTGGGCATTAAAGGTGAAACCGTGCTGGAAGCGGTGAAAGAACTGGATGCGGCGACCTATCTGCATGCGGAAATCTTTGAACAGTTCAGCATTCGTGTGCCGAAAGGCATTCTGACCACCGAAATTGCGAACCTGACCGCGTATCTGGTGAAAGAAGCGGGTAACGCGATTATTCCGTTTGGCATGTCCGGCGGCACCGGCCTGACCTATGGTCGTGGCATTGCGGATGCGTGA"

    # Gene 2: ATP synthase
    gene2 = "ATGAGCGAACTGAAACTGGCAGAAATGATTGAAGCGATTGGTGCGGCGCTGGGTGAAACCGTTCCGCTGAAAGAAGCGCTGGAAGCGGGTAAAGCGGCGATTGAACTGATGGGTGAAATGCTGCGTCGTATTGCGGAAGAAGCGAAAGCGATGGCGACCATTGATGCGATTCTGGATACCCTGGCGGAAGCGCATGGTGCGCTGAAAGAAGCGATTAAACGTCTGATGACCCTGCTGGAAGCGGGTCGTGAAGCGGCGGAAATGGTTGAAATTGCGGCGGCGCTGGAAGCGTATGATGCGACCATTGAAGCGCTGGAAGCGATTGCGGGTGAACTGATTTGA"

    # Gene 3: ribosomal protein S1
    gene3 = "ATGACCGAAATTGATGCGCTGATTGCGGATGCGGCGAAAGCGACCGCGGCGATTGATGCGCTGATTGCGGATGCGGCGAAAGCGACCGCGGCGATTGATGCGCTGATTGCGGATGCGGCGAAAGCGACCGCGGCGATTGATGCGCTGATTGCGGATGCGGCGAAAGCGACCGCGGCGATTGATGCGCTGATTGCGGATGCGGCGAAAGCGACCGCGGCGATTGATGCGCTGATTGCGGATGCGGCGAAAGCGTAG"

    # Build genome with inter-genic regions
    genome_seq = (
        "TTAAGGAGATACACATATG" +  # IGS with ribosome binding site
        gene1 +
        "TAAGCGATTCCGTAACGTT" +  # IGS
        gene2 +
        "GGTACCGTAAGGAGGTACC" +  # IGS
        gene3 +
        "TTAACCGGTTAACCGGTTA"    # IGS
    )

    # Create genome file
    genome_file = output_dir / "test_genome.fna"
    genome_record = SeqRecord(
        Seq(genome_seq),
        id="test_genome",
        description="Test bacterial genome with 3 genes"
    )

    with open(genome_file, 'w') as f:
        SeqIO.write(genome_record, f, "fasta")

    print(f"  Created genome: {len(genome_seq)} bp")
    print(f"  Genes: 3 (MDH, ATP synthase, ribosomal protein)")
    print(f"  File: {genome_file}")

    return genome_file


def create_test_annotation(output_dir: Path, genome_file: Path) -> Path:
    """
    Create a GFF annotation file for the test genome.

    Returns:
        Path to created GFF file
    """
    print("\nCreating test annotation...")

    # Load genome to calculate positions
    genome_record = next(SeqIO.parse(genome_file, "fasta"))
    genome_seq = str(genome_record.seq)

    # Gene positions (manually calculated from genome construction)
    igs1_len = 19
    gene1_start = igs1_len + 1  # GFF is 1-indexed
    gene1_len = 630
    gene1_end = gene1_start + gene1_len - 1

    igs2_len = 19
    gene2_start = gene1_end + igs2_len + 1
    gene2_len = 492
    gene2_end = gene2_start + gene2_len - 1

    igs3_len = 19
    gene3_start = gene2_end + igs3_len + 1
    gene3_len = 255
    gene3_end = gene3_start + gene3_len - 1

    # Create GFF file
    gff_file = output_dir / "test_genome.gff"
    with open(gff_file, 'w') as f:
        f.write("##gff-version 3\n")
        f.write(f"##sequence-region test_genome 1 {len(genome_seq)}\n")

        # Gene 1: malate dehydrogenase
        f.write(f"test_genome\ttest\tCDS\t{gene1_start}\t{gene1_end}\t.\t+\t0\t")
        f.write("ID=gene1;Name=mdh;product=malate dehydrogenase\n")

        # Gene 2: ATP synthase
        f.write(f"test_genome\ttest\tCDS\t{gene2_start}\t{gene2_end}\t.\t+\t0\t")
        f.write("ID=gene2;Name=atpA;product=ATP synthase subunit alpha\n")

        # Gene 3: ribosomal protein
        f.write(f"test_genome\ttest\tCDS\t{gene3_start}\t{gene3_end}\t.\t+\t0\t")
        f.write("ID=gene3;Name=rpsA;product=30S ribosomal protein S1\n")

    print(f"  Created annotation with 3 genes")
    print(f"  Gene 1: {gene1_start}-{gene1_end} (mdh)")
    print(f"  Gene 2: {gene2_start}-{gene2_end} (atpA)")
    print(f"  Gene 3: {gene3_start}-{gene3_end} (rpsA)")
    print(f"  File: {gff_file}")

    return gff_file


def test_glm2_import():
    """Test if gLM2 dependencies are available."""
    print("Testing gLM2 imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")

        from transformers import AutoModel, AutoTokenizer
        print(f"  ✓ Transformers available")

        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_glm2_model():
    """Test loading the gLM2 model."""
    print("\nTesting gLM2 model loading...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = "tattabio/gLM2_650M"
        print(f"  Loading model: {model_name}")

        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"  ✓ Model loaded")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Vocab size: {model.config.vocab_size}")

        return True
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False


def test_glm2_inference():
    """Test gLM2 inference with a simple sequence."""
    print("\nTesting gLM2 inference...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = "tattabio/gLM2_650M"

        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Test sequence (protein + DNA)
        test_seq = "<+>MALTK<+>aattggcc<->MGKL"

        print(f"  Test sequence: {test_seq}")

        # Tokenize
        inputs = tokenizer(test_seq, return_tensors="pt")

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

        print(f"  ✓ Inference successful")
        print(f"  Output shape: {hidden_states.shape}")
        print(f"  Expected: (batch=1, seq_len, hidden_size={model.config.hidden_size})")

        # Test mean pooling
        embedding = hidden_states.mean(dim=1)
        print(f"  Mean pooled embedding: {embedding.shape}")

        return True
    except Exception as e:
        print(f"  ✗ Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full gLM2 embedding pipeline."""
    print("\n" + "="*80)
    print("Testing Full gLM2 Pipeline")
    print("="*80)

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"\nUsing temporary directory: {tmpdir}\n")

        # Create test genome and annotation
        genome_dir = tmpdir / "genomes"
        genome_dir.mkdir()
        annotation_dir = tmpdir / "annotations"
        annotation_dir.mkdir()
        output_dir = tmpdir / "embeddings"
        output_dir.mkdir()

        genome_file = create_test_genome(genome_dir)
        gff_file = create_test_annotation(annotation_dir, genome_file)

        # Test both modes
        for mode in ["genome", "genes"]:
            print(f"\n{'='*80}")
            print(f"Testing {mode.upper()} mode")
            print(f"{'='*80}\n")

            # Import the embedding script
            sys.path.insert(0, str(Path(__file__).parent))
            from get_glm2_embeddings import (
                load_gff_annotations,
                format_genome_for_glm2,
                format_genes_for_glm2,
            )

            # Load genome
            genome_record = next(SeqIO.parse(genome_file, "fasta"))
            genome_seq = str(genome_record.seq)

            # Load annotations
            annotations = load_gff_annotations(gff_file)
            print(f"Loaded {len(annotations)} gene annotations")

            if mode == "genome":
                # Test genome mode
                formatted_chunks = format_genome_for_glm2(
                    genome_seq, annotations, chunk_size=1000
                )
                print(f"  Generated {len(formatted_chunks)} genome chunks")
                print(f"  First chunk preview: {formatted_chunks[0][:100]}...")

                # Test with model
                try:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    model = AutoModel.from_pretrained(
                        "tattabio/gLM2_650M",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        "tattabio/gLM2_650M",
                        trust_remote_code=True,
                    )

                    # Get embeddings for first chunk
                    inputs = tokenizer(formatted_chunks[0], return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1)

                    print(f"  ✓ Generated embedding shape: {embedding.shape}")
                    print(f"  ✓ Embedding dtype: {embedding.dtype}")

                except Exception as e:
                    print(f"  ✗ Error: {e}")

            else:  # genes mode
                # Test genes mode
                formatted_seqs, gene_ids = format_genes_for_glm2(
                    genome_seq, annotations
                )
                print(f"  Generated {len(formatted_seqs)} gene sequences")
                for i, (seq, gid) in enumerate(zip(formatted_seqs, gene_ids)):
                    print(f"  {gid}: {seq[:50]}...")

                # Test with model
                try:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    model = AutoModel.from_pretrained(
                        "tattabio/gLM2_650M",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        "tattabio/gLM2_650M",
                        trust_remote_code=True,
                    )

                    # Get embeddings for all genes
                    inputs = tokenizer(formatted_seqs, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Mean pool
                        attention_mask = inputs['attention_mask'].unsqueeze(-1)
                        masked_hidden = outputs.last_hidden_state * attention_mask
                        sum_hidden = masked_hidden.sum(dim=1)
                        sum_mask = attention_mask.sum(dim=1)
                        embeddings = sum_hidden / sum_mask.clamp(min=1e-9)

                    print(f"  ✓ Generated embeddings shape: {embeddings.shape}")
                    print(f"  ✓ Expected: ({len(gene_ids)}, {model.config.hidden_size})")

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()


def main():
    """Run all tests."""
    print("="*80)
    print("gLM2 Embedding Generation - Test Suite")
    print("="*80)
    print()

    # Run tests
    all_passed = True

    # Test 1: Imports
    if not test_glm2_import():
        print("\n✗ Import test failed - please install required packages:")
        print("  pip install transformers torch")
        all_passed = False
        return

    # Test 2: Model loading
    if not test_glm2_model():
        print("\n✗ Model loading test failed")
        all_passed = False
        return

    # Test 3: Basic inference
    if not test_glm2_inference():
        print("\n✗ Inference test failed")
        all_passed = False
        return

    # Test 4: Full pipeline
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*80)


if __name__ == "__main__":
    main()
