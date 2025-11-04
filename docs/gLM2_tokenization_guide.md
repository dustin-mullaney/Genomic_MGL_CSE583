# gLM2 Tokenization Guide

## Critical Information for Jacobian Analysis and Token Manipulation

This document describes the exact tokenization behavior of the gLM2 model, which is **critical** for correctly interpreting embeddings and performing mutation analyses.

## Key Findings

### Special Token Tokenization

gLM2 has unusual tokenization behavior for its special tokens:

1. **`<|>` (DNA marker) is tokenized as 3 `<unk>` tokens**
   - Each `<|>` in your input becomes 3 separate tokens
   - Token IDs: `[3, 3, 3]`
   - All decode as `'<unk>'`

2. **`<+>` (Protein marker) is a single token**
   - Token ID: `33`
   - Decodes as `'<+>'`

### Token Position Offsets

When you tokenize a sequence and get embeddings, you must account for these special tokens to correctly map between sequence positions and embedding positions.

#### DNA-only sequences: `<|>SEQUENCE`

```
Input: <|>ATCGATCG (8 nucleotides)
Tokens: [3, 3, 3, 5, 11, 23, 6, 5, 11, 23, 6]
         └─<|>──┘  └─────DNA sequence────┘

Total tokens: 11
- Positions 0-2: <|> marker (3 <unk> tokens)
- Positions 3-10: DNA sequence (8 nucleotides)

OFFSET = 3
Use embedding[3:3+L] where L = sequence length
```

#### Protein-only sequences: `<+>SEQUENCE`

```
Input: <+>ACDEFGH (7 amino acids)
Tokens: [33, 5, 23, 13, 9, 18, 6, 21]
         └┘  └────Protein sequence────┘

Total tokens: 8
- Position 0: <+> marker (1 token)
- Positions 1-7: Protein sequence (7 amino acids)

OFFSET = 1
Use embedding[1:1+L] where L = sequence length
```

#### Mixed genomic sequences: `<|>DNA5<+>PROTEIN<|>DNA3`

```
Input: <|>ATCG<+>ACDE<|>ATCG
       (4 nt + 4 aa + 4 nt)

Tokens: [3, 3, 3, 5, 11, 23, 6, 33, 5, 23, 13, 9, 3, 3, 3, 5, 11, 23, 6]
         └─<|>──┘ └──DNA5─┘ └┘ └PROT─┘ └─<|>──┘ └──DNA3─┘

Total tokens: 19
- Positions 0-2: First <|> (3 tokens)
- Positions 3-6: 5' DNA (4 nt)
- Position 7: <+> (1 token)
- Positions 8-11: Protein (4 aa)
- Positions 12-14: Second <|> (3 tokens)
- Positions 15-18: 3' DNA (4 nt)

Structure breakdown:
- 5' DNA region: embedding[3:3+L_DNA5]
- Protein region: embedding[3+L_DNA5+1:3+L_DNA5+1+L_PROTEIN]
- 3' DNA region: embedding[3+L_DNA5+1+L_PROTEIN+3:3+L_DNA5+1+L_PROTEIN+3+L_DNA3]

Simplified for equal segments (L_DNA5 = L_DNA3):
- Each <|> adds 3 tokens
- <+> adds 1 token
- Total overhead: 3 + 1 + 3 = 7 tokens
```

## Implementation Guidelines

### For Single-Region Analysis (DNA or Protein only)

```python
# DNA sequence
dna_input = f"<|>{dna_sequence}"
tokens = tokenizer(dna_input, return_tensors="pt")
tokens = {k: v.to(device) for k, v in tokens.items() if k != 'token_type_ids'}

output = model(**tokens)
embedding = output.last_hidden_state[0].float().cpu().numpy()

# Extract DNA positions only
start_offset = 3  # Skip the 3 <unk> tokens from <|>
L = len(dna_sequence)
dna_embedding = embedding[start_offset:start_offset+L]
```

```python
# Protein sequence
protein_input = f"<+>{protein_sequence}"
tokens = tokenizer(protein_input, return_tensors="pt")
tokens = {k: v.to(device) for k, v in tokens.items() if k != 'token_type_ids'}

output = model(**tokens)
embedding = output.last_hidden_state[0].float().cpu().numpy()

# Extract protein positions only
start_offset = 1  # Skip the 1 token from <+>
L = len(protein_sequence)
protein_embedding = embedding[start_offset:start_offset+L]
```

### For Multi-Region Analysis (Genomic context)

```python
# Genomic context: 5'UTR + CDS + 3'UTR
L_dna5 = len(dna_5prime)
L_protein = len(protein_sequence)
L_dna3 = len(dna_3prime)

full_input = f"<|>{dna_5prime}<+>{protein_sequence}<|>{dna_3prime}"
tokens = tokenizer(full_input, return_tensors="pt")
tokens = {k: v.to(device) for k, v in tokens.items() if k != 'token_type_ids'}

output = model(**tokens)
embedding = output.last_hidden_state[0].float().cpu().numpy()

# Extract each region
# 5' DNA: starts after first <|> (3 tokens)
dna5_start = 3
dna5_embedding = embedding[dna5_start:dna5_start+L_dna5]

# Protein: starts after 5' DNA + <+> (1 token)
protein_start = dna5_start + L_dna5 + 1
protein_embedding = embedding[protein_start:protein_start+L_protein]

# 3' DNA: starts after protein + second <|> (3 tokens)
dna3_start = protein_start + L_protein + 3
dna3_embedding = embedding[dna3_start:dna3_start+L_dna3]
```

## Important Notes

1. **Always remove `token_type_ids`**: gLM2's forward method doesn't accept this parameter
   ```python
   tokens = {k: v for k, v in tokens.items() if k != 'token_type_ids'}
   ```

2. **Verify embedding dimensions**: Always check that your extracted embedding dimensions match your expected sequence length

3. **Off-by-one errors**: The most common bug is using the wrong offset. Use the diagnostic script to verify:
   ```bash
   python scripts/embeddings/diagnose_tokenization.py
   ```

4. **Mutation analysis**: When mutating positions, remember that the mutation affects the sequence position, but you need to extract embeddings accounting for the offset

## Diagnostic Job

To verify tokenization behavior, run:
```bash
sbatch scripts/embeddings/submit_diagnose_tokenization.sh
```

This will show:
- Exact token IDs for different input formats
- Embedding shapes
- Correct offset strategies

## Historical Context

During development of the Jacobian analysis tools, we initially used `offset=1` which worked for some analyses but was incorrect. The diagnostic revealed that:
- `<|>` is **3 tokens**, not 1
- This explains why we had shape mismatches (503 tokens for 500 nt)
- Correct offset for DNA-only is **3**, not 1

## Files Using These Offsets

- `calculate_5utr_jacobian.py`: Uses offset=3 for DNA
- `calculate_protein_jacobian.py`: Uses offset=1 for protein (gLM2)
- `calculate_genomic_jacobian.py`: **May need correction** - currently uses offset=1
- `calculate_dna_vs_protein_jacobian.py`: Handles both DNA and protein

## TODO

- [ ] Verify and potentially fix offset in `calculate_genomic_jacobian.py`
- [ ] Verify and potentially fix offset in `compare_protein_context_effect.py`
- [ ] Add automated tests to verify correct position extraction
