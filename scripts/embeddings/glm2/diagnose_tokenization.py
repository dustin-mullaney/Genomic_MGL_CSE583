#!/usr/bin/env python3
"""
Diagnose gLM2 tokenization to understand special token placement.
"""

import torch
from transformers import AutoModel, AutoTokenizer

print("=" * 80)
print("gLM2 Tokenization Diagnostic")
print("=" * 80)
print()

# Load model and tokenizer
print("Loading gLM2 model and tokenizer...")
model = AutoModel.from_pretrained(
    "tattabio/gLM2_650M",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "tattabio/gLM2_650M",
    trust_remote_code=True,
)
print(f"Model loaded on {device}")
print()

# Test cases
test_cases = [
    ("Short DNA", "ATCGATCG", "<|>ATCGATCG"),
    ("Short Protein", "ACDEFGH", "<+>ACDEFGH"),
    ("DNA+Protein", "ATCGATCG", "<|>ATCG<+>ACDE<|>ATCG"),
]

for name, seq, full_input in test_cases:
    print("=" * 80)
    print(f"Test: {name}")
    print("=" * 80)
    print(f"Sequence: {seq}")
    print(f"Full input: {full_input}")
    print(f"Sequence length: {len(seq)}")
    print()

    # Tokenize
    tokens = tokenizer(full_input, return_tensors="pt")
    token_ids = tokens['input_ids'][0]

    print(f"Number of tokens: {len(token_ids)}")
    print(f"Token IDs: {token_ids.tolist()}")
    print()

    # Try to decode each token
    print("Token-by-token decoding:")
    for i, token_id in enumerate(token_ids.tolist()):
        decoded = tokenizer.decode([token_id])
        print(f"  Position {i}: token_id={token_id:5d}, decoded='{decoded}'")
    print()

    # Get embedding
    tokens_filtered = {k: v.to(device) for k, v in tokens.items() if k != 'token_type_ids'}
    with torch.no_grad():
        output = model(**tokens_filtered)

    embedding = output.last_hidden_state[0]
    print(f"Embedding shape: {embedding.shape}")
    print(f"Expected positions for sequence: {len(seq)}")
    print(f"Extra tokens: {embedding.shape[0] - len(seq)}")
    print()

    # Try different offset strategies
    print("Testing offset strategies:")
    for offset in [0, 1, 2, 3]:
        end = offset + len(seq)
        if end <= embedding.shape[0]:
            print(f"  offset={offset}: positions [{offset}:{end}] = {end-offset} positions")
        else:
            print(f"  offset={offset}: OUT OF RANGE")
    print()

print("=" * 80)
print("Diagnostic Complete!")
print("=" * 80)
