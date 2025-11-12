#!/bin/bash
# Quick fix for missing packages in glm2_env

source ~/.bashrc
micromamba activate glm2_env

echo "Installing missing packages..."
pip install einops

echo "Testing gLM2 model download..."
python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('Downloading gLM2_650M model...')
model = AutoModel.from_pretrained(
    'tattabio/gLM2_650M',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print('✓ Model downloaded successfully')

tokenizer = AutoTokenizer.from_pretrained(
    'tattabio/gLM2_650M',
    trust_remote_code=True,
)
print('✓ Tokenizer downloaded successfully')
"

echo "Done!"
