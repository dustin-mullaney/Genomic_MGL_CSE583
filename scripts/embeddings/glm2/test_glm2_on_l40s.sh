#!/bin/bash
#SBATCH --job-name=test_glm2
#SBATCH --partition=chorus
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/test_glm2_%j.out
#SBATCH --error=logs/test_glm2_%j.err

echo "========================================"
echo "Testing gLM2 on L40S GPU"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Load CUDA module
module load CUDA/11.7.0

# Load environment
source ~/.bashrc
micromamba activate glm2_env

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv
echo ""

# Install einops if missing
echo "Installing einops..."
pip install einops -q
echo ""

# Test PyTorch CUDA
echo "Testing PyTorch CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available!')
    print('This may be due to PyTorch not being built with CUDA support.')
"
echo ""

# Download and test gLM2 model
echo "Testing gLM2 model..."
python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('Loading gLM2_650M model...')
model = AutoModel.from_pretrained(
    'tattabio/gLM2_650M',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Try to move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

if device == 'cuda':
    model = model.to(device)
    print(f'Model on GPU: {next(model.parameters()).is_cuda}')

tokenizer = AutoTokenizer.from_pretrained(
    'tattabio/gLM2_650M',
    trust_remote_code=True,
)

# Test inference
test_seq = '<+>MALTK'
print(f'\\nTesting inference with: {test_seq}')
inputs = tokenizer(test_seq, return_tensors='pt')

if device == 'cuda':
    inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)

print(f'✓ Inference successful!')
print(f'  Output shape: {embedding.shape}')
print(f'  Output device: {embedding.device}')
print(f'  Expected: (1, {model.config.hidden_size})')
"

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
