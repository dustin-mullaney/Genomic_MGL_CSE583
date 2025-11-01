#!/bin/bash
#SBATCH --job-name=setup_glm2_env
#SBATCH --partition=chorus
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/setup_glm2_env_%j.out
#SBATCH --error=logs/setup_glm2_env_%j.err

################################################################################
# Setup and Test gLM2 Environment on L40S GPU
#
# This script:
# 1. Creates the glm2_env micromamba environment
# 2. Tests the installation
# 3. Downloads the gLM2 model
# 4. Runs the test suite
#
# Usage:
#   sbatch scripts/embeddings/setup_glm2_env_l40s.sh
################################################################################

echo "========================================"
echo "gLM2 Environment Setup - L40S GPU"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load conda/micromamba
source ~/.bashrc

# Check GPU
echo "Checking GPU..."
nvidia-smi
echo ""

# Check CUDA version
echo "CUDA version:"
nvcc --version 2>/dev/null || echo "nvcc not found, checking nvidia-smi for driver info"
nvidia-smi | grep "CUDA Version"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Step 1: Create environment
echo "========================================"
echo "Step 1: Creating glm2_env environment"
echo "========================================"
echo ""

# Try with flexible channel priority first
echo "Attempting to create environment with flexible channel priority..."
micromamba env create -f environment_glm2.yml --channel-priority flexible -y

if [ $? -ne 0 ]; then
    echo ""
    echo "Failed with flexible priority, trying without specific CUDA version..."

    # Create a temporary environment file without strict CUDA version
    cat > /tmp/environment_glm2_flexible.yml <<EOF
name: glm2_env
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  # Python
  - python=3.10

  # PyTorch with CUDA support (let it auto-detect)
  - pytorch>=2.0
  - pytorch-cuda

  # Transformers and HuggingFace
  - transformers>=4.35.0
  - tokenizers
  - huggingface_hub
  - accelerate

  # Scientific computing
  - numpy>=1.24
  - scipy>=1.10
  - pandas>=2.0

  # Biology tools
  - biopython>=1.81

  # Data storage
  - h5py>=3.8

  # Dimensionality reduction & clustering
  - scikit-learn>=1.3
  - umap-learn
  - hdbscan

  # Visualization
  - matplotlib>=3.7
  - seaborn

  # Progress bars
  - tqdm

  # Jupyter for analysis
  - jupyterlab
  - ipywidgets

  # Development tools
  - pytest
  - ipython

  # Pip packages
  - pip
  - pip:
    - sentencepiece
    - safetensors
EOF

    micromamba env create -f /tmp/environment_glm2_flexible.yml --channel-priority flexible -y

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to create environment!"
        echo "Please check the error messages above."
        exit 1
    fi
fi

echo ""
echo "✓ Environment created successfully!"
echo ""

# Step 2: Activate and test
echo "========================================"
echo "Step 2: Testing Installation"
echo "========================================"
echo ""

micromamba activate glm2_env

echo "Python version:"
python --version
echo ""

echo "PyTorch version and CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "Transformers version:"
python -c "from transformers import __version__; print(f'Transformers: {__version__}')"
echo ""

echo "BioPython version:"
python -c "import Bio; print(f'BioPython: {Bio.__version__}')"
echo ""

echo "Other key packages:"
python -c "import numpy, scipy, pandas, sklearn, umap, hdbscan, matplotlib, seaborn; print('✓ All key packages imported successfully')"
echo ""

# Step 3: Download gLM2 model
echo "========================================"
echo "Step 3: Downloading gLM2 Model"
echo "========================================"
echo ""

python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('Downloading gLM2_650M model...')
print('This may take a few minutes (~2.5GB download)')
print()

try:
    model = AutoModel.from_pretrained(
        'tattabio/gLM2_650M',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print('✓ Model downloaded successfully')
    print(f'  Hidden size: {model.config.hidden_size}')
    print(f'  Vocab size: {model.config.vocab_size}')
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        'tattabio/gLM2_650M',
        trust_remote_code=True,
    )
    print('✓ Tokenizer downloaded successfully')
    print(f'  Vocab size: {len(tokenizer)}')

except Exception as e:
    print(f'✗ Error downloading model: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download gLM2 model!"
    exit 1
fi

echo ""

# Step 4: Run test suite
echo "========================================"
echo "Step 4: Running Test Suite"
echo "========================================"
echo ""

python scripts/embeddings/test_glm2_embeddings.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ All tests passed!"
    echo "========================================"
    echo ""
    echo "Environment is ready for use!"
    echo ""
    echo "To use the environment:"
    echo "  micromamba activate glm2_env"
    echo ""
    echo "To run gLM2 embeddings:"
    echo "  python scripts/embeddings/get_glm2_embeddings.py --help"
    echo ""
    echo "To compare with ESM-C:"
    echo "  python scripts/embeddings/compare_glm2_esmc.py --help"
    echo ""
else
    echo ""
    echo "========================================"
    echo "✗ Some tests failed"
    echo "========================================"
    echo ""
    echo "Please check the error messages above."
    echo "The environment may still be usable, but some features may not work."
fi

echo ""
echo "End time: $(date)"
echo "========================================"
