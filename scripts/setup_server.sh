#!/bin/bash
# Server Setup Script for Cancer Research
# Run this on a fresh vast.ai server to restore environment
# Usage: bash setup_server.sh

set -e

echo "=== Cancer Research Server Setup ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"

# Create workspace
mkdir -p /workspace/cancer_research/{data,models,results,checkpoints,logs,scripts}
mkdir -p /workspace/cancer_research/data/{tcga,drugbank,depmap,drugcomb}

# Install PyTorch with CUDA
echo "=== Installing PyTorch ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install core ML packages
echo "=== Installing ML packages ==="
pip install pandas scikit-learn scipy matplotlib seaborn
pip install lifelines umap-learn statsmodels adjustText
pip install tqdm requests jupyter plotly

# Install bioinformatics packages
echo "=== Installing Bio packages ==="
pip install gseapy mygene pybiomart
pip install scanpy anndata
pip install pydeseq2

# Install graph ML
echo "=== Installing Graph ML ==="
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# Install transformers for foundation models
echo "=== Installing Transformers ==="
pip install transformers accelerate peft datasets

# Install chemistry tools
echo "=== Installing Chemistry ==="
pip install rdkit-pypi

echo "=== Setup Complete ==="
echo "Verify GPU:"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
