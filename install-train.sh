#!/bin/bash

# Installation script for Cross-View Siamese Training
# This script installs only the dependencies needed for training

set -e

echo "=========================================="
echo "Cross-View Siamese Training Setup"
echo "=========================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed"
    exit 1
fi

# Check CUDA availability (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA available:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "Warning: nvidia-smi not found. Training will use CPU (very slow)"
fi

echo ""
echo "=========================================="
echo "Step 1: Installing Python dependencies"
echo "=========================================="

# Install basic requirements
pip install --upgrade pip

# Install requirements from requirements.txt if exists
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, installing manually..."
    pip install opencv-python yacs tqdm pyyaml matplotlib colorama cython tensorboardX
fi

# Install PyTorch (check if already installed)
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch is already installed:"
    python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
else
    echo "PyTorch not found. Please install PyTorch manually:"
    echo "  For CUDA 10.1: conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch"
    echo "  For CUDA 11.0: conda install pytorch torchvision cudatoolkit=11.0 -c pytorch"
    echo "  For CPU only: conda install pytorch torchvision cpuonly -c pytorch"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Step 2: Building Cython extensions"
echo "=========================================="

# Build toolkit extensions
if [ -f "setup.py" ]; then
    echo "Building toolkit extensions..."
    python setup.py build_ext --inplace
    echo "✓ Toolkit extensions built"
else
    echo "Warning: setup.py not found, skipping toolkit extensions"
fi

# Build DCN extensions
if [ -d "pysot/models/head/dcn" ]; then
    echo "Building DCN (Deformable Convolution) extensions..."
    cd pysot/models/head/dcn/
    if [ -f "setup.py" ]; then
        python setup.py build_ext --inplace
        echo "✓ DCN extensions built"
    else
        echo "Warning: DCN setup.py not found"
    fi
    cd ../../../../..
else
    echo "Warning: DCN directory not found, skipping DCN extensions"
fi

echo ""
echo "=========================================="
echo "Step 3: Verifying installation"
echo "=========================================="

# Verify imports
echo "Verifying Python imports..."
python << EOF
import sys
errors = []

try:
    import torch
    print("✓ PyTorch")
except ImportError as e:
    errors.append("PyTorch")
    print("✗ PyTorch: {}".format(e))

try:
    import cv2
    print("✓ OpenCV")
except ImportError as e:
    errors.append("OpenCV")
    print("✗ OpenCV: {}".format(e))

try:
    import yacs
    print("✓ yacs")
except ImportError as e:
    errors.append("yacs")
    print("✗ yacs: {}".format(e))

try:
    import tensorboardX
    print("✓ tensorboardX")
except ImportError as e:
    errors.append("tensorboardX")
    print("✗ tensorboardX: {}".format(e))

try:
    from pysot.core.config import cfg
    print("✓ pysot.core.config")
except ImportError as e:
    errors.append("pysot.core.config")
    print("✗ pysot.core.config: {}".format(e))

if errors:
    print("\n✗ Some packages failed to import: {}".format(", ".join(errors)))
    sys.exit(1)
else:
    print("\n✓ All required packages imported successfully")
EOF

if [ $? -ne 0 ]; then
    echo "Error: Some packages failed to import"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 4: Checking dataset"
echo "=========================================="

# Check if dataset exists
if [ -d "training_dataset/observing/train" ]; then
    echo "✓ Training dataset found: training_dataset/observing/train"
    
    # Count samples
    SAMPLE_COUNT=$(find training_dataset/observing/train/samples -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Number of video samples: $SAMPLE_COUNT"
    
    # Check annotations
    if [ -f "training_dataset/observing/train/annotations/annotations.json" ]; then
        echo "✓ Annotations file found"
    else
        echo "✗ Annotations file not found: training_dataset/observing/train/annotations/annotations.json"
    fi
else
    echo "✗ Training dataset not found: training_dataset/observing/train"
    echo "  Please ensure the dataset is in the correct location"
fi

echo ""
echo "=========================================="
echo "Step 5: Checking pretrained models"
echo "=========================================="

# Check for pretrained backbone
if [ -f "pretrained_models/resnet50.model" ]; then
    echo "✓ Pretrained ResNet-50 found"
elif [ -d "pretrained_models" ]; then
    echo "⚠ Pretrained ResNet-50 not found in pretrained_models/"
    echo "  You may need to download it separately"
else
    echo "⚠ pretrained_models/ directory not found"
    echo "  Creating directory..."
    mkdir -p pretrained_models
    echo "  Please download pretrained ResNet-50 and place it in pretrained_models/resnet50.model"
fi

echo ""
echo "=========================================="
echo "Step 6: Creating necessary directories"
echo "=========================================="

# Create log and snapshot directories
mkdir -p logs/cross_view
mkdir -p snapshot/cross_view
echo "✓ Created logs/cross_view/"
echo "✓ Created snapshot/cross_view/"

echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo "✓ Python dependencies installed"
echo "✓ Cython extensions built"
echo "✓ Directories created"
echo ""
echo "Next steps:"
echo "1. Download pretrained ResNet-50 to pretrained_models/resnet50.model (if not done)"
echo "2. Verify dataset is in training_dataset/observing/train/"
echo "3. Test dataset: python tools/test_cross_view_dataset.py"
echo "4. Start training: ./run_cross_view_training.sh"
echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="

