#!/bin/bash

# Script to build Cython extensions with better error handling
# This is mainly for evaluation toolkit, not critical for training

set -e

echo "Building Cython extensions..."

# Build toolkit extensions (optional for training)
if [ -f "setup.py" ]; then
    echo "Attempting to build toolkit extensions..."
    if python setup.py build_ext --inplace 2>&1 | tee /tmp/build_log.txt; then
        echo "✓ Toolkit extensions built successfully"
    else
        echo "⚠ Warning: Toolkit extensions build failed"
        echo "  This is usually not critical for training"
        echo "  The region.pyx extension is mainly used for evaluation metrics"
        echo "  Training can proceed without it"
        
        # Check if it's a .pxd issue
        if grep -q "c_region.pxd" /tmp/build_log.txt; then
            echo ""
            echo "Note: The c_region.pxd issue is a known problem with some Cython versions"
            echo "  You can skip this step and proceed with training"
            echo "  The training code does not require this extension"
        fi
    fi
fi

# Build DCN extensions (required for training if using deformable conv)
if [ -d "pysot/models/head/dcn" ]; then
    echo ""
    echo "Building DCN (Deformable Convolution) extensions..."
    cd pysot/models/head/dcn/
    if [ -f "setup.py" ]; then
        if python setup.py build_ext --inplace; then
            echo "✓ DCN extensions built successfully"
        else
            echo "✗ Error: DCN extensions build failed"
            echo "  This IS required for training with deformable convolution"
            exit 1
        fi
    else
        echo "Warning: DCN setup.py not found"
    fi
    cd ../../../../..
fi

echo ""
echo "Extension build process completed"

