#!/bin/bash

# Script to build DCN extensions
# This is required before training

set -e

echo "=========================================="
echo "Building DCN Extensions"
echo "=========================================="

DCN_DIR="pysot/models/head/dcn"

if [ ! -d "$DCN_DIR" ]; then
    echo "Error: DCN directory not found: $DCN_DIR"
    exit 1
fi

cd "$DCN_DIR"

if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found in $DCN_DIR"
    exit 1
fi

echo "Building DCN extensions (this may take a few minutes)..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✓ DCN extensions built successfully"
    echo "  You can now run training"
else
    echo "✗ Error: DCN extensions build failed"
    echo "  Please check the error messages above"
    exit 1
fi

cd - > /dev/null

echo "=========================================="
echo "Build completed!"
echo "=========================================="

