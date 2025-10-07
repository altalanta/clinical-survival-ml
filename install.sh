#!/bin/bash
# Installation script for clinical-survival-ml
# This script tries multiple installation methods to handle pip compatibility issues

set -e

echo "Installing clinical-survival-ml..."

# Method 1: Try pip install with requirements.txt (most compatible)
echo "Trying installation with requirements.txt..."
if command -v pip &> /dev/null; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Successfully installed with requirements.txt"
        echo "Installing package in development mode..."
        pip install -e .
        echo "✅ Installation complete!"
        exit 0
    fi
fi

# Method 2: Try conda/mamba if available
echo "Trying installation with conda/mamba..."
if command -v mamba &> /dev/null; then
    echo "Using mamba..."
    mamba install -c conda-forge --file requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Successfully installed with mamba"
        echo "Installing package in development mode..."
        pip install -e .
        echo "✅ Installation complete!"
        exit 0
    fi
elif command -v conda &> /dev/null; then
    echo "Using conda..."
    conda install -c conda-forge --file requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Successfully installed with conda"
        echo "Installing package in development mode..."
        pip install -e .
        echo "✅ Installation complete!"
        exit 0
    fi
fi

# Method 3: Try direct pip install of package
echo "Trying direct pip install..."
pip install -e . || {
    echo "❌ Direct pip install failed"
    echo "Trying installation without development dependencies..."
    pip install .
    echo "✅ Basic installation complete (without dev dependencies)"
    echo "Install development dependencies manually if needed:"
    echo "pip install pytest ruff black"
    exit 0
}

echo "✅ Installation complete!"

