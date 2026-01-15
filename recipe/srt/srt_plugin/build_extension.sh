#!/bin/bash
# Build the SRT plugin C++ extension
#
# PREREQUISITES:
#   - cmake >= 3.18
#   - nanobind (pip install nanobind)
#   - OpenMP (libomp-dev on Ubuntu, or brew install libomp on macOS)
#   - ninja (optional, for faster builds)
#
# USAGE:
#   cd recipe/srt/srt_plugin
#   ./build_extension.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building SRT suffix_cache C++ extension..."
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check dependencies
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Install with: apt install cmake"
    exit 1
fi

python -c "import nanobind" 2>/dev/null || {
    echo "Installing nanobind..."
    pip install nanobind
}

# Build the extension
# Note: --inplace doesn't work correctly with nested packages, so we build and copy manually
python setup.py build_ext

# Find and copy the built extension
SO_FILE=$(find build -name "_C.cpython-*.so" -type f 2>/dev/null | head -1)
if [ -n "$SO_FILE" ]; then
    echo "Copying $SO_FILE to suffix_cache/"
    cp "$SO_FILE" suffix_cache/
fi

# Verify it was built
if ls suffix_cache/_C.*.so 1> /dev/null 2>&1; then
    echo "SUCCESS: C++ extension built successfully!"
    ls -la suffix_cache/_C.*.so
else
    echo "ERROR: C++ extension not found after build"
    exit 1
fi
