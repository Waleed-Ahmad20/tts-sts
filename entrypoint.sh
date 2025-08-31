#!/bin/bash
set -e

echo "=== XTTS Audiobook Pipeline ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# Check GPU availability
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits || true
fi

echo "================================"

# Run the application
exec python app.py "$@"