#!/usr/bin/env python3
"""Check GPU availability for ML analysis"""

try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU name:', torch.cuda.get_device_name(0))
        print('GPU count:', torch.cuda.device_count())
except ImportError as e:
    print(f'PyTorch not installed: {e}')

try:
    import tensorflow as tf
    print('\nTensorFlow version:', tf.__version__)
    print('CUDA available:', tf.test.is_built_with_cuda())
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('GPU count:', len(gpus))
        for gpu in gpus:
            print(f'  GPU: {gpu}')
except ImportError as e:
    print(f'TensorFlow not installed: {e}')

print('\nChecking available Python ML libraries:')
import sys
for pkg in ['numpy', 'pandas', 'scipy', 'sklearn']:
    try:
        __import__(pkg)
        mod = sys.modules[pkg]
        print(f'  ✅ {pkg} v{getattr(mod, "__version__", "unknown")}')
    except ImportError:
        print(f'  ❌ {pkg} not installed')
