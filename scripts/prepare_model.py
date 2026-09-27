"""Run locally once; commit the resulting small NPZ model for deployment."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import sys
from pathlib import Path
import json
import importlib.metadata
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hog_detector import train_model

if __name__ == '__main__':
    coef, intercept, metadata = train_model()
    metadata['versions'] = {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'scikit-image', 'opencv-python-headless')}
    target = ROOT / 'models' / 'hog_svm.npz'
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target, coef=coef, intercept=intercept, metadata=json.dumps(metadata, ensure_ascii=False))
    print('Saved', target.name, target.stat().st_size, 'bytes')
