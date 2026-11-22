#!/usr/bin/env python3
"""compare.py A.txt B.txt  -> max |centroid difference| between two result files (exit 1 if > tol)."""
import sys
import numpy as np
a = np.loadtxt(sys.argv[1]); b = np.loadtxt(sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
d = float(np.abs(a - b).max())
print(f"max centroid difference = {d:.3e}  ->", "OK" if d <= tol else "MISMATCH")
sys.exit(0 if d <= tol else 1)
