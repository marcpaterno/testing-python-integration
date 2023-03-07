"""
Demonstration of numerical integration routines using Genz's f3 test function.
"""
import numpy as np
from integrator import do_tplquad_integration, do_vegas_integration

INDICES = np.arange(1, 4)
ANS = -0.53117994723428651


def f1(x: np.ndarray) -> np.float64:
    """Genz's test function f1, here in 3d."""
    return np.cos(np.sum(INDICES * x))


def f1s(v, w, x):
    """Genz's test function f1, here in 3d."""
    return np.cos(v + 2 * w + 3 * x)


if __name__ == "__main__":
    tplvolume = [[0, 1]] * len(INDICES)
    tplvolume = [item for sublist in tplvolume for item in sublist]
    print("*" * 10, "Starting TPLQUAD")
    do_tplquad_integration(f1s, tplvolume, 1e-4, ANS)

    volume = [[0.0, 1.0]] * len(INDICES)
    print("*" * 10, "Starting VEGAS")
    do_vegas_integration(f1, volume, 1e-4, ANS)
