"""
Demonstration of numerical integration routines using Genz's f1 test function.
"""
import numpy as np
from integrator import do_tplquad_integration, do_vegas_integration


INDICES = np.arange(1, 4)
ANS = 0.028039353051767744


def f3(x: np.ndarray) -> np.float64:
    """Genz's test function f3, here in 3d."""
    return 1.0 / ((1 + np.sum(INDICES * x)) ** 3)


def f3s(x, y, z):
    """Genz's test function f3, here in 3d."""
    return 1.0 / ((1 + x + 2 * y + 3 * z) ** 3)


if __name__ == "__main__":
    volume = [[0.0, 1.0]] * len(INDICES)
    print("*" * 10, "Starting VEGAS")
    do_vegas_integration(f3, volume, 1.0e-3, ANS)

    tplvolume = [[0, 1]] * 3
    tplvolume = [item for sublist in tplvolume for item in sublist]
    print("*" * 10, "Starting TPLQUAD")
    do_tplquad_integration(f3s, tplvolume, 1e-3, ANS)
