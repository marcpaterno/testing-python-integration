"""
Integration timing and testing routines.

This module uses the Python version of the MC algorithm VEGAS as well as the
scipy.integrate routine tplquad.
"""
import time
import vegas
from scipy.integrate import tplquad


class Integrator:
    """Integrator wraps the vegas.Integrator class and provides a primitive
    method to continue refining the VEGAS calculation until the desired
    accuracy has been reached.

    Much more refinement of this primitive algorithm would be needed for any
    serious work."""

    def __init__(self, volume):
        """Create an integrator that will integration functions over the
        given volume of integration. For an integral in N dimensions, volume
        should be list of length 2*N providing the lower and upper bounds for
        each variable of integration, in the order that the variables appear
        in the integrand.
        """
        self.alg = vegas.Integrator(volume)
        self.n_refinements = 0
        self.current_neval = 10 * 1000

    def __call__(self, func, epsrel=1e-6, epsabs=1e-12):
        """Integrate the given function "func" over the current volume to the
        desired relative and absolute tolerances.

        The function "func" should take a single np.ndarray."""
        result = self.alg(func, nitn=10, neval=self.current_neval)
        self.n_refinements += 1
        while result.sdev > abs(epsrel * result.mean):
            self.current_neval *= 3
            self.n_refinements += 1
            result = self.alg(func, nitn=10, neval=self.current_neval)
        return result


def do_vegas_integration(func, volume, epsrel, ans):
    """Time and analyze the results of the VEGAS integrator."""
    start = time.time()
    integ = Integrator(volume)
    result = integ(func, epsrel=epsrel)
    stop = time.time()
    print(f"Integration with VEGAS took {stop-start} seconds")
    print(f"Result: {result.mean} +/- {result.sdev}")
    true_error = abs(result.mean - ans)
    print(f"True error is: {true_error}")
    print(f"Ratio (true error)/(estimated error) = {true_error/result.sdev}")
    print(f"True fractional error = {true_error/ans}")


def do_tplquad_integration(func, volume, epsrel, ans):
    """Time and analyze the results of the TPLQUAD integrator."""
    start = time.time()
    res, err = tplquad(func, *volume, epsrel=epsrel)
    stop = time.time()
    print(f"Integration with TPLQUAD took {stop-start} seconds")
    print(f"Result: {res} +/- {err}")
    true_error = abs(res - ans)
    print(f"True error is: {true_error}")
    print(f"Ratio (true error)/(estimated error) = {true_error/err}")
    print(f"True fractional error = {true_error/ans}")
