import numpy as np
from typing import Callable


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Numerical gradient checker for testing Physika autograd correctness.

    Computes a central-difference numerical gradient that can be compared
    against Physika's tape-based ``compute_grad`` to validate that autograd
    produces the correct derivatives.

    For each component *i* the partial derivative is approximated by:

        grad[i] = (f(x + h·e_i) - f(x - h·e_i)) / (2h)

    Parameters
    ----------
    f : Callable[[np.ndarray], float]
        Function ``f(x)`` where `x` is a 1-D numpy array.
        Both scalar and vector inputs are supported. Scalar inputs should
        be wrapped in a 1D array.
    x : numpy.ndarray
        1-D array of evaluation point(s). A copy is made internally so
        the original array is not modified.
    h : float, default 1e-5
        Step size for the central difference.

    Returns
    -------
    numpy.ndarray
        Gradient array with the same shape as `x`.

    Examples
    --------
    >>> from tests.test_gradients.gradient_checker import numerical_gradient
    >>> import numpy as np
    >>> import math
    >>> #Scalar case: f(x) = x^2  -> grad = 2x
    >>> f = lambda x: float(x[0] ** 2)
    >>> grad = numerical_gradient(f, np.array([3.0]))
    >>> assert abs(grad[0] - 6.0) < 1e-5
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)
    return grad