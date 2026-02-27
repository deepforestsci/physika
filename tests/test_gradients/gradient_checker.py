
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

        grad[i] = (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h)

    Parameters
    ----------
    f : Callable[[np.ndarray], float]
        Function ``f(x) -> float`` where *x* is a 1-D numpy array.
        Both scalar and vector inputs are supported; scalar inputs should
        be wrapped in a length-1 array (e.g. ``np.array([3.0])``).
    x : numpy.ndarray
        1-D array of evaluation point(s).  A copy is made internally so
        the original array is never modified.
    h : float, default 1e-5
        Step size for the central difference.

    Returns
    -------
    numpy.ndarray
        Gradient array with the same shape as *x*.

    Examples
    --------
    >>> from tests.test_gradients.gradient_checker import numerical_gradient
    >>> import numpy as np
    >>> import math
    >>> f = lambda x: float(x[0] ** 2)
    >>> grad = numerical_gradient(f, np.array([3.0]))
    >>> abs(grad[0] - 6.0) < 1e-4
    True
    >>> # Vector case: f(x) = sum(sin(x))  -> grad = cos(x)
    >>> import math
    >>> f_vec = lambda x: float(sum(math.sin(xi) for xi in x))
    >>> x0 = np.array([0.5, 1.0, 1.5])
    >>> grad_vec = numerical_gradient(f_vec, x0)
    >>> all(abs(grad_vec[i] - math.cos(x0[i])) < 1e-4 for i in range(3))
    True
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)
    return grad
