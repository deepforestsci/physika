from typing import Callable
import torch


def numerical_gradient(
    f: Callable[[torch.Tensor], float],
    x: torch.Tensor,
    h: float = 1e-5,
) -> torch.Tensor:
    """Numerical gradient checker for testing Physika autograd correctness.

    Computes a central-difference numerical gradient that can be compared
    against Physika's tape-based ``compute_grad`` to validate that autograd
    produces the correct derivatives.

    For each component *i* the partial derivative is approximated by:

        grad[i] = (f(x + h·e_i) - f(x - h·e_i)) / (2h)

    Parameters
    ----------
    f : Callable[[torch.Tensor], float]
        Function ``f(x)`` where `x` is a 1-D torch tensor.
        Both scalar and vector inputs are supported. Scalar inputs should
        be wrapped in a 1D tensor.
    x : torch.Tensor
        1-D tensor of evaluation point(s). A copy is made internally so
        the original array is not modified.
    h : float, default 1e-5
        Step size for the central difference.

    Returns
    -------
    torch.Tensor
        Gradient tensor with the same shape as `x`.

    Examples
    --------
    >>> from tests.test_gradients.gradient_checker import numerical_gradient
    >>> import torch
    >>> import math
    >>> #Scalar case: f(x) = x^2  -> grad = 2x
    >>> f = lambda x: float(x[0] ** 2)
    >>> grad = numerical_gradient(f, torch.tensor([3.0]))
    >>> assert abs(grad[0] - 6) < 1e-2
    """
    if x.dim() == 0:
        x = x.unsqueeze(0)
    grad = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_forward = x.clone()
        x_backward = x.clone()
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)
    return grad
