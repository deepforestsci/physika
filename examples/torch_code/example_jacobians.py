import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def single_arg_jacobian(x):
    a = x[int(0)]
    b = x[int(1)]
    return torch.stack([torch.as_tensor((a * b)), torch.as_tensor((a + b))])

def double_arg_jacobians(state, theta):
    x = state[int(0)]
    y = state[int(1)]
    α = theta[int(0)]
    β = theta[int(1)]
    γ = theta[int(2)]
    δ = theta[int(3)]
    dx = ((α * x) - ((β * x) * y))
    dy = (((δ * x) * y) - (γ * y))
    return torch.stack([torch.as_tensor(dx), torch.as_tensor(dy)])

def three_arg_jacobians(a, b, c):
    x1 = a[int(0)]
    x2 = b[int(0)]
    x3 = c[int(0)]
    y1 = ((x1 * x2) + x3)
    y2 = (x1 + (x2 * x3))
    return torch.stack([torch.as_tensor(y1), torch.as_tensor(y2)])

# === Program ===
x = torch.as_tensor(torch.tensor([2.0, 3.0])).requires_grad_(True)
J = compute_grad(single_arg_jacobian, x)
physika_print(J)
state = torch.as_tensor(torch.tensor([1.0, 2.0])).requires_grad_(True)
θ = torch.as_tensor(torch.tensor([1.0, 0.3, 1.3, 0.5])).requires_grad_(True)
J_state = compute_grad(lambda _dstate: double_arg_jacobians(_dstate, θ), state)
J_theta = compute_grad(lambda _dθ: double_arg_jacobians(state, _dθ), θ)
physika_print(J_state)
physika_print(J_theta)
a = torch.as_tensor(torch.tensor([1.0, 2.0])).requires_grad_(True)
b = torch.as_tensor(torch.tensor([3.0, 4.0])).requires_grad_(True)
c = torch.as_tensor(torch.tensor([5.0, 6.0])).requires_grad_(True)
J_a = compute_grad(lambda _da: three_arg_jacobians(_da, b, c), a)
J_b = compute_grad(lambda _db: three_arg_jacobians(a, _db, c), b)
J_c = compute_grad(lambda _dc: three_arg_jacobians(a, b, _dc), c)
physika_print(J_a)
physika_print(J_b)
physika_print(J_c)