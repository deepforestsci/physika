import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def single_arg_jacobian(x, m=None):
    if m is None:
        m = int(x.shape[0])
    a = x[int(0)]
    b = x[int(1)]
    return torch.stack([torch.as_tensor((a * b)), torch.as_tensor((a + b))])

def double_arg_jacobians(state, theta):
    x = state[int(0)]
    y = state[int((1 + 0))]
    α = theta[int(0)]
    β = theta[int((1 + 0))]
    γ = theta[int((1 + (1 + 0)))]
    δ = theta[int((1 + (1 + (1 + 0))))]
    dx = ((α * x) - ((β * x) * y))
    dy = (((δ * x) * y) - (γ * y))
    return torch.stack([torch.as_tensor(dx), torch.as_tensor(dy)])

def three_arg_jacobians(a, b, c, m=None, n=None, o=None):
    if m is None:
        m = int(a.shape[0])
    if n is None:
        n = int(b.shape[0])
    if o is None:
        o = int(c.shape[0])
    x1 = a[int(0)]
    x2 = b[int(0)]
    x3 = c[int(0)]
    y1 = ((x1 * x2) + x3)
    y2 = (x1 + (x2 * x3))
    return torch.stack([torch.as_tensor(y1), torch.as_tensor(y2)])

# === Program ===
x = torch.as_tensor(torch.stack([torch.as_tensor(2.0), torch.as_tensor(3.0)])).requires_grad_(True).to(DEVICE)
J = compute_grad(lambda _dx: single_arg_jacobian(_dx, 2), x)
print(J)
state = torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0)])).requires_grad_(True).to(DEVICE)
θ = torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(0.3), torch.as_tensor(1.3), torch.as_tensor(0.5)])).requires_grad_(True).to(DEVICE)
J_state = compute_grad(lambda _dstate: double_arg_jacobians(_dstate, θ), state)
J_theta = compute_grad(lambda _dθ: double_arg_jacobians(state, _dθ), θ)
print(J_state)
print(J_theta)
a = torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0)])).requires_grad_(True).to(DEVICE)
b = torch.as_tensor(torch.stack([torch.as_tensor(3.0), torch.as_tensor(4.0)])).requires_grad_(True).to(DEVICE)
c = torch.as_tensor(torch.stack([torch.as_tensor(5.0), torch.as_tensor(6.0)])).requires_grad_(True).to(DEVICE)
J_a = compute_grad(lambda _da: three_arg_jacobians(_da, b, c, 2, 2, 2), a)
J_b = compute_grad(lambda _db: three_arg_jacobians(a, _db, c, 2, 2, 2), b)
J_c = compute_grad(lambda _dc: three_arg_jacobians(a, b, _dc, 2, 2, 2), c)
print(J_a)
print(J_b)
print(J_c)