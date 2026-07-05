import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def neighbor_sum(eta):
    s = (((torch.roll(eta, 1, 0) + torch.roll(eta, (-1), 0)) + torch.roll(eta, 1, 1)) + torch.roll(eta, (-1), 1))
    return s

def semi_implicit_step(eta, eps):
    coeff = ((dt * (eps ** 2)) / (dx ** 2))
    diag = ((1.0 + (dt * stab)) + (4.0 * coeff))
    rhs = (((1.0 + (dt * stab)) * eta) - (dt * ((eta ** 3) - eta)))
    next_eta = eta
    for k in range(int(0), int(jacobi_iters)):
        next_eta = ((rhs + (coeff * neighbor_sum(next_eta))) / diag)
    return next_eta

def solver(eps):
    eta = ic
    for s in range(int(0), int(num_steps)):
        eta = semi_implicit_step(eta, eps)
    return eta

def calculate_loss(eps):
    pred = solver(eps)
    loss = torch.mean(((pred - true_values) ** 2) if isinstance(((pred - true_values) ** 2), torch.Tensor) else torch.tensor(float(((pred - true_values) ** 2))))
    return loss

# === Program ===
Nx = 64
dx = (1.0 / Nx)
dt = 0.02
stab = 2.0
jacobi_iters = 12
num_steps = 80
ic = torch.stack([torch.distributions.Uniform((-0.1), 0.1).rsample((int(Nx),)) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
true_eps = 0.03
true_values = solver(true_eps)
eps = torch.tensor(0.06, requires_grad=True)
learning_rate = 0.5
epochs = 1
for epoch in range(int(0), int(epochs)):
    g = compute_grad(calculate_loss, eps)
    eps = (eps - (learning_rate * g))
    physika_print(eps)