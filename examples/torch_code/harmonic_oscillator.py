import torch
import torch.nn as nn
import torch.optim as optim
import re

from runtime import physika_print
from runtime import solve
from runtime import compute_grad
from runtime import animate

# === Functions ===
def U(k, m, t, x0, v0):
    k = torch.as_tensor(k).float()
    m = torch.as_tensor(m).float()
    t = torch.as_tensor(t).float()
    x0 = torch.as_tensor(x0).float()
    v0 = torch.as_tensor(v0).float()
    omega = ((k / m) ** 0.5)
    eq1 = 'x0 = a + b'
    eq2 = 'v0 = i * omega * a - i * omega * b'
    a, b = solve(eq1, eq2, k=k, m=m, t=t, x0=x0, v0=v0, omega=omega)
    return ((a * torch.exp(((torch.tensor(1j) * t) * omega))) + (b * torch.exp((0.0 - ((torch.tensor(1j) * t) * omega)))))

# === Program ===
k = 1.0
m = 1.0
x0 = 1.0
v0 = 0.0
physika_print(U(k, m, 0.0, x0, v0))
physika_print(U(k, m, 1.5708, x0, v0))
physika_print(U(k, m, 3.1416, x0, v0))
physika_print(U(k, m, 4.7124, x0, v0))
physika_print(U(k, m, 6.2832, x0, v0))
physika_print(U(k, m, 0.0, 0.0, 1.0))
physika_print(U(k, m, 1.5708, 0.0, 1.0))
physika_print(U(k, m, 3.1416, 0.0, 1.0))
animate(U, k, m, x0, v0, 0.0, 31.1416)
t0 = torch.tensor(0.0, requires_grad=True)
physika_print(compute_grad(torch.real(U(k, m, t0, x0, v0)), t0))
t1 = torch.tensor(1.5708, requires_grad=True)
physika_print(compute_grad(torch.real(U(k, m, t1, x0, v0)), t1))
t2 = torch.tensor(3.1416, requires_grad=True)
physika_print(compute_grad(torch.real(U(k, m, t2, x0, v0)), t2))