import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def linspace(start, end, n):
    x = zero_1d_array(n)
    dx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * dx))
    return x

def psi(x):
    return torch.exp((-(x ** 2)) if isinstance((-(x ** 2)), torch.Tensor) else torch.tensor(float((-(x ** 2)))))

def norm_psi(x):
    return (norm * psi(x))

def rho(x):
    return (torch.abs(norm_psi(x) if isinstance(norm_psi(x), torch.Tensor) else torch.tensor(float(norm_psi(x)))) ** 2)

def mom(x):
    value = compute_grad(lambda _dx: psi(_dx), x)
    return (((-1j) * h) * value)

def dpsi_dx(x):
    return compute_grad(lambda _dx: norm_psi(_dx), x)

def d2psi_dx2(x):
    return compute_grad(lambda _dx: dpsi_dx(_dx), x)

def V(x):
    return 0.0

def hamiltonian(x):
    kinetic = (((-(h ** 2)) / (2 * m)) * d2psi_dx2(x))
    potential = (V(x) * norm_psi(x))
    return (kinetic + potential)

# === Program ===
x = linspace((-5), 5, 100)
dx = (x[int(1)] - x[int(0)])
total = 0.0
for i in range(int(0), int(100)):
    total = total + (torch.abs(psi(x[int(i)]) if isinstance(psi(x[int(i)]), torch.Tensor) else torch.tensor(float(psi(x[int(i)])))) ** 2)
norm = ((1 / (total * dx)) ** 0.5)
total = 0.0
for i in range(int(0), int(100)):
    total = total + (torch.abs(norm_psi(x[int(i)]) if isinstance(norm_psi(x[int(i)]), torch.Tensor) else torch.tensor(float(norm_psi(x[int(i)])))) ** 2)
print((total * dx))
print(rho(0.01))
h = 6.582e-16
print((mom(0.001) * 10000000000.0))
m = 5.6856e-32
print(hamiltonian(0.001))
energy_numerator = 0.0
energy_denominator = 0.0
for i in range(int(0), int(100)):
    energy_numerator = energy_numerator + (norm_psi(x[int(i)]) * hamiltonian(x[int(i)]))
    energy_denominator = energy_denominator + (norm_psi(x[int(i)]) ** 2)
E = (energy_numerator / energy_denominator)
print(E)