import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i))]])
    return results

def linspace(start, end, n):
    x = zero_1d_array(n)
    dx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * dx))
    return x

def heat_equation(T, dx, α):
    nx = get_1d_array_length(T)
    f = zero_1d_array(nx)
    for i in range(int(1), int((nx - 1))):
        f[int(i)] = ((α / (dx ** 2)) * ((T[int((i - 1))] - (2 * T[int(i)])) + T[int((i + 1))]))
    return f

def solver(α, T0, dx, dt, nt):
    T = T0
    last_index = get_1d_array_length(T)
    for i in range(int(0), int(nt)):
        T = (T + (dt * heat_equation(T, dx, α)))
    T[int(0)] = 0
    T[int((last_index - 1))] = 0
    return T

def calculate_loss(α):
    predictions = solver(α, T0, dx, dt, nt)
    loss = 0.0
    for i in range(int(0), int(nx)):
        diff = (predictions[int(i)] - true_values[int(i)])
        loss = loss + (diff ** 2)
    return (loss / nx)

# === Program ===
lx = 1.0
nx = 21
dx = (lx / (nx - 1))
x = linspace(0, lx, nx)
true_alpha = 0.4
fourier = 0.49
dt = (((fourier * dx) ** 2) / true_alpha)
nt = 100
T0 = zero_1d_array(nx)
for i in range(int(0), int(nx)):
    T0[int(i)] = torch.exp(((-50) * ((x[int(i)] - 0.5) ** 2)) if isinstance(((-50) * ((x[int(i)] - 0.5) ** 2)), torch.Tensor) else torch.tensor(float(((-50) * ((x[int(i)] - 0.5) ** 2)))))
true_values = solver(true_alpha, T0, dx, dt, nt)
α = torch.tensor(0.1, requires_grad=True)
learning_rate = 0.1
epochs = 1
for i in range(int(0), int(epochs)):
    g = compute_grad(calculate_loss, α)
    α = (α - (learning_rate * g))
pred_values = solver(α, T0, dx, dt, nt)