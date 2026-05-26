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

def wave_equation(u, dx, c):
    nx = get_1d_array_length(u)
    f = zero_1d_array(nx)
    for i in range(int(1), int((nx - 1))):
        f[int(i)] = (((c ** 2) / (dx ** 2)) * ((u[int((i - 1))] - (2 * u[int(i)])) + u[int((i + 1))]))
    return f

def solver(c, u0, dx, dt, nt):
    u_prev = u0
    u_curr = u0
    nx = get_1d_array_length(u0)
    for n in range(int(0), int(nt)):
        accel = wave_equation(u_curr, dx, c)
        u_next = (((2 * u_curr) - u_prev) + ((dt ** 2) * accel))
        u_next[int(0)] = 0
        u_next[int((nx - 1))] = 0
        u_prev = u_curr
        u_curr = u_next
    return u_curr

def calculate_loss(c):
    predictions = solver(c, u0, dx, dt, nt)
    loss = 0.0
    for i in range(int(0), int(nx)):
        diff = (predictions[int(i)] - true_values[int(i)])
        loss = loss + (diff ** 2)
    return (loss / nx)

# === Program ===
nx = 30
nt = 30
a = 0
b = 1
t0 = 0
tf = 1
dx = ((b - a) / (nx - 1))
dt = ((tf - t0) / (nt - 1))
pi = 3.14
true_c = 0.5
x = linspace(a, b, nx)
u0 = zero_1d_array(nx)
for i in range(int(0), int(nx)):
    u0[int(i)] = torch.sin(((2 * pi) * x[int(i)]) if isinstance(((2 * pi) * x[int(i)]), torch.Tensor) else torch.tensor(float(((2 * pi) * x[int(i)]))))
true_values = solver(true_c, u0, dx, dt, nt)
c = torch.tensor(0.1, requires_grad=True)
learning_rate = 0.01
epochs = 1
for i in range(int(0), int(epochs)):
    physika_print(i)
    g = compute_grad(calculate_loss, c)
    c = (c - (learning_rate * g))
pred_values = solver(c, u0, dx, dt, nt)