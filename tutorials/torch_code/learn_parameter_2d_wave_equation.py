import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def get_1d_array_length(x, m=None):
    if m is None:
        m = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def linspace(start, end, n):
    x = zero_1d_array(n)
    Δx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * Δx))
    return x

def wave_equation(u, Δx, Δy, c):
    lap = zero_2d_array(nx, ny)
    lap[int(1):int((nx - 1)), int(1):int((ny - 1))] = ((c ** 2) * ((((u[int(0):int((nx - 2)), int(1):int((ny - 1))] - (2 * u[int(1):int((nx - 1)), int(1):int((ny - 1))])) + u[int(2):int(nx), int(1):int((ny - 1))]) / (Δx ** 2)) + (((u[int(1):int((nx - 1)), int(0):int((ny - 2))] - (2 * u[int(1):int((nx - 1)), int(1):int((ny - 1))])) + u[int(1):int((nx - 1)), int(2):int(ny)]) / (Δy ** 2))))
    return lap

def solver(c, u0, v0, Δx, Δy, Δt, nt):
    u_prev = u0
    u_curr = u0
    for step in range(int(0), int(nt)):
        accel = wave_equation(u_curr, Δx, Δy, c)
        u_next = (((2 * u_curr) - u_prev) + ((Δt ** 2) * accel))
        u_next[:, int(0)] = 0
        u_next[:, int((ny - 1))] = 0
        u_next[int(0), :] = 0
        u_next[int((nx - 1)), :] = 0
        u_prev = u_curr
        u_curr = u_next
    return u_curr

def calculate_loss(c):
    predictions = solver(c, u0, v0, Δx, Δy, Δt, nt)
    diff = (predictions - true_solution)
    loss = torch.mean((diff ** 2) if isinstance((diff ** 2), torch.Tensor) else torch.tensor(float((diff ** 2))))
    return loss

def adam(c, g, m, v, t, lr):
    β1 = 0.9
    β2 = 0.999
    ε = 1e-08
    m_new = ((β1 * m) + ((1.0 - β1) * g))
    v_new = ((β2 * v) + ((1.0 - β2) * (g ** 2)))
    m_hat = (m_new / (1.0 - (β1 ** t)))
    v_hat = (v_new / (1.0 - (β2 ** t)))
    c_new = (c - ((lr * m_hat) / (torch.sqrt(torch.as_tensor(v_hat).float()) + ε)))
    return torch.stack([torch.as_tensor(c_new), torch.as_tensor(m_new), torch.as_tensor(v_new), torch.as_tensor((t + 1.0))])

# === Program ===
Lx, Ly, nx, ny, tf = 1.0, 1.0, 40, 40, 2.0
Δx = (Lx / (nx - 1))
Δy = (Ly / (ny - 1))
true_c = 1.0
cfl = 0.4
Δt = (cfl / (5.0 * torch.sqrt((((1 / Δx) ** 2) + ((1 / Δy) ** 2)) if isinstance((((1 / Δx) ** 2) + ((1 / Δy) ** 2)), torch.Tensor) else torch.tensor(float((((1 / Δx) ** 2) + ((1 / Δy) ** 2)))))))
nt = 50
x = linspace(0, Lx, nx)
y = linspace(0, Ly, ny)
π = 3.14
u0 = zero_2d_array(nx, ny)
for i in range(int(0), int(nx)):
    for j in range(int(0), int(ny)):
        u0[int(i), int(j)] = (torch.sin(((2 * π) * x[int(i)]) if isinstance(((2 * π) * x[int(i)]), torch.Tensor) else torch.tensor(float(((2 * π) * x[int(i)])))) * torch.sin((π * y[int(j)]) if isinstance((π * y[int(j)]), torch.Tensor) else torch.tensor(float((π * y[int(j)])))))
v0 = zero_2d_array(nx, ny)
true_solution = solver(true_c, u0, v0, Δx, Δy, Δt, nt)
c = torch.tensor(3.0, requires_grad=True)
m_adam, v_adam, t_adam, lr = 0.0, 0.0, 1.0, 0.01
epochs = 1
for i in range(int(0), int(epochs)):
    print(i)
    g = compute_grad(calculate_loss, c)
    result = adam(c, g, m_adam, v_adam, t_adam, lr)
    c = result[int(0)]
    m_adam = result[int(1)]
    v_adam = result[int(2)]
    t_adam = result[int(3)]
    print(c)
pred_solution = solver(c, u0, v0, Δx, Δy, Δt, nt)