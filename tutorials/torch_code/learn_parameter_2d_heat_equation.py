import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
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

def heat_equation(T, Δx, Δy, α):
    f = zero_2d_array(nx, ny)
    f[int(1):int((nx - 1)), int(1):int((ny - 1))] = (α * ((((T[int(0):int((nx - 2)), int(1):int((ny - 1))] - (2 * T[int(1):int((nx - 1)), int(1):int((ny - 1))])) + T[int(2):int(nx), int(1):int((ny - 1))]) / (Δx ** 2)) + (((T[int(1):int((nx - 1)), int(0):int((ny - 2))] - (2 * T[int(1):int((nx - 1)), int(1):int((ny - 1))])) + T[int(1):int((nx - 1)), int(2):int(ny)]) / (Δy ** 2))))
    return f

def solver(α, T0, Δx, Δy, Δt, nt):
    T = T0
    for step in range(int(0), int(nt)):
        T = (T + (Δt * heat_equation(T, Δx, Δy, α)))
        T[:, int(0)] = 0
        T[:, int((ny - 1))] = 0
        T[int(0), :] = 0
        T[int((nx - 1)), :] = 0
    return T

def calculate_loss(α):
    predictions = solver(α, T0, Δx, Δy, Δt, nt)
    diff = (predictions - true_solution)
    loss = torch.mean((diff ** 2) if isinstance((diff ** 2), torch.Tensor) else torch.tensor(float((diff ** 2))))
    return loss

def adam(α, g, m, v, t, lr):
    β1 = 0.9
    β2 = 0.999
    ε = 1e-08
    m_new = ((β1 * m) + ((1.0 - β1) * g))
    v_new = ((β2 * v) + ((1.0 - β2) * (g ** 2)))
    m_hat = (m_new / (1.0 - (β1 ** t)))
    v_hat = (v_new / (1.0 - (β2 ** t)))
    α_new = (α - ((lr * m_hat) / (torch.sqrt(torch.as_tensor(v_hat).float()) + ε)))
    return torch.stack([torch.as_tensor(α_new), torch.as_tensor(m_new), torch.as_tensor(v_new), torch.as_tensor((t + 1.0))])

# === Program ===
true_α = 2.0
Lx, Ly, nx, ny, tf = 1.0, 1.0, 40, 40, 10
Δx = (Lx / (nx - 1))
Δy = (Ly / (ny - 1))
fourier = 0.49
Δt = ((fourier / (((1 / Δx) ** 2) + ((1 / Δy) ** 2))) / 10.0)
nt = 100
T1, T2, T3, T4 = 0, 0, 0, 0
x = linspace(0, Lx, nx)
y = linspace(0, Ly, ny)
T0 = torch.stack([torch.stack([torch.exp(((-20) * (((x[int(i)] - 0.5) ** 2) + ((y[int(j)] - 0.5) ** 2))) if isinstance(((-20) * (((x[int(i)] - 0.5) ** 2) + ((y[int(j)] - 0.5) ** 2))), torch.Tensor) else torch.tensor(float(((-20) * (((x[int(i)] - 0.5) ** 2) + ((y[int(j)] - 0.5) ** 2)))))) for _fi_j in range(int(ny)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
true_solution = solver(true_α, T0, Δx, Δy, Δt, nt)
α = torch.tensor(4.0, requires_grad=True)
guess_solution = solver(α, T0, Δx, Δy, Δt, nt)
m_adam, v_adam, t_adam, lr = 0.0, 0.0, 1.0, 0.01
epochs = 1
for i in range(int(0), int(epochs)):
    print(i)
    g = compute_grad(calculate_loss, α)
    result = adam(α, g, m_adam, v_adam, t_adam, lr)
    α = result[int(0)]
    m_adam = result[int(1)]
    v_adam = result[int(2)]
    t_adam = result[int(3)]
    print(α)
print(α)
pred_solution = solver(α, T0, Δx, Δy, Δt, nt)