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

def linspace(start, end, n):
    x = zero_1d_array(n)
    Δx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * Δx))
    return x

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def central_difference_x(f):
    diff = zero_2d_array(n_points, n_points)
    diff[int(1):int((n_points - 1)), int(1):int((n_points - 1))] = ((f[int(1):int((n_points - 1)), int(2):int(n_points)] - f[int(1):int((n_points - 1)), int(0):int((n_points - 2))]) / (2 * element_length))
    return diff

def central_difference_y(f):
    diff = zero_2d_array(n_points, n_points)
    diff[int(1):int((n_points - 1)), int(1):int((n_points - 1))] = ((f[int(2):int(n_points), int(1):int((n_points - 1))] - f[int(0):int((n_points - 2)), int(1):int((n_points - 1))]) / (2 * element_length))
    return diff

def laplace(f):
    diff = zero_2d_array(n_points, n_points)
    diff[int(1):int((n_points - 1)), int(1):int((n_points - 1))] = (((((f[int(1):int((n_points - 1)), int(0):int((n_points - 2))] + f[int(0):int((n_points - 2)), int(1):int((n_points - 1))]) + f[int(1):int((n_points - 1)), int(2):int(n_points)]) + f[int(2):int(n_points), int(1):int((n_points - 1))]) - (4 * f[int(1):int((n_points - 1)), int(1):int((n_points - 1))])) / (element_length ** 2))
    return diff

def apply_u_bc(u):
    u[int(0), :] = 0.0
    u[int((-1)), :] = horizontal_velocity_top
    u[:, int(0)] = 0.0
    u[:, int((-1))] = 0.0
    return u

def apply_v_bc(v):
    v[int(0), :] = 0.0
    v[int((-1)), :] = 0.0
    v[:, int(0)] = 0.0
    v[:, int((-1))] = 0.0
    return v

def apply_p_bc(p):
    p[:, int((-1))] = p[:, int((-2))]
    p[int(0), :] = p[int(1), :]
    p[:, int(0)] = p[:, int(1)]
    p[int((-1)), :] = 0.0
    return p

def advect_diffuse(f, u, v):
    return ((-((u * central_difference_x(f)) + (v * central_difference_y(f)))) + (ν * laplace(f)))

def solve_pressure(p_init, rhs):
    p_prev = p_init
    for k in range(int(0), int(n_pressure_poisson_iterations)):
        p_next = zero_2d_array(n_points, n_points)
        p_next[int(1):int((-1)), int(1):int((-1))] = (0.25 * ((((p_prev[int(1):int((-1)), :int((-2))] + p_prev[:int((-2)), int(1):int((-1))]) + p_prev[int(1):int((-1)), int(2):]) + p_prev[int(2):, int(1):int((-1))]) - ((element_length ** 2) * rhs[int(1):int((-1)), int(1):int((-1))])))
        p_prev = apply_p_bc(p_next)
    return p_prev

def solver(ρ):
    u = zero_2d_array(n_points, n_points)
    v = zero_2d_array(n_points, n_points)
    p = zero_2d_array(n_points, n_points)
    for i in range(int(0), int(n_iterations)):
        u_tent = apply_u_bc((u + (time_step_length * advect_diffuse(u, u, v))))
        v_tent = apply_v_bc((v + (time_step_length * advect_diffuse(v, u, v))))
        rhs = ((ρ / time_step_length) * (central_difference_x(u_tent) + central_difference_y(v_tent)))
        p = solve_pressure(p, rhs)
        u = apply_u_bc((u_tent - ((time_step_length / ρ) * central_difference_x(p))))
        v = apply_v_bc((v_tent - ((time_step_length / ρ) * central_difference_y(p))))
    return torch.stack([torch.as_tensor(u), torch.as_tensor(v), torch.as_tensor(p)])

def calculate_loss(ρ):
    predictions = solver(ρ)
    pred_u = predictions[int(0)]
    pred_v = predictions[int(1)]
    pred_p = predictions[int(2)]
    loss_u = torch.mean(((pred_u - true_u) ** 2) if isinstance(((pred_u - true_u) ** 2), torch.Tensor) else torch.tensor(float(((pred_u - true_u) ** 2))))
    loss_v = torch.mean(((pred_v - true_v) ** 2) if isinstance(((pred_v - true_v) ** 2), torch.Tensor) else torch.tensor(float(((pred_v - true_v) ** 2))))
    loss_p = torch.mean(((pred_p - true_p) ** 2) if isinstance(((pred_p - true_p) ** 2), torch.Tensor) else torch.tensor(float(((pred_p - true_p) ** 2))))
    loss = ((loss_u + loss_v) + loss_p)
    return loss

def adam(ρ, g, m, v, t, lr):
    β1 = 0.9
    β2 = 0.999
    ε = 1e-08
    m_new = ((β1 * m) + ((1.0 - β1) * g))
    v_new = ((β2 * v) + ((1.0 - β2) * (g ** 2)))
    m_hat = (m_new / (1.0 - (β1 ** t)))
    v_hat = (v_new / (1.0 - (β2 ** t)))
    ρ_new = (ρ - ((lr * m_hat) / (torch.sqrt(torch.as_tensor(v_hat).float()) + ε)))
    return torch.stack([torch.as_tensor(ρ_new), torch.as_tensor(m_new), torch.as_tensor(v_new), torch.as_tensor((t + 1.0))])

# === Program ===
n_points = 21
domain_size = 1.0
n_iterations = 500
time_step_length = 0.001
ν = 0.1
true_ρ = 1.0
horizontal_velocity_top = 1.0
n_pressure_poisson_iterations = 10
stability_safety_factor = 0.5
element_length = (domain_size / (n_points - float(1)))
x = linspace(0.0, domain_size, n_points)
y = linspace(0.0, domain_size, n_points)
f = zero_2d_array(n_points, n_points)
for i in range(int(0), int(n_points)):
    for j in range(int(0), int(n_points)):
        t_x = (j * element_length)
        t_y = (i * element_length)
        f[int(i), int(j)] = ((t_x ** 2) + (t_y ** 2))
n_iterations = 5
true_solution = solver(true_ρ)
true_u = true_solution[int(0)]
true_v = true_solution[int(1)]
true_p = true_solution[int(2)]
ρ = torch.tensor(3.0, requires_grad=True)
m_adam, v_adam, t_adam, lr = 0.0, 0.0, 1.0, 0.01
epochs = 1
for i in range(int(0), int(epochs)):
    print(i)
    g = compute_grad(calculate_loss, ρ)
    result = adam(ρ, g, m_adam, v_adam, t_adam, lr)
    ρ = result[int(0)]
    m_adam = result[int(1)]
    v_adam = result[int(2)]
    t_adam = result[int(3)]
    print(ρ)
print(ρ)
pred_solution = solver(ρ)
pred_u = pred_solution[int(0)]
pred_v = pred_solution[int(1)]
pred_p = pred_solution[int(2)]
X = zero_2d_array(n_points, n_points)
Y = zero_2d_array(n_points, n_points)
for i in range(int(0), int(n_points)):
    for j in range(int(0), int(n_points)):
        X[int(i), int(j)] = (j * element_length)
        Y[int(i), int(j)] = (i * element_length)