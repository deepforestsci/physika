import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i))]])
    return results

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def append(x, var):
    new_length = (len(x) + 1)
    results = zero_1d_array(new_length)
    len_x = get_1d_array_length(x)
    for i in range(int(0), int(new_length)):
        if i < len_x:
            results[int(i)] = x[int(i)]
        else:
            results[int(i)] = var
    return results

def f(state, θ):
    x = state[int(0)]
    y = state[int(1)]
    α = θ[int(0)]
    β = θ[int(1)]
    γ = θ[int(2)]
    δ = θ[int(3)]
    dx = ((α * x) - ((β * x) * y))
    dy = ((-(γ * y)) + ((δ * x) * y))
    return torch.stack([torch.as_tensor(dx).float(), torch.as_tensor(dy).float()])

def rk4_step(state, θ):
    k1 = f(state, θ)
    k2_state = (state + ((0.5 * dt) * k1))
    k2 = f(k2_state, θ)
    k3_state = (state + ((0.5 * dt) * k2))
    k3 = f(k3_state, θ)
    k4_state = (state + (dt * k3))
    k4 = f(k4_state, θ)
    return (state + ((dt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))

def solver(θ):
    state = torch.tensor([10.0, 1.0])
    x_array = torch.tensor([10.0])
    y_array = torch.tensor([1.0])
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        x = results[int(0)]
        y = results[int(1)]
        x_array = append(x_array, x)
        y_array = append(y_array, y)
        state = results
    return torch.stack([torch.as_tensor(x_array).float(), torch.as_tensor(y_array).float()])

def adjoint_grad(θ):
    states = solver(θ)
    x_array = states[int(0)]
    y_array = states[int(1)]
    m = get_1d_array_length(x_array)
    s = torch.stack([torch.as_tensor((2 * (x_array[int((m - 1))] - true_x[int((m - 1))]))).float(), torch.as_tensor((2 * (y_array[int((m - 1))] - true_y[int((m - 1))]))).float()])
    L = zero_1d_array(4)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 1) - i)
        x = x_array[int(idx)]
        y = y_array[int(idx)]
        state = torch.stack([torch.as_tensor(x).float(), torch.as_tensor(y).float()])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        s = (s @ J_state)
    return L

# === Program ===
dt = 0.1
timesteps = 100
true_theta = torch.tensor([1.5, 1.0, 3.0, 1.0])
true_results = solver(true_theta)
true_x = true_results[int(0)]
true_y = true_results[int(1)]
θ = torch.tensor([1.0, 0.7, 2.5, 0.7])
learning_rate = 0.0005
epochs = 1
for i in range(int(0), int(epochs)):
    g = adjoint_grad(θ)
    θ = (θ - (learning_rate * g))
pred_results = solver(θ)