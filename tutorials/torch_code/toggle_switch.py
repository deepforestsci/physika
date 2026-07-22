import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
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
    u1 = state[int(0)]
    v1 = state[int(1)]
    u2 = state[int(2)]
    v2 = state[int(3)]
    a1 = θ[int(0)]
    a2 = θ[int(1)]
    du1 = ((a1 / (1.0 + (v1 ** 2))) - u1)
    dv1 = ((a2 / (1.0 + (u1 ** 2))) - v1)
    du2 = ((a1 / (1.0 + (v2 ** 2))) - u2)
    dv2 = ((a2 / (1.0 + (u2 ** 2))) - v2)
    return torch.stack([torch.as_tensor(du1), torch.as_tensor(dv1), torch.as_tensor(du2), torch.as_tensor(dv2)])

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
    state = torch.tensor([2.0, 0.1, 0.1, 2.0], device=DEVICE)
    u1_array = torch.tensor([2.0], device=DEVICE)
    v1_array = torch.tensor([0.1], device=DEVICE)
    u2_array = torch.tensor([0.1], device=DEVICE)
    v2_array = torch.tensor([2.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        u1 = results[int(0)]
        v1 = results[int(1)]
        u2 = results[int(2)]
        v2 = results[int(3)]
        u1_array = append(u1_array, u1)
        v1_array = append(v1_array, v1)
        u2_array = append(u2_array, u2)
        v2_array = append(v2_array, v2)
        state = results
    return torch.stack([torch.as_tensor(u1_array), torch.as_tensor(v1_array), torch.as_tensor(u2_array), torch.as_tensor(v2_array)])

def adjoint_grad(θ):
    states = solver(θ)
    u1_array = states[int(0)]
    v1_array = states[int(1)]
    u2_array = states[int(2)]
    v2_array = states[int(3)]
    m = get_1d_array_length(u1_array)
    s = torch.stack([torch.as_tensor(((u1_array[int((m - 1))] - true_u1[int((m - 1))]) / m)), torch.as_tensor(((v1_array[int((m - 1))] - true_v1[int((m - 1))]) / m)), torch.as_tensor(((u2_array[int((m - 1))] - true_u2[int((m - 1))]) / m)), torch.as_tensor(((v2_array[int((m - 1))] - true_v2[int((m - 1))]) / m))])
    L = zero_1d_array(2)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        u1 = u1_array[int(idx)]
        v1 = v1_array[int(idx)]
        u2 = u2_array[int(idx)]
        v2 = v2_array[int(idx)]
        state = torch.stack([torch.as_tensor(u1), torch.as_tensor(v1), torch.as_tensor(u2), torch.as_tensor(v2)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        r1 = ((u1_array[int(idx)] - true_u1[int(idx)]) / m)
        r2 = ((v1_array[int(idx)] - true_v1[int(idx)]) / m)
        r3 = ((u2_array[int(idx)] - true_u2[int(idx)]) / m)
        r4 = ((v2_array[int(idx)] - true_v2[int(idx)]) / m)
        residual = torch.stack([torch.as_tensor(r1), torch.as_tensor(r2), torch.as_tensor(r3), torch.as_tensor(r4)])
        s = (residual + (s @ J_state))
    return L

# === Program ===
dt = 0.1
timesteps = 100
true_theta = torch.tensor([3.0, 3.0], device=DEVICE)
true_results = solver(true_theta)
true_u1 = true_results[int(0)]
true_v1 = true_results[int(1)]
true_u2 = true_results[int(2)]
true_v2 = true_results[int(3)]
θ = torch.tensor([2.0, 2.0], device=DEVICE)
learning_rate = 1.0
epochs = 1
for i in range(int(0), int(epochs)):
    g = adjoint_grad(θ)
    θ = (θ - (learning_rate * g))
pred_results = solver(θ)