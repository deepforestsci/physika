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
    S = state[int(0)]
    I = state[int(1)]
    beta = θ[int(0)]
    gamma = θ[int(1)]
    dS = (0.0 - ((beta * S) * I))
    dI = (((beta * S) * I) - (gamma * I))
    dR = (gamma * I)
    return torch.stack([torch.as_tensor(dS), torch.as_tensor(dI), torch.as_tensor(dR)])

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
    state = torch.tensor([0.99, 0.01, 0.0], device=DEVICE)
    S_array = torch.tensor([0.99], device=DEVICE)
    I_array = torch.tensor([0.01], device=DEVICE)
    R_array = torch.tensor([0.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        S = results[int(0)]
        I = results[int(1)]
        Rec = results[int(2)]
        S_array = append(S_array, S)
        I_array = append(I_array, I)
        R_array = append(R_array, Rec)
        state = results
    return torch.stack([torch.as_tensor(S_array), torch.as_tensor(I_array), torch.as_tensor(R_array)])

def adjoint_grad(θ):
    states = solver(θ)
    S_array = states[int(0)]
    I_array = states[int(1)]
    R_array = states[int(2)]
    m = get_1d_array_length(I_array)
    s = torch.stack([torch.as_tensor(0.0), torch.as_tensor(((I_array[int((m - 1))] - true_I[int((m - 1))]) / m)), torch.as_tensor(0.0)])
    L = zero_1d_array(2)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        S = S_array[int(idx)]
        I = I_array[int(idx)]
        Rec = R_array[int(idx)]
        state = torch.stack([torch.as_tensor(S), torch.as_tensor(I), torch.as_tensor(Rec)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        residual = torch.stack([torch.as_tensor(0.0), torch.as_tensor(((I_array[int(idx)] - true_I[int(idx)]) / m)), torch.as_tensor(0.0)])
        s = (residual + (s @ J_state))
    return L

# === Program ===
dt = 0.5
timesteps = 200
true_theta = torch.tensor([0.3, 0.1], device=DEVICE)
true_results = solver(true_theta)
true_S = true_results[int(0)]
true_I = true_results[int(1)]
true_R = true_results[int(2)]
θ = torch.tensor([0.2, 0.2], device=DEVICE)
learning_rate = 0.5
epochs = 1
for i in range(int(0), int(epochs)):
    g = adjoint_grad(θ)
    θ = (θ - (learning_rate * g))
pred_results = solver(θ)