import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def get_1d_array_length(x, m=None):
    if m is None:
        m = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

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
    v = state[int(0)]
    w = state[int((1 + 0))]
    a = θ[int(0)]
    b = θ[int((1 + 0))]
    eps = θ[int((1 + (1 + 0)))]
    Iext = θ[int((1 + (1 + (1 + 0))))]
    dv = (((v - ((v ** 3) / 3.0)) - w) + Iext)
    dw = (eps * ((v + a) - (b * w)))
    return torch.stack([torch.as_tensor(dv), torch.as_tensor(dw)])

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
    state = torch.tensor([(-1.0), 1.0], device=DEVICE)
    v_array = torch.tensor([(-1.0)], device=DEVICE)
    w_array = torch.tensor([1.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        v = results[int(0)]
        w = results[int(1)]
        v_array = append(v_array, v)
        w_array = append(w_array, w)
        state = results
    return torch.stack([torch.as_tensor(v_array), torch.as_tensor(w_array)])

def adjoint_grad(θ):
    states = solver(θ)
    v_array = states[int(0)]
    w_array = states[int(1)]
    m = get_1d_array_length(v_array)
    s = torch.stack([torch.as_tensor(((v_array[int((m - 1))] - true_v[int((m - 1))]) / m)), torch.as_tensor(((w_array[int((m - 1))] - true_w[int((m - 1))]) / m))])
    L = zero_1d_array(4)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        v = v_array[int(idx)]
        w = w_array[int(idx)]
        state = torch.stack([torch.as_tensor(v), torch.as_tensor(w)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        rv = ((v_array[int(idx)] - true_v[int(idx)]) / m)
        rw = ((w_array[int(idx)] - true_w[int(idx)]) / m)
        residual = torch.stack([torch.as_tensor(rv), torch.as_tensor(rw)])
        s = (residual + (s @ J_state))
    return L

# === Program ===
dt = 0.1
timesteps = 200
true_theta = torch.stack([torch.as_tensor(0.7), torch.as_tensor(0.8), torch.as_tensor(0.08), torch.as_tensor(0.5)])
true_results = solver(true_theta)
true_v = true_results[int(0)]
true_w = true_results[int(1)]
θ = torch.stack([torch.as_tensor(0.6), torch.as_tensor(0.7), torch.as_tensor(0.1), torch.as_tensor(0.6)])
learning_rate = 0.003
epochs = 1
for i in range(int(0), int(epochs)):
    g = adjoint_grad(θ)
    θ = (θ - (learning_rate * g))
pred_results = solver(θ)