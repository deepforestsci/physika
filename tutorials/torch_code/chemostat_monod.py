import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

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
    X = state[int(0)]
    S = state[int(1)]
    mumax = θ[int(0)]
    Ks = θ[int(1)]
    Y = θ[int(2)]
    mu = ((mumax * S) / (Ks + S))
    dX = ((mu - D) * X)
    dS = ((D * (Sin - S)) - ((mu * X) / Y))
    return torch.stack([torch.as_tensor(dX), torch.as_tensor(dS)])

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
    state = torch.tensor([0.05, 10.0], device=DEVICE)
    X_array = torch.tensor([0.05], device=DEVICE)
    S_array = torch.tensor([10.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        X = results[int(0)]
        S = results[int(1)]
        X_array = append(X_array, X)
        S_array = append(S_array, S)
        state = results
    return torch.stack([torch.as_tensor(X_array), torch.as_tensor(S_array)])

def adjoint_grad(θ):
    states = solver(θ)
    X_array = states[int(0)]
    S_array = states[int(1)]
    m = get_1d_array_length(X_array)
    s = torch.stack([torch.as_tensor(((X_array[int((m - 1))] - true_X[int((m - 1))]) / m)), torch.as_tensor(((S_array[int((m - 1))] - true_S[int((m - 1))]) / m))])
    L = zero_1d_array(3)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        X = X_array[int(idx)]
        S = S_array[int(idx)]
        state = torch.stack([torch.as_tensor(X), torch.as_tensor(S)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        residual = torch.stack([torch.as_tensor(((X_array[int(idx)] - true_X[int(idx)]) / m)), torch.as_tensor(((S_array[int(idx)] - true_S[int(idx)]) / m))])
        s = (residual + (s @ J_state))
    return L

# === Program ===
D = 0.3
Sin = 10.0
dt = 0.1
timesteps = 300
true_theta = torch.tensor([0.5, 0.5, 0.5], device=DEVICE)
true_results = solver(true_theta)
true_X = true_results[int(0)]
true_S = true_results[int(1)]
θ = torch.tensor([0.7, 0.2, 0.7], device=DEVICE)
learning_rate = 0.03
beta1 = 0.9
beta2 = 0.999
eps_adam = 1e-08
m_adam = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
v_adam = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
t_adam = 0.0
epochs = 1
for i in range(int(0), int(epochs)):
    g = adjoint_grad(θ)
    t_adam = (t_adam + 1.0)
    m_adam = ((beta1 * m_adam) + ((1.0 - beta1) * g))
    v_adam = ((beta2 * v_adam) + ((1.0 - beta2) * (g * g)))
    mhat = (m_adam / (1.0 - (beta1 ** t_adam)))
    vhat = (v_adam / (1.0 - (beta2 ** t_adam)))
    θ = (θ - ((learning_rate * mhat) / (torch.sqrt(vhat if isinstance(vhat, torch.Tensor) else torch.tensor(float(vhat))) + eps_adam)))
pred_results = solver(θ)