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
    w1 = state[int(0)]
    w2 = state[int(1)]
    w3 = state[int(2)]
    w4 = state[int(3)]
    w5 = state[int(4)]
    w6 = state[int(5)]
    w7 = state[int(6)]
    K1 = θ[int(0)]
    K2 = θ[int(1)]
    dw1 = (((u1 * (1.0 - w1)) / (K1 + (1.0 - w1))) - ((Vp * w1) / (K2 + w1)))
    dw2 = (((u2 * (1.0 - w2)) / (K1 + (1.0 - w2))) - ((Vp * w2) / (K2 + w2)))
    dw3 = (((u3 * (1.0 - w3)) / (K1 + (1.0 - w3))) - ((Vp * w3) / (K2 + w3)))
    dw4 = (((u4 * (1.0 - w4)) / (K1 + (1.0 - w4))) - ((Vp * w4) / (K2 + w4)))
    dw5 = (((u5 * (1.0 - w5)) / (K1 + (1.0 - w5))) - ((Vp * w5) / (K2 + w5)))
    dw6 = (((u6 * (1.0 - w6)) / (K1 + (1.0 - w6))) - ((Vp * w6) / (K2 + w6)))
    dw7 = (((u7 * (1.0 - w7)) / (K1 + (1.0 - w7))) - ((Vp * w7) / (K2 + w7)))
    return torch.stack([torch.as_tensor(dw1), torch.as_tensor(dw2), torch.as_tensor(dw3), torch.as_tensor(dw4), torch.as_tensor(dw5), torch.as_tensor(dw6), torch.as_tensor(dw7)])

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
    state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=DEVICE)
    w1_array = torch.tensor([0.0], device=DEVICE)
    w2_array = torch.tensor([0.0], device=DEVICE)
    w3_array = torch.tensor([0.0], device=DEVICE)
    w4_array = torch.tensor([0.0], device=DEVICE)
    w5_array = torch.tensor([0.0], device=DEVICE)
    w6_array = torch.tensor([0.0], device=DEVICE)
    w7_array = torch.tensor([0.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        w1 = results[int(0)]
        w2 = results[int(1)]
        w3 = results[int(2)]
        w4 = results[int(3)]
        w5 = results[int(4)]
        w6 = results[int(5)]
        w7 = results[int(6)]
        w1_array = append(w1_array, w1)
        w2_array = append(w2_array, w2)
        w3_array = append(w3_array, w3)
        w4_array = append(w4_array, w4)
        w5_array = append(w5_array, w5)
        w6_array = append(w6_array, w6)
        w7_array = append(w7_array, w7)
        state = results
    return torch.stack([torch.as_tensor(w1_array), torch.as_tensor(w2_array), torch.as_tensor(w3_array), torch.as_tensor(w4_array), torch.as_tensor(w5_array), torch.as_tensor(w6_array), torch.as_tensor(w7_array)])

def adjoint_grad(θ):
    states = solver(θ)
    w1_array = states[int(0)]
    w2_array = states[int(1)]
    w3_array = states[int(2)]
    w4_array = states[int(3)]
    w5_array = states[int(4)]
    w6_array = states[int(5)]
    w7_array = states[int(6)]
    m = get_1d_array_length(w1_array)
    s = torch.stack([torch.as_tensor(((w1_array[int((m - 1))] - true_w1[int((m - 1))]) / n_exp)), torch.as_tensor(((w2_array[int((m - 1))] - true_w2[int((m - 1))]) / n_exp)), torch.as_tensor(((w3_array[int((m - 1))] - true_w3[int((m - 1))]) / n_exp)), torch.as_tensor(((w4_array[int((m - 1))] - true_w4[int((m - 1))]) / n_exp)), torch.as_tensor(((w5_array[int((m - 1))] - true_w5[int((m - 1))]) / n_exp)), torch.as_tensor(((w6_array[int((m - 1))] - true_w6[int((m - 1))]) / n_exp)), torch.as_tensor(((w7_array[int((m - 1))] - true_w7[int((m - 1))]) / n_exp))])
    L = zero_1d_array(2)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        w1 = w1_array[int(idx)]
        w2 = w2_array[int(idx)]
        w3 = w3_array[int(idx)]
        w4 = w4_array[int(idx)]
        w5 = w5_array[int(idx)]
        w6 = w6_array[int(idx)]
        w7 = w7_array[int(idx)]
        state = torch.stack([torch.as_tensor(w1), torch.as_tensor(w2), torch.as_tensor(w3), torch.as_tensor(w4), torch.as_tensor(w5), torch.as_tensor(w6), torch.as_tensor(w7)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        s = (s @ J_state)
    return L

# === Program ===
Vp = 1.0
u1 = 0.7
u2 = 0.8
u3 = 0.9
u4 = 1.0
u5 = 1.1
u6 = 1.2
u7 = 1.3
dt = 0.1
timesteps = 200
true_theta = torch.tensor([0.1, 0.1], device=DEVICE)
true_results = solver(true_theta)
true_w1 = true_results[int(0)]
true_w2 = true_results[int(1)]
true_w3 = true_results[int(2)]
true_w4 = true_results[int(3)]
true_w5 = true_results[int(4)]
true_w6 = true_results[int(5)]
true_w7 = true_results[int(6)]
n_exp = 7.0
θ = torch.tensor([0.02, 0.5], device=DEVICE)
learning_rate = 0.02
beta1 = 0.9
beta2 = 0.999
eps_adam = 1e-08
m_adam = torch.tensor([0.0, 0.0], device=DEVICE)
v_adam = torch.tensor([0.0, 0.0], device=DEVICE)
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