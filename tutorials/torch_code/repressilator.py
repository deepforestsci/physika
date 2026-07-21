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
    m1 = state[int(0)]
    m2 = state[int(1)]
    m3 = state[int(2)]
    p1 = state[int(3)]
    p2 = state[int(4)]
    p3 = state[int(5)]
    a = θ[int(0)]
    a0 = θ[int(1)]
    beta = θ[int(2)]
    dm1 = ((a0 + (a / (1.0 + (p3 ** 2)))) - m1)
    dm2 = ((a0 + (a / (1.0 + (p1 ** 2)))) - m2)
    dm3 = ((a0 + (a / (1.0 + (p2 ** 2)))) - m3)
    dp1 = (beta * (m1 - p1))
    dp2 = (beta * (m2 - p2))
    dp3 = (beta * (m3 - p3))
    return torch.stack([torch.as_tensor(dm1), torch.as_tensor(dm2), torch.as_tensor(dm3), torch.as_tensor(dp1), torch.as_tensor(dp2), torch.as_tensor(dp3)])

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
    state = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], device=DEVICE)
    m1_array = torch.tensor([0.0], device=DEVICE)
    m2_array = torch.tensor([0.0], device=DEVICE)
    m3_array = torch.tensor([0.0], device=DEVICE)
    p1_array = torch.tensor([1.0], device=DEVICE)
    p2_array = torch.tensor([2.0], device=DEVICE)
    p3_array = torch.tensor([3.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        m1 = results[int(0)]
        m2 = results[int(1)]
        m3 = results[int(2)]
        p1 = results[int(3)]
        p2 = results[int(4)]
        p3 = results[int(5)]
        m1_array = append(m1_array, m1)
        m2_array = append(m2_array, m2)
        m3_array = append(m3_array, m3)
        p1_array = append(p1_array, p1)
        p2_array = append(p2_array, p2)
        p3_array = append(p3_array, p3)
        state = results
    return torch.stack([torch.as_tensor(m1_array), torch.as_tensor(m2_array), torch.as_tensor(m3_array), torch.as_tensor(p1_array), torch.as_tensor(p2_array), torch.as_tensor(p3_array)])

def adjoint_grad(θ):
    states = solver(θ)
    m1_array = states[int(0)]
    m2_array = states[int(1)]
    m3_array = states[int(2)]
    p1_array = states[int(3)]
    p2_array = states[int(4)]
    p3_array = states[int(5)]
    m = get_1d_array_length(m1_array)
    s = torch.stack([torch.as_tensor(((m1_array[int((m - 1))] - true_m1[int((m - 1))]) / m)), torch.as_tensor(((m2_array[int((m - 1))] - true_m2[int((m - 1))]) / m)), torch.as_tensor(((m3_array[int((m - 1))] - true_m3[int((m - 1))]) / m)), torch.as_tensor(((p1_array[int((m - 1))] - true_p1[int((m - 1))]) / m)), torch.as_tensor(((p2_array[int((m - 1))] - true_p2[int((m - 1))]) / m)), torch.as_tensor(((p3_array[int((m - 1))] - true_p3[int((m - 1))]) / m))])
    L = zero_1d_array(3)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        m1 = m1_array[int(idx)]
        m2 = m2_array[int(idx)]
        m3 = m3_array[int(idx)]
        p1 = p1_array[int(idx)]
        p2 = p2_array[int(idx)]
        p3 = p3_array[int(idx)]
        state = torch.stack([torch.as_tensor(m1), torch.as_tensor(m2), torch.as_tensor(m3), torch.as_tensor(p1), torch.as_tensor(p2), torch.as_tensor(p3)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        r1 = ((m1_array[int(idx)] - true_m1[int(idx)]) / m)
        r2 = ((m2_array[int(idx)] - true_m2[int(idx)]) / m)
        r3 = ((m3_array[int(idx)] - true_m3[int(idx)]) / m)
        r4 = ((p1_array[int(idx)] - true_p1[int(idx)]) / m)
        r5 = ((p2_array[int(idx)] - true_p2[int(idx)]) / m)
        r6 = ((p3_array[int(idx)] - true_p3[int(idx)]) / m)
        residual = torch.stack([torch.as_tensor(r1), torch.as_tensor(r2), torch.as_tensor(r3), torch.as_tensor(r4), torch.as_tensor(r5), torch.as_tensor(r6)])
        s = (residual + (s @ J_state))
    return L

# === Program ===
dt = 0.1
timesteps = 200
true_theta = torch.tensor([40.0, 1.0, 1.0], device=DEVICE)
true_results = solver(true_theta)
true_m1 = true_results[int(0)]
true_m2 = true_results[int(1)]
true_m3 = true_results[int(2)]
true_p1 = true_results[int(3)]
true_p2 = true_results[int(4)]
true_p3 = true_results[int(5)]
θ = torch.tensor([30.0, 1.5, 0.7], device=DEVICE)
learning_rate = 0.05
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