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
    f1 = state[int(0)]
    f2 = state[int(1)]
    f3 = state[int(2)]
    kcat = θ[int(0)]
    Km = θ[int(1)]
    Vmax = (kcat * Etot)
    df1 = ((Vmax * (1.0 - f1)) / (Km + (S0_1 * (1.0 - f1))))
    df2 = ((Vmax * (1.0 - f2)) / (Km + (S0_2 * (1.0 - f2))))
    df3 = ((Vmax * (1.0 - f3)) / (Km + (S0_3 * (1.0 - f3))))
    return torch.stack([torch.as_tensor(df1), torch.as_tensor(df2), torch.as_tensor(df3)])

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
    state = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
    f1_array = torch.tensor([0.0], device=DEVICE)
    f2_array = torch.tensor([0.0], device=DEVICE)
    f3_array = torch.tensor([0.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        f1 = results[int(0)]
        f2 = results[int(1)]
        f3 = results[int(2)]
        f1_array = append(f1_array, f1)
        f2_array = append(f2_array, f2)
        f3_array = append(f3_array, f3)
        state = results
    return torch.stack([torch.as_tensor(f1_array), torch.as_tensor(f2_array), torch.as_tensor(f3_array)])

def adjoint_grad(θ):
    states = solver(θ)
    f1_array = states[int(0)]
    f2_array = states[int(1)]
    f3_array = states[int(2)]
    m = get_1d_array_length(f1_array)
    s = torch.stack([torch.as_tensor(((f1_array[int((m - 1))] - true_f1[int((m - 1))]) / m)), torch.as_tensor(((f2_array[int((m - 1))] - true_f2[int((m - 1))]) / m)), torch.as_tensor(((f3_array[int((m - 1))] - true_f3[int((m - 1))]) / m))])
    L = zero_1d_array(2)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        f1 = f1_array[int(idx)]
        f2 = f2_array[int(idx)]
        f3 = f3_array[int(idx)]
        state = torch.stack([torch.as_tensor(f1), torch.as_tensor(f2), torch.as_tensor(f3)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        r1 = ((f1_array[int(idx)] - true_f1[int(idx)]) / m)
        r2 = ((f2_array[int(idx)] - true_f2[int(idx)]) / m)
        r3 = ((f3_array[int(idx)] - true_f3[int(idx)]) / m)
        residual = torch.stack([torch.as_tensor(r1), torch.as_tensor(r2), torch.as_tensor(r3)])
        s = (residual + (s @ J_state))
    return L

# === Program ===
Etot = 1.0
S0_1 = 1.0
S0_2 = 10.0
S0_3 = 40.0
dt = 0.005
timesteps = 200
true_theta = torch.tensor([100.0, 5.0], device=DEVICE)
true_results = solver(true_theta)
true_f1 = true_results[int(0)]
true_f2 = true_results[int(1)]
true_f3 = true_results[int(2)]
θ = torch.tensor([50.0, 20.0], device=DEVICE)
learning_rate = 0.2
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