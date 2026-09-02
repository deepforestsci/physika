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
    mr = state[int(0)]
    pr = state[int(1)]
    ktx = θ[int(0)]
    ktl = θ[int(1)]
    gm = θ[int(2)]
    gp = θ[int(3)]
    dmr = (ktx - (gm * mr))
    dpr = ((ktl * mr) - (gp * pr))
    return torch.stack([torch.as_tensor(dmr), torch.as_tensor(dpr)])

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
    state = torch.tensor([0.0, 0.0], device=DEVICE)
    mr_array = torch.tensor([0.0], device=DEVICE)
    pr_array = torch.tensor([0.0], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        mr = results[int(0)]
        pr = results[int(1)]
        mr_array = append(mr_array, mr)
        pr_array = append(pr_array, pr)
        state = results
    return torch.stack([torch.as_tensor(mr_array), torch.as_tensor(pr_array)])

def adjoint_grad(θ):
    states = solver(θ)
    mr_array = states[int(0)]
    pr_array = states[int(1)]
    m = get_1d_array_length(pr_array)
    s = torch.stack([torch.as_tensor(0.0), torch.as_tensor(((pr_array[int((m - 1))] - true_pr[int((m - 1))]) / m))])
    L = zero_1d_array(4)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        mr = mr_array[int(idx)]
        pr = pr_array[int(idx)]
        state = torch.stack([torch.as_tensor(mr), torch.as_tensor(pr)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        residual = torch.stack([torch.as_tensor(0.0), torch.as_tensor(((pr_array[int(idx)] - true_pr[int(idx)]) / m))])
        s = (residual + (s @ J_state))
    return L

# === Program ===
dt = 0.1
timesteps = 250
true_theta = torch.tensor([4.0, 2.0, 1.0, 0.25], device=DEVICE)
true_results = solver(true_theta)
true_mr = true_results[int(0)]
true_pr = true_results[int(1)]
θ = torch.tensor([3.0, 3.0, 0.7, 0.4], device=DEVICE)
learning_rate = 0.02
beta1 = 0.9
beta2 = 0.999
eps_adam = 1e-08
m_adam = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
v_adam = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
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