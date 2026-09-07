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
    new_length = (get_1d_array_length(x) + 1)
    results = zero_1d_array(new_length)
    len_x = get_1d_array_length(x)
    for i in range(int(0), int(new_length)):
        if i < len_x:
            results[int(i)] = x[int(i)]
        else:
            results[int(i)] = var
    return results

def alpha_m(V):
    x = (V + 40.0)
    return (1.0 if (torch.abs(torch.as_tensor(x)) < 0.0001) else ((0.1 * x) / (1.0 - torch.exp(torch.as_tensor((0.0 - (x / 10.0))).float()))))

def beta_m(V):
    return (4.0 * torch.exp(torch.as_tensor((0.0 - ((V + 65.0) / 18.0))).float()))

def alpha_h(V):
    return (0.07 * torch.exp(torch.as_tensor((0.0 - ((V + 65.0) / 20.0))).float()))

def beta_h(V):
    return (1.0 / (1.0 + torch.exp(torch.as_tensor((0.0 - ((V + 35.0) / 10.0))).float())))

def alpha_n(V):
    x = (V + 55.0)
    return (0.1 if (torch.abs(torch.as_tensor(x)) < 0.0001) else ((0.01 * x) / (1.0 - torch.exp(torch.as_tensor((0.0 - (x / 10.0))).float()))))

def beta_n(V):
    return (0.125 * torch.exp(torch.as_tensor((0.0 - ((V + 65.0) / 80.0))).float()))

def f(state, θ):
    V = state[int(0)]
    mg = state[int(1)]
    hg = state[int(2)]
    ng = state[int(3)]
    gNa = θ[int(0)]
    gK = θ[int(1)]
    gL = θ[int(2)]
    am = alpha_m(V)
    bm = beta_m(V)
    ah = alpha_h(V)
    bh = beta_h(V)
    an = alpha_n(V)
    bn = beta_n(V)
    iNa = (((((gNa * mg) * mg) * mg) * hg) * (V - ENa))
    iK = (((((gK * ng) * ng) * ng) * ng) * (V - EK))
    iL = (gL * (V - EL))
    dV = ((((Iapp - iNa) - iK) - iL) / Cm)
    dmg = ((am * (1.0 - mg)) - (bm * mg))
    dhg = ((ah * (1.0 - hg)) - (bh * hg))
    dng = ((an * (1.0 - ng)) - (bn * ng))
    return torch.stack([torch.as_tensor(dV), torch.as_tensor(dmg), torch.as_tensor(dhg), torch.as_tensor(dng)])

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
    state = torch.tensor([(-65.0), 0.0529, 0.5961, 0.3177], device=DEVICE)
    V_array = torch.tensor([(-65.0)], device=DEVICE)
    mg_array = torch.tensor([0.0529], device=DEVICE)
    hg_array = torch.tensor([0.5961], device=DEVICE)
    ng_array = torch.tensor([0.3177], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        results = rk4_step(state, θ)
        V = results[int(0)]
        mg = results[int(1)]
        hg = results[int(2)]
        ng = results[int(3)]
        V_array = append(V_array, V)
        mg_array = append(mg_array, mg)
        hg_array = append(hg_array, hg)
        ng_array = append(ng_array, ng)
        state = results
    return torch.stack([torch.as_tensor(V_array), torch.as_tensor(mg_array), torch.as_tensor(hg_array), torch.as_tensor(ng_array)])

def adjoint_grad(θ):
    states = solver(θ)
    V_array = states[int(0)]
    mg_array = states[int(1)]
    hg_array = states[int(2)]
    ng_array = states[int(3)]
    m = get_1d_array_length(V_array)
    s = torch.stack([torch.as_tensor(((V_array[int((m - 1))] - true_V[int((m - 1))]) / m)), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])
    L = zero_1d_array(3)
    for i in range(int(0), int((m - 1))):
        idx = ((m - 2) - i)
        V = V_array[int(idx)]
        mg = mg_array[int(idx)]
        hg = hg_array[int(idx)]
        ng = ng_array[int(idx)]
        state = torch.stack([torch.as_tensor(V), torch.as_tensor(mg), torch.as_tensor(hg), torch.as_tensor(ng)])
        J_state = compute_grad(lambda _dstate: rk4_step(_dstate, θ), state)
        J_theta = compute_grad(lambda _dθ: rk4_step(state, _dθ), θ)
        L = L + (s @ J_theta)
        residual = torch.stack([torch.as_tensor(((V_array[int(idx)] - true_V[int(idx)]) / m)), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])
        s = (residual + (s @ J_state))
    return L

# === Program ===
Cm = 1.0
ENa = 50.0
EK = (-77.0)
EL = (-54.387)
Iapp = 10.0
dt = 0.05
timesteps = 400
true_theta = torch.stack([torch.as_tensor(120.0), torch.as_tensor(36.0), torch.as_tensor(0.3)])
true_results = solver(true_theta)
true_V = true_results[int(0)]
θ = torch.stack([torch.as_tensor(100.0), torch.as_tensor(30.0), torch.as_tensor(0.5)])
learning_rate = 0.2
beta1 = 0.9
beta2 = 0.999
eps_adam = 1e-08
m_adam = torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])
v_adam = torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])
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