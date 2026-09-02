import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

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
    new_length = (get_1d_array_length(x) + 1)
    results = zero_1d_array(new_length)
    len_x = get_1d_array_length(x)
    for i in range(int(0), int(new_length)):
        if i < len_x:
            results[int(i)] = x[int(i)]
        else:
            results[int(i)] = var
    return results

def solver_hard(a):
    v = (-65.0)
    u = (b * v)
    v_array = torch.tensor([(-65.0)], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        v = v + ((0.5 * dt) * ((((((0.04 * v) * v) + (5.0 * v)) + 140.0) - u) + I))
        v = v + ((0.5 * dt) * ((((((0.04 * v) * v) + (5.0 * v)) + 140.0) - u) + I))
        u = u + (dt * (a * ((b * v) - u)))
        if v >= 30.0:
            v = c
            u = (u + d)
        v_array = append(v_array, v)
    return v_array

def solver_salt(a):
    v = (-65.0)
    u = (b * v)
    vfull = 0.0
    ufull = 0.0
    frac = 0.0
    hf = 0.0
    vc = 0.0
    u_cross = 0.0
    hr = 0.0
    ur = 0.0
    vv = 0.0
    v_array = torch.tensor([(-65.0)], device=DEVICE)
    for i in range(int(0), int(timesteps)):
        vfull = (v + ((0.5 * dt) * ((((((0.04 * v) * v) + (5.0 * v)) + 140.0) - u) + I)))
        vfull = (vfull + ((0.5 * dt) * ((((((0.04 * vfull) * vfull) + (5.0 * vfull)) + 140.0) - u) + I)))
        ufull = (u + (dt * (a * ((b * vfull) - u))))
        if vfull >= 30.0:
            frac = ((30.0 - v) / (vfull - v))
            hf = (frac * dt)
            vc = (v + ((0.5 * hf) * ((((((0.04 * v) * v) + (5.0 * v)) + 140.0) - u) + I)))
            vc = (vc + ((0.5 * hf) * ((((((0.04 * vc) * vc) + (5.0 * vc)) + 140.0) - u) + I)))
            u_cross = (u + (hf * (a * ((b * vc) - u))))
            hr = ((1.0 - frac) * dt)
            ur = (u_cross + d)
            vv = (c + ((0.5 * hr) * ((((((0.04 * c) * c) + (5.0 * c)) + 140.0) - ur) + I)))
            vv = (vv + ((0.5 * hr) * ((((((0.04 * vv) * vv) + (5.0 * vv)) + 140.0) - ur) + I)))
            u = (ur + (hr * (a * ((b * vv) - ur))))
            v = vv
        else:
            v = vfull
            u = ufull
        v_array = append(v_array, v)
    return v_array

def loss_hard(a):
    v_pred = solver_hard(a)
    L = 0.0
    m = get_1d_array_length(v_pred)
    for i in range(int(0), int(m)):
        L = L + ((v_pred[int(i)] - v_data[int(i)]) ** 2)
    return (L / m)

def loss_salt(a):
    v_pred = solver_salt(a)
    L = 0.0
    m = get_1d_array_length(v_pred)
    for i in range(int(0), int(m)):
        L = L + ((v_pred[int(i)] - v_data[int(i)]) ** 2)
    return (L / m)

# === Program ===
b = 0.2
c = (-65.0)
d = 8.0
dt = 0.5
timesteps = 200
I = 10.0
true_a = 0.02
v_data = solver_salt(true_a)
a_probe = torch.tensor(0.025, requires_grad=True)
print(compute_grad(loss_hard, a_probe))
print(compute_grad(loss_salt, a_probe))
a_log = torch.log(0.04 if isinstance(0.04, torch.Tensor) else torch.tensor(float(0.04)))
learning_rate = 0.004
beta1 = 0.9
beta2 = 0.999
eps_adam = 1e-08
m_adam = 0.0
v_adam = 0.0
t_adam = 0.0
epochs = 1
for i in range(int(0), int(epochs)):
    a_lin = torch.exp(a_log if isinstance(a_log, torch.Tensor) else torch.tensor(float(a_log)))
    g = compute_grad(loss_salt, a_lin)
    g_log = (g * a_lin)
    t_adam = (t_adam + 1.0)
    m_adam = ((beta1 * m_adam) + ((1.0 - beta1) * g_log))
    v_adam = ((beta2 * v_adam) + ((1.0 - beta2) * (g_log * g_log)))
    mhat = (m_adam / (1.0 - (beta1 ** t_adam)))
    vhat = (v_adam / (1.0 - (beta2 ** t_adam)))
    a_log = (a_log - ((learning_rate * mhat) / (torch.sqrt(vhat if isinstance(vhat, torch.Tensor) else torch.tensor(float(vhat))) + eps_adam)))
a_final = torch.exp(a_log if isinstance(a_log, torch.Tensor) else torch.tensor(float(a_log)))
print(print(a_final))