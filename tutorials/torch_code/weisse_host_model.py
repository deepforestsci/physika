import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def f(y, vt, Kt, s):
    si = y[int(0)]
    a = y[int(1)]
    r = y[int(2)]
    et = y[int(3)]
    em = y[int(4)]
    q = y[int(5)]
    mt = y[int(6)]
    mm = y[int(7)]
    mq = y[int(8)]
    mr = y[int(9)]
    ct = y[int(10)]
    cm = y[int(11)]
    cq = y[int(12)]
    cr = y[int(13)]
    gam = ((gmax * a) / (Kgamma + a))
    nu_t = ((gam * ct) / nx)
    nu_m = ((gam * cm) / nx)
    nu_q = ((gam * cq) / nx)
    nu_r = ((gam * cr) / nr)
    lam = ((gam * (((ct + cm) + cq) + cr)) / M)
    nu_imp = (((et * vt) * s) / (Kt + s))
    nu_cat = (((em * vm) * si) / (Km + si))
    w_r = ((wr * a) / (thetar + a))
    w_t = ((we * a) / (thetax + a))
    w_m = ((we * a) / (thetax + a))
    w_q = (((wq * a) / (thetax + a)) / (1.0 + ((q / Kq) ** 4.0)))
    dsi = ((nu_imp - nu_cat) - (lam * si))
    da = (((ns * nu_cat) - ((nr * nu_r) + (nx * ((nu_t + nu_m) + nu_q)))) - (lam * a))
    dr = ((((nu_r - (lam * r)) + (((nu_r + nu_t) + nu_m) + nu_q)) - ((kb * r) * (((mr + mt) + mm) + mq))) + (ku * (((cr + ct) + cm) + cq)))
    det = (nu_t - (lam * et))
    dem = (nu_m - (lam * em))
    dq = (nu_q - (lam * q))
    dmt = (((((w_t - ((kb * mt) * r)) + (ku * ct)) + nu_t) - (dm * mt)) - (lam * mt))
    dmm = (((((w_m - ((kb * mm) * r)) + (ku * cm)) + nu_m) - (dm * mm)) - (lam * mm))
    dmq = (((((w_q - ((kb * mq) * r)) + (ku * cq)) + nu_q) - (dm * mq)) - (lam * mq))
    dmr = (((((w_r - ((kb * mr) * r)) + (ku * cr)) + nu_r) - (dm * mr)) - (lam * mr))
    dct = (((((kb * mt) * r) - (ku * ct)) - nu_t) - (lam * ct))
    dcm = (((((kb * mm) * r) - (ku * cm)) - nu_m) - (lam * cm))
    dcq = (((((kb * mq) * r) - (ku * cq)) - nu_q) - (lam * cq))
    dcr = (((((kb * mr) * r) - (ku * cr)) - nu_r) - (lam * cr))
    return torch.stack([torch.as_tensor(dsi), torch.as_tensor(da), torch.as_tensor(dr), torch.as_tensor(det), torch.as_tensor(dem), torch.as_tensor(dq), torch.as_tensor(dmt), torch.as_tensor(dmm), torch.as_tensor(dmq), torch.as_tensor(dmr), torch.as_tensor(dct), torch.as_tensor(dcm), torch.as_tensor(dcq), torch.as_tensor(dcr)])

def growth(y):
    a = y[int(1)]
    return ((((gmax * a) / (Kgamma + a)) * (((y[int(10)] + y[int(11)]) + y[int(12)]) + y[int(13)])) / M)

def steady(vt, Kt, s, y_init):
    z = torch.log(y_init if isinstance(y_init, torch.Tensor) else torch.tensor(float(y_init)))
    zk = z
    y = y_init
    yk = y_init
    fk = y_init
    w = y_init
    d = y_init
    J = Eye
    A = Eye
    dt = 0.05
    for k in range(int(0), int(30)):
        zk = z
        y = torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))
        for j in range(int(0), int(4)):
            yk = torch.exp(zk if isinstance(zk, torch.Tensor) else torch.tensor(float(zk)))
            fk = f(yk, vt, Kt, s)
            J = compute_grad(lambda _dyk: f(_dyk, vt, Kt, s), yk)
            A = (Eye - (dt * J))
            w = gaussian_solve(A, ((yk - y) - (dt * fk)))
            d = (w / yk)
            zk = (zk - (d / (1.0 + torch.abs(d if isinstance(d, torch.Tensor) else torch.tensor(float(d))))))
        z = zk
        dt = (dt * 1.6)
    return torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))

def loss(θ):
    vt = torch.exp(θ[int(0)] if isinstance(θ[int(0)], torch.Tensor) else torch.tensor(float(θ[int(0)])))
    Kt = torch.exp(θ[int(1)] if isinstance(θ[int(1)], torch.Tensor) else torch.tensor(float(θ[int(1)])))
    e1 = ((growth(steady(vt, Kt, 500.0, yss)) - lt1) / lt1)
    e2 = ((growth(steady(vt, Kt, 1500.0, yss)) - lt2) / lt2)
    e3 = ((growth(steady(vt, Kt, 5000.0, yss)) - lt3) / lt3)
    e4 = ((growth(steady(vt, Kt, 20000.0, yss)) - lt4) / lt4)
    e5 = ((growth(steady(vt, Kt, 100000.0, yss)) - lt5) / lt5)
    return ((((((e1 * e1) + (e2 * e2)) + (e3 * e3)) + (e4 * e4)) + (e5 * e5)) / 5.0)

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_cols(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return get_1d_array_length(x[int(0)])

def max(x, y):
    if x > y:
        return x
    else:
        return y

def min(x, y):
    if x < y:
        return x
    else:
        return y

def gaussian_solve(A, b):
    a_row = get_2d_array_num_rows(A)
    a_col = get_2d_array_num_cols(A)
    new_col = (a_col + 1)
    aug = torch.zeros(int(a_row), int(new_col), device=DEVICE)
    for i in range(int(0), int(a_row)):
        aug[int(i), :int(a_col)] = A[int(i), :]
        aug[int(i), int(a_col)] = b[int(i)]
    for i in range(int(0), int(a_row)):
        max_row = i
        for k in range(int((i + 1)), int(a_row)):
            if torch.abs(aug[int(k), int(i)] if isinstance(aug[int(k), int(i)], torch.Tensor) else torch.tensor(float(aug[int(k), int(i)]))) > torch.abs(aug[int(max_row), int(i)] if isinstance(aug[int(max_row), int(i)], torch.Tensor) else torch.tensor(float(aug[int(max_row), int(i)]))):
                max_row = k
        if max_row != i:
            lo = min(i, max_row)
            hi = max(i, max_row)
            rows_above = aug[:int(lo), :]
            pivot_row = aug[int(hi):int((hi + 1)), :]
            rows_between = aug[int((lo + 1)):int(hi), :]
            current_row = aug[int(lo):int((lo + 1)), :]
            rows_below = aug[int((hi + 1)):, :]
            aug = torch.cat([rows_above, pivot_row, rows_between, current_row, rows_below])
        pivot_row = aug[int(i):int((i + 1)), :]
        pivot_value = aug[int(i), int(i)]
        rows_below = aug[int((i + 1)):, :]
        elimination_factors = (rows_below[:, int(i):int((i + 1))] / pivot_value)
        eliminated_rows = (rows_below - (elimination_factors * pivot_row))
        aug = torch.cat([aug[:int((i + 1)), :], eliminated_rows])
    x = torch.zeros(int(0), device=DEVICE)
    for i in range(int(0), int(a_row)):
        idx = ((a_col - 1) - i)
        total = aug[int(idx), int(a_col)]
        for j in range(int((idx + 1)), int(a_row)):
            total = (total - (aug[int(idx), int(j)] * x[int(((j - idx) - 1))]))
        val = (total / aug[int(idx), int(idx)])
        val = torch.reshape(val, (1,))
        x = torch.cat([val, x])
    return x

# === Program ===
dm = 0.1
kb = 1.0
ku = 1.0
gmax = 1260.0
Kp = 180.137803092828
Kgamma = (gmax / Kp)
vm = 5800.0
Km = 1000.0
ns = 0.5
nr = 7549.0
nx = 300.0
wr = 929.967887456483
we = 4.13917218782445
wq = 948.93498829479
thetar = 426.869333896869
thetax = 4.37973339483464
Kq = 152219.040373749
M = 100000000.0
Eye = torch.zeros(int(14), int(14), device=DEVICE)
for k in range(int(0), int(14)):
    Eye[int(k), int(k)] = 1.0
y0 = torch.tensor([500.0, 100.0, 1000.0, 1000.0, 1000.0, 200000.0, 50.0, 50.0, 5000.0, 500.0, 100.0, 100.0, 500.0, 1000.0], device=DEVICE)
yss = steady(726.0, 1000.0, 10000.0, y0)
print(print(growth(yss)))
lt1 = growth(steady(726.0, 1000.0, 500.0, yss))
lt2 = growth(steady(726.0, 1000.0, 1500.0, yss))
lt3 = growth(steady(726.0, 1000.0, 5000.0, yss))
lt4 = growth(steady(726.0, 1000.0, 20000.0, yss))
lt5 = growth(steady(726.0, 1000.0, 100000.0, yss))
print(print(torch.stack([torch.as_tensor(lt1), torch.as_tensor(lt2), torch.as_tensor(lt3), torch.as_tensor(lt4), torch.as_tensor(lt5)])))
θ = torch.as_tensor(torch.stack([torch.as_tensor(torch.log(400.0 if isinstance(400.0, torch.Tensor) else torch.tensor(float(400.0)))), torch.as_tensor(torch.log(3000.0 if isinstance(3000.0, torch.Tensor) else torch.tensor(float(3000.0))))])).requires_grad_(True).to(DEVICE)
print(print(loss(θ)))
learning_rate = 0.05
beta1 = 0.9
beta2 = 0.999
eps_adam = 1e-08
m_adam = torch.tensor([0.0, 0.0], device=DEVICE)
v_adam = torch.tensor([0.0, 0.0], device=DEVICE)
t_adam = 0.0
epochs = 1
for ep in range(int(0), int(epochs)):
    g = compute_grad(loss, θ)
    t_adam = (t_adam + 1.0)
    m_adam = ((beta1 * m_adam) + ((1.0 - beta1) * g))
    v_adam = ((beta2 * v_adam) + ((1.0 - beta2) * (g * g)))
    mhat = (m_adam / (1.0 - (beta1 ** t_adam)))
    vhat = (v_adam / (1.0 - (beta2 ** t_adam)))
    θ = (θ - ((learning_rate * mhat) / (torch.sqrt(vhat if isinstance(vhat, torch.Tensor) else torch.tensor(float(vhat))) + eps_adam)))
    print(θ)
print(print(loss(θ)))