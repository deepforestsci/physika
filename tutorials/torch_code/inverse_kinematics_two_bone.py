import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def hand(θ, l, n):
    x, y, φ = 0.0, 0.0, 0.0
    for i in range(int(0), int(n)):
        φ = (φ + θ[int(i)])
        x = (x + (l[int(i)] * torch.cos(φ if isinstance(φ, torch.Tensor) else torch.tensor(float(φ)))))
        y = (y + (l[int(i)] * torch.sin(φ if isinstance(φ, torch.Tensor) else torch.tensor(float(φ)))))
    return torch.stack([torch.as_tensor(x), torch.as_tensor(y)])

def L(θ, l, n, target):
    d = (hand(θ, l, n) - target)
    return torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d))))

def clip_reach(t, reach):
    dist = torch.sqrt(torch.sum((t * t) if isinstance((t * t), torch.Tensor) else torch.tensor(float((t * t)))) if isinstance(torch.sum((t * t) if isinstance((t * t), torch.Tensor) else torch.tensor(float((t * t)))), torch.Tensor) else torch.tensor(float(torch.sum((t * t) if isinstance((t * t), torch.Tensor) else torch.tensor(float((t * t)))))))
    return torch.where(torch.as_tensor(dist > reach), (t * (reach / dist)), t)

def closed_form(t, l1, l2):
    r2 = ((t[int(0)] * t[int(0)]) + (t[int(1)] * t[int(1)]))
    ca = ((((l1 * l1) + r2) - (l2 * l2)) / ((2.0 * l1) * torch.sqrt(r2 if isinstance(r2, torch.Tensor) else torch.tensor(float(r2)))))
    cb = ((((l1 * l1) + (l2 * l2)) - r2) / ((2.0 * l1) * l2))
    a = torch.atan2(torch.sqrt(torch.abs((1.0 - (ca * ca)) if isinstance((1.0 - (ca * ca)), torch.Tensor) else torch.tensor(float((1.0 - (ca * ca))))) if isinstance(torch.abs((1.0 - (ca * ca)) if isinstance((1.0 - (ca * ca)), torch.Tensor) else torch.tensor(float((1.0 - (ca * ca))))), torch.Tensor) else torch.tensor(float(torch.abs((1.0 - (ca * ca)) if isinstance((1.0 - (ca * ca)), torch.Tensor) else torch.tensor(float((1.0 - (ca * ca)))))))), ca)
    b = torch.atan2(torch.sqrt(torch.abs((1.0 - (cb * cb)) if isinstance((1.0 - (cb * cb)), torch.Tensor) else torch.tensor(float((1.0 - (cb * cb))))) if isinstance(torch.abs((1.0 - (cb * cb)) if isinstance((1.0 - (cb * cb)), torch.Tensor) else torch.tensor(float((1.0 - (cb * cb))))), torch.Tensor) else torch.tensor(float(torch.abs((1.0 - (cb * cb)) if isinstance((1.0 - (cb * cb)), torch.Tensor) else torch.tensor(float((1.0 - (cb * cb)))))))), cb)
    ψ = torch.atan2(t[int(1)], t[int(0)])
    return torch.stack([torch.as_tensor((ψ - a)), torch.as_tensor((π - b))])

def make_obs(angs, l_true, count):
    obs = torch.zeros(int(count), int(2), device=DEVICE)
    for k in range(int(0), int(count)):
        obs[int(k)] = hand(angs[int(k)], l_true, len(l_true))
    return obs

def fit_loss(guess, angs, obs, count):
    total = 0.0
    for k in range(int(0), int(count)):
        d = (hand(angs[int(k)], guess, len(guess)) - obs[int(k)])
        total = (total + torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d)))))
    return total

# === Program ===
l = torch.tensor([4.0, 4.0], device=DEVICE)
n_bones = len(l)
reach = torch.sum(l if isinstance(l, torch.Tensor) else torch.tensor(float(l)))
target = torch.tensor([5.0, 5.0], device=DEVICE)
θ = torch.as_tensor(torch.tensor([0.1, 0.5], device=DEVICE)).requires_grad_(True).to(DEVICE)
η = 0.005
steps = 1000
J = compute_grad(lambda _dθ: hand(_dθ, l, n_bones), θ)
print(print(J))
d = (hand(θ, l, n_bones) - target)
print(print(compute_grad(lambda _dθ: L(_dθ, l, n_bones, target), θ)))
print(print((2.0 * (d @ J))))
history = torch.zeros(int(steps), int(n_bones), device=DEVICE)
gradients = torch.zeros(int(steps), int(n_bones), device=DEVICE)
for i in range(int(0), int(steps)):
    history[int(i)] = θ
    g = compute_grad(lambda _dθ: L(_dθ, l, n_bones, target), θ)
    gradients[int(i)] = g
    θ = (θ - (η * g))
print(print(θ))
print(print(hand(θ, l, n_bones)))
print(print(target))
print(print(L(θ, l, n_bones, target)))
π = 3.141592653589793
goal = clip_reach(target, reach)
exact = closed_form(goal, l[int(0)], l[int(1)])
print(print(exact))
print(print(hand(exact, l, n_bones)))
angs = torch.tensor([[0.2, 0.6], [0.9, (-0.4)], [1.4, 0.8], [2.0, (-0.7)]], device=DEVICE)
n_obs = len(angs)
l_true = l
obs = make_obs(angs, l_true, n_obs)
l_fit = torch.as_tensor(torch.tensor([3.0, 5.0], device=DEVICE)).requires_grad_(True).to(DEVICE)
η_l = 0.02
length_steps = 500
for s in range(int(0), int(length_steps)):
    g_fit = compute_grad(lambda _dl_fit: fit_loss(_dl_fit, angs, obs, n_obs), l_fit)
    l_fit = (l_fit - (η_l * g_fit))
print(print(l_fit))