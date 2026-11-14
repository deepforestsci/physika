import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def sample_normal1D(x):
    t = torch.distributions.Normal(0, 1).rsample((int(x),)).to(DEVICE)
    return t

def sample_normal2D(x):
    t = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)).to(DEVICE) for _fi_i in range(int(x)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return t

# === Program ===
m_true = 3.0
b_true = 1.0
m_hat = 2.0
b_hat = 0.0
μ = torch.tensor(2.0, requires_grad=True)
σ = torch.tensor(0.5, requires_grad=True)
n = int(100)
x = torch.distributions.Normal(μ, σ).rsample((int(n),)).to(DEVICE)
y_true = ((m_true * x) + b_true)
y_pred = ((m_hat * x) + b_hat)
loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
print(loss)
grad_mu = compute_grad(loss, μ)
grad_sigma = compute_grad(loss, σ)
print(grad_mu)
print(grad_sigma)
lr = 0.01
for step in range(int(0), int(300)):
    x = torch.distributions.Normal(μ, σ).rsample((int(n),)).to(DEVICE)
    y_true = ((m_true * x) + b_true)
    y_pred = ((m_hat * x) + b_hat)
    loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
    grad_mu = compute_grad(loss, μ)
    grad_sigma = compute_grad(loss, σ)
    μ = (μ - (lr * grad_mu))
    σ = (σ - (lr * grad_sigma))
print(μ)
print(σ)
x = torch.distributions.Normal(μ, σ).rsample((int(n),)).to(DEVICE)
y_true = ((m_true * x) + b_true)
y_pred = ((m_hat * x) + b_hat)
loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
print(loss)
z = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)).to(DEVICE) for _fi_i in range(int(10)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(z)
z_3d = torch.stack([torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)).to(DEVICE) for _fi_j in range(int(5)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(10)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(z_3d)
x = 3
y = sample_normal1D(x)
print(y)
y = sample_normal2D(x)
print(y)
u = torch.distributions.Normal(0.0, 1.0).rsample().to(DEVICE)
print(u)
u_vec = torch.distributions.Normal(0.0, 1.0).rsample((int(5),)).to(DEVICE)
print(u_vec)
v = torch.distributions.Uniform(0.0, 1.0).rsample().to(DEVICE)
print(v)
v_vec = torch.distributions.Uniform(0.0, 1.0).rsample((int(5),)).to(DEVICE)
print(v_vec)
α = 2.0
β_b = 5.0
bs = torch.distributions.Beta(α, β_b).rsample().to(DEVICE)
print(bs)
bs_vec = torch.distributions.Beta(α, β_b).rsample((int(5),)).to(DEVICE)
print(bs_vec)
conc = 2.0
rate = 1.0
gs = torch.distributions.Gamma(conc, rate).rsample().to(DEVICE)
print(gs)
gs_vec = torch.distributions.Gamma(conc, rate).rsample((int(5),)).to(DEVICE)
print(gs_vec)
p = 0.5
coin = torch.distributions.Bernoulli(p).sample().detach().to(DEVICE)
print(coin)
coin_vec = torch.distributions.Bernoulli(p).sample((int(10),)).detach().to(DEVICE)
print(coin_vec)