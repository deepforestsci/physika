import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def sample_normal1D(x):
    t = torch.distributions.Normal(0, 1).rsample((int(x),))
    return t

def sample_normal2D(x):
    t = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)) for _fi_i in range(int(x)) for i in [torch.tensor(float(_fi_i))]])
    return t

# === Program ===
m_true = 3.0
b_true = 1.0
m_hat = 2.0
b_hat = 0.0
μ = torch.tensor(2.0, requires_grad=True)
σ = torch.tensor(0.5, requires_grad=True)
n = int(100)
x = torch.distributions.Normal(μ, σ).rsample((int(n),))
y_true = ((m_true * x) + b_true)
y_pred = ((m_hat * x) + b_hat)
loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
physika_print(loss)
grad_mu = compute_grad(loss, μ)
grad_sigma = compute_grad(loss, σ)
physika_print(grad_mu)
physika_print(grad_sigma)
lr = 0.01
for step in range(int(0), int(300)):
    x = torch.distributions.Normal(μ, σ).rsample((int(n),))
    y_true = ((m_true * x) + b_true)
    y_pred = ((m_hat * x) + b_hat)
    loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
    grad_mu = compute_grad(loss, μ)
    grad_sigma = compute_grad(loss, σ)
    μ = (μ - (lr * grad_mu))
    σ = (σ - (lr * grad_sigma))
physika_print(μ)
physika_print(σ)
x = torch.distributions.Normal(μ, σ).rsample((int(n),))
y_true = ((m_true * x) + b_true)
y_pred = ((m_hat * x) + b_hat)
loss = (torch.sum(((y_pred - y_true) ** 2) if isinstance(((y_pred - y_true) ** 2), torch.Tensor) else torch.tensor(float(((y_pred - y_true) ** 2)))) / n)
physika_print(loss)
z = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)) for _fi_i in range(int(10)) for i in [torch.tensor(float(_fi_i))]])
physika_print(z)
z_3d = torch.stack([torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)) for _fi_j in range(int(5)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(10)) for i in [torch.tensor(float(_fi_i))]])
physika_print(z_3d)
x = 3
y = sample_normal1D(x)
physika_print(y)
y = sample_normal2D(x)
physika_print(y)
u = torch.distributions.Normal(0.0, 1.0).rsample()
physika_print(u)
u_vec = torch.distributions.Normal(0.0, 1.0).rsample((int(5),))
physika_print(u_vec)
v = torch.distributions.Uniform(0.0, 1.0).rsample()
physika_print(v)
v_vec = torch.distributions.Uniform(0.0, 1.0).rsample((int(5),))
physika_print(v_vec)
α = 2.0
β_b = 5.0
bs = torch.distributions.Beta(α, β_b).rsample()
physika_print(bs)
bs_vec = torch.distributions.Beta(α, β_b).rsample((int(5),))
physika_print(bs_vec)
conc = 2.0
rate = 1.0
gs = torch.distributions.Gamma(conc, rate).rsample()
physika_print(gs)
gs_vec = torch.distributions.Gamma(conc, rate).rsample((int(5),))
physika_print(gs_vec)
p = 0.5
coin = torch.distributions.Bernoulli(p).sample().detach()
physika_print(coin)
coin_vec = torch.distributions.Bernoulli(p).sample((int(10),)).detach()
physika_print(coin_vec)