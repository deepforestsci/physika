import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE
import re

from physika.runtime import print
from physika.runtime import solve
from physika.runtime import compute_grad

# === Functions ===
def bathymetry(θ):
    A, x0, σ = θ[int(0)], θ[int(1)], θ[int(2)]
    return torch.stack([(A * torch.exp(((-((x[int(i)] - x0) ** 2)) / (2.0 * (σ ** 2))) if isinstance(((-((x[int(i)] - x0) ** 2)) / (2.0 * (σ ** 2))), torch.Tensor) else torch.tensor(float(((-((x[int(i)] - x0) ** 2)) / (2.0 * (σ ** 2))))))) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rhs(q, b):
    h, hu = q[int(0)], q[int(1)]
    η = (h + b)
    h_right, hu_right = torch.roll(h, (-1), 0), torch.roll(hu, (-1), 0)
    b_right, η_right = torch.roll(b, (-1), 0), torch.roll(η, (-1), 0)
    z_face = torch.stack([(0.5 * ((b[int(i)] + b_right[int(i)]) + torch.abs((b[int(i)] - b_right[int(i)]) if isinstance((b[int(i)] - b_right[int(i)]), torch.Tensor) else torch.tensor(float((b[int(i)] - b_right[int(i)])))))) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    h_left_face = torch.stack([(0.5 * ((η[int(i)] - z_face[int(i)]) + torch.abs((η[int(i)] - z_face[int(i)]) if isinstance((η[int(i)] - z_face[int(i)]), torch.Tensor) else torch.tensor(float((η[int(i)] - z_face[int(i)])))))) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    h_right_face = torch.stack([(0.5 * ((η_right[int(i)] - z_face[int(i)]) + torch.abs((η_right[int(i)] - z_face[int(i)]) if isinstance((η_right[int(i)] - z_face[int(i)]), torch.Tensor) else torch.tensor(float((η_right[int(i)] - z_face[int(i)])))))) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    u = (hu / h)
    u_right = (hu_right / h_right)
    a = torch.stack([(0.5 * ((((torch.abs(u[int(i)] if isinstance(u[int(i)], torch.Tensor) else torch.tensor(float(u[int(i)]))) + torch.sqrt((g * h_left_face[int(i)]) if isinstance((g * h_left_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_left_face[int(i)]))))) + torch.abs(u_right[int(i)] if isinstance(u_right[int(i)], torch.Tensor) else torch.tensor(float(u_right[int(i)])))) + torch.sqrt((g * h_right_face[int(i)]) if isinstance((g * h_right_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_right_face[int(i)]))))) + torch.abs((((torch.abs(u[int(i)] if isinstance(u[int(i)], torch.Tensor) else torch.tensor(float(u[int(i)]))) + torch.sqrt((g * h_left_face[int(i)]) if isinstance((g * h_left_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_left_face[int(i)]))))) - torch.abs(u_right[int(i)] if isinstance(u_right[int(i)], torch.Tensor) else torch.tensor(float(u_right[int(i)])))) - torch.sqrt((g * h_right_face[int(i)]) if isinstance((g * h_right_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_right_face[int(i)]))))) if isinstance((((torch.abs(u[int(i)] if isinstance(u[int(i)], torch.Tensor) else torch.tensor(float(u[int(i)]))) + torch.sqrt((g * h_left_face[int(i)]) if isinstance((g * h_left_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_left_face[int(i)]))))) - torch.abs(u_right[int(i)] if isinstance(u_right[int(i)], torch.Tensor) else torch.tensor(float(u_right[int(i)])))) - torch.sqrt((g * h_right_face[int(i)]) if isinstance((g * h_right_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_right_face[int(i)]))))), torch.Tensor) else torch.tensor(float((((torch.abs(u[int(i)] if isinstance(u[int(i)], torch.Tensor) else torch.tensor(float(u[int(i)]))) + torch.sqrt((g * h_left_face[int(i)]) if isinstance((g * h_left_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_left_face[int(i)]))))) - torch.abs(u_right[int(i)] if isinstance(u_right[int(i)], torch.Tensor) else torch.tensor(float(u_right[int(i)])))) - torch.sqrt((g * h_right_face[int(i)]) if isinstance((g * h_right_face[int(i)]), torch.Tensor) else torch.tensor(float((g * h_right_face[int(i)])))))))))) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    Fh = ((0.5 * (hu + hu_right)) - ((0.5 * a) * (h_right_face - h_left_face)))
    Fhu = (0.5 * ((((hu * u) + ((0.5 * g) * (h_left_face ** 2))) + (hu_right * u_right)) + ((0.5 * g) * (h_right_face ** 2))))
    Fhu = (Fhu - ((0.5 * a) * (hu_right - hu)))
    Fhu_right = (Fhu + ((0.5 * g) * ((h ** 2) - (h_left_face ** 2))))
    Fhu_left = (torch.roll(Fhu, 1, 0) + ((0.5 * g) * ((h ** 2) - torch.roll((h_right_face ** 2), 1, 0))))
    dh = ((-(Fh - torch.roll(Fh, 1, 0))) / Δx)
    dhu = ((-(Fhu_right - Fhu_left)) / Δx)
    return torch.stack([torch.as_tensor(dh), torch.as_tensor(dhu)])

def rk3_step(q, b):
    q1 = (q + (Δt * rhs(q, b)))
    q2 = ((0.75 * q) + (0.25 * (q1 + (Δt * rhs(q1, b)))))
    return ((q / 3.0) + ((2.0 / 3.0) * (q2 + (Δt * rhs(q2, b)))))

def initial_state(b):
    η0 = torch.stack([(1.0 + (0.025 * torch.exp(((-((x[int(i)] - 0.25) ** 2)) / 0.008) if isinstance(((-((x[int(i)] - 0.25) ** 2)) / 0.008), torch.Tensor) else torch.tensor(float(((-((x[int(i)] - 0.25) ** 2)) / 0.008)))))) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    h0 = (η0 - b)
    zero = (0.0 * h0)
    return torch.stack([torch.as_tensor(h0), torch.as_tensor(zero)])

def solve(θ):
    b = bathymetry(θ)
    q = initial_state(b)
    for step in range(int(0), int(steps)):
        q = rk3_step(q, b)
    return q

def gauges(q, b):
    η = (q[int(0)] + b)
    return torch.stack([torch.as_tensor(η[int(6)]), torch.as_tensor(η[int(16)]), torch.as_tensor(η[int(26)])])

def loss(θ):
    b = bathymetry(θ)
    predicted = gauges(solve(θ), b)
    return torch.mean(((predicted - gauge_obs) ** 2) if isinstance(((predicted - gauge_obs) ** 2), torch.Tensor) else torch.tensor(float(((predicted - gauge_obs) ** 2))))

# === Program ===
Nx = 32
g = 1.0
Δx = (1.0 / Nx)
Δt = 0.002
steps = 30
x = torch.stack([((i + 0.5) / Nx) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
θ_true = torch.tensor([0.14, 0.58, 0.11], device=DEVICE)
b_true = bathymetry(θ_true)
gauge_obs = gauges(solve(θ_true), b_true)
θ = torch.as_tensor(torch.tensor([0.09, 0.45, 0.14], device=DEVICE)).requires_grad_(True).to(DEVICE)
loss_gradient = compute_grad(loss, θ)
print(loss_gradient)