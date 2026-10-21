import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def face_speed(a, axis):
    neighbor = torch.roll(a, (-1), axis)
    return (0.5 * ((a + neighbor) + torch.abs((a - neighbor) if isinstance((a - neighbor), torch.Tensor) else torch.tensor(float((a - neighbor))))))

def divergence(w, f, flux_y, ax, ay, dx, dy):
    F = (0.5 * (f + torch.roll(f, (-1), 0)))
    G = (0.5 * (flux_y + torch.roll(flux_y, (-1), 1)))
    F = (F - ((0.5 * ax) * (torch.roll(w, (-1), 0) - w)))
    G = (G - ((0.5 * ay) * (torch.roll(w, (-1), 1) - w)))
    return (((-(F - torch.roll(F, 1, 0))) / dx) - ((G - torch.roll(G, 1, 1)) / dy))

def rhs(q, g, dx, dy):
    h, hu, hv = q[int(0)], q[int(1)], q[int(2)]
    u, v = (hu / h), (hv / h)
    c = torch.sqrt((g * h) if isinstance((g * h), torch.Tensor) else torch.tensor(float((g * h))))
    ax = face_speed((torch.abs(u if isinstance(u, torch.Tensor) else torch.tensor(float(u))) + c), 0)
    ay = face_speed((torch.abs(v if isinstance(v, torch.Tensor) else torch.tensor(float(v))) + c), 1)
    pressure = (((0.5 * g) * h) * h)
    dh, dhu, dhv = divergence(h, hu, hv, ax, ay, dx, dy), divergence(hu, ((hu * u) + pressure), (hv * u), ax, ay, dx, dy), divergence(hv, (hu * v), ((hv * v) + pressure), ax, ay, dx, dy)
    return torch.stack([torch.as_tensor(dh), torch.as_tensor(dhu), torch.as_tensor(dhv)])

def rk3_step(q, g, dx, dy, Δt):
    q1 = (q + (Δt * rhs(q, g, dx, dy)))
    q2 = ((0.75 * q) + (0.25 * (q1 + (Δt * rhs(q1, g, dx, dy)))))
    q_next = ((q / 3.0) + ((2.0 / 3.0) * (q2 + (Δt * rhs(q2, g, dx, dy)))))
    return q_next

def initial_state(Nx, Ny, H, amplitude):
    x = torch.stack([((i + 0.5) / Nx) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    y = torch.stack([((j + 0.5) / Ny) for _fi_j in range(int(Ny)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]])
    pi = 3.141592653589793
    bump = torch.stack([torch.stack([torch.exp((((torch.cos(((2 * pi) * (x[int(i)] - 0.5)) if isinstance(((2 * pi) * (x[int(i)] - 0.5)), torch.Tensor) else torch.tensor(float(((2 * pi) * (x[int(i)] - 0.5))))) + torch.cos(((2 * pi) * (y[int(j)] - 0.5)) if isinstance(((2 * pi) * (y[int(j)] - 0.5)), torch.Tensor) else torch.tensor(float(((2 * pi) * (y[int(j)] - 0.5)))))) - 2.0) / 0.12) if isinstance((((torch.cos(((2 * pi) * (x[int(i)] - 0.5)) if isinstance(((2 * pi) * (x[int(i)] - 0.5)), torch.Tensor) else torch.tensor(float(((2 * pi) * (x[int(i)] - 0.5))))) + torch.cos(((2 * pi) * (y[int(j)] - 0.5)) if isinstance(((2 * pi) * (y[int(j)] - 0.5)), torch.Tensor) else torch.tensor(float(((2 * pi) * (y[int(j)] - 0.5)))))) - 2.0) / 0.12), torch.Tensor) else torch.tensor(float((((torch.cos(((2 * pi) * (x[int(i)] - 0.5)) if isinstance(((2 * pi) * (x[int(i)] - 0.5)), torch.Tensor) else torch.tensor(float(((2 * pi) * (x[int(i)] - 0.5))))) + torch.cos(((2 * pi) * (y[int(j)] - 0.5)) if isinstance(((2 * pi) * (y[int(j)] - 0.5)), torch.Tensor) else torch.tensor(float(((2 * pi) * (y[int(j)] - 0.5)))))) - 2.0) / 0.12)))) for _fi_j in range(int(Ny)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(Nx)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    h = (H + (amplitude * bump))
    zero = (0.0 * h)
    return torch.stack([torch.as_tensor(h), torch.as_tensor(zero), torch.as_tensor(zero)])

# === Program ===
Nx = 24
Ny = 24
H, g = 1.0, 1.0
dx, dy = (1.0 / Nx), (1.0 / Ny)
Δt = 0.003
steps = 40
initial = initial_state(Nx, Ny, H, 0.05)
final = initial
for step in range(int(0), int(steps)):
    final = rk3_step(final, g, dx, dy, Δt)
mean_depth_change = torch.mean((final[int(0)] - initial[int(0)]) if isinstance((final[int(0)] - initial[int(0)]), torch.Tensor) else torch.tensor(float((final[int(0)] - initial[int(0)]))))
print(mean_depth_change)