import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import simulate

# === Functions ===
def pendulum(x):
    return torch.stack([torch.as_tensor(x[int(1.0)]).float(), torch.as_tensor((0.0 - ((9.81 / 1.0) * torch.sin(x[int(0.0)])))).float()])

# === Classes ===
class RK4(nn.Module):
    def __init__(self, f, dt, n):
        super().__init__()
        self.f = f
        self.dt = nn.Parameter(torch.tensor(dt).float() if not isinstance(dt, torch.Tensor) else dt.clone().detach().float())
        self.n = n

    def forward(self, x):
        x = torch.as_tensor(x).float()
        n = int(self.n) if hasattr(self, 'n') else self.n.shape[0] if hasattr(self.n, 'shape') else 2
        for k in range(n):
            k1 = self.f(x)
            k2 = self.f((x + ((0.5 * self.dt) * k1)))
            k3 = self.f((x + ((0.5 * self.dt) * k2)))
            k4 = self.f((x + (self.dt * k3)))
            x = (x + ((self.dt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))
        return x

# === Program ===
dt = 0.01
solver = RK4(pendulum, dt, 1000.0)
physika_print(solver(torch.tensor([0.5, 0.0])))
physika_print(solver(torch.tensor([1.0, 0.0])))
step = RK4(pendulum, dt, 1.0)
simulate(step, torch.tensor([0.5, 0.0]), 1000.0, dt)