import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import simulate

# === Functions ===
def spring_pendulum(x):
    return torch.stack([torch.as_tensor(x[int(2.0)]).float(), torch.as_tensor(x[int(3.0)]).float(), torch.as_tensor(((((x[int(0.0)] * x[int(3.0)]) * x[int(3.0)]) - (9.81 * torch.cos(x[int(1.0)]))) - (50.0 * (x[int(0.0)] - 1.0)))).float(), torch.as_tensor((((0.0 - ((2.0 * x[int(2.0)]) * x[int(3.0)])) - (9.81 * torch.sin(x[int(1.0)]))) / x[int(0.0)])).float()])

# === Classes ===
class RK4(nn.Module):
    def __init__(self, f, dt, n):
        super().__init__()
        self.f = f
        self.dt = nn.Parameter(torch.tensor(dt).float() if not isinstance(dt, torch.Tensor) else dt.clone().detach().float())
        self.n = n

    def forward(self, x):
        x = torch.as_tensor(x).float()
        for k in range(self.n):
            k1 = self.f(x)
            k2 = self.f((x + ((0.5 * self.dt) * k1)))
            k3 = self.f((x + ((0.5 * self.dt) * k2)))
            k4 = self.f((x + (self.dt * k3)))
            x = (x + ((self.dt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))
        return x

# === Program ===
dt = 0.01
solver = RK4(spring_pendulum, dt, 10000.0)
physika_print(solver(torch.tensor([1.2, 0.3, 0.0, 0.0])))
physika_print(solver(torch.tensor([1.3, 0.8, 0.0, 0.0])))
step = RK4(spring_pendulum, dt, 1.0)
simulate(step, torch.tensor([1.2, 0.3, 0.0, 0.0]), 500.0, dt)