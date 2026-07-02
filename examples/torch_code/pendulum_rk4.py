import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import simulate

# === Functions ===
def pendulum(x):
    return torch.stack([torch.as_tensor(x[int(1)]), torch.as_tensor((0.0 - ((9.81 / 1.0) * torch.sin(x[int(0)] if isinstance(x[int(0)], torch.Tensor) else torch.tensor(float(x[int(0)]))))))])

# === Classes ===
class RK4(nn.Module):
    def __init__(self, dt):
        super().__init__()
        self.dt = nn.Parameter(torch.as_tensor(dt))
        self.learnable_params = [self.dt]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x).float()
        k1 = pendulum(x)
        k2 = pendulum((x + ((0.5 * dt) * k1)))
        k3 = pendulum((x + ((0.5 * dt) * k2)))
        k4 = pendulum((x + (dt * k3)))
        x = (x + ((dt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))
        return x

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
dt = 0.01
solver = RK4(dt)
physika_print(solver(torch.tensor([0.5, 0.0])))
physika_print(solver(torch.tensor([1.0, 0.0])))
step = RK4(dt)
simulate(step, torch.tensor([0.5, 0.0]), 1000, dt)