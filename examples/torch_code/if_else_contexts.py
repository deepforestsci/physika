import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import compute_grad

# === Functions ===
def f(x):
    x = torch.as_tensor(x).float()
    return torch.where(x > 0.0, torch.as_tensor((x * x)).float(), torch.as_tensor((-x)).float())

def clamp(x):
    x = torch.as_tensor(x).float()
    if x > 1.0:
        y = 1.0
    else:
        y = x
    if y < (-1.0):
        y = (-1.0)
    return y

def classify(x):
    x = torch.as_tensor(x).float()
    if x > 0.5:
        return torch.where(x > 1.5, torch.as_tensor(2.0).float(), torch.as_tensor(1.0).float())
    else:
        return torch.where(x < (-0.5), torch.as_tensor((-1.0)).float(), torch.as_tensor(0.0).float())

# === Classes ===
class PiecewiseNet(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold).float() if not isinstance(threshold, torch.Tensor) else threshold.clone().detach().float())

    def forward(self, x):
        x = torch.as_tensor(x).float()
        if x > self.threshold:
            y = (x * x)
        else:
            y = (-x)
        return y

    def loss(self, pred, target):
        return ((pred - target) ** 2.0)

# === Program ===
a = 2.0
b = (-1.5)
physika_print(f(a))
physika_print(compute_grad(f, a))
physika_print(f(b))
physika_print(compute_grad(f, b))
physika_print(clamp(3.0))
physika_print(clamp(0.5))
physika_print(clamp((-5.0)))
physika_print(classify(2.0))
physika_print(classify(0.0))
physika_print(classify((-1.0)))
net = PiecewiseNet(0.0)
a = 2.0
physika_print(net(a))
physika_print(compute_grad(net, a))
b = (-1.5)
physika_print(net(b))
physika_print(compute_grad(net, b))
x = 0.3
if x > 0.5:
    y = (3.0 * ((x - 0.75) ** 2.0))
else:
    y = ((x ** 2.0) + 2.0)
physika_print(y)