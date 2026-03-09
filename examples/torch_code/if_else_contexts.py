import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def f(x):
    if x > 0.0:
        return (x * x)
    else:
        return (-x)

def clamp(x):
    if x > 1.0:
        y = 1.0
    else:
        y = x
    if y < (-1.0):
        y = (-1.0)
    return y

def classify(x):
    if x > 0.5:
        if x > 1.5:
            return 2.0
        else:
            return 1.0
    else:
        if x < (-0.5):
            return (-1.0)
        else:
            return 0.0

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
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor((-1.5), requires_grad=True)
physika_print(f(a))
physika_print(compute_grad(f(a), a))
physika_print(f(b))
physika_print(compute_grad(f(b), b))
physika_print(clamp(3.0))
physika_print(clamp(0.5))
physika_print(clamp((-5.0)))
physika_print(classify(2.0))
physika_print(classify(0.0))
physika_print(classify((-1.0)))
net = PiecewiseNet(0.0)
a = torch.tensor(2.0, requires_grad=True)
physika_print(net(a))
physika_print(compute_grad(net(a), a))
b = torch.tensor((-1.5), requires_grad=True)
physika_print(net(b))
physika_print(compute_grad(net(b), b))
x = 0.3
if x > 0.5:
    y = (3.0 * ((x - 0.75) ** 2.0))
else:
    y = ((x ** 2.0) + 2.0)
physika_print(y)