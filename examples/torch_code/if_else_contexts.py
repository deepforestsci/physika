import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def f(x):
    if x > 0:
        return (x * x)
    else:
        return (-x)

def clamp(x):
    if x > 1:
        y = 1
    else:
        y = x
    if y < (-1):
        y = (-1)
    return y

def classify(x):
    if x > 0.5:
        if x > 1.5:
            return 2
        else:
            return 1
    else:
        if x < (-0.5):
            return (-1)
        else:
            return 0

# === Classes ===
class PiecewiseNet(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = nn.Parameter(torch.as_tensor(threshold))
        self.learnable_params = [self.threshold]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x).float()
        if x > self.threshold:
            y = (x * x)
        else:
            y = (-x)
        return y

    def loss(self, pred, target):
        this = self
        pred = torch.as_tensor(pred).float()
        target = torch.as_tensor(target).float()
        return ((pred - target) ** 2)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor((-1.5), requires_grad=True)
physika_print(f(a))
physika_print(compute_grad(lambda _da: f(_da), a))
physika_print(f(b))
physika_print(compute_grad(lambda _db: f(_db), b))
physika_print(clamp(3))
physika_print(clamp(0.5))
physika_print(clamp((-5)))
physika_print(classify(2))
physika_print(classify(0))
physika_print(classify((-1)))
net = PiecewiseNet(0.0)
a = torch.tensor(2.0, requires_grad=True)
physika_print(net(a))
physika_print(compute_grad(lambda _da: net(_da), a))
b = torch.tensor((-1.5), requires_grad=True)
physika_print(net(b))
physika_print(compute_grad(lambda _db: net(_db), b))
x = 0.3
if x > 0.5:
    y = (3 * ((x - 0.75) ** 2))
else:
    y = ((x ** 2) + 2)
physika_print(y)