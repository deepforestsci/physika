import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def f(x):
    if x > 0.0:
        return torch.cos(x)
    else:
        return torch.sin(x)

# === Program ===
x0 = torch.tensor((-1.5), requires_grad=True)
physika_print(f(x0))
physika_print(compute_grad(f(x0), x0))
x1 = torch.tensor((-0.5), requires_grad=True)
physika_print(f(x1))
physika_print(compute_grad(f(x1), x1))
x2 = torch.tensor(0.5, requires_grad=True)
physika_print(f(x2))
physika_print(compute_grad(f(x2), x2))
x3 = torch.tensor(1.5, requires_grad=True)
physika_print(f(x3))
physika_print(compute_grad(f(x3), x3))
x4 = torch.tensor(3.14, requires_grad=True)
physika_print(f(x4))
physika_print(compute_grad(f(x4), x4))