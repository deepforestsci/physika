import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import compute_grad

# === Functions ===
def f(x):
    x = torch.as_tensor(x).float()
    return torch.where(x > 0.0, torch.as_tensor(torch.cos(x)).float(), torch.as_tensor(torch.sin(x)).float())

# === Program ===
x0 = (-1.5)
physika_print(f(x0))
physika_print(compute_grad(f, x0))
x1 = (-0.5)
physika_print(f(x1))
physika_print(compute_grad(f, x1))
x2 = 0.5
physika_print(f(x2))
physika_print(compute_grad(f, x2))
x3 = 1.5
physika_print(f(x3))
physika_print(compute_grad(f, x3))
x4 = 3.14
physika_print(f(x4))
physika_print(compute_grad(f, x4))