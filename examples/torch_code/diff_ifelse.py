import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import compute_grad

# === Functions ===
def f(x):
    x = torch.as_tensor(x).float()
    return torch.where(x > 0.0, torch.as_tensor((x * x)).float(), torch.as_tensor((0.0 - x)).float())

# === Program ===
a = 3.0
physika_print(f(a))
physika_print(compute_grad(f, a))
b = (-2.0)
physika_print(f(b))
physika_print(compute_grad(f, b))