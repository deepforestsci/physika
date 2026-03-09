import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import compute_grad

# === Functions ===
def f_with_grad(x):
    g = compute_grad(x, x)
    return g

# === Program ===
v = torch.tensor(3.0, requires_grad=True)
physika_print(f_with_grad(v))