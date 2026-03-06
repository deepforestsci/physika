import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import compute_grad

# === Functions ===
def f(x):
    if x > 0.0:
        return (x * x)
    else:
        return (-x)

# === Program ===
a = torch.tensor(3.0, requires_grad=True)
physika_print(f(a))
physika_print(compute_grad(f(a), a))
b = torch.tensor((-2.0), requires_grad=True)
physika_print(f(b))
physika_print(compute_grad(f(b), b))