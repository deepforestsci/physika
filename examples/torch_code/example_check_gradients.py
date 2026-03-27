import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def f(x):
    y = (x[int(0.0)] ** 2.0)
    g = compute_grad(y, x)
    return g

# === Program ===
v = torch.tensor([1.0], requires_grad=True)
physika_print(f(v))