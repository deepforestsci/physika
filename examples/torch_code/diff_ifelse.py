import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def f(x):
    return ((x * x) if (x > 0.0) else (-x))

# === Program ===
a = torch.tensor(3.0, requires_grad=True)
print(f(a))
print(compute_grad(lambda _da: f(_da), a))
b = torch.tensor((-2.0), requires_grad=True)
print(f(b))
print(compute_grad(lambda _db: f(_db), b))