import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def U(x):
    return ((((0.5 * 4.0) * x) * x) - ((2.0 * 9.8) * x))

# === Program ===
x = torch.tensor(0.0, requires_grad=True)
for i in range(int(0), int(60)):
    x = (x - (0.05 * compute_grad(lambda _dx: U(_dx), x)))
physika_print(x)