import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def f(x):
    return ((x ** 2) + 1)

# === Program ===
α = 1.0
β = 2.0
x = 100000.0
y = 300000.0
results = (α + β)
physika_print(results)
z = (x + y)
physika_print(z)
greek_letters_array = torch.stack([torch.as_tensor(α).float(), torch.as_tensor(β).float()])
physika_print(greek_letters_array)
μ = torch.as_tensor(torch.tensor([2])).float().requires_grad_(True)
physika_print(compute_grad(f, μ))