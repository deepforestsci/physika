import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
a = int(10)
b = int(3)
z_add = (a + b)
z_array = torch.stack([torch.as_tensor(1), torch.as_tensor(2)])
print(a)
print(b)
print(z_add)
print(z_array)
x = 3.14
y = 2
r_mul = (x * y)
print(x)
print(y)
print(r_mul)
z_number = int(1)
r_number = 2.0
result = (float(z_number) * r_number)
print(result)
neg_int = int((-7))
neg_float = (-3.14)
neg_array = torch.tensor([(-1), (-2.0), (-3)], device=DEVICE)
print(neg_int)
print(neg_float)
print(neg_array)