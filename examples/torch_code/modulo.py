import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
print(print((10 % 2)))
print(print((12 % 5)))
a = 10
b = 4
print(print((a % b)))
a = torch.tensor([10, 12, 15, 17], device=DEVICE)
b = 4
print(print((a % b)))
a = torch.tensor([10, 12, 15, 17], device=DEVICE)
b = torch.tensor([1, 3, 5, 7], device=DEVICE)
print(print((a % b)))
