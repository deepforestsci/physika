import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def fact(n):
    return (1.0 if (n == 0.0) else (n * fact((n - 1.0))))

# === Program ===
print(fact(0.0))
print(fact(1.0))
print(fact(2.0))
print(fact(3.0))
print(fact(4.0))
print(fact(5.0))
print(fact(10.0))