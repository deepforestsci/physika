import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def transform(M):
    return ((M * scale) + 1.0)

# === Program ===
A = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)])), torch.as_tensor(torch.stack([torch.as_tensor(4.0), torch.as_tensor(5.0), torch.as_tensor(6.0)]))])
print(A)
B = (2.0 * A)
print(B)
C = (A + B)
print(C)
scale = 2.0
X = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0)])), torch.as_tensor(torch.stack([torch.as_tensor(3.0), torch.as_tensor(4.0)]))])
result = transform(X)
print(result)
a = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)])
b = torch.stack([torch.as_tensor(4.0), torch.as_tensor(5.0), torch.as_tensor(6.0)])
result = (a @ b)
print(result)
A = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)])), torch.as_tensor(torch.stack([torch.as_tensor(4.0), torch.as_tensor(5.0), torch.as_tensor(6.0)]))])
x = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(2.0)])), torch.as_tensor(torch.stack([torch.as_tensor(3.0)]))])
result = torch.matmul(A, x)
print(result)