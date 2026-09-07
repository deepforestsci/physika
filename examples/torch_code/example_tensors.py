import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
v = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)])
print(v)
omega = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)])
print(omega)
T = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(2.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(3.0), torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(4.0)]))])
v = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(2.0)])), torch.as_tensor(torch.stack([torch.as_tensor(3.0)]))])
w = torch.matmul(T, v)
print(w)
T = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(2.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(3.0), torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(4.0)]))])
v = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(2.0)])), torch.as_tensor(torch.stack([torch.as_tensor(3.0)]))])
w = torch.matmul(T, v)
print(w)
print(T)