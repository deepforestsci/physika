import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
r_unicode = 3.14
r_mathbb = 2.5
r_macro = 1.5
r_ascii = 0.5
print(r_unicode)
print(r_mathbb)
print(r_macro)
print(r_ascii)
r_vector = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)])
r_matrix = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0)])), torch.as_tensor(torch.stack([torch.as_tensor(3.0), torch.as_tensor(4.0)]))])
print(r_vector)
print(r_matrix)
z_unicode = int(10)
z_mathbb = int((-4))
z_macro = int(7)
z_ascii = int(42)
print(z_unicode)
print(z_mathbb)
print(z_macro)
print(z_ascii)
z_vector = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3)])
print(z_vector)
n_unicode = 5
n_mathbb = 8
n_macro = 3
n_ascii = 1
print(n_unicode)
print(n_mathbb)
print(n_macro)
print(n_ascii)
n_vector = torch.stack([torch.as_tensor(0), torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3)])
print(n_vector)
c_unicode = torch.tensor((3 + 1j), dtype=torch.complex64)
c_mathbb = torch.tensor((5 + 3j), dtype=torch.complex64)
print(c_unicode)
print(c_mathbb)
c_vector = torch.stack([torch.as_tensor((1 + 2j), dtype=torch.complex64), torch.as_tensor((3 + 4j), dtype=torch.complex64)])
print(c_vector)