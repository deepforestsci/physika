import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def transform(M):
    return ((M * scale) + 1.0)

# === Program ===
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cpu')
physika_print(A)
B = (2.0 * A)
physika_print(B)
C = (A + B)
physika_print(C)
scale = 2.0
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cpu')
result = transform(X)
physika_print(result)
a = torch.tensor([1.0, 2.0, 3.0], device='cpu')
b = torch.tensor([4.0, 5.0, 6.0], device='cpu')
result = (a @ b)
physika_print(result)
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cpu')
x = torch.tensor([[1.0], [2.0], [3.0]], device='cpu')
result = (A @ x)
physika_print(result)