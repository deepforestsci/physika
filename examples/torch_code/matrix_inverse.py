import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def solve_via_inv(M, b, n=None):
    if n is None:
        n = int(M.shape[0])
    return torch.matmul(torch.linalg.inv(M), b)

def trace_of_inv(M, s):
    C = torch.linalg.inv((M * s))
    return (C[int(0), int(0)] + C[int(1), int(1)])

# === Program ===
A = torch.tensor([[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]], device=DEVICE)
B = torch.tensor([[4.0, 7.0], [2.0, 6.0]], device=DEVICE)
rhs = torch.tensor([[1.0], [2.0]], device=DEVICE)
print(torch.linalg.inv(A))
print((torch.linalg.inv(A) @ A))
print(solve_via_inv(B, rhs, 2))
s0 = torch.tensor(1.0, requires_grad=True)
print(trace_of_inv(B, s0))
print(compute_grad(lambda _ds0: trace_of_inv(B, _ds0), s0))