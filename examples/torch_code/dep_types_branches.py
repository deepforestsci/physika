import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def is_same_order(u, v, a=None, b=None, m=None, n=None):
    if a is None:
        a = int(u.shape[0])
    if b is None:
        b = int(u.shape[1])
    if m is None:
        m = int(v.shape[0])
    if n is None:
        n = int(v.shape[1])
    return ((1 if (b == n) else 0) if (a == m) else 0)

def is_same_order2(u, v, a=None, b=None, m=None, n=None):
    if a is None:
        a = int(u.shape[0])
    if b is None:
        b = int(u.shape[1])
    if m is None:
        m = int(v.shape[0])
    if n is None:
        n = int(v.shape[1])
    o1 = 0
    o2 = 0
    o1 = (1 if (a == m) else o1)
    o2 = (1 if (b == n) else o2)
    return (1 if ((o1 + o2) == 2) else 0)

# === Program ===
A = torch.tensor([[2, 1], [7, 6], [4, 5]], device=DEVICE)
B = torch.tensor([[2, 1], [7, 6], [4, 5]], device=DEVICE)
C = torch.tensor([[2, 1], [7, 6]], device=DEVICE)
print(is_same_order(A, B, 3, 2, 3, 2))
print(is_same_order(A, C, 3, 2, 2, 2))
print(is_same_order2(A, B, 3, 2, 3, 2))
print(is_same_order2(A, C, 3, 2, 2, 2))