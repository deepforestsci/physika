import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import compute_grad

# === Functions ===
def L(t):
    t = torch.as_tensor(t).float()
    return torch.where(t > 0.5, torch.as_tensor((((3.0 * (t - 0.75)) * (t - 0.75)) + 0.1)).float(), torch.as_tensor(((t * t) + 2.0)).float())

# === Program ===
t0 = 0.9
physika_print(L(t0))
physika_print(compute_grad(L, t0))
t1 = (t0 - (0.2 * compute_grad(L, t0)))
physika_print(L(t1))
physika_print(compute_grad(L, t1))
t2 = (t1 - (0.2 * compute_grad(L, t1)))
physika_print(L(t2))
physika_print(compute_grad(L, t2))
t3 = (t2 - (0.2 * compute_grad(L, t2)))
physika_print(L(t3))
s0 = 0.3
physika_print(L(s0))
physika_print(compute_grad(L, s0))
s1 = (s0 - (0.2 * compute_grad(L, s0)))
physika_print(L(s1))
physika_print(compute_grad(L, s1))
s2 = (s1 - (0.2 * compute_grad(L, s1)))
physika_print(L(s2))