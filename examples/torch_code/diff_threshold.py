import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def L(t):
    if t > 0.5:
        return ((3 * ((t - 0.75) ** 2)) + 0.1)
    else:
        return ((t ** 2) + 2)

# === Program ===
t0 = torch.tensor(0.9, requires_grad=True)
physika_print(L(t0))
physika_print(compute_grad(lambda _dt0: L(_dt0), t0))
t1 = torch.tensor((t0 - (0.2 * compute_grad(lambda _dt0: L(_dt0), t0))), requires_grad=True)
physika_print(L(t1))
physika_print(compute_grad(lambda _dt1: L(_dt1), t1))
t2 = torch.tensor((t1 - (0.2 * compute_grad(lambda _dt1: L(_dt1), t1))), requires_grad=True)
physika_print(L(t2))
physika_print(compute_grad(lambda _dt2: L(_dt2), t2))
t3 = torch.tensor((t2 - (0.2 * compute_grad(lambda _dt2: L(_dt2), t2))), requires_grad=True)
physika_print(L(t3))
physika_print(compute_grad(lambda _dt3: L(_dt3), t3))
physika_print(t3)
s0 = torch.tensor(0.3, requires_grad=True)
physika_print(L(s0))
physika_print(compute_grad(lambda _ds0: L(_ds0), s0))
s1 = torch.tensor((s0 - (0.2 * compute_grad(lambda _ds0: L(_ds0), s0))), requires_grad=True)
physika_print(L(s1))
physika_print(compute_grad(lambda _ds1: L(_ds1), s1))
s2 = torch.tensor((s1 - (0.2 * compute_grad(lambda _ds1: L(_ds1), s1))), requires_grad=True)
physika_print(L(s2))
physika_print(compute_grad(lambda _ds2: L(_ds2), s2))
s3 = torch.tensor((s2 - (0.2 * compute_grad(lambda _ds2: L(_ds2), s2))), requires_grad=True)
physika_print(L(s3))
physika_print(compute_grad(lambda _ds3: L(_ds3), s3))
s4 = torch.tensor((s3 - (0.2 * compute_grad(lambda _ds3: L(_ds3), s3))), requires_grad=True)
physika_print(L(s4))
physika_print(compute_grad(lambda _ds4: L(_ds4), s4))
physika_print(s4)