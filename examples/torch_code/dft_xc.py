import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def lda_x(n):
    cx = (((-3.0) / 4.0) * ((3.0 / (2.0 * π)) ** (2.0 / 3.0)))
    rs = ((3.0 / ((4.0 * π) * n)) ** (1.0 / 3.0))
    ex = (cx / rs)
    vx = ((4.0 / 3.0) * ex)
    return torch.stack([torch.as_tensor(ex), torch.as_tensor(vx)])

def lda_c_chachiyo(n):
    a = (-0.01554535)
    b = 20.4562557
    rs = ((3.0 / ((4.0 * π) * n)) ** (1.0 / 3.0))
    ec = (a * torch.log(((1.0 + (b / rs)) + (b / (rs ** 2.0))) if isinstance(((1.0 + (b / rs)) + (b / (rs ** 2.0))), torch.Tensor) else torch.tensor(float(((1.0 + (b / rs)) + (b / (rs ** 2.0)))))))
    vc = (ec + (((a * b) * (2.0 + rs)) / (3.0 * ((b + (b * rs)) + (rs ** 2.0)))))
    return torch.stack([torch.as_tensor(ec), torch.as_tensor(vc)])

# === Program ===
π = 3.141592653589793