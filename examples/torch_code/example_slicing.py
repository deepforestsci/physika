
=== Physika generated Pytorch code ===
import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def update(y):
    z = y[:, int(2)]
    return y

# === Classes ===
class A(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = torch.as_tensor(x).float()
        self.learnable_params = [self.x]

    def abe(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = x[:, int(2)]
        return z

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
x = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], device=DEVICE)
physika_print(x[1:5])
physika_print(x[:4])
physika_print(x[:])
physika_print(x[2:])
physika_print(x[int((-1))])
y = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], device=DEVICE)
physika_print(y[:1, :2])
physika_print(y[1:3, 1:3])
physika_print(y[:, int(2)])
physika_print(y[int(1), :])
physika_print(y[1:, 1:3])
physika_print(y[:, :])
z = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], device=DEVICE)
physika_print(z[int(0), :, :])
physika_print(z[:, int(1), :])
physika_print(z[:, :, int(1)])
physika_print(z[int(1), :, 1:])
physika_print(z[:, :, :])
physika_print(update(y))
obj = A(1.0)
physika_print(obj.abe(y))
=== End Pytorch code ===

[20, 30, 40, 50] ∈ ℝ[4]
[10, 20, 30, 40] ∈ ℝ[4]
[10, 20, 30, 40, 50, 60, 70, 80] ∈ ℝ[8]
[30, 40, 50, 60, 70, 80] ∈ ℝ[6]
80 ∈ ℝ
[[1, 2]] ∈ ℝ[1,2]
[[6, 7], [10, 11]] ∈ ℝ[2,2]
[3, 7, 11, 15] ∈ ℝ[4]
[5, 6, 7, 8] ∈ ℝ[4]
[[6, 7], [10, 11], [14, 15]] ∈ ℝ[3,2]
[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]] ∈ ℝ[4,4]
[[1, 2, 3], [4, 5, 6]] ∈ ℝ[2,3]
[[4, 5, 6], [10, 11, 12]] ∈ ℝ[2,3]
[[2, 5], [8, 11]] ∈ ℝ[2,2]
[[8, 9], [11, 12]] ∈ ℝ[2,2]
[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]] ∈ ℝ[2,2,3]
[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]] ∈ ℝ[4,4]
[3.0, 7.0, 11.0, 15.0] ∈ ℝ[4]
