import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def slice_demo_1d(a):
    return a[:2]

def slice_demo_2d(a):
    return a[:, int(0)]

def slice_demo_3d(a):
    return a[:, int(0), int(0)]

# === Classes ===
class SliceDemo(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.learnable_params = []

    def return_x(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        return x[:4]

    def return_y(self, y):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        return y[:, int(1)]

    def return_z(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        return z[:, int(1), int(0)]

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=DEVICE)
y = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], device=DEVICE)
z = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=DEVICE)
physika_print(x[1:3])
physika_print(x[:4])
physika_print(y[1:3, :])
physika_print(y[:, int(2)])
physika_print(z[int(1), :, :])
physika_print(z[:, int(0), int(0)])
physika_print(slice_demo_1d(x))
physika_print(slice_demo_2d(y))
physika_print(slice_demo_3d(z))
obj_slice_demo = SliceDemo().to(DEVICE)
physika_print(obj_slice_demo.return_x(x))
physika_print(obj_slice_demo.return_y(y))
physika_print(obj_slice_demo.return_z(z))
x[1:3] = torch.tensor([10, 20], device=DEVICE)
y[:, int(0)] = torch.tensor([10, 20, 30, 40], device=DEVICE)
z[:, :, int(0)] = torch.tensor([10, 20], device=DEVICE)
physika_print(x)
physika_print(y)
physika_print(z)