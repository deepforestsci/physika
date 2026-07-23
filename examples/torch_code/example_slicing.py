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
x_slice_1_to_3 = x[1:3]
x_slice_start_to_4 = x[:4]
y_rows_1_to_3 = y[1:3, :]
y_column_2 = y[:, int(2)]
z_layer_1 = z[int(1), :, :]
z_first_element_each_layer = z[:, int(0), int(0)]
physika_print(x_slice_1_to_3)
physika_print(x_slice_start_to_4)
physika_print(y_rows_1_to_3)
physika_print(y_column_2)
physika_print(z_layer_1)
physika_print(z_first_element_each_layer)
func_x = slice_demo_1d(x)
func_y = slice_demo_2d(y)
func_z = slice_demo_3d(z)
physika_print(func_x)
physika_print(func_y)
physika_print(func_z)
obj_slice_demo = SliceDemo().to(DEVICE)
class_x = obj_slice_demo.return_x(x)
class_y = obj_slice_demo.return_y(y)
class_z = obj_slice_demo.return_z(z)
physika_print(class_x)
physika_print(class_y)
physika_print(class_z)
x_assign = x
y_assign = y
z_assign = z
x_assign[1:3] = torch.tensor([10, 20], device=DEVICE)
y_assign[:, int(0)] = torch.tensor([10, 20, 30, 40], device=DEVICE)
z_assign[:, :, int(0)] = torch.tensor([10, 20], device=DEVICE)
x_after_assignment = x_assign
y_after_assignment = y_assign
z_after_assignment = z_assign
physika_print(x_after_assignment)
physika_print(y_after_assignment)
physika_print(z_after_assignment)
for i in range(int(0), int(2)):
    y[:, int(0)] = torch.tensor([1, 2, 3, 4], device=DEVICE)
for i in range(int(0), int(2)):
    z[:, int(0), int(0)] = torch.tensor([9, 10], device=DEVICE)
loop_y = y
loop_z = z
physika_print(loop_y)
physika_print(loop_z)