import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def update_1d_array(x):
    x[int(1)] = 3
    return x

def update_3d_array(x):
    x[int(1), int(1), int(1)] = 3
    return x

def update_2d_array(x):
    x[int(1), int(1)] = 3
    return x

# === Program ===
x = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(5), torch.as_tensor(6), torch.as_tensor(7)])
y = (x[int(0):int(3)] + x[int(0):int(3)])
z = (y + torch.tensor([1, 3, 4], device=DEVICE))
print(x)
print(y)
print(z)
u0 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.75), torch.as_tensor(0.75), torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.75), torch.as_tensor(0.75), torch.as_tensor(10.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)]))])
u00 = u0[int(0)][int(0)]
u01 = u0[int(0)][int((1 + 0))]
u02 = u0[int(0)][int((1 + (1 + 0)))]
u03 = u0[int(0)][int((1 + (1 + (1 + 0))))]
u10 = u0[int((1 + 0))][int(0)]
u11 = u0[int((1 + 0))][int((1 + 0))]
u12 = u0[int((1 + 0))][int((1 + (1 + 0)))]
u13 = u0[int((1 + 0))][int((1 + (1 + (1 + 0))))]
u20 = u0[int((1 + (1 + 0)))][int(0)]
u21 = u0[int((1 + (1 + 0)))][int((1 + 0))]
u22 = u0[int((1 + (1 + 0)))][int((1 + (1 + 0)))]
u23 = u0[int((1 + (1 + 0)))][int((1 + (1 + (1 + 0))))]
u30 = u0[int((1 + (1 + (1 + 0))))][int(0)]
u31 = u0[int((1 + (1 + (1 + 0))))][int((1 + 0))]
u32 = u0[int((1 + (1 + (1 + 0))))][int((1 + (1 + 0)))]
u33 = u0[int((1 + (1 + (1 + 0))))][int((1 + (1 + (1 + 0))))]
A = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(0), torch.as_tensor(0)])), torch.as_tensor(torch.stack([torch.as_tensor(0), torch.as_tensor(2), torch.as_tensor(0)])), torch.as_tensor(torch.stack([torch.as_tensor(0), torch.as_tensor(0), torch.as_tensor(3)]))])
A00 = A[int(0)][int(0)]
A11 = A[int((1 + 0))][int((1 + 0))]
A22 = A[int((1 + (1 + 0)))][int((1 + (1 + 0)))]
T = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(4)])), torch.as_tensor(torch.stack([torch.as_tensor(5), torch.as_tensor(6), torch.as_tensor(7), torch.as_tensor(8)])), torch.as_tensor(torch.stack([torch.as_tensor(9), torch.as_tensor(10), torch.as_tensor(11), torch.as_tensor(12)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(13), torch.as_tensor(14), torch.as_tensor(15), torch.as_tensor(16)])), torch.as_tensor(torch.stack([torch.as_tensor(17), torch.as_tensor(18), torch.as_tensor(19), torch.as_tensor(20)])), torch.as_tensor(torch.stack([torch.as_tensor(21), torch.as_tensor(22), torch.as_tensor(23), torch.as_tensor(24)]))]))])
T0 = T[int(0)]
T12 = T[int((1 + 0))][int((1 + (1 + 0)))]
T000 = T[int(0)][int(0)][int(0)]
T123 = T[int((1 + 0))][int((1 + (1 + 0)))][int((1 + (1 + (1 + 0))))]
T012 = T[int(0)][int((1 + 0))][int((1 + (1 + 0)))]
prog_1d = torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])
prog_2d = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)]))])
prog_3d = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)]))]))])
prog_1d[int(1)] = 2
prog_2d[int(1), int(1)] = 2
prog_3d[int(1), int(1), int(1)] = 2
print(prog_1d)
print(prog_2d)
print(prog_3d)
func_1d = torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])
func_2d = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)]))])
func_3d = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(1.0)]))]))])
print(update_1d_array(func_1d))
print(update_2d_array(func_2d))
print(update_3d_array(func_3d))