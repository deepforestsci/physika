import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def mod_scalar(x, y):
    return (x % y)

def mod_twice(x, y):
    r = (x % y)
    return (r % 2)

def mod_tensor(v, k):
    return (v % k)

def wrap_angle(theta):
    return (theta % 6.283185307179586)

def count_odd(x):
    total = 0
    for i in range(len(x)):
        total = total + (x[int(i)] % 2)
    return total

def mod_wrt_dividend(x):
    return (x % 2)

# === Classes ===
class Clock(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = torch.as_tensor(period).float()

    def wrap(self, t):
        this = self
        t = torch.as_tensor(t, device=DEVICE).float()
        return (t % self.period)

    def wrap_all(self, ts):
        this = self
        ts = torch.as_tensor(ts, device=DEVICE).float()
        return (ts % self.period)

# === Program ===
a = 17
b = 5
mod_literals = (17 % 5)
mod_variables = (a % b)
mod_float = (7.5 % 2)
mod_negative = ((-7) % 3)
mod_precedence = (2 + ((7 % 4) * 3))
print(mod_literals)
print(mod_variables)
print(mod_float)
print(mod_negative)
print(mod_precedence)
w = torch.tensor([10, 12, 15, 17], device=DEVICE)
func_scalar = mod_scalar(17, 5)
func_twice = mod_twice(17, 5)
func_tensor = mod_tensor(w, 4)
func_angle = wrap_angle(7.0)
print(func_scalar)
print(func_twice)
print(func_tensor)
print(func_angle)
v = torch.tensor([10, 11, 12, 13, 14, 15], device=DEVICE)
loop_comprehension = torch.stack([(v[int(i)] % 4) for _fi_i in range(int(len(v))) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
loop_count_odd = count_odd(v)
print(loop_comprehension)
print(loop_count_odd)
clock = Clock((36 % 24)).to(DEVICE)
class_wrap = clock.wrap(27)
class_wrap_all = clock.wrap_all(torch.tensor([5, 13, 25], device=DEVICE))
print(class_wrap)
print(class_wrap_all)
s = 10
t = torch.tensor([10, 12, 15, 17], device=DEVICE)
u = torch.tensor([3, 5, 7, 4], device=DEVICE)
A = torch.tensor([[7, 8, 9], [10, 11, 12]], device=DEVICE)
B = torch.tensor([[2, 3, 4], [5, 6, 7]], device=DEVICE)
scalar_mod_scalar = (s % 3)
tensor_mod_scalar = (t % 4)
tensor_mod_tensor = (t % u)
matrix_mod_scalar = (A % 5)
matrix_mod_matrix = (A % B)
print(scalar_mod_scalar)
print(tensor_mod_scalar)
print(tensor_mod_tensor)
print(matrix_mod_scalar)
print(matrix_mod_matrix)
grad_dividend = compute_grad(mod_wrt_dividend, 5.5)
print(grad_dividend)