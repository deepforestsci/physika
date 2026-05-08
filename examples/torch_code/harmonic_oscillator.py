import torch
import torch.nn as nn
import torch.optim as optim
import re

from physika.runtime import physika_print
from physika.runtime import solve
from physika.runtime import compute_grad
from physika.runtime import animate

# === Functions ===
def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i))]])
    return results

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i))]])
    return results

def solve(A, b):
    n = get_1d_array_length(b)
    Aug = zero_2d_array(n, (n + 1))
    result = zero_1d_array(n)
    for i in range(int(0), int(n)):
        for j in range(int(0), int(n)):
            Aug[int(i), int(j)] = A[int(i), int(j)]
        Aug[int(i), int(n)] = b[int(i)]
    for pivot in range(int(0), int(n)):
        for i in range(int((pivot + 1)), int(n)):
            factor = (Aug[int(i), int(pivot)] / Aug[int(pivot), int(pivot)])
            for j in range(int(pivot), int((n + 1))):
                Aug[int(i), int(j)] = (Aug[int(i), int(j)] - (factor * Aug[int(pivot), int(j)]))
    for i in range(int(0), int(n)):
        ri = ((n - 1) - i)
        result[int(ri)] = Aug[int(ri), int(n)]
        for j in range(int((ri + 1)), int(n)):
            result[int(ri)] = (result[int(ri)] - (Aug[int(ri), int(j)] * result[int(j)]))
        result[int(ri)] = (result[int(ri)] / Aug[int(ri), int(ri)])
    return result

def U(k, m, t, x0, v0):
    omega = ((k / m) ** 0.5)
    A = torch.tensor([[1.0, 0.0], [0.0, omega]])
    B = torch.stack([torch.as_tensor(x0).float(), torch.as_tensor(v0).float()])
    coeffs = solve(A, B)
    a = coeffs[int(0)]
    b = coeffs[int(1)]
    return ((a * torch.cos((omega * t) if isinstance((omega * t), torch.Tensor) else torch.tensor(float((omega * t))))) + (b * torch.sin((omega * t) if isinstance((omega * t), torch.Tensor) else torch.tensor(float((omega * t))))))

# === Program ===
k = 1.0
m = 1.0
x0 = 1.0
v0 = 0.0
physika_print(U(k, m, 0.0, x0, v0))
physika_print(U(k, m, 1.5708, x0, v0))
physika_print(U(k, m, 3.1416, x0, v0))
physika_print(U(k, m, 4.7124, x0, v0))
physika_print(U(k, m, 6.2832, x0, v0))
physika_print(U(k, m, 0.0, 0.0, 1.0))
physika_print(U(k, m, 1.5708, 0.0, 1.0))
physika_print(U(k, m, 3.1416, 0.0, 1.0))
animate(U, k, m, x0, v0, 0.0, 31.1416)
t0 = torch.tensor(0.0, requires_grad=True)
physika_print(compute_grad(lambda _dt0: U(k, m, _dt0, x0, v0), t0))
t1 = torch.tensor(1.5708, requires_grad=True)
physika_print(compute_grad(lambda _dt1: U(k, m, _dt1, x0, v0), t1))
t2 = torch.tensor(3.1416, requires_grad=True)
physika_print(compute_grad(lambda _dt2: U(k, m, _dt2, x0, v0), t2))