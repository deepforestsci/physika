import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_cols(x):
    return get_1d_array_length(x[int(0)])

def arange(n):
    arr = torch.stack([i for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return arr

def gaussian_solve(A, b):
    a_row = get_2d_array_num_rows(A)
    a_col = get_2d_array_num_cols(A)
    new_col = (a_col + 1)
    aug = torch.zeros(a_row, new_col)
    for i in range(int(0), int(a_row)):
        aug[int(i), :a_col] = A[int(i), :]
        aug[int(i), int(a_col)] = b[int(i)]
    row_buffer = torch.zeros(new_col)
    for i in range(int(0), int(a_row)):
        max_row = i
        for k in range(int((i + 1)), int(a_row)):
            if torch.abs(aug[int(k), int(i)] if isinstance(aug[int(k), int(i)], torch.Tensor) else torch.tensor(float(aug[int(k), int(i)]))) > torch.abs(aug[int(max_row), int(i)] if isinstance(aug[int(max_row), int(i)], torch.Tensor) else torch.tensor(float(aug[int(max_row), int(i)]))):
                max_row = k
        if max_row != i:
            for k in range(int(0), int(new_col)):
                row_buffer[int(k)] = aug[int(i), int(k)]
            for k in range(int(0), int(new_col)):
                aug[int(i), int(k)] = aug[int(max_row), int(k)]
            for k in range(int(0), int(new_col)):
                aug[int(max_row), int(k)] = row_buffer[int(k)]
        for j in range(int((i + 1)), int(a_row)):
            factor = (aug[int(j), int(i)] / aug[int(i), int(i)])
            for k in range(int(i), int(new_col)):
                aug[int(j), int(k)] = (aug[int(j), int(k)] - (factor * aug[int(i), int(k)]))
    x = torch.zeros(a_col)
    for i in range(int(0), int(a_col)):
        idx = ((a_col - 1) - i)
        total = aug[int(idx), int(a_col)]
        for j in range(int((idx + 1)), int(a_row)):
            total = (total - (aug[int(idx), int(j)] * x[int(j)]))
        x[int(idx)] = (total / aug[int(idx), int(idx)])
    return x

# === Program ===
A = torch.tensor([[1, 2, 1], [3, 1, (-1)], [2, (-1), 1]], device=DEVICE)
b = torch.tensor([8, 2, 3], device=DEVICE)
print(gaussian_solve(A, b))