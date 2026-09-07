import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def get_1d_array_length(x, m=None):
    if m is None:
        m = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def get_2d_array_num_rows(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def get_2d_array_num_cols(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return get_1d_array_length(x[int(0)], n)

def arange(n):
    arr = torch.stack([i for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return arr

def gaussian_solve(A, b):
    a_row = get_2d_array_num_rows(A)
    a_col = get_2d_array_num_cols(A)
    new_col = (a_col + 1)
    aug = zero_2d_array(a_row, new_col)
    for i in range(int(0), int(a_row)):
        for c in range(int(0), int(a_col)):
            aug[int(i), int(c)] = A[int(i), int(c)]
        aug[int(i), int(a_col)] = b[int(i)]
    for i in range(int(0), int(a_row)):
        max_row = i
        for k in range(int((i + 1)), int(a_row)):
            if torch.abs(aug[int(k), int(i)] if isinstance(aug[int(k), int(i)], torch.Tensor) else torch.tensor(float(aug[int(k), int(i)]))) > torch.abs(aug[int(max_row), int(i)] if isinstance(aug[int(max_row), int(i)], torch.Tensor) else torch.tensor(float(aug[int(max_row), int(i)]))):
                max_row = k
        pivot_row = zero_1d_array(new_col)
        displaced_row = zero_1d_array(new_col)
        for c in range(int(0), int(new_col)):
            pivot_row[int(c)] = aug[int(max_row), int(c)]
            displaced_row[int(c)] = aug[int(i), int(c)]
        aug_next = zero_2d_array(a_row, new_col)
        for row_idx in range(int(0), int(a_row)):
            if row_idx < i:
                for c in range(int(0), int(new_col)):
                    aug_next[int(row_idx), int(c)] = aug[int(row_idx), int(c)]
            else:
                if row_idx == i:
                    for c in range(int(0), int(new_col)):
                        aug_next[int(row_idx), int(c)] = pivot_row[int(c)]
                else:
                    source_row = zero_1d_array(new_col)
                    if row_idx == max_row:
                        for c in range(int(0), int(new_col)):
                            source_row[int(c)] = displaced_row[int(c)]
                    else:
                        for c in range(int(0), int(new_col)):
                            source_row[int(c)] = aug[int(row_idx), int(c)]
                    factor = (source_row[int(i)] / pivot_row[int(i)])
                    for c in range(int(0), int(new_col)):
                        aug_next[int(row_idx), int(c)] = (source_row[int(c)] - (factor * pivot_row[int(c)]))
        aug = aug_next
    x = zero_1d_array(a_col)
    for i in range(int(0), int(a_col)):
        idx = ((a_col - 1) - i)
        total = aug[int(idx), int(a_col)]
        for j in range(int((idx + 1)), int(a_row)):
            total = (total - (aug[int(idx), int(j)] * x[int(j)]))
        solved_val = (total / aug[int(idx), int(idx)])
        x_next = zero_1d_array(a_col)
        for c in range(int(0), int(a_col)):
            if c == idx:
                x_next[int(c)] = solved_val
            else:
                x_next[int(c)] = x[int(c)]
        x = x_next
    return x

# === Program ===
A = torch.tensor([[1, 2, 1], [3, 1, (-1)], [2, (-1), 1]], device=DEVICE)
b = torch.stack([torch.as_tensor(8), torch.as_tensor(2), torch.as_tensor(3)])
print(gaussian_solve(A, b))