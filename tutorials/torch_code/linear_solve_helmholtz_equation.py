import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def linspace(start, end, n):
    x = zero_1d_array(n)
    Δx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * Δx))
    return x

def append(x, var):
    new_length = (len(x) + 1)
    results = zero_1d_array(new_length)
    len_x = get_1d_array_length(x)
    for i in range(int(0), int(new_length)):
        if i < len_x:
            results[int(i)] = x[int(i)]
        else:
            results[int(i)] = var
    return results

def central_difference(u, j):
    return (((u[int((j - 1))] - (2 * u[int(j)])) + u[int((j + 1))]) / (Δx ** 2))

def helmholtz_equation(u, j, k, m=None):
    if m is None:
        m = int(u.shape[0])
    return (central_difference(u, j) + ((k ** 2) * u[int(j)]))

def get_row_coeffs(j, k):
    e_left = zero_1d_array((n + 1))
    e_left[int((j - 1))] = 1
    a = helmholtz_equation(e_left, j, k)
    e_center = zero_1d_array((n + 1))
    e_center[int(j)] = 1
    b = helmholtz_equation(e_center, j, k)
    e_right = zero_1d_array((n + 1))
    e_right[int((j + 1))] = 1
    c = helmholtz_equation(e_right, j, k)
    return torch.stack([torch.as_tensor(a), torch.as_tensor(b), torch.as_tensor(c)])

def assemble_matrix(k, n):
    n_size = (n + 1)
    A = zero_2d_array(n_size, n_size)
    b = zero_1d_array(n_size)
    for j in range(int(1), int(n)):
        c1, c2, c3 = get_row_coeffs(j, k)
        if (j - 1) == 0:
            b[int(j)] = (b[int(j)] - (c1 * u_x0))
        else:
            A[int(j), int((j - 1))] = c1
        A[int(j), int(j)] = c2
        if (j + 1) == (n + 0):
            b[int(j)] = (b[int(j)] - (c3 * u_x1))
        else:
            A[int(j), int((j + 1))] = c3
    A[int(0), int(0)] = 1
    b[int(0)] = u_x0
    A[int(n), int(n)] = 1
    b[int(n)] = u_x1
    results = [A, b]
    return results

def solver(k, n):
    results = assemble_matrix(k, n)
    A = results[int(0)]
    b = results[int(1)]
    u = gaussian_solve(A, b)
    return u

def mse_loss(true_u, pred_u):
    total_len = get_1d_array_length(pred_u)
    square_diff = ((true_u - pred_u) ** 2)
    total = 0
    for i in range(int(0), int(total_len)):
        total = (total + square_diff[int(i)])
    return (total / total_len)

def gaussian_solve(A, b):
    a_row = get_2d_array_num_rows(A)
    a_col = get_2d_array_num_cols(A)
    new_col = (a_col + 1)
    aug = torch.zeros(int(a_row), int(new_col))
    for i in range(int(0), int(a_row)):
        aug[int(i), :int(a_col)] = A[int(i), :]
        aug[int(i), int(a_col)] = b[int(i)]
    for i in range(int(0), int(a_row)):
        max_row = i
        for k in range(int((i + 1)), int(a_row)):
            if torch.abs(aug[int(k), int(i)] if isinstance(aug[int(k), int(i)], torch.Tensor) else torch.tensor(float(aug[int(k), int(i)]))) > torch.abs(aug[int(max_row), int(i)] if isinstance(aug[int(max_row), int(i)], torch.Tensor) else torch.tensor(float(aug[int(max_row), int(i)]))):
                max_row = k
        if max_row != i:
            lo = min(i, max_row)
            hi = max(i, max_row)
            rows_above = aug[:int(lo), :]
            pivot_row = aug[int(hi):int((hi + 1)), :]
            rows_between = aug[int((lo + 1)):int(hi), :]
            current_row = aug[int(lo):int((lo + 1)), :]
            rows_below = aug[int((hi + 1)):, :]
            aug = torch.cat([rows_above, pivot_row, rows_between, current_row, rows_below])
        pivot_row = aug[int(i):int((i + 1)), :]
        pivot_value = aug[int(i), int(i)]
        rows_below = aug[int((i + 1)):, :]
        elimination_factors = (rows_below[:, int(i):int((i + 1))] / pivot_value)
        eliminated_rows = (rows_below - (elimination_factors * pivot_row))
        aug = torch.cat([aug[:int((i + 1)), :], eliminated_rows])
    x = torch.zeros(int(0))
    for i in range(int(0), int(a_row)):
        idx = ((a_col - 1) - i)
        total = aug[int(idx), int(a_col)]
        for j in range(int((idx + 1)), int(a_row)):
            total = (total - (aug[int(idx), int(j)] * x[int(((j - idx) - 1))]))
        val = (total / aug[int(idx), int(idx)])
        val = reshape(val, 1)
        x = torch.cat([val, x])
    return x

def get_2d_array_num_cols(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return get_1d_array_length(x[int(0)])

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def min(x, y):
    if x < y:
        return x
    else:
        return y

def max(x, y):
    if x > y:
        return x
    else:
        return y

# === Program ===
x0, x1, n = 0, 1, 10
Δx = ((x1 - x0) / n)
u_x0, u_x1 = 0, 1
k = 2
X = linspace(0, 1, (n + 1))
true_u = solver(k, n)
losses = torch.tensor([100], device=DEVICE)
guess_k = torch.tensor(2.6, requires_grad=True)
epochs = 1
lr = 0.01
for i in range(int(0), int(epochs)):
    print(i)
    pred_u = solver(guess_k, n)
    loss = mse_loss(true_u, pred_u)
    losses = append(losses, loss)
    grad = compute_grad(loss, guess_k)
    guess_k = (guess_k - (lr * grad))
    print(guess_k)
print(print(guess_k))
pred_traj = solver(guess_k, n)