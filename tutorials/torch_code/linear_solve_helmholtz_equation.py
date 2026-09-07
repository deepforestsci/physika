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

def get_2d_array_num_cols(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return get_1d_array_length(x[int(0)], n)

def get_2d_array_num_rows(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def get_1d_array_length(x, m=None):
    if m is None:
        m = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

# === Program ===
x0, x1, n = 0, 1, 10
Δx = ((x1 - x0) / n)
u_x0, u_x1 = 0, 1
k = 2
X = linspace(0, 1, (n + 1))
true_u = solver(k, n)
losses = torch.stack([torch.as_tensor(100)])
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