import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def lu_decomposition(A):
    n_size = get_2d_array_num_rows(A)
    L = torch.zeros(n_size, n_size)
    U = torch.zeros(n_size, n_size)
    P = torch.zeros(n_size, n_size)
    for i in range(int(0), int(n_size)):
        P[int(i), int(i)] = 1.0
    buf_a = torch.zeros(n_size)
    buf_l = torch.zeros(n_size)
    buf_p = torch.zeros(n_size)
    for j in range(int(0), int(n_size)):
        max_row = j
        for i in range(int((j + 1)), int(n_size)):
            if torch.abs(A[int(i), int(j)] if isinstance(A[int(i), int(j)], torch.Tensor) else torch.tensor(float(A[int(i), int(j)]))) > torch.abs(A[int(max_row), int(j)] if isinstance(A[int(max_row), int(j)], torch.Tensor) else torch.tensor(float(A[int(max_row), int(j)]))):
                max_row = i
        if max_row != j:
            for k in range(int(0), int(n_size)):
                buf_a[int(k)] = A[int(j), int(k)]
            for k in range(int(0), int(n_size)):
                A[int(j), int(k)] = A[int(max_row), int(k)]
            for k in range(int(0), int(n_size)):
                A[int(max_row), int(k)] = buf_a[int(k)]
            for k in range(int(0), int(j)):
                buf_l[int(k)] = L[int(j), int(k)]
            for k in range(int(0), int(j)):
                L[int(j), int(k)] = L[int(max_row), int(k)]
            for k in range(int(0), int(j)):
                L[int(max_row), int(k)] = buf_l[int(k)]
            for k in range(int(0), int(n_size)):
                buf_p[int(k)] = P[int(j), int(k)]
            for k in range(int(0), int(n_size)):
                P[int(j), int(k)] = P[int(max_row), int(k)]
            for k in range(int(0), int(n_size)):
                P[int(max_row), int(k)] = buf_p[int(k)]
        for i in range(int(0), int((j + 1))):
            partial = 0.0
            for k in range(int(0), int(i)):
                partial = (partial + (U[int(k), int(j)] * L[int(i), int(k)]))
            U[int(i), int(j)] = (A[int(i), int(j)] - partial)
        for i in range(int(j), int(n_size)):
            partial = 0.0
            for k in range(int(0), int(j)):
                partial = (partial + (U[int(k), int(j)] * L[int(i), int(k)]))
            L[int(i), int(j)] = ((A[int(i), int(j)] - partial) / U[int(j), int(j)])
    return torch.stack([torch.as_tensor(P), torch.as_tensor(L), torch.as_tensor(U)])

# === Program ===
A = torch.tensor([[(-1.0), 0.0, 3.0], [2.0, 1.0, 3.0], [1.0, 1.0, 2.0]], device=DEVICE)
A_original = torch.stack([torch.stack([A[int(i), int(j)] for _fi_j in range(int(3)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
results = lu_decomposition(A)
P_matrix = results[int(0)]
L_matrix = results[int(1)]
U_matrix = results[int(2)]
print(print(P_matrix))
print(print(L_matrix))
print(print(U_matrix))
LU = (L_matrix @ U_matrix)
PA = (P_matrix @ A_original)
print(print(LU))
print(print(PA))