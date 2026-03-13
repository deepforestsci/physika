import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def outer_product(u, v):
    return torch.stack([torch.stack([(u[int(i)] * v[int(j)]) for _fi_j in range(int(len(v))) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(len(u))) for i in [torch.tensor(float(_fi_i))]])

def matmul_physika(A, B):
    C = torch.stack([torch.stack([torch.sum(torch.stack([(A[int(i), int(k)] * B[int(k), int(j)]) for k in range(A.shape[1])])) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    return C

def chain_mm(A, B):
    C = torch.stack([torch.stack([torch.sum(torch.stack([(A[int(i), int(k)] * B[int(k), int(j)]) for k in range(A.shape[1])])) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    D = torch.stack([torch.stack([torch.sum(torch.stack([(C[int(i), int(k)] * A[int(k), int(j)]) for k in range(C.shape[1])])) for j in range(A.shape[1])]) for i in range(C.shape[0])])
    return D

def both_matmuls(A, B):
    AB = torch.stack([torch.stack([torch.sum(torch.stack([(A[int(i), int(k)] * B[int(k), int(j)]) for k in range(A.shape[1])])) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    BA = torch.stack([torch.stack([torch.sum(torch.stack([(B[int(i), int(k)] * A[int(k), int(j)]) for k in range(B.shape[1])])) for j in range(A.shape[1])]) for i in range(B.shape[0])])
    return (AB + BA)

def tensor_contraction(A, B, C):
    T = torch.stack([torch.stack([torch.stack([torch.sum(torch.stack([((A[int(i), int(k)] * B[int(k), int(j)]) * C[int(k), int(l)]) for k in range(A.shape[1])])) for l in range(C.shape[1])]) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    return T

# === Program ===
arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
total = 0.0
for i in range(len(arr)):
    total = total + arr[int(i)]
physika_print(total)
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
sum_sq = 0.0
for i in range(len(X)):
    sum_sq = sum_sq + (X[int(i)] ** 2.0)
physika_print(sum_sq)
y = torch.tensor([2.0, 4.0, 6.0, 8.0])
mse = 0.0
for i in range(len(X)):
    mse = mse + ((X[int(i)] - y[int(i)]) ** 2.0)
physika_print(mse)
a = torch.stack([(i * 1.0) for _fi_i in range(int(5.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(a)
cos_wave = torch.stack([torch.cos((i * 0.5)) for _fi_i in range(int(6.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(cos_wave)
add = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4.0)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(add)
outer = torch.stack([torch.stack([(i * j) for _fi_j in range(int(4.0)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(4.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(outer)
t = torch.stack([torch.stack([torch.stack([((i + j) + k) for _fi_k in range(int(4.0)) for k in [torch.tensor(float(_fi_k))]]) for _fi_j in range(int(3.0)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(2.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(t)
arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
doubled = torch.stack([(arr[int(i)] * 2.0) for _fi_i in range(int(len(arr))) for i in [torch.tensor(float(_fi_i))]])
physika_print(doubled)
u = torch.tensor([1.0, 2.0, 3.0, 4.0])
v = torch.tensor([4.0, 3.0, 2.0, 1.0])
dot_elems = torch.stack([(u[int(i)] * v[int(i)]) for _fi_i in range(int(len(u))) for i in [torch.tensor(float(_fi_i))]])
physika_print(dot_elems)
p = torch.tensor([1.0, 2.0, 3.0])
q = torch.tensor([10.0, 20.0, 30.0, 40.0])
physika_print(outer_product(p, q))
x = torch.tensor([1.0, 0.0, 0.0, 0.0])
y = torch.tensor([0.0, 1.0, 0.0, 0.0])
dot = torch.sum(torch.stack([(x[int(i)] * y[int(i)]) for _fi_i in range(int(len(x))) for i in [torch.tensor(float(_fi_i))]]))
physika_print(dot)
A = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 2.0]])
B = torch.tensor([[1.0], [0.0], [0.0], [2.0]])
physika_print(matmul_physika(A, B))
A2 = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
B2 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
physika_print(chain_mm(A2, B2))
A2 = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
B2 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
physika_print(both_matmuls(A2, B2))
C_mat = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
physika_print(tensor_contraction(A2, B2, C_mat))