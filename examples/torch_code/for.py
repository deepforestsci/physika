import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def outer_product(u, v):
    return torch.stack([torch.stack([(u[int(i)] * v[int(j)]) for _fi_j in range(int(len(v))) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(len(u))) for i in [torch.tensor(float(_fi_i))]])

def get_last(arr):
    cur = 0.0
    for i in range(len(arr)):
        cur = arr[int(i)]
    return cur

def iter_prod(n):
    total = 0.0
    for i in range(int(0), int(n)):
        total = total + (i * 1.0)
    return total

def partial_sum(arr, low, high):
    total = 0.0
    for i in range(int(low), int(high)):
        total = total + arr[int(i)]
    return total

def outer_accum(u, v):
    C = torch.stack([torch.stack([(u[int(i)] * v[int(j)]) for j in range(v.shape[0])]) for i in range(u.shape[0])])
    return C

def matmul_physika(A, B):
    C = torch.stack([torch.stack([torch.sum(torch.stack([(A[int(i), int(k)] * B[int(k), int(j)]) for k in range(A.shape[1])])) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    return C

def chain_mm(A, B):
    C = torch.stack([torch.stack([torch.sum(torch.stack([(A[int(i), int(k)] * B[int(k), int(j)]) for k in range(A.shape[1])])) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    D = torch.stack([torch.stack([torch.sum(torch.stack([(C[int(i), int(k)] * A[int(k), int(j)]) for k in range(C.shape[1])])) for j in range(A.shape[1])]) for i in range(C.shape[0])])
    return D

def tensor_contraction(A, B, C):
    T = torch.stack([torch.stack([torch.stack([torch.sum(torch.stack([((A[int(i), int(k)] * B[int(k), int(j)]) * C[int(k), int(l)]) for k in range(A.shape[1])])) for l in range(C.shape[1])]) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    return T

def sum_or_sum_sq(arr, sq):
    return torch.where(torch.as_tensor(sq > 0.0), torch.sum(torch.stack([(arr[int(i)] ** 2.0) for _fi_i in range(int(len(arr))) for i in [torch.tensor(float(_fi_i))]])), torch.sum(torch.stack([arr[int(i)] for _fi_i in range(int(len(arr))) for i in [torch.tensor(float(_fi_i))]])))

def abs_sum(arr):
    total = 0.0
    for i in range(len(arr)):
        total = total + arr[int(i)]
    return torch.where(torch.as_tensor(total > 0.0), total, (0.0 - total))

def sum_positive(arr):
    total = 0.0
    for i in range(len(arr)):
        if arr[int(i)] > 0.0:
            total = total + arr[int(i)]
    return total

def sum_abs(arr):
    total = 0.0
    for i in range(len(arr)):
        if arr[int(i)] > 0.0:
            total = total + arr[int(i)]
        else:
            total = total + (0.0 - arr[int(i)])
    return total

def count_above(arr, thresh):
    count = 0.0
    for i in range(len(arr)):
        if arr[int(i)] > thresh:
            count = count + 1.0
    return count

def count_above_range(arr, lo, hi, thresh):
    count = 0.0
    for i in range(int(lo), int(hi)):
        if arr[int(i)] > thresh:
            count = count + 1.0
    return count

def deep_nest(arr):
    n = 3.0
    a = 0.0
    if a > (-1.0):
        for i in range(int(0), int(2.0)):
            if arr[int(i)] > 0.0:
                for j in range(int(0), int(2.0)):
                    if arr[int(j)] < 100.0:
                        for k in range(int(0), int(2.0)):
                            if arr[int(k)] != 0.0:
                                for l in range(int(0), int(2.0)):
                                    if a < 10.0:
                                        a = a + 1.0
                                    else:
                                        a = a + 2.0
                            else:
                                a = (a + 3.0)
                    else:
                        a = (a + 4.0)
            else:
                a = (a + 5.0)
    else:
        a = (-1.0)
    return a

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
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
dst = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
for i in range(len(dst)):
    dst[int(i)] = (src[int(i)] * src[int(i)])
physika_print(dst)
start = 10.0
end = 20.0
total = 0.0
for i in range(int(start), int(end)):
    total = total + i
physika_print(total)
n = 10.0
for i in range(int(0), int(n)):
    for j in range(int(i), int(10.0)):
        total = total + i
        total = total + j
physika_print(total)
a = torch.stack([(i * 1.0) for _fi_i in range(int(5.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(a)
cos_wave = torch.stack([torch.cos((i * 0.5)) for _fi_i in range(int(6.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(cos_wave)
add = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4.0)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(add)
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
vals = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
physika_print(get_last(vals))
physika_print(iter_prod(10.0))
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
physika_print(partial_sum(data, 2.0, 5.0))
physika_print(outer_accum(p, q))
A = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 2.0]])
B = torch.tensor([[1.0], [0.0], [0.0], [2.0]])
physika_print(matmul_physika(A, B))
A2 = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
B2 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
physika_print(chain_mm(A2, B2))
C_mat = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
physika_print(tensor_contraction(A2, B2, C_mat))
w = torch.tensor([1.0, 2.0, 3.0, 4.0])
physika_print(sum_or_sum_sq(w, 1.0))
physika_print(sum_or_sum_sq(w, (0.0 - 1.0)))
pos = torch.tensor([1.0, 2.0, 3.0, 4.0])
neg = torch.stack([torch.as_tensor((0.0 - 1.0)).float(), torch.as_tensor((0.0 - 2.0)).float(), torch.as_tensor((0.0 - 3.0)).float(), torch.as_tensor((0.0 - 4.0)).float()])
physika_print(abs_sum(pos))
physika_print(abs_sum(neg))
mixed = torch.stack([torch.as_tensor(1.0).float(), torch.as_tensor((0.0 - 2.0)).float(), torch.as_tensor(3.0).float(), torch.as_tensor((0.0 - 4.0)).float(), torch.as_tensor(5.0).float()])
physika_print(sum_positive(mixed))
physika_print(sum_abs(mixed))
data = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
physika_print(count_above(data, 3.0))
data = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
physika_print(count_above_range(data, 1.0, 5.0, 3.0))
arr2 = torch.tensor([1.0, 2.0, 4.0])
physika_print(deep_nest(arr2))
arr3 = torch.tensor([1.0, (-2.0), 3.0, (-4.0), 5.0])
pos_sum = 0.0
for i in range(int(0), int(5.0)):
    if arr3[int(i)] > 0.0:
        pos_sum = pos_sum + arr3[int(i)]
physika_print(pos_sum)
abs_total = 0.0
for i in range(int(0), int(5.0)):
    if arr3[int(i)] > 0.0:
        abs_total = abs_total + arr3[int(i)]
    else:
        abs_total = abs_total + (0.0 - arr3[int(i)])
physika_print(abs_total)
pos_sum2 = 0.0
for i in range(len(arr3)):
    if arr3[int(i)] > 0.0:
        pos_sum2 = pos_sum2 + arr3[int(i)]
physika_print(pos_sum2)
mat = torch.tensor([1.0, 2.0, 3.0])
vec = torch.tensor([4.0, 5.0, 6.0])
dot2 = 0.0
for i in range(int(0), int(3.0)):
    if mat[int(i)] > 0.0:
        dot2 = dot2 + (mat[int(i)] * vec[int(i)])
physika_print(dot2)
arr4 = torch.tensor([1.0, 4.0])
a = 0.0
for i in range(int(0), int(2.0)):
    if arr4[int(i)] > 0.0:
        for j in range(int(0), int(2.0)):
            if arr4[int(j)] < 100.0:
                for k in range(int(0), int(2.0)):
                    if arr4[int(k)] != 0.0:
                        if a < 10.0:
                            a = a + 1.0
                        else:
                            a = a + 2.0
                    else:
                        a = a + 3.0
            else:
                a = a + 4.0
    else:
        a = a + 5.0
physika_print(a)
vals = torch.tensor([1.0, (-2.0), 3.0, 4.0, 5.0])
flag = 1.0
res = torch.stack([vals[int(i)] for _fi_i in range(int(len(vals))) for i in [torch.tensor(float(_fi_i))]])
if flag > 0.0:
    res = torch.stack([(vals[int(i)] ** 3.0) for _fi_i in range(int(5.0)) for i in [torch.tensor(float(_fi_i))]])
else:
    res = torch.stack([vals[int(i)] for _fi_i in range(int(5.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(res)
total_c = torch.sum(torch.stack([res[int(i)] for _fi_i in range(int(len(res))) for i in [torch.tensor(float(_fi_i))]]))
physika_print(total_c)
scale = 2.0
W = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4.0)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3.0)) for i in [torch.tensor(float(_fi_i))]])
if scale > 1.0:
    W = torch.stack([torch.stack([((i + j) * scale) for _fi_j in range(int(4.0)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3.0)) for i in [torch.tensor(float(_fi_i))]])
else:
    W = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4.0)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(W)
u2 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
row_sums = torch.stack([torch.sum(torch.stack([(u2[int(i)] * v2[int(j)]) for _fi_j in range(int(len(v2))) for j in [torch.tensor(float(_fi_j))]])) for _fi_i in range(int(len(u2))) for i in [torch.tensor(float(_fi_i))]])
physika_print(row_sums)
data2 = torch.tensor([10.0, 20.0, 30.0, 40.0])
norm_flag = 1.0
normed = torch.stack([data2[int(i)] for _fi_i in range(int(len(data2))) for i in [torch.tensor(float(_fi_i))]])
if norm_flag > 0.0:
    normed = torch.stack([(data2[int(i)] * (1.0 / (i + 1.0))) for _fi_i in range(int(4.0)) for i in [torch.tensor(float(_fi_i))]])
else:
    normed = torch.stack([data2[int(i)] for _fi_i in range(int(4.0)) for i in [torch.tensor(float(_fi_i))]])
physika_print(normed)