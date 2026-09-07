import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def outer_product(u, v):
    results = torch.stack([torch.stack([(u[int(i)] * v[int(j)]) for j in range(v.shape[0])]) for i in range(u.shape[0])])
    return results

def get_last(arr, n=None):
    if n is None:
        n = int(arr.shape[0])
    cur = 0
    return (lambda cur: ([cur := arr[int(i)] for i in range(int(n))], cur)[1])(cur)

def iter_prod(n):
    total = 0
    for i in range(int(0), int(n)):
        total = total + (i * 1)
    return total

def partial_sum(arr, low, high):
    total = 0
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

def sum_or_sum_sq(arr, sq, n=None):
    if n is None:
        n = int(arr.shape[0])
    return (torch.sum(torch.stack([torch.as_tensor((arr[int(i)] ** 2)) for i in range(int(n))]).float()) if (sq > 0.0) else torch.sum(torch.stack([torch.as_tensor(arr[int(i)]) for i in range(int(n))]).float()))

def abs_sum(arr, n=None):
    if n is None:
        n = int(arr.shape[0])
    return (torch.sum(torch.stack([torch.as_tensor(arr[int(i)]) for i in range(int(n))]).float()) if (torch.sum(torch.stack([torch.as_tensor(arr[int(i)]) for i in range(int(n))]).float()) > 0.0) else (0.0 - torch.sum(torch.stack([torch.as_tensor(arr[int(i)]) for i in range(int(n))]).float())))

def sum_positive(arr, n=None):
    if n is None:
        n = int(arr.shape[0])
    return torch.sum(torch.stack([torch.as_tensor((arr[int(i)] if (arr[int(i)] > 0.0) else 0.0)) for i in range(int(n))]).float())

def sum_abs(arr, n=None):
    if n is None:
        n = int(arr.shape[0])
    return torch.sum(torch.stack([torch.as_tensor((arr[int(i)] if (arr[int(i)] > 0.0) else (0.0 - arr[int(i)]))) for i in range(int(n))]).float())

def count_above(arr, thresh, n=None):
    if n is None:
        n = int(arr.shape[0])
    return torch.sum(torch.stack([torch.as_tensor((1 if (arr[int(i)] > thresh) else 0.0)) for i in range(int(n))]).float())

def count_above_range(arr, lo, hi, thresh):
    count = 0.0
    for i in range(int(lo), int(hi)):
        if arr[int(i)] > thresh:
            count = count + 1
    return count

def deep_nest(arr):
    n = 3
    a = 0
    if a > (-1):
        for i in range(int(0), int(2)):
            if arr[int(i)] > 0:
                for j in range(int(0), int(2)):
                    if arr[int(j)] < 100:
                        for k in range(int(0), int(2)):
                            if arr[int(k)] != 0:
                                for l in range(int(0), int(2)):
                                    if a < 10:
                                        a = a + 1
                                    else:
                                        a = a + 2
                            else:
                                a = (a + 3)
                    else:
                        a = (a + 4)
            else:
                a = (a + 5)
    else:
        a = (-1)
    return a

def get_array_length(x, m=None):
    if m is None:
        m = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def get_2d_array_num_rows(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def manipulate_1d_array(x):
    m = get_array_length(x)
    for i in range(int(0), int(m)):
        x[int(i)] = (i * 2)
    return x

def manipulate_2d_array(x):
    m = get_2d_array_num_rows(x)
    n = get_array_length(x[int(0)])
    for i in range(int(0), int(m)):
        for j in range(int(0), int(n)):
            x[int(i), int(j)] = (j * 2)
    return x

def manipulate_3d_array(x):
    for i in range(int(0), int(2)):
        for j in range(int(0), int(2)):
            for k in range(int(0), int(2)):
                x[int(i), int(j), int(k)] = (((i * 2) + j) + k)
    return x

# === Program ===
arr = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(4), torch.as_tensor(5)])
total = 0
for i in range(len(arr)):
    total = total + arr[int(i)]
print(total)
X = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(4)])
sum_sq = 0
for i in range(len(X)):
    sum_sq = sum_sq + (X[int(i)] ** 2)
print(sum_sq)
y = torch.stack([torch.as_tensor(2), torch.as_tensor(4), torch.as_tensor(6), torch.as_tensor(8)])
mse = 0
for i in range(len(X)):
    mse = mse + ((X[int(i)] - y[int(i)]) ** 2)
print(mse)
src = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(4), torch.as_tensor(5)])
dst = torch.stack([torch.as_tensor(0), torch.as_tensor(0), torch.as_tensor(0), torch.as_tensor(0), torch.as_tensor(0)])
for i in range(len(src)):
    dst[int(i)] = (src[int(i)] * src[int(i)])
print(dst)
start = 10
end = 20
total = 0
for i in range(int(start), int(end)):
    total = total + i
print(total)
n = 10
for i in range(int(0), int(n)):
    for j in range(int(i), int(10)):
        total = total + i
        total = total + j
print(total)
a = torch.stack([(i * 1) for _fi_i in range(int(5)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(a)
cos_wave = torch.stack([torch.cos((i * 0.5) if isinstance((i * 0.5), torch.Tensor) else torch.tensor(float((i * 0.5)))) for _fi_i in range(int(6)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(cos_wave)
add = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(add)
t = torch.stack([torch.stack([torch.stack([((i + j) + k) for _fi_k in range(int(4)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(3)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(2)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(t)
arr = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0), torch.as_tensor(4.0), torch.as_tensor(5.0)])
doubled = torch.stack([torch.as_tensor((arr[int(i)] * 2.0)) for i in range(int(5))]).float()
print(doubled)
u = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0), torch.as_tensor(4.0)])
v = torch.stack([torch.as_tensor(4.0), torch.as_tensor(3.0), torch.as_tensor(2.0), torch.as_tensor(1.0)])
dot_elems = torch.stack([torch.as_tensor((u[int(i)] * v[int(i)])) for i in range(int(4))]).float()
print(dot_elems)
p = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3)])
q = torch.stack([torch.as_tensor(10), torch.as_tensor(20), torch.as_tensor(30), torch.as_tensor(40)])
print(outer_product(p, q))
x = torch.stack([torch.as_tensor(1), torch.as_tensor(0), torch.as_tensor(0), torch.as_tensor(0)])
y = torch.stack([torch.as_tensor(0), torch.as_tensor(1), torch.as_tensor(0), torch.as_tensor(0)])
dot = torch.sum(torch.stack([torch.as_tensor((x[int(i)] * y[int(i)])) for i in range(int(4))]).float())
print(dot)
vals = torch.stack([torch.as_tensor(3), torch.as_tensor(1), torch.as_tensor(4), torch.as_tensor(1), torch.as_tensor(5)])
print(get_last(vals, 5))
print(iter_prod(10))
data = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(4), torch.as_tensor(5), torch.as_tensor(6)])
print(partial_sum(data, 2, 5))
print(outer_accum(p, q))
A = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(4)])), torch.as_tensor(torch.stack([torch.as_tensor(0), torch.as_tensor(1), torch.as_tensor(1), torch.as_tensor(2)]))])
B = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1)])), torch.as_tensor(torch.stack([torch.as_tensor(0)])), torch.as_tensor(torch.stack([torch.as_tensor(0)])), torch.as_tensor(torch.stack([torch.as_tensor(2)]))])
print(matmul_physika(A, B))
A2 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2)])), torch.as_tensor(torch.stack([torch.as_tensor(0), torch.as_tensor(1)]))])
B2 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(0)])), torch.as_tensor(torch.stack([torch.as_tensor(0), torch.as_tensor(2)]))])
print(chain_mm(A2, B2))
C_mat = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0), torch.as_tensor(1.0), torch.as_tensor(0.0)]))])
print(tensor_contraction(A2, B2, C_mat))
w = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0), torch.as_tensor(4.0)])
print(sum_or_sum_sq(w, 1.0, 4))
print(sum_or_sum_sq(w, (0.0 - 1.0), 4))
pos = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0), torch.as_tensor(4.0)])
neg = torch.stack([torch.as_tensor((0.0 - 1.0)), torch.as_tensor((0.0 - 2.0)), torch.as_tensor((0.0 - 3.0)), torch.as_tensor((0.0 - 4.0))])
print(abs_sum(pos, 4))
print(abs_sum(neg, 4))
mixed = torch.stack([torch.as_tensor(1.0), torch.as_tensor((0.0 - 2.0)), torch.as_tensor(3.0), torch.as_tensor((0.0 - 4.0)), torch.as_tensor(5.0)])
print(sum_positive(mixed, 5))
print(sum_abs(mixed, 5))
data = torch.stack([torch.as_tensor(1), torch.as_tensor(5), torch.as_tensor(2), torch.as_tensor(8), torch.as_tensor(3), torch.as_tensor(7)])
print(count_above(data, 3, 6))
data = torch.stack([torch.as_tensor(1), torch.as_tensor(5), torch.as_tensor(2), torch.as_tensor(8), torch.as_tensor(3), torch.as_tensor(7)])
print(count_above_range(data, 1, 5, 3))
arr2 = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(4)])
print(deep_nest(arr2))
arr3 = torch.tensor([1, (-2), 3, (-4), 5], device=DEVICE)
pos_sum = 0
for i in range(int(0), int(5)):
    if arr3[int(i)] > 0:
        pos_sum = pos_sum + arr3[int(i)]
print(pos_sum)
abs_total = 0.0
for i in range(int(0), int(5)):
    if arr3[int(i)] > 0.0:
        abs_total = abs_total + arr3[int(i)]
    else:
        abs_total = abs_total + (0.0 - arr3[int(i)])
print(abs_total)
pos_sum2 = 0.0
for i in range(len(arr3)):
    if arr3[int(i)] > 0.0:
        pos_sum2 = pos_sum2 + arr3[int(i)]
print(pos_sum2)
mat = torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)])
vec = torch.stack([torch.as_tensor(4.0), torch.as_tensor(5.0), torch.as_tensor(6.0)])
dot2 = 0.0
for i in range(int(0), int(3)):
    if mat[int(i)] > 0.0:
        dot2 = dot2 + (mat[int(i)] * vec[int(i)])
print(dot2)
arr4 = torch.stack([torch.as_tensor(1.0), torch.as_tensor(4.0)])
a = 0.0
for i in range(int(0), int(2)):
    if arr4[int(i)] > 0.0:
        for j in range(int(0), int(2)):
            if arr4[int(j)] < 100.0:
                for k in range(int(0), int(2)):
                    if arr4[int(k)] != 0.0:
                        if a < 10.0:
                            a = a + 1
                        else:
                            a = a + 2
                    else:
                        a = a + 3
            else:
                a = a + 4
    else:
        a = a + 5
print(a)
vals = torch.tensor([1, (-2), 3, 4, 5], device=DEVICE)
flag = 1
res = torch.stack([torch.as_tensor(vals[int(i)]) for i in range(int(5))]).float()
if flag > 0:
    res = torch.stack([(vals[int(i)] ** 3) for _fi_i in range(int(5)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
else:
    res = torch.stack([vals[int(i)] for _fi_i in range(int(5)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(res)
total_c = torch.sum(torch.stack([torch.as_tensor(res[int(i)]) for i in range(int(5))]).float())
print(total_c)
scale = 2
W = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
if scale > 1:
    W = torch.stack([torch.stack([((i + j) * scale) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
else:
    W = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(W)
u2 = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3)])
v2 = torch.stack([torch.as_tensor(4), torch.as_tensor(5), torch.as_tensor(6)])
row_sums = torch.stack([torch.as_tensor(torch.sum(torch.stack([torch.as_tensor((u2[int(i)] * v2[int(j)])) for j in range(int(3))]).float())) for i in range(int(3))]).float()
print(row_sums)
x = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3), torch.as_tensor(4)])
y = torch.stack([torch.as_tensor(0), torch.as_tensor(5), torch.as_tensor(6), torch.as_tensor(7)])
print(outer_product(x, y))
data2 = torch.stack([torch.as_tensor(10), torch.as_tensor(20), torch.as_tensor(30), torch.as_tensor(40)])
norm_flag = 1
normed = torch.stack([torch.as_tensor(data2[int(i)]) for i in range(int(4))]).float()
if norm_flag > 0:
    normed = torch.stack([(data2[int(i)] * (1 / (i + 1))) for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
else:
    normed = torch.stack([data2[int(i)] for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
print(normed)
sample_1d_array = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3)])
length_array = get_array_length(sample_1d_array, 3)
for i in range(int(0), int(length_array)):
    sample_1d_array[int(i)] = (i * 2)
print(sample_1d_array)
sample_2d_array = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(1)])), torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(1)]))])
rows = get_2d_array_num_rows(sample_2d_array, 2, 2)
cols = get_array_length(sample_2d_array[int(0)], 2)
for i in range(int(0), int(rows)):
    for j in range(int(0), int(cols)):
        sample_2d_array[int(i), int(j)] = (j * 2)
print(sample_2d_array)
sample_3d_array = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2)])), torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2)])), torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2)]))]))])
for i in range(int(0), int(2)):
    for j in range(int(0), int(2)):
        for k in range(int(0), int(2)):
            sample_3d_array[int(i), int(j), int(k)] = ((j * 2) + k)
print(sample_3d_array)
arr1d = torch.stack([torch.as_tensor(1), torch.as_tensor(2), torch.as_tensor(3)])
print(manipulate_1d_array(arr1d))
arr2d = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(1)])), torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(1)]))])
print(manipulate_2d_array(arr2d))
arr3d = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(3), torch.as_tensor(2)])), torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(1)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(4)])), torch.as_tensor(torch.stack([torch.as_tensor(1), torch.as_tensor(2)]))]))])
print(manipulate_3d_array(arr3d))