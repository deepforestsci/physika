import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def outer_product(u, v):
    return torch.stack([torch.stack([(u[int(i)] * v[int(j)]) for _fi_j in range(int(len(v))) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(len(u))) for i in [torch.tensor(float(_fi_i))]])

def get_last(arr):
    cur = 0
    for i in range(len(arr)):
        cur = arr[int(i)]
    return cur

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

def sum_or_sum_sq(arr, sq):
    return torch.where(torch.as_tensor(sq > 0.0), torch.sum(torch.stack([(arr[int(i)] ** 2) for _fi_i in range(int(len(arr))) for i in [torch.tensor(float(_fi_i))]])), torch.sum(torch.stack([arr[int(i)] for _fi_i in range(int(len(arr))) for i in [torch.tensor(float(_fi_i))]])))

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
            count = count + 1
    return count

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

def get_array_length(x):
    total = 0
    for i in range(len(x)):
        curr = x[int(i)]
        total = total + 1
    return total

def manipulate_1d_array(x):
    m = get_array_length(x)
    for i in range(int(0), int(m)):
        x[int(i)] = (i * 2)
    return x

def manipulate_2d_array(x):
    rows = get_array_length(x)
    cols = get_array_length(x[int(0)])
    for i in range(int(0), int(rows)):
        for j in range(int(0), int(cols)):
            x[int(i), int(j)] = (j * 2)
    return x

# === Program ===
arr = torch.tensor([1, 2, 3, 4, 5])
total = 0
for i in range(len(arr)):
    total = total + arr[int(i)]
physika_print(total)
X = torch.tensor([1, 2, 3, 4])
sum_sq = 0
for i in range(len(X)):
    sum_sq = sum_sq + (X[int(i)] ** 2)
physika_print(sum_sq)
y = torch.tensor([2, 4, 6, 8])
mse = 0
for i in range(len(X)):
    mse = mse + ((X[int(i)] - y[int(i)]) ** 2)
physika_print(mse)
src = torch.tensor([1, 2, 3, 4, 5])
dst = torch.tensor([0, 0, 0, 0, 0])
for i in range(len(src)):
    dst[int(i)] = (src[int(i)] * src[int(i)])
physika_print(dst)
start = 10
end = 20
total = 0
for i in range(int(start), int(end)):
    total = total + i
physika_print(total)
n = 10
for i in range(int(0), int(n)):
    for j in range(int(i), int(10)):
        total = total + i
        total = total + j
physika_print(total)
a = torch.stack([(i * 1) for _fi_i in range(int(5)) for i in [torch.tensor(float(_fi_i))]])
physika_print(a)
cos_wave = torch.stack([torch.cos((i * 0.5)) for _fi_i in range(int(6)) for i in [torch.tensor(float(_fi_i))]])
physika_print(cos_wave)
add = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i))]])
physika_print(add)
t = torch.stack([torch.stack([torch.stack([((i + j) + k) for _fi_k in range(int(4)) for k in [torch.tensor(float(_fi_k))]]) for _fi_j in range(int(3)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(2)) for i in [torch.tensor(float(_fi_i))]])
physika_print(t)
arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
doubled = torch.stack([(arr[int(i)] * 2.0) for _fi_i in range(int(len(arr))) for i in [torch.tensor(float(_fi_i))]])
physika_print(doubled)
u = torch.tensor([1.0, 2.0, 3.0, 4.0])
v = torch.tensor([4.0, 3.0, 2.0, 1.0])
dot_elems = torch.stack([(u[int(i)] * v[int(i)]) for _fi_i in range(int(len(u))) for i in [torch.tensor(float(_fi_i))]])
physika_print(dot_elems)
p = torch.tensor([1, 2, 3])
q = torch.tensor([10, 20, 30, 40])
physika_print(outer_product(p, q))
x = torch.tensor([1, 0, 0, 0])
y = torch.tensor([0, 1, 0, 0])
dot = torch.sum(torch.stack([(x[int(i)] * y[int(i)]) for _fi_i in range(int(len(x))) for i in [torch.tensor(float(_fi_i))]]))
physika_print(dot)
vals = torch.tensor([3, 1, 4, 1, 5])
physika_print(get_last(vals))
physika_print(iter_prod(10))
data = torch.tensor([1, 2, 3, 4, 5, 6])
physika_print(partial_sum(data, 2, 5))
physika_print(outer_accum(p, q))
A = torch.tensor([[1, 2, 3, 4], [0, 1, 1, 2]])
B = torch.tensor([[1], [0], [0], [2]])
physika_print(matmul_physika(A, B))
A2 = torch.tensor([[1, 2], [0, 1]])
B2 = torch.tensor([[1, 0], [0, 2]])
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
data = torch.tensor([1, 5, 2, 8, 3, 7])
physika_print(count_above(data, 3))
data = torch.tensor([1, 5, 2, 8, 3, 7])
physika_print(count_above_range(data, 1, 5, 3))
arr2 = torch.tensor([1, 2, 4])
physika_print(deep_nest(arr2))
arr3 = torch.tensor([1, (-2), 3, (-4), 5])
pos_sum = 0
for i in range(int(0), int(5)):
    if arr3[int(i)] > 0:
        pos_sum = pos_sum + arr3[int(i)]
physika_print(pos_sum)
abs_total = 0.0
for i in range(int(0), int(5)):
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
for i in range(int(0), int(3)):
    if mat[int(i)] > 0.0:
        dot2 = dot2 + (mat[int(i)] * vec[int(i)])
physika_print(dot2)
arr4 = torch.tensor([1.0, 4.0])
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
physika_print(a)
vals = torch.tensor([1, (-2), 3, 4, 5])
flag = 1
res = torch.stack([vals[int(i)] for _fi_i in range(int(len(vals))) for i in [torch.tensor(float(_fi_i))]])
if flag > 0:
    res = torch.stack([(vals[int(i)] ** 3) for _fi_i in range(int(5)) for i in [torch.tensor(float(_fi_i))]])
else:
    res = torch.stack([vals[int(i)] for _fi_i in range(int(5)) for i in [torch.tensor(float(_fi_i))]])
physika_print(res)
total_c = torch.sum(torch.stack([res[int(i)] for _fi_i in range(int(len(res))) for i in [torch.tensor(float(_fi_i))]]))
physika_print(total_c)
scale = 2
W = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i))]])
if scale > 1:
    W = torch.stack([torch.stack([((i + j) * scale) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i))]])
else:
    W = torch.stack([torch.stack([(i + j) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i))]])
physika_print(W)
u2 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
row_sums = torch.stack([torch.sum(torch.stack([(u2[int(i)] * v2[int(j)]) for _fi_j in range(int(len(v2))) for j in [torch.tensor(float(_fi_j))]])) for _fi_i in range(int(len(u2))) for i in [torch.tensor(float(_fi_i))]])
physika_print(row_sums)
data2 = torch.tensor([10, 20, 30, 40])
norm_flag = 1
normed = torch.stack([data2[int(i)] for _fi_i in range(int(len(data2))) for i in [torch.tensor(float(_fi_i))]])
if norm_flag > 0:
    normed = torch.stack([(data2[int(i)] * (1 / (i + 1))) for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i))]])
else:
    normed = torch.stack([data2[int(i)] for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i))]])
physika_print(normed)
sample_1d_array = torch.tensor([1, 2, 3])
length_array = get_array_length(sample_1d_array)
for i in range(int(0), int(length_array)):
    sample_1d_array[int(i)] = (i * 2)
physika_print(sample_1d_array)
sample_2d_array = torch.tensor([[1, 1], [1, 1]])
rows = get_array_length(sample_2d_array)
cols = get_array_length(sample_2d_array[int(0)])
for i in range(int(0), int(rows)):
    for j in range(int(0), int(cols)):
        sample_2d_array[int(i), int(j)] = (j * 2)
physika_print(sample_2d_array)
arr1d = torch.tensor([1, 2, 3])
physika_print(manipulate_1d_array(arr1d))
arr2d = torch.tensor([[1, 1], [1, 1]])
physika_print(manipulate_2d_array(arr2d))