import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def sum_for_expr(s):
    return torch.sum(torch.stack([(s * i) for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i))]]) if isinstance(torch.stack([(s * i) for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i))]]), torch.Tensor) else torch.tensor(float(torch.stack([(s * i) for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i))]]))))

def dot_with_arr(s):
    a3 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = 0.0
    for i in range(len(a3)):
        result = result + (s * a3[int(i)])
    return result

def matmul_scale(s):
    A3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    I = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    C3 = torch.stack([torch.stack([torch.sum(torch.stack([((s * A3[int(i), int(k)]) * I[int(k), int(j)]) for k in range(A3.shape[1])])) for j in range(I.shape[1])]) for i in range(A3.shape[0])])
    return torch.sum(C3 if isinstance(C3, torch.Tensor) else torch.tensor(float(C3)))

def nested_sum(s):
    result = 0
    for i in range(int(0), int(10)):
        for j in range(int(i), int(10)):
            result = result + (((s * i) * 1.0) + ((s * j) * 1.0))
    return result

def scale_vec(x):
    return torch.stack([(x * (i + 1)) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i))]])

def sq_vec(x):
    return torch.stack([((x ** 2) * (i + 1)) for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i))]])

def cos_freqs(x):
    return torch.stack([torch.cos((x * (i + 1)) if isinstance((x * (i + 1)), torch.Tensor) else torch.tensor(float((x * (i + 1))))) for _fi_i in range(int(4)) for i in [torch.tensor(float(_fi_i))]])

def elementwise_sq(x):
    return torch.stack([(x[int(i)] ** 2) for _fi_i in range(int(len(x))) for i in [torch.tensor(float(_fi_i))]])

# === Program ===
s0 = torch.tensor(2.0, requires_grad=True)
physika_print(sum_for_expr(s0))
physika_print(compute_grad(lambda _ds0: sum_for_expr(_ds0), s0))
s1 = torch.tensor(1.0, requires_grad=True)
physika_print(dot_with_arr(s1))
physika_print(compute_grad(lambda _ds1: dot_with_arr(_ds1), s1))
s2 = torch.tensor(1.0, requires_grad=True)
physika_print(matmul_scale(s2))
physika_print(compute_grad(lambda _ds2: matmul_scale(_ds2), s2))
s3 = torch.tensor(1.0, requires_grad=True)
physika_print(nested_sum(s3))
physika_print(compute_grad(lambda _ds3: nested_sum(_ds3), s3))
s = torch.tensor(2.0, requires_grad=True)
physika_print(scale_vec(s))
physika_print(compute_grad(lambda _ds: scale_vec(_ds), s))
sv = torch.tensor(3.0, requires_grad=True)
physika_print(sq_vec(sv))
physika_print(compute_grad(lambda _dsv: sq_vec(_dsv), sv))
x = torch.tensor(0.5, requires_grad=True)
physika_print(cos_freqs(x))
physika_print(compute_grad(lambda _dx: cos_freqs(_dx), x))
ev = torch.as_tensor(torch.tensor([1.0, 2.0, 3.0])).float().requires_grad_(True)
physika_print(elementwise_sq(ev))
physika_print(compute_grad(lambda _dev: elementwise_sq(_dev), ev))