import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def f_x(x):
    return (x ** 2)

def fact(n):
    if n == 0.0:
        return 1.0
    else:
        return (n * fact((n - 1.0)))

def f(x):
    if x > 0:
        return torch.cos(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    else:
        return torch.sin(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))

def torch_funcs_with_scalar_R(x):
    result_sin = torch.sin(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_cos = torch.cos(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_exp = torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_sqrt = torch.sqrt(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_log = torch.log(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_abs = torch.abs(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    return torch.stack([torch.as_tensor(result_sin), torch.as_tensor(result_cos), torch.as_tensor(result_exp), torch.as_tensor(result_sqrt), torch.as_tensor(result_log), torch.as_tensor(result_abs)])

def superbee(r):
    s1 = (0.5 * (((2.0 * r) + 1.0) - torch.abs(((2.0 * r) - 1.0) if isinstance(((2.0 * r) - 1.0), torch.Tensor) else torch.tensor(float(((2.0 * r) - 1.0))))))
    s2 = (0.5 * ((r + 2.0) - torch.abs((r - 2.0) if isinstance((r - 2.0), torch.Tensor) else torch.tensor(float((r - 2.0))))))
    s3 = (0.5 * ((s1 + s2) + torch.abs((s1 - s2) if isinstance((s1 - s2), torch.Tensor) else torch.tensor(float((s1 - s2))))))
    phi = (0.5 * (s3 + torch.abs(s3 if isinstance(s3, torch.Tensor) else torch.tensor(float(s3)))))
    return phi

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

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

# === Classes ===
class ExampleClass(nn.Module):
    def __init__(self, ):
        super().__init__()

    def class_method(self):
        this = self
        return 1

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
x = 1.0
fact_results = fact(x)
print(fact_results)
torch_funcs_results = torch_funcs_with_scalar_R(x)
print(torch_funcs_results)
f_results = f(x)
print(f_results)
obj_example_class = ExampleClass().to(DEVICE)
class_value = obj_example_class.class_method()
print(class_value)
v = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
for i in range(int(0), int(3)):
    if v[int(i)] > 2:
        v[int(i)] = (v[int(i)] * 2)
print(v)
grad_f_x = compute_grad(f_x, v[int(2)])
print(grad_f_x)
r = torch.tensor([(-1.0), 0.0, 0.5, 1.0, 2.0], device=DEVICE)
φ = superbee(r)
print(φ)
A = torch.tensor([[1, 2, 1], [3, 1, (-1)], [2, (-1), 1]], device=DEVICE)
b = torch.tensor([8, 2, 3], device=DEVICE)
gaussian_results = gaussian_solve(A, b)
print(gaussian_results)