import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def torch_funcs_with_scalar_R(x):
    result_sin = torch.sin(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_cos = torch.cos(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_exp = torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_sqrt = torch.sqrt(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_log = torch.log(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    result_abs = torch.abs(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    return torch.stack([torch.as_tensor(result_sin).float(), torch.as_tensor(result_cos).float(), torch.as_tensor(result_exp).float(), torch.as_tensor(result_sqrt).float(), torch.as_tensor(result_log).float(), torch.as_tensor(result_abs).float()])

def check_diff_torch_funcs(x):
    results = compute_grad(torch_funcs_with_scalar_R, x)
    return results

def f(x):
    if x > 0:
        return torch.cos(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    else:
        return torch.sin(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))

# === Program ===
x = 1.0
physika_print(torch_funcs_with_scalar_R(x))
physika_print(check_diff_torch_funcs(5.0))
x_matrix = torch.tensor([[1, (-1), 0], [(-1), 0, 0], [0, 0, 0]])
rolled_neg = torch.roll(x_matrix, (-1))
rolled_pos = torch.roll(x_matrix, 1)
physika_print(rolled_neg)
physika_print(rolled_pos)
x0 = torch.tensor((-1.5), requires_grad=True)
physika_print(f(x0))
physika_print(compute_grad(lambda _dx0: f(_dx0), x0))
x1 = torch.tensor((-0.5), requires_grad=True)
physika_print(f(x1))
physika_print(compute_grad(lambda _dx1: f(_dx1), x1))
x2 = torch.tensor(0.5, requires_grad=True)
physika_print(f(x2))
physika_print(compute_grad(lambda _dx2: f(_dx2), x2))
x3 = torch.tensor(1.5, requires_grad=True)
physika_print(f(x3))
physika_print(compute_grad(lambda _dx3: f(_dx3), x3))
x4 = torch.tensor(3.14, requires_grad=True)
physika_print(f(x4))
physika_print(compute_grad(lambda _dx4: f(_dx4), x4))