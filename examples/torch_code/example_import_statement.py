import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
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

# === Program ===
x = 1.0
fact_results = fact(x)
physika_print(fact_results)
torch_funcs_results = torch_funcs_with_scalar_R(x)
physika_print(torch_funcs_results)
f_results = f(x)
physika_print(f_results)
r = torch.tensor([(-1.0), 0.0, 0.5, 1.0, 2.0], device=DEVICE)
phi = superbee(r)
physika_print(phi)