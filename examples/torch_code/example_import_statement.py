import torch
import torch.nn as nn
import torch.optim as optim

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

# === Program ===
x = 1.0
fact_results = fact(x)
physika_print(fact_results)
torch_funcs_results = torch_funcs_with_scalar_R(x)
physika_print(torch_funcs_results)
f_results = f(x)
physika_print(f_results)