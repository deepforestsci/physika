import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print
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

# === Classes ===
class ExampleClass(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.learnable_params = []

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
physika_print(fact_results)
torch_funcs_results = torch_funcs_with_scalar_R(x)
physika_print(torch_funcs_results)
f_results = f(x)
physika_print(f_results)
obj_example_class = ExampleClass().to(DEVICE)
class_value = obj_example_class.class_method()
physika_print(class_value)
v = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
for i in range(int(0), int(3)):
    if v[int(i)] > 2:
        v[int(i)] = (v[int(i)] * 2)
physika_print(v)
grad_f_x = compute_grad(f_x, v[int(2)])
physika_print(grad_f_x)
r = torch.tensor([(-1.0), 0.0, 0.5, 1.0, 2.0], device=DEVICE)
φ = superbee(r)
physika_print(φ)