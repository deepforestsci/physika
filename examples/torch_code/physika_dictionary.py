import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def declare_dict(value1, value2):
    function_dict = {0: value1, 1: value2}
    return function_dict

def scalar_f(x):
    return (x ** 2)

def array_f(x, n=None):
    if n is None:
        n = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor((x[int(i)] ** 2)) for i in range(int(n))]).float())

# === Classes ===
class A(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = torch.as_tensor(x).float() if isinstance(x, (int, float, torch.Tensor)) else x

    def return_dict(self):
        this = self
        return self.x

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
empty_dict = {}
print(print(empty_dict))
simple_dict = {0: 1.6, 1: 3.2, 2: 5.5}
print(print(simple_dict))
union_dict = {0: 1, 1: 3j, 2: 30, 3: torch.tensor([1, 2, 3], device=DEVICE)}
print(print(union_dict))
func_dict = declare_dict(1.0, 2.0)
print(print(func_dict))
objA = A(simple_dict).to(DEVICE)
class_dict = objA.return_dict()
print(print(class_dict))
first_value = union_dict[int(0)]
last_value = union_dict[int(3)]
print(print(first_value))
print(print(last_value))
example_dict = {0: 1.6, 1: 3.2, 2: 5.5}
print(print(example_dict))
example_dict[int(2)] = torch.tensor([1, 2, 3], device=DEVICE)
print(print(example_dict))
scalar_diff = compute_grad(scalar_f, union_dict[int(0)])
print(print(scalar_diff))
array_diff = compute_grad(array_f, union_dict[int(3)])
print(print(array_diff))