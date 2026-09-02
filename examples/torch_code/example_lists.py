import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def return_diff_list(x):
    A = torch.tensor([[1, 2], [3, 4]], device=DEVICE)
    b = torch.tensor([5, 6], device=DEVICE)
    results = [A, b]
    return results

def f_scalar(x):
    return (x ** 2)

def f_tensor(x):
    return torch.sum((x ** 2) if isinstance((x ** 2), torch.Tensor) else torch.tensor(float((x ** 2))))

def get_1d_list_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def square_list(values):
    len_list = get_1d_list_length(values)
    results = torch.stack([(values[int(i)] ** 2) for _fi_i in range(int(len_list)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def f_return_list(x, y, z):
    results = [x, y, z]
    return results

# === Classes ===
class A(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = torch.as_tensor(x).float() if isinstance(x, (int, float, torch.Tensor)) else x
        self.learnable_params = []

    def get_list(self):
        this = self
        return self.x

    def get_value_list(self, value1, value2):
        this = self
        value1 = torch.as_tensor(value1, device=DEVICE).float()
        value2 = torch.as_tensor(value2, device=DEVICE).float()
        results = [value1, value2]
        return results

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
x = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
y = torch.tensor([9.0, 3.0, 5.0, 1.0, 4.0], device=DEVICE)
complex_list = torch.tensor([1j, 2j, 3j], dtype=torch.complex64)
simple_nested_list = [x, y]
nested_list = [1, 2, x, [x, y]]
mixed_list = [1, x, complex_list, [y, complex_list]]
print(x)
print(y)
print(complex_list)
print(simple_nested_list)
print(nested_list)
print(mixed_list)
simple_index_x = x[int(0)]
simple_complex_index = torch.tensor(complex_list[int(1)], dtype=torch.complex64)
index_simple_nested_list = simple_nested_list[int(0)]
index_nested_list = nested_list[int(3)]
index_mixed_list = mixed_list[int(3)]
print(simple_index_x)
print(simple_complex_index)
print(index_simple_nested_list)
print(index_nested_list)
print(index_mixed_list)
diff_list_results = return_diff_list(1.0)
diff_list_first_index = diff_list_results[int(0)]
diff_list_second_index = diff_list_results[int(1)]
print(diff_list_first_index)
print(diff_list_second_index)
scalar_grad = compute_grad(f_scalar, x[int(0)])
tensor_grad = compute_grad(f_tensor, y)
nested_grad = compute_grad(f_tensor, mixed_list[int(1)])
print(scalar_grad)
print(tensor_grad)
print(nested_grad)
squared_tensor = square_list(x)
print(squared_tensor)
f_results = f_return_list(1, 2, 3)
print(f_results)
obj = A(nested_list).to(DEVICE)
obj_list = obj.get_list()
print(obj_list)
obj_value_results = obj.get_value_list(1, 2)
print(obj_value_results)