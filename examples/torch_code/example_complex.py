import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def magnitude(z):
    return torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))

def f(x):
    return torch.abs((x ** 2) if isinstance((x ** 2), torch.Tensor) else torch.tensor(float((x ** 2))))

# === Classes ===
class A(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = nn.Parameter(torch.as_tensor(x))

    def f(self, y):
        this = self
        y = torch.as_tensor(y).float()
        return torch.abs((self.x ** y) if isinstance((self.x ** y), torch.Tensor) else torch.tensor(float((self.x ** y))))

    def forward(self, y):
        this = self
        y = torch.as_tensor(y).float()
        f_value = self.f(y)
        return compute_grad(f_value, self.x)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
x = torch.tensor((3 + 1j), dtype=torch.complex64)
y = torch.tensor((5 + 3j), dtype=torch.complex64)
physika_print(x)
physika_print(y)
complex_add = torch.tensor((x + y), dtype=torch.complex64)
physika_print(complex_add)
physika_print((x - y))
complex_mul = torch.tensor((x * y), dtype=torch.complex64)
physika_print(complex_mul)
physika_print(magnitude(x))
array_imag = torch.tensor([1j, 2j, 3j], dtype=torch.complex64)
array_complex = torch.stack([torch.as_tensor((1 + 9j), dtype=torch.complex64), torch.as_tensor((7 + 2j), dtype=torch.complex64), torch.as_tensor((3 + 5j), dtype=torch.complex64)])
nested_complex = torch.tensor([[(1 + 2j), (3 + 4j)], [(4 + 9j), (7 + 2j)]], dtype=torch.complex64)
physika_print(array_imag)
physika_print(array_complex)
physika_print(nested_complex)
scalar_x = torch.tensor((1 + 3j), dtype=torch.complex64)
scalar_grad = compute_grad(f, scalar_x)
physika_print(scalar_grad)
z = torch.as_tensor(torch.stack([torch.as_tensor((1 + 2j), dtype=torch.complex64), torch.as_tensor((3 + 1j), dtype=torch.complex64)])).requires_grad_(True)
loss = torch.sum((torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) * torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))) if isinstance((torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) * torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))), torch.Tensor) else torch.tensor(float((torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) * torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))))))
tensor_grad = compute_grad(loss, z)
physika_print(tensor_grad)
objA = A((1 + 2j))
class_grad = objA(2)
physika_print(class_grad)