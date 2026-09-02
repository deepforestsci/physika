import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def unpack_simple_array(arr):
    a, b, c = arr
    return a

def return_array(n):
    results = torch.stack([(i * 1) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def unpack_array_in_function_call(n):
    a, b, c = return_array(n)
    return c

# === Classes ===
class Simple(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = torch.as_tensor(v).float()
        self.learnable_params = [self.v]

    def get(self):
        this = self
        return (self.v, self.v)

    def sum_pairs(self, n):
        this = self
        total = 0.0
        for k in range(int(0), int(n)):
            a, b = self.get()
            total = ((total + a) + b)
        return total

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Pair(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = torch.as_tensor(a).float()
        self.b = torch.as_tensor(b).float()
        self.learnable_params = [self.a, self.b]

    def get(self):
        this = self
        return (self.a, self.b)

    def sum(self):
        this = self
        x, y = self.get()
        return (x + y)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Model(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = torch.as_tensor(a).float()
        self.b = torch.as_tensor(b).float()
        self.learnable_params = [self.a, self.b]

    def pair(self):
        this = self
        return (self.a, self.b)

    def run(self, steps):
        this = self
        x, y = self.pair()
        total = (x + y)
        for k in range(int(0), int(steps)):
            p, q = self.pair()
            total = ((total + p) + q)
        return total

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Grid(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = torch.as_tensor(v).float()
        self.learnable_params = [self.v]

    def compute(self, n):
        this = self
        arr = torch.stack([self.v for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
        total = 0.0
        for k in range(int(0), int(n)):
            a, b, c = arr
            total = (((total + a) + b) + c)
        return total

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Point(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.as_tensor(x).float()
        self.y = torch.as_tensor(y).float()
        self.learnable_params = [self.x, self.y]

    def get(self):
        this = self
        return (self.x, self.y)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Vec4(nn.Module):
    def __init__(self, w, x, y, z):
        super().__init__()
        self.w = torch.as_tensor(w).float()
        self.x = torch.as_tensor(x).float()
        self.y = torch.as_tensor(y).float()
        self.z = torch.as_tensor(z).float()
        self.learnable_params = [self.w, self.x, self.y, self.z]

    def f(self):
        this = self
        return ((self.w * 10), (self.x * 10), (self.y * 10), (self.z * 10))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Tensors(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = torch.as_tensor(a).float()
        self.b = torch.as_tensor(b).float()
        self.learnable_params = [self.a, self.b]

    def sum_parts(self):
        this = self
        x, y = self.a, self.b
        return (x + y)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Vec2(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.as_tensor(x).float()
        self.y = torch.as_tensor(y).float()
        self.learnable_params = [self.x, self.y]

    def f(self):
        this = self
        return ((self.x * 10), (self.y * 10))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class A(nn.Module):
    def __init__(self, x, y, z, c1, c2, total1, total2):
        super().__init__()
        self.x = torch.as_tensor(x).float()
        self.y = torch.as_tensor(y).float()
        self.z = torch.as_tensor(z).float()
        self.c1 = torch.as_tensor(c1).float()
        self.c2 = torch.as_tensor(c2).float()
        self.total1 = int(total1)
        self.total2 = int(total2)
        self.learnable_params = [self.x, self.y, self.z, self.c1, self.c2]

    def return_Real_type(self):
        this = self
        return self.x

    def return_Complex_type(self):
        this = self
        return self.c1

    def return_Natural_type(self):
        this = self
        return self.total1

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
a, b, c, d = 1, 2, 3, 4
print(a)
print(b)
print(c)
print(d)
a, b = 7, 8
print(a)
print(b)
x_complex, y_complex = 1j, 2j
print(x_complex)
print(y_complex)
x_arr, y_arr = torch.tensor([[1, 1], [1, 1]], device=DEVICE), torch.tensor([[1, 1], [1, 1]], device=DEVICE)
print(x_arr)
print(y_arr)
print(unpack_simple_array(torch.tensor([1, 2, 3], device=DEVICE)))
print(unpack_array_in_function_call(3))
sample_arr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=DEVICE)
for i in range(int(0), int(3)):
    a, b, c = sample_arr[int(i)]
    print(a)
    print(b)
    print(c)
p = Point(1.0, 2.0).to(DEVICE)
a, b = p.get()
result = (a + b)
print(a)
print(b)
print(result)
v = Vec4(0.5, 1.0, 2.0, 3.0).to(DEVICE)
a, b, c, d = v.f()
print(a)
print(b)
print(c)
print(d)
v = Vec2(0.5, 1.0).to(DEVICE)
a, b = v.f()
print(a)
print(b)
obj_A = A(1.0, 2.0, 3.0, 1j, 2j, 10, 20).to(DEVICE)
print(obj_A.return_Real_type())
print(obj_A.return_Complex_type())
print(obj_A.return_Natural_type())