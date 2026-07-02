import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

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
        arr = torch.stack([self.v for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i))]])
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

# === Program ===
p = Point(1.0, 2.0)
a, b = p.get()
result = (a + b)
physika_print(a)
physika_print(b)
physika_print(result)
v = Vec4(0.5, 1.0, 2.0, 3.0)
a, b, c, d = v.f()
physika_print(a)
physika_print(b)
physika_print(c)
physika_print(d)
v = Vec2(0.5, 1.0)
a, b = v.f()
physika_print(a)
physika_print(b)
a, b, c, d = 1, 2, 3, 4
physika_print(a)
physika_print(b)
physika_print(c)
physika_print(d)