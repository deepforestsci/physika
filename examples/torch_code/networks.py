import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def get_1d_array_length(x, m=None):
    if m is None:
        m = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def get_2d_array_num_rows(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return torch.sum(torch.stack([torch.as_tensor(1) for i in range(int(m))]).float())

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def sigma(x):
    rows = get_2d_array_num_rows(x)
    cols = get_1d_array_length(x[int(0)])
    results = zero_2d_array(rows, cols)
    for i in range(int(0), int(rows)):
        for j in range(int(0), int(cols)):
            results[int(i), int(j)] = (1.0 / (1.0 + torch.exp((0.0 - x[int(i), int(j)]) if isinstance((0.0 - x[int(i), int(j)]), torch.Tensor) else torch.tensor(float((0.0 - x[int(i), int(j)]))))))
    return results

# === Classes ===
class OneLayerNet(nn.Module):
    def __init__(self, W0, c0, w1, b1):
        super().__init__()
        self.W0 = nn.Parameter(torch.as_tensor(W0))
        self.c0 = nn.Parameter(torch.as_tensor(c0))
        self.w1 = nn.Parameter(torch.as_tensor(w1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.learnable_params = [self.W0, self.c0, self.w1, self.b1]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = sigma(((self.W0 @ x) + self.c0))
        results = ((self.w1 @ z) + self.b1)
        return results[int(0), int(0)]

    def loss(self, y, target):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        target = torch.as_tensor(target, device=DEVICE).float()
        return ((y - target) ** 2.0)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class FullyConnectedNetwork(nn.Module):
    def __init__(self, W, B, w, b, n):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.B = nn.Parameter(torch.as_tensor(B))
        self.w = nn.Parameter(torch.as_tensor(w))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.n = int(n)
        self.learnable_params = [self.W, self.B, self.w, self.b]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        for k in range(len(self.W)):
            x = sigma(((self.W[int(k)] @ x) + self.B[int(k)]))
        return ((self.w @ x) + self.b)

    def loss(self, y, target):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        target = torch.as_tensor(target, device=DEVICE).float()
        return ((y - target) ** 2.0)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
W0 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.2), torch.as_tensor(0.3)])), torch.as_tensor(torch.stack([torch.as_tensor(0.4), torch.as_tensor(0.5), torch.as_tensor(0.6)]))])
c0 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.2)]))])
w1 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.7), torch.as_tensor(0.8)]))])
b1 = 0.3
net1 = OneLayerNet(W0, c0, w1, b1)
print(net1(torch.tensor([1.0, 2.0, 3.0], device=DEVICE)))
W = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.2), torch.as_tensor(0.3)])), torch.as_tensor(torch.stack([torch.as_tensor(0.4), torch.as_tensor(0.5), torch.as_tensor(0.6)])), torch.as_tensor(torch.stack([torch.as_tensor(0.7), torch.as_tensor(0.8), torch.as_tensor(0.9)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.2), torch.as_tensor(0.3), torch.as_tensor(0.4)])), torch.as_tensor(torch.stack([torch.as_tensor(0.5), torch.as_tensor(0.6), torch.as_tensor(0.7)])), torch.as_tensor(torch.stack([torch.as_tensor(0.8), torch.as_tensor(0.9), torch.as_tensor(0.1)]))]))])
B = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.2)])), torch.as_tensor(torch.stack([torch.as_tensor(0.3)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.2)])), torch.as_tensor(torch.stack([torch.as_tensor(0.3)]))]))])
w = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.5), torch.as_tensor(0.5), torch.as_tensor(0.5)]))])
b = 0.1
net2 = FullyConnectedNetwork(W, B, w, b, 2)
print(net2(torch.tensor([[1.0], [2.0], [3.0]], device=DEVICE)))
print(net2(torch.tensor([[0.0], [0.0], [0.0]], device=DEVICE)))
print(net2(torch.tensor([[1.0], [1.0], [1.0]], device=DEVICE)))