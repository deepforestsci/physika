import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def σ(x):
    return (1.0 / (1.0 + torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))))

def σ1d(x):
    len_x = get_1d_array_length(x)
    results = torch.zeros(int(len_x), device=DEVICE)
    for i in range(int(0), int(len_x)):
        results[int(i)] = (1.0 / (1.0 + torch.exp(x[int(i)] if isinstance(x[int(i)], torch.Tensor) else torch.tensor(float(x[int(i)])))))
    return results

# === Classes ===
class OneLayerNet(nn.Module):
    def __init__(self, W0, b0, W1, b1):
        super().__init__()
        self.W0 = nn.Parameter(torch.as_tensor(W0))
        self.b0 = nn.Parameter(torch.as_tensor(b0))
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        return σ(((self.W1 * σ(((self.W0 @ x) + self.b0))) + self.b1))

    def loss(self, y, target):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        target = torch.as_tensor(target, device=DEVICE).float()
        return ((y - target) ** 2.0)

class FullyConnectedNetwork(nn.Module):
    def __init__(self, W, B, w, b):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.B = nn.Parameter(torch.as_tensor(B))
        self.w = nn.Parameter(torch.as_tensor(w))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        for i in range(len(self.W)):
            x = σ1d(((self.W[int(i)] @ x) + self.B[int(i)]))
        return ((self.w @ x) + self.b)

    def loss(self, y, target):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        target = torch.as_tensor(target, device=DEVICE).float()
        return ((y - target) ** 2.0)

# === Program ===
W0 = torch.tensor([0.1, 0.2], device=DEVICE)
b0 = 0.1
w1 = 0.1
b1 = 0.3
net1 = OneLayerNet(W0, b0, w1, b1)
x = torch.tensor([1.0, 2.0], device=DEVICE)
print(net1(x))
W = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]], device=DEVICE)
B = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], device=DEVICE)
w = torch.tensor([0.5, 0.5, 0.5], device=DEVICE)
b = 0.1
net2 = FullyConnectedNetwork(W, B, w, b)
print(net2(torch.tensor([1.0, 2.0, 3.0], device=DEVICE)))
print(net2(torch.tensor([0.0, 0.0, 0.0], device=DEVICE)))
print(net2(torch.tensor([1.0, 1.0, 1.0], device=DEVICE)))