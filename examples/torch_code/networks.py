import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def sigma(x):
    return (1.0 / (1.0 + torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))))

# === Classes ===
class OneLayerNet(nn.Module):
    def __init__(self, W0, c0, w1, b1):
        super().__init__()
        self.W0 = nn.Parameter(torch.as_tensor(W0).float())
        self.c0 = nn.Parameter(torch.as_tensor(c0).float())
        self.w1 = nn.Parameter(torch.as_tensor(w1).float())
        self.b1 = nn.Parameter(torch.as_tensor(b1).float())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x).float()
        return sigma(((self.w1 @ sigma(((self.W0 @ x) + self.c0))) + self.b1))

    def loss(self, y, target):
        this = self
        y = torch.as_tensor(y).float()
        target = torch.as_tensor(target).float()
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
    def __init__(self, f, W, B, w, b, n):
        super().__init__()
        self.f = torch.as_tensor(f).float() if isinstance(f, (int, float, torch.Tensor)) else f
        self.W = nn.Parameter(torch.as_tensor(W).float())
        self.B = nn.Parameter(torch.as_tensor(B).float())
        self.w = nn.Parameter(torch.as_tensor(w).float())
        self.b = nn.Parameter(torch.as_tensor(b).float())
        self.n = torch.as_tensor(n).float() if isinstance(n, (int, float, torch.Tensor)) else n

    def forward(self, x):
        this = self
        x = torch.as_tensor(x).float()
        for k in range(len(self.W)):
            x = self.f(((self.W[int(k)] @ x) + self.B[int(k)]))
        return ((self.w @ x) + self.b)

    def loss(self, y, target):
        this = self
        y = torch.as_tensor(y).float()
        target = torch.as_tensor(target).float()
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
W0 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
c0 = torch.tensor([0.1, 0.2])
w1 = torch.tensor([0.7, 0.8])
b1 = 0.3
net1 = OneLayerNet(W0, c0, w1, b1)
physika_print(net1(torch.tensor([1.0, 2.0, 3.0])))
W = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]])
B = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
w = torch.tensor([0.5, 0.5, 0.5])
b = 0.1
net2 = FullyConnectedNetwork(sigma, W, B, w, b, 2)
physika_print(net2(torch.tensor([1.0, 2.0, 3.0])))
physika_print(net2(torch.tensor([0.0, 0.0, 0.0])))
physika_print(net2(torch.tensor([1.0, 1.0, 1.0])))