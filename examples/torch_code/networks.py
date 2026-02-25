import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print

# === Functions ===
def sigma(x):
    x = torch.as_tensor(x).float()
    return (1.0 / (1.0 + torch.exp((0.0 - x))))

# === Classes ===
class OneLayerNet(nn.Module):
    def __init__(self, W0, c0, w1, b1):
        super().__init__()
        self.W0 = nn.Parameter(torch.tensor(W0).float() if not isinstance(W0, torch.Tensor) else W0.clone().detach().float())
        self.c0 = nn.Parameter(torch.tensor(c0).float() if not isinstance(c0, torch.Tensor) else c0.clone().detach().float())
        self.w1 = nn.Parameter(torch.tensor(w1).float() if not isinstance(w1, torch.Tensor) else w1.clone().detach().float())
        self.b1 = nn.Parameter(torch.tensor(b1).float() if not isinstance(b1, torch.Tensor) else b1.clone().detach().float())

    def forward(self, x):
        x = torch.as_tensor(x).float()
        return sigma(((self.w1 @ sigma(((self.W0 @ x) + self.c0))) + self.b1))

    def loss(self, y, target):
        return ((y - target) ** 2.0)

class FullyConnectedNetwork(nn.Module):
    def __init__(self, f, W, B, w, b, n):
        super().__init__()
        self.f = f
        self.W = nn.Parameter(torch.tensor(W).float() if not isinstance(W, torch.Tensor) else W.clone().detach().float())
        self.B = nn.Parameter(torch.tensor(B).float() if not isinstance(B, torch.Tensor) else B.clone().detach().float())
        self.w = nn.Parameter(torch.tensor(w).float() if not isinstance(w, torch.Tensor) else w.clone().detach().float())
        self.b = nn.Parameter(torch.tensor(b).float() if not isinstance(b, torch.Tensor) else b.clone().detach().float())
        self.n = n

    def forward(self, x):
        x = torch.as_tensor(x).float()
        n = int(self.n) if hasattr(self, 'n') else self.n.shape[0] if hasattr(self.n, 'shape') else 2
        for k in range(n):
            x = self.f(((self.W[int(k)] @ x) + self.B[int(k)]))
        return ((self.w @ x) + self.b)

    def loss(self, y, target):
        return ((y - target) ** 2.0)

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
net2 = FullyConnectedNetwork(sigma, W, B, w, b, 2.0)
physika_print(net2(torch.tensor([1.0, 2.0, 3.0])))
physika_print(net2(torch.tensor([0.0, 0.0, 0.0])))
physika_print(net2(torch.tensor([1.0, 1.0, 1.0])))