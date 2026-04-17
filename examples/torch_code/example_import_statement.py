import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def fact(n):
    if n == 0.0:
        return 1.0
    else:
        return (n * fact((n - 1.0)))

def sigma(x):
    return (1.0 / (1.0 + torch.exp((0.0 - x))))

# === Classes ===
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
        for k in range(len(self.W)):
            x = self.f(((self.W[int(k)] @ x) + self.B[int(k)]))
        return ((self.w @ x) + self.b)

    def loss(self, y, target):
        return ((y - target) ** 2.0)

# === Program ===
physika_print(fact(1.0))
physika_print(sigma(torch.tensor([1.0])))