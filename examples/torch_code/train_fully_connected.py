import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import train
from runtime import evaluate

# === Functions ===
def sigma(x):
    x = torch.as_tensor(x).float()
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
        n = int(self.n) if hasattr(self, 'n') else self.n.shape[0] if hasattr(self.n, 'shape') else 2
        for k in range(n):
            x = self.f(((self.W[int(k)] @ x) + self.B[int(k)]))
        return ((self.w @ x) + self.b)

    def loss(self, y, target):
        return ((y - target) ** 2.0)

# === Program ===
X = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
y = torch.tensor([0.2, 0.4, 0.6, 0.9])
W = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]])
B = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
w = torch.tensor([0.5, 0.5, 0.5])
b = 0.1
net = FullyConnectedNetwork(sigma, W, B, w, b, 2.0)
loss_before = evaluate(net, X, y)
physika_print(loss_before)
epochs = 1000.0
lr = 0.1
net_trained = train(net, X, y, epochs, lr)
loss_after = evaluate(net_trained, X, y)
physika_print(loss_after)
physika_print(net_trained(torch.tensor([1.0, 0.0, 0.0])))
physika_print(net_trained(torch.tensor([0.0, 1.0, 0.0])))
physika_print(net_trained(torch.tensor([0.0, 0.0, 1.0])))
physika_print(net_trained(torch.tensor([1.0, 1.0, 1.0])))