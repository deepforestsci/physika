import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print
from runtime import train
from runtime import evaluate
from runtime import compute_grad

# === Functions ===
def tanh(x):
    x = torch.as_tensor(x).float()
    return ((torch.exp(x) - torch.exp((0.0 - x))) / (torch.exp(x) + torch.exp((0.0 - x))))

# === Classes ===
class HamiltonianNet(nn.Module):
    def __init__(self, W1, b1, w2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.tensor(W1).float() if not isinstance(W1, torch.Tensor) else W1.clone().detach().float())
        self.b1 = nn.Parameter(torch.tensor(b1).float() if not isinstance(b1, torch.Tensor) else b1.clone().detach().float())
        self.w2 = nn.Parameter(torch.tensor(w2).float() if not isinstance(w2, torch.Tensor) else w2.clone().detach().float())
        self.b2 = nn.Parameter(torch.tensor(b2).float() if not isinstance(b2, torch.Tensor) else b2.clone().detach().float())

    def forward(self, x):
        x = torch.as_tensor(x).float()
        h = ((self.w2 @ torch.tanh(((self.W1 @ x) + self.b1))) + self.b2)
        return h

    def loss(self, H, target, x):
        lo = (((compute_grad(H, x)[int(1.0)] - target[int(0.0)]) ** 2.0) + (((0.0 - compute_grad(H, x)[int(0.0)]) - target[int(1.0)]) ** 2.0))
        return lo

# === Program ===
X = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, (-1.0)], [(-1.0), 0.0], [0.5, 0.5], [(-0.5), (-0.5)], [0.7, (-0.7)], [(-0.7), 0.7]])
y = torch.tensor([[1.0, 0.0], [0.0, (-1.0)], [(-1.0), 0.0], [0.0, 1.0], [0.5, (-0.5)], [(-0.5), 0.5], [(-0.7), (-0.7)], [0.7, 0.7]])
W1 = torch.tensor([[0.5, 0.1], [0.1, 0.5], [0.3, 0.3], [0.4, 0.2], [0.2, 0.4], [0.1, 0.1], [0.3, 0.1], [0.1, 0.3], [0.2, 0.2], [0.4, 0.4], [0.5, 0.3], [0.3, 0.5], [0.2, 0.1], [0.1, 0.2], [0.4, 0.1], [0.1, 0.4]])
b1 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
w2 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
b2 = 0.0
H_net = HamiltonianNet(W1, b1, w2, b2)
loss_before = evaluate(H_net, X, y)
physika_print(loss_before)
epochs = 500.0
lr = 0.01
H_trained = train(H_net, X, y, epochs, lr)
loss_after = evaluate(H_trained, X, y)
physika_print(loss_after)