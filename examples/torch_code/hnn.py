import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import train
from physika.runtime import evaluate

# === Functions ===
def tanh(x):
    return ((torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x))) - torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))) / (torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x))) + torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))))

# === Classes ===
class HamiltonianNet(nn.Module):
    def __init__(self, W1, b1, w2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1).float())
        self.b1 = nn.Parameter(torch.as_tensor(b1).float())
        self.w2 = nn.Parameter(torch.as_tensor(w2).float())
        self.b2 = nn.Parameter(torch.as_tensor(b2).float())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x).float()
        h = ((self.w2 @ tanh(((self.W1 @ x) + self.b1))) + self.b2)
        return h

    def loss(self, H, target):
        this = self
        H = torch.as_tensor(H).float()
        target = torch.as_tensor(target).float()
        lo = (((compute_grad(H, x)[int(1)] - target[int(0)]) ** 2.0) + (((0.0 - compute_grad(H, x)[int(0)]) - target[int(1)]) ** 2.0))
        return lo

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

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
epochs = 500
lr = 0.01
H_trained = train(H_net, X, y, epochs, lr)
loss_after = evaluate(H_trained, X, y)
physika_print(loss_after)