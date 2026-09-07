import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import train
from physika.runtime import evaluate

# === Functions ===
def tanh(x):
    return ((torch.exp(torch.as_tensor(x).float()) - torch.exp(torch.as_tensor((0.0 - x)).float())) / (torch.exp(torch.as_tensor(x).float()) + torch.exp(torch.as_tensor((0.0 - x)).float())))

# === Classes ===
class HamiltonianNet(nn.Module):
    def __init__(self, W1, b1, w2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.w2 = nn.Parameter(torch.as_tensor(w2))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.learnable_params = [self.W1, self.b1, self.w2, self.b2]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        h = ((self.w2 @ tanh(((self.W1 @ x) + self.b1))) + self.b2)
        dh_grad = compute_grad(h, x)
        dh_dp = dh_grad[int(1), int(0)]
        dh_dq = (-dh_grad[int(0), int(0)])
        return torch.tensor([[dh_dp], [dh_dq]], device=DEVICE)

    def loss(self, symplectic_gradient, target):
        this = self
        symplectic_gradient = torch.as_tensor(symplectic_gradient, device=DEVICE).float()
        target = torch.as_tensor(target, device=DEVICE).float()
        lo = (((symplectic_gradient[int(0)] - target[int(0)]) ** 2.0) + ((symplectic_gradient[int(1)] - target[int(1)]) ** 2.0))
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
X = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor((-1.0))]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor((-1.0))])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.5)])), torch.as_tensor(torch.stack([torch.as_tensor(0.5)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor((-0.5))])), torch.as_tensor(torch.stack([torch.as_tensor((-0.5))]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.7)])), torch.as_tensor(torch.stack([torch.as_tensor((-0.7))]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor((-0.7))])), torch.as_tensor(torch.stack([torch.as_tensor(0.7)]))]))])
y = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(1.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor((-1.0))]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor((-1.0))])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(1.0)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.5)])), torch.as_tensor(torch.stack([torch.as_tensor((-0.5))]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor((-0.5))])), torch.as_tensor(torch.stack([torch.as_tensor(0.5)]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor((-0.7))])), torch.as_tensor(torch.stack([torch.as_tensor((-0.7))]))])), torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.7)])), torch.as_tensor(torch.stack([torch.as_tensor(0.7)]))]))])
W1 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.5), torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.5)])), torch.as_tensor(torch.stack([torch.as_tensor(0.3), torch.as_tensor(0.3)])), torch.as_tensor(torch.stack([torch.as_tensor(0.4), torch.as_tensor(0.2)])), torch.as_tensor(torch.stack([torch.as_tensor(0.2), torch.as_tensor(0.4)])), torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.3), torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.3)])), torch.as_tensor(torch.stack([torch.as_tensor(0.2), torch.as_tensor(0.2)])), torch.as_tensor(torch.stack([torch.as_tensor(0.4), torch.as_tensor(0.4)])), torch.as_tensor(torch.stack([torch.as_tensor(0.5), torch.as_tensor(0.3)])), torch.as_tensor(torch.stack([torch.as_tensor(0.3), torch.as_tensor(0.5)])), torch.as_tensor(torch.stack([torch.as_tensor(0.2), torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.2)])), torch.as_tensor(torch.stack([torch.as_tensor(0.4), torch.as_tensor(0.1)])), torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.4)]))])
b1 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)])), torch.as_tensor(torch.stack([torch.as_tensor(0.0)]))])
w2 = torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1), torch.as_tensor(0.1)]))])
b2 = 0.0
H_net = HamiltonianNet(W1, b1, w2, b2)
loss_before = evaluate(H_net, X, y)
print(loss_before)
epochs = 500
lr = 0.01
H_trained = train(H_net, X, y, epochs, lr)
loss_after = evaluate(H_trained, X, y)
print(loss_after)