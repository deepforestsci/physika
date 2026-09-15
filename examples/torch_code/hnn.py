import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def tanh(x):
    return ((torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x))) - torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))) / (torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x))) + torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))))

def mse_loss(symplectic_gradient, target_gradient, N=None):
    if N is None:
        N = int(target_gradient.shape[0])
    loss = (((symplectic_gradient[int(0)] - target_gradient[int(0)]) ** 2.0) + ((symplectic_gradient[int((1 + 0))] - target_gradient[int(1)]) ** 2.0))
    return loss

def rand_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

# === Classes ===
class HamiltonianNet(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.W2 = nn.Parameter(torch.as_tensor(W2))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.learnable_params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float().requires_grad_(True)
        h = ((self.W2 @ tanh(((self.W1 @ x) + self.b1))) + self.b2)
        dh_grad = compute_grad(h, x)
        dh_dp = dh_grad[int(1), int(0)]
        dh_dq = (-dh_grad[int(0), int(0)])
        return torch.stack([torch.as_tensor(dh_dp), torch.as_tensor(dh_dq)])

    def train(self, train_X, train_y, epochs, lr):
        this = self
        train_X = torch.as_tensor(train_X, device=DEVICE).float()
        train_y = torch.as_tensor(train_y, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        for epoch in range(int(0), int(epochs)):
            loss = 0
            for i in range(int(0), int(8)):
                state = train_X[int(i)]
                target = train_y[int(i)]
                prediction = self(state)
                loss = mse_loss(prediction, target)
                learnable_grads = compute_grad(loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
        return loss

    def evaluate(self, X, y):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        y = torch.as_tensor(y, device=DEVICE).float()
        total_loss = 0
        for j in range(int(0), int(8)):
            pred = self(X[int(j)])
            current_loss = mse_loss(pred, y[int(j)])
            total_loss = (total_loss + current_loss)
        return (total_loss / 8)

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.W1.copy_((self.W1 - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b1.copy_((self.b1 - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.W2.copy_((self.W2 - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.b2.copy_((self.b2 - (lr * learnable_grads[int(3)])))

# === Program ===
X = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]], [[0.0], [(-1.0)]], [[(-1.0)], [0.0]], [[0.5], [0.5]], [[(-0.5)], [(-0.5)]], [[0.7], [(-0.7)]], [[(-0.7)], [0.7]]], device=DEVICE)
y = torch.tensor([[1.0, 0.0], [0.0, (-1.0)], [(-1.0), 0.0], [0.0, 1.0], [0.5, (-0.5)], [(-0.5), 0.5], [(-0.7), (-0.7)], [0.7, 0.7]], device=DEVICE)
W1 = rand_array(16, 2, 0.05)
b1 = rand_array(16, 1, 0.05)
W2 = rand_array(1, 16, 0.05)
b2 = 0.0
H_net = HamiltonianNet(W1, b1, W2, b2).to(DEVICE)
loss_before = H_net.evaluate(X, y)
print(print(loss_before))
epochs = 1000
lr = 0.01
train_loss = H_net.train(X, y, epochs, lr)
print(print(train_loss))
loss_after = H_net.evaluate(X, y)
print(print(loss_after))