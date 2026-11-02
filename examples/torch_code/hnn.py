import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def get_1d_array_length(x):
    total, temp = 0, 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total
def hamiltonian(x):
    q, p = x[int(0)], x[int(1)]
    m, ω = 1, 1
    H = (((p ** 2) / (2 * m)) + (((m * (ω ** 2)) * (q ** 2)) / 2))
    return H

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
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float().requires_grad_(True)
        H = ((self.W2 @ tanh(((self.W1 @ x) + self.b1))) + self.b2)
        partial_H_over_partial_x = compute_grad(H, x)
        partial_h_over_partial_q = partial_H_over_partial_x[int(0), int(0)]
        partial_h_over_partial_p = partial_H_over_partial_x[int(1), int(0)]
        return torch.stack([torch.as_tensor(partial_h_over_partial_p), torch.as_tensor((-partial_h_over_partial_q))])

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
        for idx in range(len(learnable_grads)):
            with torch.no_grad():
                self.learnable_params[int(idx)].copy_((self.learnable_params[int(idx)] - (lr * learnable_grads[int(idx)])))

# === Program ===
x = torch.as_tensor(torch.tensor([1, 0.5], device=DEVICE)).requires_grad_(True).to(DEVICE)
H = hamiltonian(x)
partial_H_over_partial_x = compute_grad(H, x)
partial_H_over_partial_q = partial_H_over_partial_x[int(0)]
partial_H_over_partial_p = partial_H_over_partial_x[int(1)]
print(print(partial_H_over_partial_x))
print(print(partial_H_over_partial_q))
print(print(partial_H_over_partial_p))
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