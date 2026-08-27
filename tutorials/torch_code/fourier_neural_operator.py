import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Classes ===
class Conv1d(nn.Module):
    def __init__(self, W, b):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.learnable_params = [self.W, self.b]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = ((self.W @ x) + self.b)
        return z

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.W.copy_((self.W - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b.copy_((self.b - (lr * learnable_grads[int(1)])))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class MLP(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.W2 = nn.Parameter(torch.as_tensor(W2))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.learnable_params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z1 = ((self.W1 @ x) + self.b1)
        a1 = torch.nn.functional.gelu(z1)
        z2 = ((self.W2 @ a1) + self.b2)
        return z2

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

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class SpectralConv(nn.Module):
    def __init__(self, weights1, in_ch, out_ch, modes):
        super().__init__()
        self.weights1 = nn.Parameter(torch.as_tensor(weights1))
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.modes = int(modes)
        self.learnable_params = [self.weights1]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x_ft = torch.fft.rfft(x)
        x_ft = x_ft[:, :self.modes]
        out_ft = compl_mul1d(x_ft, self.weights1)
        results = torch.fft.irfft(out_ft, len(x[int(0)]))
        return results

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.weights1.copy_((self.weights1 - (lr * learnable_grads[int(0)])))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class FNO1d(nn.Module):
    def __init__(self, p, conv0, conv1, conv2, conv3, mlp0, mlp1, mlp2, mlp3, w0, w1, w2, w3, q):
        super().__init__()
        self.add_module('p', p)
        self.add_module('conv0', conv0)
        self.add_module('conv1', conv1)
        self.add_module('conv2', conv2)
        self.add_module('conv3', conv3)
        self.add_module('mlp0', mlp0)
        self.add_module('mlp1', mlp1)
        self.add_module('mlp2', mlp2)
        self.add_module('mlp3', mlp3)
        self.add_module('w0', w0)
        self.add_module('w1', w1)
        self.add_module('w2', w2)
        self.add_module('w3', w3)
        self.add_module('q', q)
        self.learnable_params = []

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x = self.p(x)
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = torch.nn.functional.gelu((x1 + x2))
        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = torch.nn.functional.gelu((x1 + x2))
        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = torch.nn.functional.gelu((x1 + x2))
        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = (x1 + x2)
        x = self.q(x)
        return x

    def loss(self, pred, label):
        this = self
        pred = torch.as_tensor(pred, device=DEVICE).float()
        label = torch.as_tensor(label, device=DEVICE).float()
        diff = (pred - label)
        return torch.mean((diff ** 2) if isinstance((diff ** 2), torch.Tensor) else torch.tensor(float((diff ** 2))))

    def train(self, X, y, epochs, lr):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        y = torch.as_tensor(y, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        len_dataset = len(X)
        for i in range(int(0), int(epochs)):
            epoch_loss = 0
            for j in range(int(0), int(len_dataset)):
                pred = self(X[int(j)])
                current_loss = self.loss(pred, y[int(j)])
                epoch_loss = (epoch_loss + current_loss)
                dp = compute_grad(current_loss, self.p.learnable_params)
                dconv0 = compute_grad(current_loss, self.conv0.learnable_params)
                dconv1 = compute_grad(current_loss, self.conv1.learnable_params)
                dconv2 = compute_grad(current_loss, self.conv2.learnable_params)
                dconv3 = compute_grad(current_loss, self.conv3.learnable_params)
                dmlp0 = compute_grad(current_loss, self.mlp0.learnable_params)
                dmlp1 = compute_grad(current_loss, self.mlp1.learnable_params)
                dmlp2 = compute_grad(current_loss, self.mlp2.learnable_params)
                dmlp3 = compute_grad(current_loss, self.mlp3.learnable_params)
                dw0 = compute_grad(current_loss, self.w0.learnable_params)
                dw1 = compute_grad(current_loss, self.w1.learnable_params)
                dw2 = compute_grad(current_loss, self.w2.learnable_params)
                dw3 = compute_grad(current_loss, self.w3.learnable_params)
                dq = compute_grad(current_loss, self.q.learnable_params)
                self.p.update_params(lr, dp)
                self.conv0.update_params(lr, dconv0)
                self.conv1.update_params(lr, dconv1)
                self.conv2.update_params(lr, dconv2)
                self.conv3.update_params(lr, dconv3)
                self.mlp0.update_params(lr, dmlp0)
                self.mlp1.update_params(lr, dmlp1)
                self.mlp2.update_params(lr, dmlp2)
                self.mlp3.update_params(lr, dmlp3)
                self.q.update_params(lr, dq)
            last_loss = (epoch_loss / len_dataset)
        return last_loss

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
width = 16
modes = 8
Wp = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(1),)) for _fi_i in range(int(width)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
Bp = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(1),)) for _fi_i in range(int(width)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
p = Conv1d(Wp, Bp)
weights0 = random_complex(width, width, modes)
weights1 = random_complex(width, width, modes)
weights2 = random_complex(width, width, modes)
weights3 = random_complex(width, width, modes)
conv0 = SpectralConv(weights0, width, width, modes).to(DEVICE)
conv1 = SpectralConv(weights1, width, width, modes).to(DEVICE)
conv2 = SpectralConv(weights2, width, width, modes).to(DEVICE)
conv3 = SpectralConv(weights3, width, width, modes).to(DEVICE)
Ww0 = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(width),)) for _fi_i in range(int(width)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
Bw0 = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(1),)) for _fi_i in range(int(width)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
w0 = Conv1d(Ww0, Bw0).to(DEVICE)
w1 = Conv1d(Ww0, Bw0).to(DEVICE)
w2 = Conv1d(Ww0, Bw0).to(DEVICE)
w3 = Conv1d(Ww0, Bw0).to(DEVICE)
W1 = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(width),)) for _fi_i in range(int(width)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1 = torch.zeros(width, 1)
W2 = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(width),)) for _fi_i in range(int(width)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2 = torch.zeros(width, 1)
mlp0 = MLP(W1, b1, W2, b2).to(DEVICE)
mlp1 = MLP(W1, b1, W2, b2).to(DEVICE)
mlp2 = MLP(W1, b1, W2, b2).to(DEVICE)
mlp3 = MLP(W1, b1, W2, b2).to(DEVICE)
q_width = (2 * width)
Wq1 = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(width),)) for _fi_i in range(int(q_width)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
bq1 = torch.zeros(q_width, 1)
Wq2 = torch.stack([torch.distributions.Normal(0.0, 0.1).rsample((int(q_width),)) for _fi_i in range(int(1)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
bq2 = torch.zeros(1, 1)
q = MLP(Wq1, bq1, Wq2, bq2)
train_X = torch.stack([torch.stack([torch.stack([torch.distributions.Normal(0, 1).rsample() for _fi_k in range(int(100)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(1)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(1)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
train_y = torch.stack([torch.stack([torch.stack([torch.distributions.Normal(0, 1).rsample() for _fi_k in range(int(100)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(1)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(1)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
fno_obj = FNO1d(p, conv0, conv1, conv2, conv3, mlp0, mlp1, mlp2, mlp3, w0, w1, w2, w3, q).to(DEVICE)
epochs = 1
loss = fno_obj.train(train_X, train_y, epochs, 0.0001)
print(loss)