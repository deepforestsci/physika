import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def rand_2d_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

# === Classes ===
class Conv1d(nn.Module):
    def __init__(self, W, b):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = ((self.W @ x) + self.b)
        return z

    def update_params(self, lr, grad_W, grad_b):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        grad_W = torch.as_tensor(grad_W, device=DEVICE).float()
        grad_b = torch.as_tensor(grad_b, device=DEVICE).float()
        with torch.no_grad():
            self.W.copy_((self.W - (lr * grad_W)))
        with torch.no_grad():
            self.b.copy_((self.b - (lr * grad_b)))

class MLP(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.W2 = nn.Parameter(torch.as_tensor(W2))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z1 = ((self.W1 @ x) + self.b1)
        a1 = torch.nn.functional.gelu(z1)
        z2 = ((self.W2 @ a1) + self.b2)
        return z2

    def update_params(self, lr, grad_W1, grad_b1, grad_W2, grad_b2):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        grad_W1 = torch.as_tensor(grad_W1, device=DEVICE).float()
        grad_b1 = torch.as_tensor(grad_b1, device=DEVICE).float()
        grad_W2 = torch.as_tensor(grad_W2, device=DEVICE).float()
        grad_b2 = torch.as_tensor(grad_b2, device=DEVICE).float()
        with torch.no_grad():
            self.W1.copy_((self.W1 - (lr * grad_W1)))
        with torch.no_grad():
            self.b1.copy_((self.b1 - (lr * grad_b1)))
        with torch.no_grad():
            self.W2.copy_((self.W2 - (lr * grad_W2)))
        with torch.no_grad():
            self.b2.copy_((self.b2 - (lr * grad_b2)))

class SpectralConv(nn.Module):
    def __init__(self, weights1, in_ch, out_ch, modes):
        super().__init__()
        self.weights1 = nn.Parameter(torch.as_tensor(weights1))
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.modes = int(modes)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x_ft = torch.fft.rfft(x)
        x_ft = x_ft[:, :int(self.modes)]
        out_ft = compl_mul1d(x_ft, self.weights1)
        results = torch.fft.irfft(out_ft, len(x[int(0)]))
        return results

    def update_params(self, lr, grad_weights1):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        grad_weights1 = torch.as_tensor(grad_weights1, device=DEVICE).float()
        with torch.no_grad():
            self.weights1.copy_((self.weights1 - (lr * grad_weights1)))

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
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x = self.p(x)
        x = torch.nn.functional.gelu((self.mlp0(self.conv0(x)) + self.w0(x)))
        x = torch.nn.functional.gelu((self.mlp1(self.conv1(x)) + self.w1(x)))
        x = torch.nn.functional.gelu((self.mlp2(self.conv2(x)) + self.w2(x)))
        x = (self.mlp3(self.conv3(x)) + self.w3(x))
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
        for idx in range(int(0), int(epochs)):
            print(idx)
            epoch_loss = 0
            for j in range(int(0), int(len_dataset)):
                pred = self(X[int(j)])
                current_loss = self.loss(pred, y[int(j)])
                epoch_loss = (epoch_loss + current_loss)
                learnable_grads = compute_grad(current_loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
            last_loss = (epoch_loss / len_dataset)
            print(last_loss)
        return last_loss

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        self.p.update_params(lr, learnable_grads[int(0)], learnable_grads[int(1)])
        self.conv0.update_params(lr, learnable_grads[int(2)])
        self.conv1.update_params(lr, learnable_grads[int(3)])
        self.conv2.update_params(lr, learnable_grads[int(4)])
        self.conv3.update_params(lr, learnable_grads[int(5)])
        self.mlp0.update_params(lr, learnable_grads[int(6)], learnable_grads[int(7)], learnable_grads[int(8)], learnable_grads[int(9)])
        self.mlp1.update_params(lr, learnable_grads[int(10)], learnable_grads[int(11)], learnable_grads[int(12)], learnable_grads[int(13)])
        self.mlp2.update_params(lr, learnable_grads[int(14)], learnable_grads[int(15)], learnable_grads[int(16)], learnable_grads[int(17)])
        self.mlp3.update_params(lr, learnable_grads[int(18)], learnable_grads[int(19)], learnable_grads[int(20)], learnable_grads[int(21)])
        self.w0.update_params(lr, learnable_grads[int(22)], learnable_grads[int(23)])
        self.w1.update_params(lr, learnable_grads[int(24)], learnable_grads[int(25)])
        self.w2.update_params(lr, learnable_grads[int(26)], learnable_grads[int(27)])
        self.w3.update_params(lr, learnable_grads[int(28)], learnable_grads[int(29)])
        self.q.update_params(lr, learnable_grads[int(30)], learnable_grads[int(31)], learnable_grads[int(32)], learnable_grads[int(33)])

# === Program ===
width = 16
modes = 8
Wp = rand_2d_array(width, 1, 0.1)
Bp = rand_2d_array(width, 1, 0.1)
p = Conv1d(Wp, Bp).to(DEVICE)
weights0 = random_complex(width, width, modes)
weights1 = random_complex(width, width, modes)
weights2 = random_complex(width, width, modes)
weights3 = random_complex(width, width, modes)
conv0 = SpectralConv(weights0, width, width, modes).to(DEVICE)
conv1 = SpectralConv(weights1, width, width, modes).to(DEVICE)
conv2 = SpectralConv(weights2, width, width, modes).to(DEVICE)
conv3 = SpectralConv(weights3, width, width, modes).to(DEVICE)
Ww0 = rand_2d_array(width, width, 0.1)
Bw0 = rand_2d_array(width, 1, 0.1)
w0 = Conv1d(Ww0, Bw0).to(DEVICE)
w1 = Conv1d(Ww0, Bw0).to(DEVICE)
w2 = Conv1d(Ww0, Bw0).to(DEVICE)
w3 = Conv1d(Ww0, Bw0).to(DEVICE)
W1 = rand_2d_array(width, width, 0.1)
b1 = torch.zeros(int(width), int(1), device=DEVICE)
W2 = rand_2d_array(width, width, 0.1)
b2 = torch.zeros(int(width), int(1), device=DEVICE)
mlp0 = MLP(W1, b1, W2, b2).to(DEVICE)
mlp1 = MLP(W1, b1, W2, b2).to(DEVICE)
mlp2 = MLP(W1, b1, W2, b2).to(DEVICE)
mlp3 = MLP(W1, b1, W2, b2).to(DEVICE)
q_width = (2 * width)
Wq1 = rand_2d_array(q_width, width, 0.1)
bq1 = torch.zeros(int(q_width), int(1), device=DEVICE)
Wq2 = rand_2d_array(1, q_width, 0.1)
bq2 = torch.zeros(int(1), int(1), device=DEVICE)
q = MLP(Wq1, bq1, Wq2, bq2).to(DEVICE)
train_X = torch.stack([torch.stack([torch.stack([torch.distributions.Normal(0, 1).rsample() for _fi_k in range(int(100)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(1)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(1)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
train_y = torch.stack([torch.stack([torch.stack([torch.distributions.Normal(0, 1).rsample() for _fi_k in range(int(100)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(1)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(1)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
fno_obj = FNO1d(p, conv0, conv1, conv2, conv3, mlp0, mlp1, mlp2, mlp3, w0, w1, w2, w3, q).to(DEVICE)
epochs = 1
loss = fno_obj.train(train_X, train_y, epochs, 0.0001)
print(print(loss))