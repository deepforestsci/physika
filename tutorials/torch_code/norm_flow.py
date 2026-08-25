import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def len1d(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def relu(x):
    return ((x + torch.abs(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) * 0.5)

def linear(x, W, b):
    in_dim = len1d(x)
    col = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(1)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(in_dim)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    col[:, int(0)] = x
    res = (W @ col)
    return (res[:, int(0)] + b)

def flatten(img, rows, cols):
    d = (rows * cols)
    out = torch.stack([(i * 0) for _fi_i in range(int(d)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    for i in range(int(0), int(rows)):
        out[(i * cols):((i + 1) * cols)] = img[int(i), :]
    return out

def unflatten(x, rows, cols):
    img = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    for i in range(int(0), int(rows)):
        img[int(i), :] = x[(i * cols):((i + 1) * cols)]
    return img

def dequantize(x, d):
    u = torch.distributions.Uniform(0.0, 1.0).rsample((int(d),)).to(DEVICE)
    return ((x + u) / 256.0)

def log_pz(x, d):
    return (torch.sum((((-0.5) * x) * x) if isinstance((((-0.5) * x) * x), torch.Tensor) else torch.tensor(float((((-0.5) * x) * x)))) - ((d * 0.5) * torch.log((2.0 * 3.14159265) if isinstance((2.0 * 3.14159265), torch.Tensor) else torch.tensor(float((2.0 * 3.14159265))))))

# === Classes ===
class RealNVP(nn.Module):
    def __init__(self, W1a_s, b1a_s, W2a_s, b2a_s, W1a_m, b1a_m, W2a_m, b2a_m, W1b_s, b1b_s, W2b_s, b2b_s, W1b_m, b1b_m, W2b_m, b2b_m, n, d):
        super().__init__()
        self.W1a_s = nn.Parameter(torch.as_tensor(W1a_s))
        self.b1a_s = nn.Parameter(torch.as_tensor(b1a_s))
        self.W2a_s = nn.Parameter(torch.as_tensor(W2a_s))
        self.b2a_s = nn.Parameter(torch.as_tensor(b2a_s))
        self.W1a_m = nn.Parameter(torch.as_tensor(W1a_m))
        self.b1a_m = nn.Parameter(torch.as_tensor(b1a_m))
        self.W2a_m = nn.Parameter(torch.as_tensor(W2a_m))
        self.b2a_m = nn.Parameter(torch.as_tensor(b2a_m))
        self.W1b_s = nn.Parameter(torch.as_tensor(W1b_s))
        self.b1b_s = nn.Parameter(torch.as_tensor(b1b_s))
        self.W2b_s = nn.Parameter(torch.as_tensor(W2b_s))
        self.b2b_s = nn.Parameter(torch.as_tensor(b2b_s))
        self.W1b_m = nn.Parameter(torch.as_tensor(W1b_m))
        self.b1b_m = nn.Parameter(torch.as_tensor(b1b_m))
        self.W2b_m = nn.Parameter(torch.as_tensor(W2b_m))
        self.b2b_m = nn.Parameter(torch.as_tensor(b2b_m))
        self.n = int(n)
        self.d = int(d)
        self.learnable_params = [self.W1a_s, self.b1a_s, self.W2a_s, self.b2a_s, self.W1a_m, self.b1a_m, self.W2a_m, self.b2a_m, self.W1b_s, self.b1b_s, self.W2b_s, self.b2b_s, self.W1b_m, self.b1b_m, self.W2b_m, self.b2b_m]

    def coupling(self, x, W1s, b1s, W2s, b2s, W1m, b1m, W2m, b2m):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        W1s = torch.as_tensor(W1s, device=DEVICE).float()
        b1s = torch.as_tensor(b1s, device=DEVICE).float()
        W2s = torch.as_tensor(W2s, device=DEVICE).float()
        b2s = torch.as_tensor(b2s, device=DEVICE).float()
        W1m = torch.as_tensor(W1m, device=DEVICE).float()
        b1m = torch.as_tensor(b1m, device=DEVICE).float()
        W2m = torch.as_tensor(W2m, device=DEVICE).float()
        b2m = torch.as_tensor(b2m, device=DEVICE).float()
        x1 = x[:self.n]
        x2 = x[self.n:]
        s = linear(relu(linear(x1, W1s, b1s)), W2s, b2s)
        m = linear(relu(linear(x1, W1m, b1m)), W2m, b2m)
        return torch.cat([x1, ((torch.exp(s if isinstance(s, torch.Tensor) else torch.tensor(float(s))) * x2) + m)])

    def coupling_inv(self, y, W1s, b1s, W2s, b2s, W1m, b1m, W2m, b2m):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        W1s = torch.as_tensor(W1s, device=DEVICE).float()
        b1s = torch.as_tensor(b1s, device=DEVICE).float()
        W2s = torch.as_tensor(W2s, device=DEVICE).float()
        b2s = torch.as_tensor(b2s, device=DEVICE).float()
        W1m = torch.as_tensor(W1m, device=DEVICE).float()
        b1m = torch.as_tensor(b1m, device=DEVICE).float()
        W2m = torch.as_tensor(W2m, device=DEVICE).float()
        b2m = torch.as_tensor(b2m, device=DEVICE).float()
        y1 = y[:self.n]
        y2 = y[self.n:]
        s = linear(relu(linear(y1, W1s, b1s)), W2s, b2s)
        m = linear(relu(linear(y1, W1m, b1m)), W2m, b2m)
        return torch.cat([y1, ((y2 - m) * torch.exp((-s) if isinstance((-s), torch.Tensor) else torch.tensor(float((-s)))))])

    def coupling_log_det(self, x, W1s, b1s, W2s, b2s):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        W1s = torch.as_tensor(W1s, device=DEVICE).float()
        b1s = torch.as_tensor(b1s, device=DEVICE).float()
        W2s = torch.as_tensor(W2s, device=DEVICE).float()
        b2s = torch.as_tensor(b2s, device=DEVICE).float()
        x1 = x[:self.n]
        s = linear(relu(linear(x1, W1s, b1s)), W2s, b2s)
        return torch.sum(s if isinstance(s, torch.Tensor) else torch.tensor(float(s)))

    def swap(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        return torch.cat([x[self.n:], x[:self.n]])

    def forward_z(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        h = self.coupling(x, self.W1a_s, self.b1a_s, self.W2a_s, self.b2a_s, self.W1a_m, self.b1a_m, self.W2a_m, self.b2a_m)
        h = self.swap(h)
        h = self.coupling(h, self.W1b_s, self.b1b_s, self.W2b_s, self.b2b_s, self.W1b_m, self.b1b_m, self.W2b_m, self.b2b_m)
        h = self.swap(h)
        return h

    def log_det(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        ld_a = self.coupling_log_det(x, self.W1a_s, self.b1a_s, self.W2a_s, self.b2a_s)
        h = self.coupling(x, self.W1a_s, self.b1a_s, self.W2a_s, self.b2a_s, self.W1a_m, self.b1a_m, self.W2a_m, self.b2a_m)
        h = self.swap(h)
        ld_b = self.coupling_log_det(h, self.W1b_s, self.b1b_s, self.W2b_s, self.b2b_s)
        return (ld_a + ld_b)

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = self.forward_z(x)
        return (log_pz(z, self.d) + self.log_det(x))

    def inverse(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        h = self.swap(z)
        h = self.coupling_inv(h, self.W1b_s, self.b1b_s, self.W2b_s, self.b2b_s, self.W1b_m, self.b1b_m, self.W2b_m, self.b2b_m)
        h = self.swap(h)
        return self.coupling_inv(h, self.W1a_s, self.b1a_s, self.W2a_s, self.b2a_s, self.W1a_m, self.b1a_m, self.W2a_m, self.b2a_m)

    def sample(self):
        this = self
        z = torch.distributions.Normal(0.0, 1.0).rsample((int(self.d),)).to(DEVICE)
        return self.inverse(z)

    def loss(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        return (-self(x))

    def train(self, X, epochs, lr, len_train):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        len_train = torch.as_tensor(len_train, device=DEVICE).float()
        for epoch in range(int(0), int(epochs)):
            for i in range(int(0), int(len_train)):
                L = self.loss(X[int(i)])
                grads = compute_grad(L, self.params)
                self.update_params(lr, grads)
            total = 0
            for i in range(int(0), int(len_train)):
                total = total + self.loss(X[int(i)])
            epoch_loss = (total / len_train)
            print(epoch_loss)
            bits = self.evaluate(epoch_loss, len_train)
            print(bits)

    def test(self, Y, len_test):
        this = self
        Y = torch.as_tensor(Y, device=DEVICE).float()
        len_test = torch.as_tensor(len_test, device=DEVICE).float()
        total = 0
        for i in range(int(0), int(len_test)):
            total = total + self.loss(X[int(i)])
        return (total / len_test)

    def evaluate(self, num, len):
        this = self
        num = torch.as_tensor(num, device=DEVICE).float()
        len = torch.as_tensor(len, device=DEVICE).float()
        return ((num / (self.d * torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))))) + 8.0)

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.W1a_s.copy_((self.W1a_s - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b1a_s.copy_((self.b1a_s - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.W2a_s.copy_((self.W2a_s - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.b2a_s.copy_((self.b2a_s - (lr * learnable_grads[int(3)])))
        with torch.no_grad():
            self.W1a_m.copy_((self.W1a_m - (lr * learnable_grads[int(4)])))
        with torch.no_grad():
            self.b1a_m.copy_((self.b1a_m - (lr * learnable_grads[int(5)])))
        with torch.no_grad():
            self.W2a_m.copy_((self.W2a_m - (lr * learnable_grads[int(6)])))
        with torch.no_grad():
            self.b2a_m.copy_((self.b2a_m - (lr * learnable_grads[int(7)])))
        with torch.no_grad():
            self.W1b_s.copy_((self.W1b_s - (lr * learnable_grads[int(8)])))
        with torch.no_grad():
            self.b1b_s.copy_((self.b1b_s - (lr * learnable_grads[int(9)])))
        with torch.no_grad():
            self.W2b_s.copy_((self.W2b_s - (lr * learnable_grads[int(10)])))
        with torch.no_grad():
            self.b2b_s.copy_((self.b2b_s - (lr * learnable_grads[int(11)])))
        with torch.no_grad():
            self.W1b_m.copy_((self.W1b_m - (lr * learnable_grads[int(12)])))
        with torch.no_grad():
            self.b1b_m.copy_((self.b1b_m - (lr * learnable_grads[int(13)])))
        with torch.no_grad():
            self.W2b_m.copy_((self.W2b_m - (lr * learnable_grads[int(14)])))
        with torch.no_grad():
            self.b2b_m.copy_((self.b2b_m - (lr * learnable_grads[int(15)])))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
torch.manual_seed(int(0))
dataset = create_dataset(80, 200)
train_dataset = dataset[int(0)]
test_dataset = dataset[int(1)]
train_X = train_dataset[int(0)]
test_X = test_dataset[int(0)]
len_train, len_test = len1d(train_X), len1d(test_X)
d = 784
h = 128
n = 392
s1, s2 = torch.sqrt((2.0 / n) if isinstance((2.0 / n), torch.Tensor) else torch.tensor(float((2.0 / n)))), (torch.sqrt((2.0 / h) if isinstance((2.0 / h), torch.Tensor) else torch.tensor(float((2.0 / h)))) * 0.01)
W1a_s = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1a_s = torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2a_s = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2a_s = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W1a_m = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1a_m = torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2a_m = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2a_m = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W1b_s = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1b_s = torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2b_s = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2b_s = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W1b_m = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1b_m = torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2b_m = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2b_m = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
realnvp = RealNVP(W1a_s, b1a_s, W2a_s, b2a_s, W1a_m, b1a_m, W2a_m, b2a_m, W1b_s, b1b_s, W2b_s, b2b_s, W1b_m, b1b_m, W2b_m, b2b_m, n, d).to(DEVICE)
print(print(DEVICE))
train_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(d)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_train)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_train)):
    train_flat[int(i), :] = dequantize(flatten(train_X[int(i)], 28, 28), d)
test_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(d)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_test)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_test)):
    test_flat[int(i), :] = dequantize(flatten(test_X[int(i)], 28, 28), d)
epochs = 1
lr = 0.0001
X = train_flat
Y = test_flat
print(realnvp.train(X, epochs, lr, len_train))
test_loss = realnvp.test(Y, len_test)
print(print(test_loss))
bits = realnvp.evaluate(test_loss, len_test)
print(print(bits))
gen_flat = realnvp.sample()
gen_img = unflatten(gen_flat, 28, 28)
print(print(gen_img))