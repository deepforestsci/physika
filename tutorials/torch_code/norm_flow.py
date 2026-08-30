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

def concat(a, b):
    la = len1d(a)
    lb = len1d(b)
    d = (la + lb)
    out = torch.stack([(i * 0) for _fi_i in range(int(d)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    out[:la] = a
    out[la:] = b
    return out

# === Classes ===
class RealNVP(nn.Module):
    def __init__(self, W1_s, b1_s, W2_s, b2_s, W1_m, b1_m, W2_m, b2_m, n, d):
        super().__init__()
        self.W1_s = nn.Parameter(torch.as_tensor(W1_s))
        self.b1_s = nn.Parameter(torch.as_tensor(b1_s))
        self.W2_s = nn.Parameter(torch.as_tensor(W2_s))
        self.b2_s = nn.Parameter(torch.as_tensor(b2_s))
        self.W1_m = nn.Parameter(torch.as_tensor(W1_m))
        self.b1_m = nn.Parameter(torch.as_tensor(b1_m))
        self.W2_m = nn.Parameter(torch.as_tensor(W2_m))
        self.b2_m = nn.Parameter(torch.as_tensor(b2_m))
        self.n = int(n)
        self.d = int(d)
        self.learnable_params = [self.W1_s, self.b1_s, self.W2_s, self.b2_s, self.W1_m, self.b1_m, self.W2_m, self.b2_m]

    def coupling(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x1 = x[:self.n]
        x2 = x[self.n:]
        s = linear(relu(linear(x1, self.W1_s, self.b1_s)), self.W2_s, self.b2_s)
        m = linear(relu(linear(x1, self.W1_m, self.b1_m)), self.W2_m, self.b2_m)
        return torch.cat([x1, ((torch.exp(s if isinstance(s, torch.Tensor) else torch.tensor(float(s))) * x2) + m)])

    def coupling_inv(self, y):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        y1 = y[:self.n]
        y2 = y[self.n:]
        s = linear(relu(linear(y1, self.W1_s, self.b1_s)), self.W2_s, self.b2_s)
        m = linear(relu(linear(y1, self.W1_m, self.b1_m)), self.W2_m, self.b2_m)
        return torch.cat([y1, ((y2 - m) * torch.exp((-s) if isinstance((-s), torch.Tensor) else torch.tensor(float((-s)))))])

    def log_det(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x1 = x[:self.n]
        s = linear(relu(linear(x1, self.W1_s, self.b1_s)), self.W2_s, self.b2_s)
        return torch.sum(s if isinstance(s, torch.Tensor) else torch.tensor(float(s)))

    def forward_z(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        return self.coupling(x)

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = self.forward_z(x)
        return (log_pz(z, self.d) + self.log_det(x))

    def inverse(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        return self.coupling_inv(z)

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
            total = total + self.loss(Y[int(i)])
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
            self.W1_s.copy_((self.W1_s - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b1_s.copy_((self.b1_s - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.W2_s.copy_((self.W2_s - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.b2_s.copy_((self.b2_s - (lr * learnable_grads[int(3)])))
        with torch.no_grad():
            self.W1_m.copy_((self.W1_m - (lr * learnable_grads[int(4)])))
        with torch.no_grad():
            self.b1_m.copy_((self.b1_m - (lr * learnable_grads[int(5)])))
        with torch.no_grad():
            self.W2_m.copy_((self.W2_m - (lr * learnable_grads[int(6)])))
        with torch.no_grad():
            self.b2_m.copy_((self.b2_m - (lr * learnable_grads[int(7)])))

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
train_X = torch.stack([torch.stack([torch.stack([(k * 0) for _fi_k in range(int(28)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(28)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(160)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
test_X = torch.stack([torch.stack([torch.stack([(k * 0) for _fi_k in range(int(28)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(28)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(40)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
len_train, len_test = 160, 40
d, h, n = 784, 128, 392
s1, s2 = torch.sqrt((2.0 / n) if isinstance((2.0 / n), torch.Tensor) else torch.tensor(float((2.0 / n)))), (torch.sqrt((2.0 / h) if isinstance((2.0 / h), torch.Tensor) else torch.tensor(float((2.0 / h)))) * 0.01)
W1_s, W1_m = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]), torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1_s, b1_m = torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]), torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2_s, W2_m = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]), torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2_s, b2_m = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]), torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
realnvp = RealNVP(W1_s, b1_s, W2_s, b2_s, W1_m, b1_m, W2_m, b2_m, n, d).to(DEVICE)
print(print(DEVICE))
train_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(d)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_train)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_train)):
    train_flat[int(i), :] = dequantize(flatten(train_X[int(i)], 28, 28), d)
test_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(d)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_test)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_test)):
    test_flat[int(i), :] = dequantize(flatten(test_X[int(i)], 28, 28), d)
epochs = 20
lr = 0.00015
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