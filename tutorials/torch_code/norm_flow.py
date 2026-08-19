import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

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

def neg_loglik(log_px):
    return (-log_px)

def nll(model, X, count):
    total = 0
    for i in range(int(0), int(count)):
        total = total + neg_loglik(model(X[int(i)]))
    return (total / count)

# === Classes ===
class NICE(nn.Module):
    def __init__(self, W1a, b1a, W2a, b2a, W1b, b1b, W2b, b2b, a, n, d):
        super().__init__()
        self.W1a = nn.Parameter(torch.as_tensor(W1a))
        self.b1a = nn.Parameter(torch.as_tensor(b1a))
        self.W2a = nn.Parameter(torch.as_tensor(W2a))
        self.b2a = nn.Parameter(torch.as_tensor(b2a))
        self.W1b = nn.Parameter(torch.as_tensor(W1b))
        self.b1b = nn.Parameter(torch.as_tensor(b1b))
        self.W2b = nn.Parameter(torch.as_tensor(W2b))
        self.b2b = nn.Parameter(torch.as_tensor(b2b))
        self.a = nn.Parameter(torch.as_tensor(a))
        self.n = int(n)
        self.d = int(d)
        self.learnable_params = [self.W1a, self.b1a, self.W2a, self.b2a, self.W1b, self.b1b, self.W2b, self.b2b, self.a]

    def coupling(self, x, W1, b1, W2, b2):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        W1 = torch.as_tensor(W1, device=DEVICE).float()
        b1 = torch.as_tensor(b1, device=DEVICE).float()
        W2 = torch.as_tensor(W2, device=DEVICE).float()
        b2 = torch.as_tensor(b2, device=DEVICE).float()
        x1 = x[:self.n]
        x2 = x[self.n:]
        m = linear(relu(linear(x1, W1, b1)), W2, b2)
        return torch.cat([x1, (x2 + m)])

    def coupling_inv(self, y, W1, b1, W2, b2):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        W1 = torch.as_tensor(W1, device=DEVICE).float()
        b1 = torch.as_tensor(b1, device=DEVICE).float()
        W2 = torch.as_tensor(W2, device=DEVICE).float()
        b2 = torch.as_tensor(b2, device=DEVICE).float()
        y1 = y[:self.n]
        y2 = y[self.n:]
        m = linear(relu(linear(y1, W1, b1)), W2, b2)
        return torch.cat([y1, (y2 - m)])

    def swap(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        return torch.cat([x[self.n:], x[:self.n]])

    def rescale(self, h):
        this = self
        h = torch.as_tensor(h, device=DEVICE).float()
        return (h * torch.exp(self.a if isinstance(self.a, torch.Tensor) else torch.tensor(float(self.a))))

    def rescale_inv(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        return (z * torch.exp((-self.a) if isinstance((-self.a), torch.Tensor) else torch.tensor(float((-self.a)))))

    def log_det(self):
        this = self
        return torch.sum(self.a if isinstance(self.a, torch.Tensor) else torch.tensor(float(self.a)))

    def forward_z(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        h = self.coupling(x, self.W1a, self.b1a, self.W2a, self.b2a)
        h = self.swap(h)
        h = self.coupling(h, self.W1b, self.b1b, self.W2b, self.b2b)
        h = self.swap(h)
        return self.rescale(h)

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = self.forward_z(x)
        return (log_pz(z, self.d) + self.log_det())

    def inverse(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        h = self.rescale_inv(z)
        h = self.swap(h)
        h = self.coupling_inv(h, self.W1b, self.b1b, self.W2b, self.b2b)
        h = self.swap(h)
        return self.coupling_inv(h, self.W1a, self.b1a, self.W2a, self.b2a)

    def sample(self):
        this = self
        z = torch.distributions.Normal(0.0, 1.0).rsample((int(self.d),)).to(DEVICE)
        return self.inverse(z)

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
dataset = create_dataset(80, 100)
train_dataset = dataset[int(0)]
test_dataset = dataset[int(1)]
train_X = train_dataset[int(0)]
test_X = test_dataset[int(0)]
len_train, len_test = len1d(train_X), len1d(test_X)
d = 784
h = 128
n = 392
s1, s2 = torch.sqrt((2.0 / n) if isinstance((2.0 / n), torch.Tensor) else torch.tensor(float((2.0 / n)))), (torch.sqrt((2.0 / h) if isinstance((2.0 / h), torch.Tensor) else torch.tensor(float((2.0 / h)))) * 0.01)
W1a = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1a = torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2a = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2a = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W1b = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(n),)).to(DEVICE) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1b = torch.stack([(i * 0) for _fi_i in range(int(h)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2b = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(h),)).to(DEVICE) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2b = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
a0 = torch.stack([(i * 0) for _fi_i in range(int(d)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
model = NICE(W1a, b1a, W2a, b2a, W1b, b1b, W2b, b2b, a0, n, d).to(DEVICE)
x0 = dequantize(flatten(train_X[int(0)], 28, 28), d)
print(print(model(x0)))
print(print(DEVICE))
train_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(d)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_train)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_train)):
    train_flat[int(i), :] = dequantize(flatten(train_X[int(i)], 28, 28), d)
test_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(d)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_test)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_test)):
    test_flat[int(i), :] = dequantize(flatten(test_X[int(i)], 28, 28), d)
epochs = 20
lr = 0.001
losses = torch.stack([(i * 0) for _fi_i in range(int(epochs)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(epochs)):
    for j in range(int(0), int(len_train)):
        L = neg_loglik(model(train_flat[int(j)]))
        nW1a = (model.W1a - (lr * compute_grad(L, model.W1a)))
        nb1a = (model.b1a - (lr * compute_grad(L, model.b1a)))
        nW2a = (model.W2a - (lr * compute_grad(L, model.W2a)))
        nb2a = (model.b2a - (lr * compute_grad(L, model.b2a)))
        nW1b = (model.W1b - (lr * compute_grad(L, model.W1b)))
        nb1b = (model.b1b - (lr * compute_grad(L, model.b1b)))
        nW2b = (model.W2b - (lr * compute_grad(L, model.W2b)))
        nb2b = (model.b2b - (lr * compute_grad(L, model.b2b)))
        na = (model.a - (lr * compute_grad(L, model.a)))
        model = NICE(nW1a, nb1a, nW2a, nb2a, nW1b, nb1b, nW2b, nb2b, na, model.n, model.d)
    epoch_loss = nll(model, train_flat, len_train)
    losses[int(i)] = epoch_loss
    print(epoch_loss)
    bits = ((epoch_loss / (d * torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))))) + 8.0)
    print(bits)
test_loss = nll(model, test_flat, len_test)
print(print(test_loss))
bits_per_dim = ((test_loss / (d * torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))))) + 8.0)
print(print(bits_per_dim))
gen_flat = model.sample()
gen_img = unflatten(gen_flat, 28, 28)
print(print(gen_img))