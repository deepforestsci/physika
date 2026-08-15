import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import train

# === Functions ===
def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def relu1d(x):
    return ((x + torch.abs(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) * 0.5)

def linear(x, weight, bias):
    in_dim = get_1d_array_length(x)
    col = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(1)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(in_dim)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    col[:, int(0)] = x
    res = (weight @ col)
    return (res[:, int(0)] + bias)

def flatten(img, rows, cols):
    n = (rows * cols)
    results = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    for i in range(int(0), int(rows)):
        results[(i * cols):((i + 1) * cols)] = img[int(i), :]
    return results

def unflatten(x, rows, cols):
    img = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    for i in range(int(0), int(rows)):
        img[int(i), :] = x[(i * cols):((i + 1) * cols)]
    return img

def dequantize(x, n):
    ε = torch.distributions.Uniform(0.0, 1.0).rsample((int(n),)).to(DEVICE)
    return ((x + ε) / 256.0)

def gaussian_log_prob(x, n):
    return (torch.sum((((-0.5) * x) * x) if isinstance((((-0.5) * x) * x), torch.Tensor) else torch.tensor(float((((-0.5) * x) * x)))) - ((n * 0.5) * torch.log((2.0 * 3.14159265) if isinstance((2.0 * 3.14159265), torch.Tensor) else torch.tensor(float((2.0 * 3.14159265))))))

def neg_loglik(log_px):
    return (-log_px)

def nll(model, X, count):
    total = 0
    for i in range(int(0), int(count)):
        total = total + neg_loglik(model(X[int(i)]))
    return (total / count)

# === Classes ===
class NICE(nn.Module):
    def __init__(self, w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, a, n_half, ndim):
        super().__init__()
        self.w1a = nn.Parameter(torch.as_tensor(w1a))
        self.b1a = nn.Parameter(torch.as_tensor(b1a))
        self.w2a = nn.Parameter(torch.as_tensor(w2a))
        self.b2a = nn.Parameter(torch.as_tensor(b2a))
        self.w1b = nn.Parameter(torch.as_tensor(w1b))
        self.b1b = nn.Parameter(torch.as_tensor(b1b))
        self.w2b = nn.Parameter(torch.as_tensor(w2b))
        self.b2b = nn.Parameter(torch.as_tensor(b2b))
        self.a = nn.Parameter(torch.as_tensor(a))
        self.n_half = int(n_half)
        self.ndim = int(ndim)
        self.learnable_params = [self.w1a, self.b1a, self.w2a, self.b2a, self.w1b, self.b1b, self.w2b, self.b2b, self.a]

    def coupling(self, x, w1, b1, w2, b2, half):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        w1 = torch.as_tensor(w1, device=DEVICE).float()
        b1 = torch.as_tensor(b1, device=DEVICE).float()
        w2 = torch.as_tensor(w2, device=DEVICE).float()
        b2 = torch.as_tensor(b2, device=DEVICE).float()
        x1 = x[:half]
        x2 = x[half:]
        hidden_pre = linear(x1, w1, b1)
        shift = linear(relu1d(hidden_pre), w2, b2)
        y2 = (x2 + shift)
        return torch.cat([x1, y2])

    def coupling_inv(self, y, w1, b1, w2, b2, half):
        this = self
        y = torch.as_tensor(y, device=DEVICE).float()
        w1 = torch.as_tensor(w1, device=DEVICE).float()
        b1 = torch.as_tensor(b1, device=DEVICE).float()
        w2 = torch.as_tensor(w2, device=DEVICE).float()
        b2 = torch.as_tensor(b2, device=DEVICE).float()
        y1 = y[:half]
        y2 = y[half:]
        hidden_pre = linear(y1, w1, b1)
        shift = linear(relu1d(hidden_pre), w2, b2)
        x2 = (y2 - shift)
        return torch.cat([y1, x2])

    def swap(self, x, half):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        return torch.cat([x[half:], x[:half]])

    def rescale(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        s = torch.exp(self.a if isinstance(self.a, torch.Tensor) else torch.tensor(float(self.a)))
        return (x * s)

    def rescale_inv(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        s = torch.exp((-self.a) if isinstance((-self.a), torch.Tensor) else torch.tensor(float((-self.a))))
        return (z * s)

    def rescale_log_det(self):
        this = self
        return torch.sum(self.a if isinstance(self.a, torch.Tensor) else torch.tensor(float(self.a)))

    def forward_z(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        h = self.coupling(x, self.w1a, self.b1a, self.w2a, self.b2a, self.n_half)
        h = self.swap(h, self.n_half)
        h = self.coupling(h, self.w1b, self.b1b, self.w2b, self.b2b, self.n_half)
        h = self.swap(h, self.n_half)
        return self.rescale(h)

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        z = self.forward_z(x)
        log_pz = gaussian_log_prob(z, self.ndim)
        log_px = (log_pz + self.rescale_log_det())
        return log_px

    def inverse(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        h = self.rescale_inv(z)
        h = self.swap(h, self.n_half)
        h = self.coupling_inv(h, self.w1b, self.b1b, self.w2b, self.b2b, self.n_half)
        h = self.swap(h, self.n_half)
        h = self.coupling_inv(h, self.w1a, self.b1a, self.w2a, self.b2a, self.n_half)
        return h

    def sample(self):
        this = self
        z = torch.distributions.Normal(0.0, 1.0).rsample((int(self.ndim),)).to(DEVICE)
        return self.inverse(z)

    def loss(self, pred, target):
        this = self
        pred = torch.as_tensor(pred, device=DEVICE).float()
        target = torch.as_tensor(target, device=DEVICE).float()
        return (-pred)

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
len_train_X = get_1d_array_length(train_X)
len_test_X = get_1d_array_length(test_X)
ndim = 784
hidden = 128
half = 392
s1 = torch.sqrt((2.0 / half) if isinstance((2.0 / half), torch.Tensor) else torch.tensor(float((2.0 / half))))
s2 = (torch.sqrt((2.0 / hidden) if isinstance((2.0 / hidden), torch.Tensor) else torch.tensor(float((2.0 / hidden)))) * 0.01)
w1a = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(half),)).to(DEVICE) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1a = torch.stack([(i * 0) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
w2a = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(hidden),)).to(DEVICE) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2a = torch.stack([(i * 0) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
w1b = torch.stack([torch.distributions.Normal(0.0, s1).rsample((int(half),)).to(DEVICE) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1b = torch.stack([(i * 0) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
w2b = torch.stack([torch.distributions.Normal(0.0, s2).rsample((int(hidden),)).to(DEVICE) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2b = torch.stack([(i * 0) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
a_init = torch.stack([(i * 0) for _fi_i in range(int(ndim)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
nice_object = NICE(w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, a_init, half, ndim).to(DEVICE)
debug_input = dequantize(flatten(train_X[int(0)], 28, 28), ndim)
debug_log_px = nice_object(debug_input)
print(print(debug_log_px))
print(print(DEVICE))
train_X_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(ndim)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_train_X)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_train_X)):
    train_X_flat[int(i), :] = dequantize(flatten(train_X[int(i)], 28, 28), ndim)
test_X_flat = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(ndim)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(len_test_X)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(len_test_X)):
    test_X_flat[int(i), :] = dequantize(flatten(test_X[int(i)], 28, 28), ndim)
dummy_y = torch.stack([(i * 0) for _fi_i in range(int(len_train_X)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
epochs = 20
lr = 0.001
losses = torch.stack([(i * 0) for _fi_i in range(int(epochs)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for i in range(int(0), int(epochs)):
    nice_object = train(nice_object, train_X_flat, dummy_y, 1, lr)
    epoch_loss = nll(nice_object, train_X_flat, len_train_X)
    losses[int(i)] = epoch_loss
    print(epoch_loss)
    epoch_bits_per_dim = ((epoch_loss / (ndim * torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))))) + 8.0)
    print(epoch_bits_per_dim)
test_loss = nll(nice_object, test_X_flat, len_test_X)
print(print(test_loss))
bits_per_dim = ((test_loss / (ndim * torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))))) + 8.0)
print(print(bits_per_dim))
gen_flat = nice_object.sample()
gen_img = unflatten(gen_flat, 28, 28)
print(print(gen_img))