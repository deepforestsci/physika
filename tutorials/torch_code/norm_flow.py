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

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def relu(x):
    if x > 0:
        return x
    else:
        return 0.0

def relu1d(x):
    len_x = get_1d_array_length(x)
    results = zero_1d_array(len_x)
    for i in range(int(0), int(len_x)):
        results[int(i)] = relu(x[int(i)])
    return results

def linear(x, weight, bias):
    out = get_1d_array_length(bias)
    inp = get_1d_array_length(x)
    results = zero_1d_array(out)
    for i in range(int(0), int(out)):
        acc = 0
        for j in range(int(0), int(inp)):
            acc = acc + (weight[int(i), int(j)] * x[int(j)])
        results[int(i)] = (acc + bias[int(i)])
    return results

def flatten_image(img):
    rows = get_2d_array_num_rows(img)
    cols = get_1d_array_length(img[int(0)])
    n = (rows * cols)
    results = zero_1d_array(n)
    for i in range(int(0), int(rows)):
        for j in range(int(0), int(cols)):
            results[int(((i * cols) + j))] = ((img[int(i), int(j)] / 256.0) + (0.5 / 256.0))
    return results

def first_half(x):
    half = (get_1d_array_length(x) / 2)
    results = zero_1d_array(half)
    for i in range(int(0), int(half)):
        results[int(i)] = x[int(i)]
    return results

def second_half(x):
    len_x = get_1d_array_length(x)
    half = (len_x / 2)
    results = zero_1d_array(half)
    for i in range(int(0), int(half)):
        results[int(i)] = x[int((i + half))]
    return results

def concat_1d_array(x1, x2):
    len1 = get_1d_array_length(x1)
    len2 = get_1d_array_length(x2)
    total = (len1 + len2)
    results = zero_1d_array(total)
    for i in range(int(0), int(len1)):
        results[int(i)] = x1[int(i)]
    for i in range(int(0), int(len2)):
        results[int((i + len1))] = x2[int(i)]
    return results

def swap_halves(x):
    x1 = first_half(x)
    x2 = second_half(x)
    return concat_1d_array(x2, x1)

def gaussian_log_prob(x):
    len_x = get_1d_array_length(x)
    total = 0
    for i in range(int(0), int(len_x)):
        total = total + ((((-0.5) * x[int(i)]) * x[int(i)]) - (0.5 * torch.log((2 * 3.14159265) if isinstance((2 * 3.14159265), torch.Tensor) else torch.tensor(float((2 * 3.14159265))))))
    return total

def neg_loglik(log_px):
    return (-log_px)

# === Classes ===
class NICE(nn.Module):
    def __init__(self, w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, a):
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
        self.learnable_params = [self.w1a, self.b1a, self.w2a, self.b2a, self.w1b, self.b1b, self.w2b, self.b2b, self.a]

    def coupling(self, x, w1, b1, w2, b2):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        w1 = torch.as_tensor(w1, device=DEVICE).float()
        b1 = torch.as_tensor(b1, device=DEVICE).float()
        w2 = torch.as_tensor(w2, device=DEVICE).float()
        b2 = torch.as_tensor(b2, device=DEVICE).float()
        x1 = first_half(x)
        x2 = second_half(x)
        shift = linear(relu1d(linear(x1, w1, b1)), w2, b2)
        y2 = (x2 + shift)
        return concat_1d_array(x1, y2)

    def rescale(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        s = torch.exp(self.a if isinstance(self.a, torch.Tensor) else torch.tensor(float(self.a)))
        return (x * s)

    def rescale_log_det(self):
        this = self
        return torch.sum(self.a if isinstance(self.a, torch.Tensor) else torch.tensor(float(self.a)))

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        h = x
        h = self.coupling(h, self.w1a, self.b1a, self.w2a, self.b2a)
        h = swap_halves(h)
        h = self.coupling(h, self.w1b, self.b1b, self.w2b, self.b2b)
        h = swap_halves(h)
        z = self.rescale(h)
        log_pz = gaussian_log_prob(z)
        log_px = (log_pz + self.rescale_log_det())
        return log_px

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
dataset = create_dataset(80, 100)
train_dataset = dataset[int(0)]
test_dataset = dataset[int(1)]
train_X = train_dataset[int(0)]
test_X = test_dataset[int(0)]
len_train_X = get_1d_array_length(train_X)
len_test_X = get_1d_array_length(test_X)
ndim = 784
hidden = 16
half = (ndim / 2)
w1a = torch.stack([torch.stack([((torch.sin(((3.14 * i) / hidden) if isinstance(((3.14 * i) / hidden), torch.Tensor) else torch.tensor(float(((3.14 * i) / hidden)))) * torch.cos(((3.14 * j) / half) if isinstance(((3.14 * j) / half), torch.Tensor) else torch.tensor(float(((3.14 * j) / half))))) * 0.01) for _fi_j in range(int(half)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1a = torch.stack([(i * 0) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
w2a = torch.stack([torch.stack([((torch.cos(((3.14 * i) / half) if isinstance(((3.14 * i) / half), torch.Tensor) else torch.tensor(float(((3.14 * i) / half)))) * torch.sin(((3.14 * j) / hidden) if isinstance(((3.14 * j) / hidden), torch.Tensor) else torch.tensor(float(((3.14 * j) / hidden))))) * 0.01) for _fi_j in range(int(hidden)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2a = torch.stack([(i * 0) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
w1b = torch.stack([torch.stack([((torch.sin(((3.14 * i) / hidden) if isinstance(((3.14 * i) / hidden), torch.Tensor) else torch.tensor(float(((3.14 * i) / hidden)))) * torch.cos(((3.14 * j) / half) if isinstance(((3.14 * j) / half), torch.Tensor) else torch.tensor(float(((3.14 * j) / half))))) * 0.01) for _fi_j in range(int(half)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b1b = torch.stack([(i * 0) for _fi_i in range(int(hidden)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
w2b = torch.stack([torch.stack([((torch.cos(((3.14 * i) / half) if isinstance(((3.14 * i) / half), torch.Tensor) else torch.tensor(float(((3.14 * i) / half)))) * torch.sin(((3.14 * j) / hidden) if isinstance(((3.14 * j) / hidden), torch.Tensor) else torch.tensor(float(((3.14 * j) / hidden))))) * 0.01) for _fi_j in range(int(hidden)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
b2b = torch.stack([(i * 0) for _fi_i in range(int(half)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
a_init = torch.stack([(i * 0) for _fi_i in range(int(ndim)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
nice_object = NICE(w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, a_init).to(DEVICE)
debug_input = flatten_image(train_X[int(0)])
debug_log_px = nice_object(debug_input)
print(print(debug_log_px))
print(print(DEVICE))
train_X_flat = zero_2d_array(len_train_X, ndim)
for i in range(int(0), int(len_train_X)):
    flat = flatten_image(train_X[int(i)])
    for j in range(int(0), int(ndim)):
        train_X_flat[int(i), int(j)] = flat[int(j)]
test_X_flat = zero_2d_array(len_test_X, ndim)
for i in range(int(0), int(len_test_X)):
    flat = flatten_image(test_X[int(i)])
    for j in range(int(0), int(ndim)):
        test_X_flat[int(i), int(j)] = flat[int(j)]
dummy_y = zero_1d_array(len_train_X)
epochs = 10
lr = 0.005
losses = zero_1d_array(epochs)
for i in range(int(0), int(epochs)):
    nice_object = train(nice_object, train_X_flat, dummy_y, 1, lr)
    epoch_loss = 0
    for j in range(int(0), int(len_train_X)):
        log_px = nice_object(train_X_flat[int(j)])
        epoch_loss = epoch_loss + neg_loglik(log_px)
    epoch_loss = (epoch_loss / len_train_X)
    losses[int(i)] = epoch_loss
    print(epoch_loss)
test_loss = 0
for i in range(int(0), int(len_test_X)):
    log_px = nice_object(test_X_flat[int(i)])
    test_loss = test_loss + neg_loglik(log_px)
test_loss = (test_loss / len_test_X)
print(print(test_loss))