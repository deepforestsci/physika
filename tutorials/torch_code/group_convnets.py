import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def rand_5d_array(m, n, o, p, q, μ):
    return torch.stack([torch.stack([torch.stack([torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_b in range(int(q)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_s in range(int(p)) for s in [torch.tensor(float(_fi_s), device=DEVICE)]]) for _fi_k in range(int(o)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(m)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_4d_array(m, n, o, p, μ):
    return torch.stack([torch.stack([torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_s in range(int(p)) for s in [torch.tensor(float(_fi_s), device=DEVICE)]]) for _fi_k in range(int(o)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(m)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_3d_array(m, n, o, μ):
    return torch.stack([torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_k in range(int(o)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(m)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_2d_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_array(n, μ):
    return torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def pad2d(x, p):
    B, C, H, W = size(x)
    m = (H + (2 * p))
    n = (W + (2 * p))
    Y = torch.zeros(int(B), int(C), int(m), int(n), device=DEVICE)
    Y[:, :, int(p):int((p + H)), int(p):int((p + W))] = x
    return Y

def cross_entropy(probs, labels):
    B, M = size(probs)
    q = torch.stack([probs[int(i), int(labels[int(i)])] for _fi_i in range(int(B)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    l = ((-torch.sum(torch.log(q if isinstance(q, torch.Tensor) else torch.tensor(float(q))) if isinstance(torch.log(q if isinstance(q, torch.Tensor) else torch.tensor(float(q))), torch.Tensor) else torch.tensor(float(torch.log(q if isinstance(q, torch.Tensor) else torch.tensor(float(q))))))) / B)
    return l

def softmax(x):
    B, M = size(x)
    E = torch.zeros(int(B), int(M), device=DEVICE)
    Y = torch.zeros(int(B), int(M), device=DEVICE)
    for i in range(int(0), int(B)):
        for j in range(int(0), int(M)):
            E[int(i), int(j)] = torch.exp(x[int(i), int(j)] if isinstance(x[int(i), int(j)], torch.Tensor) else torch.tensor(float(x[int(i), int(j)])))
    for i in range(int(0), int(B)):
        S = torch.sum(E[int(i)] if isinstance(E[int(i)], torch.Tensor) else torch.tensor(float(E[int(i)])))
        for j in range(int(0), int(M)):
            Y[int(i), int(j)] = (E[int(i), int(j)] / S)
    return Y

def argmax1d(iterable):
    idx = 0
    max_val = iterable[int(0)]
    len_iterable = get_1d_array_length(iterable)
    for i in range(int(1), int(len_iterable)):
        if iterable[int(i)] > max_val:
            max_val = iterable[int(i)]
            idx = i
    return idx

def argmax(x):
    B, C = size(x)
    I = torch.zeros(int(B), device=DEVICE)
    for b in range(int(0), int(B)):
        I[int(b)] = argmax1d(x[int(b)])
    return I

def relu(x):
    if x > 0.0:
        return x
    else:
        return 0.0

def relu5d(x):
    B, O, r, H, W = size(x)
    results = torch.zeros(int(B), int(O), int(4), int(H), int(W), device=DEVICE)
    for b in range(int(0), int(B)):
        for o in range(int(0), int(O)):
            for r in range(int(0), int(r)):
                for i in range(int(0), int(H)):
                    for j in range(int(0), int(W)):
                        results[int(b), int(o), int(r), int(i), int(j)] = relu(x[int(b), int(o), int(r), int(i), int(j)])
    return results

def conv2d(input, kernel):
    B, I, H, W = size(input)
    O, I, K, K = size(kernel)
    H_out = ((H - K) + 1)
    W_out = ((W - K) + 1)
    results = torch.zeros(int(B), int(O), int(H_out), int(W_out), device=DEVICE)
    for b in range(int(0), int(B)):
        for o in range(int(0), int(O)):
            for h in range(int(0), int(H_out)):
                for w in range(int(0), int(W_out)):
                    patch = input[int(b), :, int(h):int((h + K)), int(w):int((w + K))]
                    weights = kernel[int(o)]
                    results[int(b), int(o), int(h), int(w)] = torch.sum((patch * weights) if isinstance((patch * weights), torch.Tensor) else torch.tensor(float((patch * weights))))
    return results

def maxpool2d(x):
    B, C, H, W = size(x)
    m = (H / 2)
    n = (W / 2)
    Y = torch.zeros(int(B), int(C), int(m), int(n), device=DEVICE)
    for b in range(int(0), int(B)):
        for i in range(int(0), int(m)):
            for j in range(int(0), int(n)):
                patch = x[int(b), :, int((i * 2)):int(((i * 2) + 2)), int((j * 2)):int(((j * 2) + 2))]
                patch = max(patch, 2)
                patch = max(patch, 1)
                Y[int(b), :, int(i), int(j)] = patch
    return Y

def gconv2d_maxpooling(x):
    return max(p4_maxpooling(x), 2)

def p4_maxpooling(x):
    B, O, r, H, W = size(x)
    x = x.view(B, (O * 4), H, W)
    x = maxpool2d(x)
    B, C, m, n = size(x)
    x = x.view(B, O, 4, m, n)
    return x

def sort_and_flatten(x):
    B, O, H, W = size(x)
    x = x.view(B, O, (H * W))
    x = torch.sort(x, dim=2).values
    x = x.view(B, ((O * H) * W))
    return x

# === Classes ===
class Z2P4Conv2d(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(w))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        B, I, H, W = size(x)
        m = ((H - K) + 1)
        n = ((W - K) + 1)
        y = torch.zeros(int(B), int(O), int(4), int(m), int(n), device=DEVICE)
        bias = torch.reshape(self.b, (1, O, 1, 1))
        for r in range(int(0), int(4)):
            w_r = torch.rot90(self.w, k=r, dims=tuple(int(d) for d in torch.tensor([2, 3], device=DEVICE)))
            z = conv2d(x, w_r)
            y[:, :, int(r), :, :] = (z + bias)
        return y

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.w.copy_((self.w - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b.copy_((self.b - (lr * learnable_grads[int(1)])))

class P4P4GConv2d(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(w))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        B, I, r, H, W = size(x)
        m = ((H - K) + 1)
        n = ((W - K) + 1)
        x = torch.reshape(x, (B, (I * 4), H, W))
        y = torch.zeros(int(B), int(O), int(4), int(m), int(n), device=DEVICE)
        bias = torch.reshape(self.b, (1, O, 1, 1))
        for r in range(int(0), int(4)):
            w_r = torch.roll(self.w, r, 2)
            w_r = torch.rot90(w_r, k=r, dims=tuple(int(d) for d in torch.tensor([3, 4], device=DEVICE)))
            w_r = torch.reshape(w_r, (O, (I * 4), K, K))
            y[:, :, int(r), :, :] = (conv2d(x, w_r) + bias)
        return y

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.w.copy_((self.w - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b.copy_((self.b - (lr * learnable_grads[int(1)])))

class Linear(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(w))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        B, I = size(x)
        O, I = size(self.w)
        y = torch.zeros(int(B), int(O), device=DEVICE)
        for i in range(int(0), int(B)):
            xb = torch.reshape(x[int(i)], (I, 1))
            yb = (self.w @ xb)
            y[int(i)] = (torch.reshape(yb, (O,)) + self.b)
        return y

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.w.copy_((self.w - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b.copy_((self.b - (lr * learnable_grads[int(1)])))

class P4GConvNet(nn.Module):
    def __init__(self, gconv1, gconv2, fc):
        super().__init__()
        self.add_module('gconv1', gconv1)
        self.add_module('gconv2', gconv2)
        self.add_module('fc', fc)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x = relu5d(self.gconv1(pad2d(x, 1)))
        x = p4_maxpooling(x)
        x = relu5d(self.gconv2(x))
        x = gconv2d_maxpooling(x)
        x = sort_and_flatten(x)
        x = self.fc(x)
        return softmax(x)

    def train(self, train_X, train_y, epochs, lr):
        this = self
        train_X = torch.as_tensor(train_X, device=DEVICE).float()
        train_y = torch.as_tensor(train_y, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        nB, B, C, H, W = size(train_X)
        print(nB)
        loss = 0
        for epoch in range(int(0), int(epochs)):
            print(epoch)
            epoch_loss = 0
            for bidx in range(int(0), int(nB)):
                x = train_X[int(bidx)]
                y = train_y[int(bidx)]
                z = self(x)
                current_loss = cross_entropy(z, y)
                epoch_loss = epoch_loss + current_loss
                learnable_grads = compute_grad(current_loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
            epoch_loss = (epoch_loss / nB)
            print(epoch_loss)
        loss = epoch_loss
        return loss

    def evaluate(self, test_X, test_y):
        this = self
        test_X = torch.as_tensor(test_X, device=DEVICE).float()
        test_y = torch.as_tensor(test_y, device=DEVICE).float()
        nB, B, C, H, W = size(test_X)
        correct = 0
        total = 0
        for bidx in range(int(0), int(nB)):
            x = test_X[int(bidx)]
            y = test_y[int(bidx)]
            logits = self(x)
            predictions = argmax(logits)
            for idx in range(int(0), int(B)):
                if predictions[int(idx)] == y[int(idx)]:
                    correct = (correct + 1)
            total = (total + B)
        print(correct)
        accuracy = (correct / total)
        return (accuracy * 100.0)

    def update_params(self, lr, learnable_grads):
        this = self
        self.gconv1.update_params(lr, learnable_grads[:int(2)])
        self.gconv2.update_params(lr, learnable_grads[int(2):int(4)])
        self.fc.update_params(lr, learnable_grads[int(4):int(6)])

# === Program ===
I = 1
O = 6
K = 3
H = 28
W = 28
B = 4
gconv1 = Z2P4Conv2d(rand_4d_array(O, I, K, K, 0.1), rand_array(O, 0.1)).to(DEVICE)
gconv2 = P4P4GConv2d(rand_5d_array(O, O, 4, K, K, 0.1), rand_array(O, 0.1)).to(DEVICE)
H1 = (H / 2)
W1 = (W / 2)
H2 = ((H1 - K) + 1)
W2 = ((W1 - K) + 1)
H3 = (H2 / 2)
W3 = (W2 / 2)
flattened_size = ((O * H3) * W3)
fc = Linear(rand_2d_array(10, flattened_size, 0.1), rand_array(10, 0.1)).to(DEVICE)
model = P4GConvNet(gconv1, gconv2, fc).to(DEVICE)
nB = 1
B = 1
train_X = rand_5d_array(1, 1, 1, 28, 28, 0.1)
train_y = rand_2d_array(1, 10, 0.1)
test_X = rand_5d_array(1, 1, 1, 28, 28, 0.1)
test_y = rand_2d_array(1, 10, 0.1)
epochs = 1
lr = 0.1
train_loss = model.train(train_X, train_y, epochs, lr)
print(print(train_loss))
accuracy = model.evaluate(train_X, train_y)
print(print(accuracy))