import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def get_1d_array_length(x):
    total, temp = 0, 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def rand_2d_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_array(x):
    t = torch.distributions.Normal(0, 1).rsample((int(x),))
    return t

def σ(x):
    n = get_1d_array_length(x)
    result = torch.zeros(int(n), device=DEVICE)
    for i in range(int(0), int(n)):
        result[int(i)] = (1.0 / (1.0 + torch.exp((-x[int(i)]) if isinstance((-x[int(i)]), torch.Tensor) else torch.tensor(float((-x[int(i)]))))))
    return result

def argmax(iterable):
    idx = 0
    max_val = iterable[int(0)]
    len_iterable = get_1d_array_length(iterable)
    for i in range(int(1), int(len_iterable)):
        if iterable[int(i)] > max_val:
            max_val = iterable[int(i)]
            idx = i
    return idx

def cross_entropy(logits, label):
    total = 0
    for i in range(int(0), int(18)):
        total = total + torch.exp(logits[int(i)] if isinstance(logits[int(i)], torch.Tensor) else torch.tensor(float(logits[int(i)])))
    return ((-logits[int(label)]) + torch.log(total if isinstance(total, torch.Tensor) else torch.tensor(float(total))))

# === Classes ===
class AdamOptimizer2D(nn.Module):
    def __init__(self, lr, beta1, beta2, eps, t, m, v):
        super().__init__()
        self.lr = torch.as_tensor(lr).float()
        self.beta1 = torch.as_tensor(beta1).float()
        self.beta2 = torch.as_tensor(beta2).float()
        self.eps = torch.as_tensor(eps).float()
        self.t = torch.as_tensor(t).float()
        self.m = torch.as_tensor(m).float()
        self.v = torch.as_tensor(v).float()

    def step(self, param, grad):
        this = self
        param = torch.as_tensor(param, device=DEVICE).float()
        grad = torch.as_tensor(grad, device=DEVICE).float()
        with torch.no_grad():
            self.t.copy_((self.t + 1.0))
        with torch.no_grad():
            self.m.copy_(((self.beta1 * self.m) + ((1.0 - self.beta1) * grad)))
        with torch.no_grad():
            self.v.copy_(((self.beta2 * self.v) + ((1.0 - self.beta2) * (grad ** 2))))
        m_hat = (self.m / (1.0 - (self.beta1 ** self.t)))
        v_hat = (self.v / (1.0 - (self.beta2 ** self.t)))
        param_new = (param - ((self.lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + self.eps)))
        return param_new

class AdamOptimizer1D(nn.Module):
    def __init__(self, lr, beta1, beta2, eps, t, m, v):
        super().__init__()
        self.lr = torch.as_tensor(lr).float()
        self.beta1 = torch.as_tensor(beta1).float()
        self.beta2 = torch.as_tensor(beta2).float()
        self.eps = torch.as_tensor(eps).float()
        self.t = torch.as_tensor(t).float()
        self.m = torch.as_tensor(m).float()
        self.v = torch.as_tensor(v).float()

    def step(self, param, grad):
        this = self
        param = torch.as_tensor(param, device=DEVICE).float()
        grad = torch.as_tensor(grad, device=DEVICE).float()
        with torch.no_grad():
            self.t.copy_((self.t + 1.0))
        with torch.no_grad():
            self.m.copy_(((self.beta1 * self.m) + ((1.0 - self.beta1) * grad)))
        with torch.no_grad():
            self.v.copy_(((self.beta2 * self.v) + ((1.0 - self.beta2) * (grad ** 2))))
        m_hat = (self.m / (1.0 - (self.beta1 ** self.t)))
        v_hat = (self.v / (1.0 - (self.beta2 ** self.t)))
        param_new = (param - ((self.lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + self.eps)))
        return param_new

class Linear(nn.Module):
    def __init__(self, W, b, adam_W, adam_b):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.add_module('adam_W', adam_W)
        self.add_module('adam_b', adam_b)
        self.learnable_params = [self.W, self.b]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        in_features = get_1d_array_length(x)
        col = torch.zeros(int(in_features), int(1), device=DEVICE)
        col[:, int(0)] = x
        res = (self.W @ col)
        out = (res[:, int(0)] + self.b)
        return out

    def update_params(self, learnable_grads):
        this = self
        with torch.no_grad():
            self.W.copy_(self.adam_W.step(self.W, learnable_grads[int(0)]))
        with torch.no_grad():
            self.b.copy_(self.adam_b.step(self.b, learnable_grads[int(1)]))

class RNN(nn.Module):
    def __init__(self, H, in2hidden, in2output):
        super().__init__()
        self.H = int(H)
        self.add_module('in2hidden', in2hidden)
        self.add_module('in2output', in2output)
        self.learnable_params = []

    def forward(self, X):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        dims = X.shape
        B = dims[int(0)]
        T = dims[int(1)]
        out = torch.zeros(int(B), int(T), int(O), device=DEVICE)
        for i in range(int(0), int(B)):
            hidden_state = self.init_hidden()
            for t in range(int(0), int(T)):
                x = X[int(i), int(t)]
                combined = torch.cat([x, hidden_state])
                hidden_state = σ(self.in2hidden(combined))
                out[int(i), int(t)] = self.in2output(hidden_state)
        return out

    def init_hidden(self):
        this = self
        return torch.zeros(int(self.H), device=DEVICE)

    def train(self, epochs):
        this = self
        for epoch in range(int(0), int(epochs)):
            print(epoch)
            epoch_loss = 0
            for idx in range(int(0), int(train_dataset_len)):
                sample = train_dataset[int(idx)]
                name = sample[int(0)]
                dims = name.shape
                T = dims[int(0)]
                name = torch.reshape(sample[int(0)], (1, T, I))
                target = sample[int(1)]
                out = self(name)
                output = out[int(0), int((T - 1))]
                loss = cross_entropy(output, target)
                epoch_loss = epoch_loss + detach(loss)
                grad_in2hidden = compute_grad(loss, self.in2hidden.learnable_params)
                grad_in2output = compute_grad(loss, self.in2output.learnable_params)
                self.in2hidden.update_params(grad_in2hidden)
                self.in2output.update_params(grad_in2output)
            epoch_loss = (epoch_loss / train_dataset_len)
            print(epoch_loss)
        return loss

    def evaluate(self):
        this = self
        correct = 0
        for idx in range(int(0), int(test_dataset_len)):
            sample = test_dataset[int(idx)]
            name = sample[int(0)]
            dims = name.shape
            T = dims[int(0)]
            name = torch.reshape(name, (1, T, 59))
            target = sample[int(1)]
            out = self(name)
            output = out[int(0), int((T - 1))]
            predicted = argmax(output)
            if predicted == target:
                correct = correct + 1
        return (correct / test_dataset_len)

# === Program ===
train_dataset_len = 162
test_dataset_len = 18
I = 59
O = 18
H = 128
W1 = rand_2d_array(H, (I + H), 0.01)
b1 = rand_array(H)
W2 = rand_2d_array(O, H, 0.01)
b2 = rand_array(O)
lr = 0.001
adam_W1 = AdamOptimizer2D(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(H), int((I + H)), device=DEVICE), torch.zeros(int(H), int((I + H)), device=DEVICE)).to(DEVICE)
adam_b1 = AdamOptimizer1D(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(H), device=DEVICE), torch.zeros(int(H), device=DEVICE)).to(DEVICE)
adam_W2 = AdamOptimizer2D(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(O), int(H), device=DEVICE), torch.zeros(int(O), int(H), device=DEVICE)).to(DEVICE)
adam_b2 = AdamOptimizer1D(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(O), device=DEVICE), torch.zeros(int(O), device=DEVICE)).to(DEVICE)
in2hidden = Linear(W1, b1, adam_W1, adam_b1).to(DEVICE)
in2output = Linear(W2, b2, adam_W2, adam_b2).to(DEVICE)
rnn = RNN(H, in2hidden, in2output).to(DEVICE)
sample_input = torch.stack([torch.stack([torch.distributions.Normal(2.0, 0.5).rsample((int(I),)) for _fi_j in range(int(7)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(1)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
fwd_pass = rnn(sample_input)