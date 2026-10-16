import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def tanh(a):
    num = (torch.exp(a if isinstance(a, torch.Tensor) else torch.tensor(float(a))) - torch.exp((-a) if isinstance((-a), torch.Tensor) else torch.tensor(float((-a)))))
    denom = (torch.exp(a if isinstance(a, torch.Tensor) else torch.tensor(float(a))) + torch.exp((-a) if isinstance((-a), torch.Tensor) else torch.tensor(float((-a)))))
    return (num / denom)

def sum_2d(a, p, q):
    s = 0.0
    for i in range(int(0), int(p)):
        for j in range(int(0), int(q)):
            s = s + a[int(i), int(j)]
    return s

def rand_array(n, m, μ):
    return torch.stack([torch.distributions.Normal(0.0, μ).rsample((int(m),)) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def zeros2d(n, m):
    return torch.stack([torch.stack([(j * 0.0) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def zeros1d(k):
    return torch.stack([(i * 0.0) for _fi_i in range(int(k)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def eye(n):
    I = torch.stack([torch.stack([(j * 0.0) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    for i in range(int(0), int(n)):
        I[int(i), int(i)] = 1.0
    return I

def df_dh(W, tanh_prime, n=None):
    if n is None:
        n = int(W.shape[0])
    return torch.stack([torch.as_tensor(torch.stack([torch.as_tensor((W[int(r)][int(c)] * tanh_prime[int(0)][int(c)])) for c in range(int(n))]).float()) for r in range(int(n))])

def gaussian_solve(A, b):
    a_row = get_2d_array_num_rows(A)
    a_col = get_2d_array_num_cols(A)
    new_col = (a_col + 1)
    aug = torch.zeros(int(a_row), int(new_col), device=DEVICE)
    for i in range(int(0), int(a_row)):
        aug[int(i), :int(a_col)] = A[int(i), :]
        aug[int(i), int(a_col)] = b[int(i)]
    for i in range(int(0), int(a_row)):
        max_row = i
        for k in range(int((i + 1)), int(a_row)):
            if torch.abs(aug[int(k), int(i)] if isinstance(aug[int(k), int(i)], torch.Tensor) else torch.tensor(float(aug[int(k), int(i)]))) > torch.abs(aug[int(max_row), int(i)] if isinstance(aug[int(max_row), int(i)], torch.Tensor) else torch.tensor(float(aug[int(max_row), int(i)]))):
                max_row = k
        if max_row != i:
            lo = min(i, max_row)
            hi = max(i, max_row)
            rows_above = aug[:int(lo), :]
            pivot_row = aug[int(hi):int((hi + 1)), :]
            rows_between = aug[int((lo + 1)):int(hi), :]
            current_row = aug[int(lo):int((lo + 1)), :]
            rows_below = aug[int((hi + 1)):, :]
            aug = torch.cat([rows_above, pivot_row, rows_between, current_row, rows_below])
        pivot_row = aug[int(i):int((i + 1)), :]
        pivot_value = aug[int(i), int(i)]
        rows_below = aug[int((i + 1)):, :]
        elimination_factors = (rows_below[:, int(i):int((i + 1))] / pivot_value)
        eliminated_rows = (rows_below - (elimination_factors * pivot_row))
        aug = torch.cat([aug[:int((i + 1)), :], eliminated_rows])
    x = torch.zeros(int(0), device=DEVICE)
    for i in range(int(0), int(a_row)):
        idx = ((a_col - 1) - i)
        total = aug[int(idx), int(a_col)]
        for j in range(int((idx + 1)), int(a_row)):
            total = (total - (aug[int(idx), int(j)] * x[int(((j - idx) - 1))]))
        val = (total / aug[int(idx), int(idx)])
        val = torch.reshape(val, (1,) if isinstance(1, int) else 1)
        x = torch.cat([val, x])
    return x

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_cols(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return get_1d_array_length(x[int(0)])

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def max(x, y):
    if x > y:
        return x
    else:
        return y

# === Classes ===
class DEQ(nn.Module):
    def __init__(self, W, U, b, Wo, bo, n, d, solver_steps):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.U = nn.Parameter(torch.as_tensor(U))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.Wo = nn.Parameter(torch.as_tensor(Wo))
        self.bo = nn.Parameter(torch.as_tensor(bo))
        self.n = int(n)
        self.d = int(d)
        self.solver_steps = int(solver_steps)
        self.learnable_params = [self.W, self.U, self.b, self.Wo, self.bo]
        self.h_star = None

    def f(self, h, x):
        this = self
        h = torch.as_tensor(h, device=DEVICE).float()
        x = torch.as_tensor(x, device=DEVICE).float()
        return tanh((((h @ self.W) + (x @ self.U)) + self.b))

    def equilibrium(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        self.h_star = zeros2d(1, self.n)
        f_h = self.f(self.h_star, x)
        tanh_prime = (1.0 - (f_h * f_h))
        J = (df_dh(self.W, tanh_prime) - eye(self.n))
        for k in range(int(0), int(self.solver_steps)):
            g = (self.f(self.h_star, x) - self.h_star)
            delta = gaussian_solve(J, g[int(0)])
            self.h_star = (self.h_star - torch.stack([torch.as_tensor(delta)]))
        return self.h_star

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        self.h_star = self.equilibrium(x)
        return ((self.h_star @ self.Wo) + self.bo)

    def loss(self, target, x_hat):
        this = self
        target = torch.as_tensor(target, device=DEVICE).float()
        x_hat = torch.as_tensor(x_hat, device=DEVICE).float()
        err = (target - x_hat)
        return sum_2d((err * err), 1, self.d)

    def train(self, X, epochs, lr, images):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        images = torch.as_tensor(images, device=DEVICE).float()
        losses = torch.stack([(i * 0) for _fi_i in range(int(epochs)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
        for epoch in range(int(0), int(epochs)):
            total = 0
            for i in range(int(0), int(images)):
                x = torch.stack([torch.as_tensor(X[int(i)])])
                preds = self(x)
                L = self.loss(x, preds)
                learnable_grads = compute_grad(L, self.learnable_params)
                self.update_params(lr, learnable_grads)
                new_preds = self(x)
                total = total + self.loss(x, new_preds)
            epoch_loss = (total / images)
            losses[int(epoch)] = epoch_loss
            print(epoch_loss)
        return losses

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.W.copy_((self.W - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.U.copy_((self.U - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.b.copy_((self.b - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.Wo.copy_((self.Wo - (lr * learnable_grads[int(3)])))
        with torch.no_grad():
            self.bo.copy_((self.bo - (lr * learnable_grads[int(4)])))

# === Program ===
torch.manual_seed(int(0))
print(print(DEVICE))
d, n, solver_steps = 784, 8, 3
W = rand_array(n, n, 0.01)
U = rand_array(d, n, 0.02)
b = zeros2d(1, n)
Wo = rand_array(n, d, 0.05)
bo = zeros2d(1, d)
deq = DEQ(W, U, b, Wo, bo, n, d, solver_steps).to(DEVICE)
images = 10
X = rand_array(10, 784, 1.0)
epochs = 1
lr = 0.001
losses = deq.train(X, epochs, lr, images)
loss_before, loss_after = losses[int(0)], losses[int((-1))]
print(print(loss_before))
print(print(loss_after))