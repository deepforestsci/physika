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

def linsolve(A, b):
    aug = zeros2d(16, 17)
    for i in range(int(0), int(16)):
        for c in range(int(0), int(16)):
            aug[int(i), int(c)] = A[int(i), int(c)]
        aug[int(i), int(16)] = b[int(i)]
    for i in range(int(0), int(16)):
        piv = zeros1d(17)
        for c in range(int(0), int(17)):
            piv[int(c)] = aug[int(i), int(c)]
        aug_next = zeros2d(16, 17)
        for r in range(int(0), int(16)):
            if r == i:
                for c in range(int(0), int(17)):
                    aug_next[int(r), int(c)] = (piv[int(c)] / piv[int(i)])
            else:
                fac = (aug[int(r), int(i)] / piv[int(i)])
                for c in range(int(0), int(17)):
                    aug_next[int(r), int(c)] = (aug[int(r), int(c)] - (fac * piv[int(c)]))
        aug = aug_next
    x = zeros1d(16)
    for i in range(int(0), int(16)):
        idx = (15 - i)
        total = aug[int(idx), int(16)]
        for j in range(int((idx + 1)), int(16)):
            total = (total - (aug[int(idx), int(j)] * x[int(j)]))
        x_next = zeros1d(16)
        for c in range(int(0), int(16)):
            if c == idx:
                x_next[int(c)] = (total / aug[int(idx), int(idx)])
            else:
                x_next[int(c)] = x[int(c)]
        x = x_next
    return x

# === Classes ===
class DEQ(nn.Module):
    def __init__(self, W, U, b, Wo, bo):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.U = nn.Parameter(torch.as_tensor(U))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.Wo = nn.Parameter(torch.as_tensor(Wo))
        self.bo = nn.Parameter(torch.as_tensor(bo))
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
        num_solver_steps = 3
        self.h_star = zeros2d(1, 16)
        f_h = self.f(self.h_star, x)
        tanh_prime = (1.0 - (f_h * f_h))
        J = (df_dh(self.W, tanh_prime) - eye(16))
        for k in range(int(0), int(num_solver_steps)):
            g = (self.f(self.h_star, x) - self.h_star)
            delta = linsolve(J, g[int(0)])
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
        diff = (target - x_hat)
        return torch.sum((diff * diff) if isinstance((diff * diff), torch.Tensor) else torch.tensor(float((diff * diff))))

    def train(self, X, epochs, lr, images):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        images = torch.as_tensor(images, device=DEVICE).float()
        for epoch in range(int(0), int(epochs)):
            for i in range(int(0), int(images)):
                x = torch.stack([torch.as_tensor(X[int(i)])])
                preds = self(x)
                L = self.loss(x, preds)
                grads = compute_grad(L, self.params)
                self.update_params(lr, grads)
            total = 0
            for i in range(int(0), int(images)):
                x = torch.stack([torch.as_tensor(X[int(i)])])
                pred = self(x)
                total = total + self.loss(x, pred)
            print((total / images))

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
print(print(DEVICE))
W = rand_array(16, 16, 0.01)
U = rand_array(784, 16, 0.02)
b = zeros2d(1, 16)
Wo = rand_array(16, 784, 0.05)
bo = zeros2d(1, 784)
deq = DEQ(W, U, b, Wo, bo).to(DEVICE)
images = 50
X = rand_array(50, 784, 1.0)
x0 = torch.stack([torch.as_tensor(X[int(0)])])
recon_before = deq(x0)
loss_before = deq.loss(x0, recon_before)
print(print(loss_before))
epochs = 1
lr = 0.001
print(deq.train(X, epochs, lr, images))
recon_after = deq(x0)
loss_after = deq.loss(x0, recon_after)
print(print(loss_after))