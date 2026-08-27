import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def tanh(z):
    num = (torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) - torch.exp((-z) if isinstance((-z), torch.Tensor) else torch.tensor(float((-z)))))
    denom = (torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) + torch.exp((-z) if isinstance((-z), torch.Tensor) else torch.tensor(float((-z)))))
    res = (num / denom)
    return res

def sigmoid(z):
    res = (1.0 / (1.0 + torch.exp((-z) if isinstance((-z), torch.Tensor) else torch.tensor(float((-z))))))
    return res

def rand_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def zeros(n, m):
    return torch.stack([torch.stack([(j * 0.0) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

# === Classes ===
class Encoder(nn.Module):
    def __init__(self, We, be, Wu, bu, Wl, bl):
        super().__init__()
        self.We = nn.Parameter(torch.as_tensor(We))
        self.be = nn.Parameter(torch.as_tensor(be))
        self.Wu = nn.Parameter(torch.as_tensor(Wu))
        self.bu = nn.Parameter(torch.as_tensor(bu))
        self.Wl = nn.Parameter(torch.as_tensor(Wl))
        self.bl = nn.Parameter(torch.as_tensor(bl))
        self.learnable_params = [self.We, self.be, self.Wu, self.bu, self.Wl, self.bl]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        he = tanh(((x @ self.We) + self.be))
        mu = ((he @ self.Wu) + self.bu)
        lv = ((he @ self.Wl) + self.bl)
        return (mu, lv)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Decoder(nn.Module):
    def __init__(self, Wa, ba, Wo, bo):
        super().__init__()
        self.Wa = nn.Parameter(torch.as_tensor(Wa))
        self.ba = nn.Parameter(torch.as_tensor(ba))
        self.Wo = nn.Parameter(torch.as_tensor(Wo))
        self.bo = nn.Parameter(torch.as_tensor(bo))
        self.learnable_params = [self.Wa, self.ba, self.Wo, self.bo]

    def forward(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        hid = tanh(((z @ self.Wa) + self.ba))
        return sigmoid(((hid @ self.Wo) + self.bo))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class VariationalAutoEncoder(nn.Module):
    def __init__(self, enc, dec, lat):
        super().__init__()
        self.add_module('enc', enc)
        self.add_module('dec', dec)
        self.lat = nn.Parameter(torch.as_tensor(lat))
        self.learnable_params = [self.lat]
        self.mu = None
        self.lv = None

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        self.mu, self.lv = self.enc(x)
        eps = torch.distributions.Normal(0.0, 1.0).rsample((int(1), int(self.lat),))
        sigma = torch.exp((self.lv * 0.5) if isinstance((self.lv * 0.5), torch.Tensor) else torch.tensor(float((self.lv * 0.5))))
        z = (self.mu + (sigma * eps))
        x_hat = self.dec(z)
        return x_hat

    def loss(self, target, x_hat):
        this = self
        target = torch.as_tensor(target, device=DEVICE).float()
        x_hat = torch.as_tensor(x_hat, device=DEVICE).float()
        kl = ((-0.5) * torch.sum((((1.0 + self.lv) - (self.mu ** 2)) - torch.exp(self.lv if isinstance(self.lv, torch.Tensor) else torch.tensor(float(self.lv)))) if isinstance((((1.0 + self.lv) - (self.mu ** 2)) - torch.exp(self.lv if isinstance(self.lv, torch.Tensor) else torch.tensor(float(self.lv)))), torch.Tensor) else torch.tensor(float((((1.0 + self.lv) - (self.mu ** 2)) - torch.exp(self.lv if isinstance(self.lv, torch.Tensor) else torch.tensor(float(self.lv))))))))
        recon = (-torch.sum(((target * torch.log((x_hat + 1e-07) if isinstance((x_hat + 1e-07), torch.Tensor) else torch.tensor(float((x_hat + 1e-07))))) + ((1.0 - target) * torch.log(((1.0 - x_hat) + 1e-07) if isinstance(((1.0 - x_hat) + 1e-07), torch.Tensor) else torch.tensor(float(((1.0 - x_hat) + 1e-07)))))) if isinstance(((target * torch.log((x_hat + 1e-07) if isinstance((x_hat + 1e-07), torch.Tensor) else torch.tensor(float((x_hat + 1e-07))))) + ((1.0 - target) * torch.log(((1.0 - x_hat) + 1e-07) if isinstance(((1.0 - x_hat) + 1e-07), torch.Tensor) else torch.tensor(float(((1.0 - x_hat) + 1e-07)))))), torch.Tensor) else torch.tensor(float(((target * torch.log((x_hat + 1e-07) if isinstance((x_hat + 1e-07), torch.Tensor) else torch.tensor(float((x_hat + 1e-07))))) + ((1.0 - target) * torch.log(((1.0 - x_hat) + 1e-07) if isinstance(((1.0 - x_hat) + 1e-07), torch.Tensor) else torch.tensor(float(((1.0 - x_hat) + 1e-07))))))))))
        return (kl + recon)

    def train(self, X, epochs, lr, images):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        images = torch.as_tensor(images, device=DEVICE).float()
        for epoch in range(int(0), int(epochs)):
            for i in range(int(0), int(images)):
                x = torch.stack([torch.as_tensor(X[int(i)])])
                preds = self(x)
                elbo = self.loss(x, preds)
                grads = compute_grad(elbo, self.params)
                self.update(lr, grads)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
torch.manual_seed(int(42))
We = rand_array(784, 256, 0.05)
be = torch.zeros(1, 256)
Wu = rand_array(256, 20, 0.01)
bu = torch.zeros(1, 20)
Wl = rand_array(256, 20, 0.01)
bl = torch.zeros(1, 20)
Wa = rand_array(20, 256, 0.2)
ba = torch.zeros(1, 256)
Wo = rand_array(256, 784, 0.05)
bo = torch.zeros(1, 784)
enc = Encoder(We, be, Wu, bu, Wl, bl)
dec = Decoder(Wa, ba, Wo, bo)
lat = 20.0
vae = VariationalAutoEncoder(enc, dec, lat)
images = 1000
X = rand_array(images, 784, 0.1)
x0 = torch.stack([torch.as_tensor(X[int(0)])])
recon_before = vae(x0)
elbo_before = vae.loss(x0, recon_before)
print(elbo_before)
epochs = 2
lr = 0.0002
print(vae.train(X, epochs, lr, images))
recon_after = vae(x0)
elbo_after = vae.loss(x0, recon_after)
print(elbo_after)