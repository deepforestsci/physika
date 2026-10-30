import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def fc(rij, rc):
    θ = ((π * rij) / rc)
    if rij <= rc:
        return (0.5 * (torch.cos(θ if isinstance(θ, torch.Tensor) else torch.tensor(float(θ))) + 1.0))
    else:
        return 0.0

def dist(pa, pb):
    d = (pa - pb)
    return torch.sqrt((d @ d) if isinstance((d @ d), torch.Tensor) else torch.tensor(float((d @ d))))

def G1(a, pos, natoms, η, rs, rc):
    total = 0.0
    for j in range(int(0), int(natoms)):
        if j != a:
            rij = dist(pos[int(a)], pos[int(j)])
            total = total + (torch.exp(((-η) * ((rij - rs) ** 2)) if isinstance(((-η) * ((rij - rs) ** 2)), torch.Tensor) else torch.tensor(float(((-η) * ((rij - rs) ** 2))))) * fc(rij, rc))
    return total

def G2(a, pos, natoms, η, ζ, lam, rc):
    total = 0.0
    for j in range(int(0), int(natoms)):
        if j != a:
            for k in range(int(0), int(natoms)):
                if k != a:
                    if k != j:
                        vij = (pos[int(a)] - pos[int(j)])
                        vik = (pos[int(a)] - pos[int(k)])
                        rij = dist(pos[int(a)], pos[int(j)])
                        rik = dist(pos[int(a)], pos[int(k)])
                        rjk = dist(pos[int(j)], pos[int(k)])
                        cosθijk = ((vij @ vik) / (rij * rik))
                        gauss = torch.exp(((-η) * (((rij ** 2) + (rik ** 2)) + (rjk ** 2))) if isinstance(((-η) * (((rij ** 2) + (rik ** 2)) + (rjk ** 2))), torch.Tensor) else torch.tensor(float(((-η) * (((rij ** 2) + (rik ** 2)) + (rjk ** 2))))))
                        total = total + ((((((1.0 + (lam * cosθijk)) ** ζ) * gauss) * fc(rij, rc)) * fc(rik, rc)) * fc(rjk, rc))
    return ((2.0 ** (1.0 - ζ)) * total)

def fa(x):
    return (1.0 - (2.0 / (torch.exp((2.0 * x) if isinstance((2.0 * x), torch.Tensor) else torch.tensor(float((2.0 * x)))) + 1.0)))

def S(g, nin, nhid, w1, b1, w2, b2):
    total = b2
    for j in range(int(0), int(nhid)):
        z = b1[int(j)]
        for m in range(int(0), int(nin)):
            z = z + (w1[int(j), int(m)] * g[int(m)])
        total = total + (w2[int(j)] * fa(z))
    return total

def mse_loss(pred, target):
    return ((pred - target) ** 2)

# === Classes ===
class Descriptor(nn.Module):
    def __init__(self, rs, ζ, lam):
        super().__init__()
        self.rs = nn.Parameter(torch.as_tensor(rs))
        self.ζ = int(ζ)
        self.lam = torch.as_tensor(lam).float() if isinstance(lam, (int, float, torch.Tensor)) else lam
        self.learnable_params = [self.rs]

    def forward(self, a, pos, natoms):
        this = self
        pos = torch.as_tensor(pos, device=DEVICE).float()
        return torch.stack([torch.as_tensor(G1(a, pos, natoms, η0, self.rs, rcut)), torch.as_tensor(G2(a, pos, natoms, ηa, self.ζ, self.lam, rcut))])

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.rs.copy_((self.rs - (lr * learnable_grads[int(0)])))

class Subnet(nn.Module):
    def __init__(self, w1, b1, w2, b2, nin, nhid):
        super().__init__()
        self.w1 = nn.Parameter(torch.as_tensor(w1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.w2 = nn.Parameter(torch.as_tensor(w2))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.nin = int(nin)
        self.nhid = int(nhid)
        self.learnable_params = [self.w1, self.b1, self.w2, self.b2]

    def forward(self, g):
        this = self
        g = torch.as_tensor(g, device=DEVICE).float()
        return S(g, self.nin, self.nhid, self.w1, self.b1, self.w2, self.b2)

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.w1.copy_((self.w1 - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b1.copy_((self.b1 - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.w2.copy_((self.w2 - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.b2.copy_((self.b2 - (lr * learnable_grads[int(3)])))

class BPNet(nn.Module):
    def __init__(self, desc, sub):
        super().__init__()
        self.add_module('desc', desc)
        self.add_module('sub', sub)
        self.learnable_params = []

    def forward(self, pos, natoms):
        this = self
        pos = torch.as_tensor(pos, device=DEVICE).float()
        total = 0.0
        for a in range(int(0), int(natoms)):
            total = total + self.sub(self.desc(a, pos, natoms))
        return total

    def desc_grads(self, pos, natoms):
        this = self
        pos = torch.as_tensor(pos, device=DEVICE).float()
        en = self(pos, natoms)
        return compute_grad(en, self.desc.learnable_params)

    def sub_grads(self, pos, natoms):
        this = self
        pos = torch.as_tensor(pos, device=DEVICE).float()
        en = self(pos, natoms)
        return compute_grad(en, self.sub.learnable_params)

    def train(self, X, y, nframes, natoms, epochs, lr):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        y = torch.as_tensor(y, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        last = 0.0
        for ep in range(int(0), int(epochs)):
            epoch_loss = 0.0
            for s in range(int(0), int(nframes)):
                pred = self(X[int(s)], natoms)
                current_loss = mse_loss(pred, y[int(s)])
                epoch_loss = (epoch_loss + current_loss)
                learnable_grads = compute_grad(current_loss, self.sub.learnable_params)
                self.sub.update_params(lr, learnable_grads)
            last = (epoch_loss / nframes)
        return last

    def train_all(self, X, y, nframes, natoms, epochs, lr, lrd):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        y = torch.as_tensor(y, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        lrd = torch.as_tensor(lrd, device=DEVICE).float()
        last = 0.0
        for ep in range(int(0), int(epochs)):
            epoch_loss = 0.0
            for s in range(int(0), int(nframes)):
                pred = self(X[int(s)], natoms)
                current_loss = mse_loss(pred, y[int(s)])
                epoch_loss = (epoch_loss + current_loss)
                sub_g = compute_grad(current_loss, self.sub.learnable_params)
                desc_g = compute_grad(current_loss, self.desc.learnable_params)
                self.sub.update_params(lr, sub_g)
                self.desc.update_params(lrd, desc_g)
            last = (epoch_loss / nframes)
        return last

# === Program ===
π = 3.141592653589793
Pc = 8
r = torch.as_tensor(torch.tensor([[3.69468256, 2.44237664, 1.57337608], [1.22577221, 0.73713025, 6.56769165], [0.94713127, 0.27985289, 2.65383696], [(-0.70279321), 0.26258477, 8.05036569], [1.77357416, 2.44520749, 3.17501106], [0.68309124, (-0.82981877), 4.72394411], [(-0.40740593), 1.70240007, 5.08154978], [(-1.40101553), (-1.92847942), 4.49800493]], device=DEVICE)).requires_grad_(True).to(DEVICE)
η0 = 0.05
ηa = 0.005
rcut = 4.0
rs0 = 0.0
ζ1 = 1
lam_p = int(1)
nin0 = 2
nhid0 = 3
w1v = torch.tensor([[0.1, (-0.2)], [0.3, 0.4], [(-0.5), 0.6]], device=DEVICE)
b1v = torch.tensor([0.05, (-0.1), 0.15], device=DEVICE)
w2v = torch.tensor([0.7, (-0.8), 0.9], device=DEVICE)
b2v = 0.25
dsc = Descriptor(rs0, ζ1, lam_p).to(DEVICE)
sbn = Subnet(w1v, b1v, w2v, b2v, nin0, nhid0).to(DEVICE)
net = BPNet(dsc, sbn).to(DEVICE)
print(net(r, Pc))
print(compute_grad(lambda _dr: net(_dr, Pc), r))
print(net.desc_grads(r, Pc))
print(net.sub_grads(r, Pc))
X3 = torch.tensor([[[3.69468256, 2.44237664, 1.57337608], [1.22577221, 0.73713025, 6.56769165], [0.94713127, 0.27985289, 2.65383696], [(-0.70279321), 0.26258477, 8.05036569], [1.77357416, 2.44520749, 3.17501106], [0.68309124, (-0.82981877), 4.72394411], [(-0.40740593), 1.70240007, 5.08154978], [(-1.40101553), (-1.92847942), 4.49800493]], [[3.70035704, 2.43735102, 1.57132253], [1.22275941, 0.72471744, 6.58336459], [0.94499653, 0.28003215, 2.65875943], [(-0.70886552), 0.25557454, 8.06916201], [1.76940601, 2.44227119, 3.17983024], [0.68112097, (-0.83334362), 4.73922893], [(-0.4129064), 1.69891291, 5.08972282], [(-1.40456399), (-1.93086651), 4.51142853]], [[3.70018932, 2.43749961, 1.57138319], [1.2228485, 0.72508445, 6.58290125], [0.94505962, 0.28002682, 2.65861393], [(-0.70868602), 0.25578177, 8.06860634], [1.76952919, 2.44235799, 3.17968776], [0.68117921, (-0.83323943), 4.73877707], [(-0.41274385), 1.69901599, 5.08948123], [(-1.40445908), (-1.93079596), 4.51103173]]], device=DEVICE)
y3 = torch.tensor([(-39.9157861), (-39.91789224), (-39.91789594)], device=DEVICE)
nframes3 = 3
nep = 10
lr0 = 0.0001
print(mse_loss(net(r, Pc), y3[int(0)]))
print(net.train(X3, y3, nframes3, Pc, nep, lr0))
print(net(X3[int(0)], Pc))
print(net(X3[int(1)], Pc))
print(net(X3[int(2)], Pc))
lrd0 = 0.0001
dsc2 = Descriptor(0.0, ζ1, lam_p).to(DEVICE)
net2 = BPNet(dsc2, Subnet(torch.tensor([[0.1, (-0.2)], [0.3, 0.4], [(-0.5), 0.6]], device=DEVICE), torch.tensor([0.05, (-0.1), 0.15], device=DEVICE), torch.tensor([0.7, (-0.8), 0.9], device=DEVICE), 0.25, nin0, nhid0)).to(DEVICE)
print(net2.train_all(X3, y3, nframes3, Pc, nep, lr0, lrd0))
print(dsc2.rs)