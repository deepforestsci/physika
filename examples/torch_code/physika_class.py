import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def ke_wrt_vel(vel):
    particle = Particle(pos0, vel, 1.0)
    return particle.kinetic_energy()

def ke_vy(vy):
    p = Particle(pos0, torch.stack([torch.as_tensor(1.0).float(), torch.as_tensor(vy).float()]), 2.0)
    return p.kinetic_energy()

def norm_sq_wrt_x(x):
    vec = Vec(x, 4.0)
    return vec.norm_sq()

# === Classes ===
class Vec(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.as_tensor(x).float()
        self.y = torch.as_tensor(y).float()

    def dot(self, other):
        this = self
        return ((self.x * other.x) + (self.y * other.y))

    def scale(self, s):
        this = self
        s = torch.as_tensor(s).float()
        return Vec((self.x * s), (self.y * s))

    def norm_sq(self):
        this = self
        return ((self.x * self.x) + (self.y * self.y))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Particle(nn.Module):
    def __init__(self, pos, vel, mass):
        super().__init__()
        self.pos = torch.as_tensor(pos).float()
        self.vel = torch.as_tensor(vel).float()
        self.mass = torch.as_tensor(mass).float()

    def kinetic_energy(self):
        this = self
        return ((0.5 * self.mass) * torch.sum((self.vel * self.vel) if isinstance((self.vel * self.vel), torch.Tensor) else torch.tensor(float((self.vel * self.vel)))))

    def step(self, force, dt):
        this = self
        force = torch.as_tensor(force).float()
        dt = torch.as_tensor(dt).float()
        acc = (force * (1.0 / self.mass))
        new_vel = (self.vel + (acc * dt))
        new_pos = (self.pos + (self.vel * dt))
        return Particle(new_pos, new_vel, self.mass)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
a = Vec(3.0, 4.0)
b = Vec(1.0, 0.0)
physika_print(a.x)
physika_print(a.y)
dot_ab = a.dot(b)
physika_print(dot_ab)
c = a.scale(4)
physika_print(c.x)
physika_print(c.y)
pos0 = torch.tensor([0.0, 10.0])
vel0 = torch.tensor([1.0, 0.0])
gravity = torch.tensor([0.0, (-9.81)])
p = Particle(pos0, vel0, 9.0)
ke0 = p.kinetic_energy()
physika_print(ke0)
p1 = p.step(gravity, 0.5)
physika_print(p1.pos)
p2 = p1.step(gravity, 0.5)
physika_print(p2.pos)
v = torch.as_tensor(torch.tensor([2.0, 3.4])).float().requires_grad_(True)
ke0_v = ke_wrt_vel(v)
physika_print(ke0_v)
dKE_dv = compute_grad(lambda _dv: ke_wrt_vel(_dv), v)
physika_print(dKE_dv)
vy0 = torch.tensor(3.0, requires_grad=True)
physika_print(ke_vy(vy0))
physika_print(compute_grad(lambda _dvy0: ke_vy(_dvy0), vy0))
x0 = torch.tensor(3.0, requires_grad=True)
physika_print(norm_sq_wrt_x(x0))
physika_print(compute_grad(lambda _dx0: norm_sq_wrt_x(_dx0), x0))
x1 = torch.tensor(5.0, requires_grad=True)
vec = Vec(x1, 4.0)
physika_print(compute_grad(vec.norm_sq(), x1))
x1 = torch.tensor(5.0, requires_grad=True)
vec = Vec(x1, 4.0)
physika_print(compute_grad(vec.x, x1))