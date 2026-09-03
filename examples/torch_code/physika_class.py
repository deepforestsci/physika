import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def ke_wrt_vel(vel):
    particle = Particle(pos0, vel, 1.0)
    return particle.kinetic_energy()

def ke_vy(vy):
    p = Particle(pos0, torch.stack([torch.as_tensor(1.0), torch.as_tensor(vy)]), 2.0)
    return p.kinetic_energy()

def norm_sq_wrt_x(x):
    vec = Vec(x, 4.0)
    return vec.norm_sq()

# === Classes ===
class ExampleClass(nn.Module):
    def __init__(self, ):
        super().__init__()

    def class_method(self):
        this = self
        return 1

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class ScalarClass(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = torch.as_tensor(x).float()

    def return_member_variable(self):
        this = self
        return self.x

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

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
        s = torch.as_tensor(s, device=DEVICE).float()
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
        force = torch.as_tensor(force, device=DEVICE).float()
        dt = torch.as_tensor(dt, device=DEVICE).float()
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

class A(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = torch.as_tensor(x).float()

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class B(nn.Module):
    def __init__(self, objA):
        super().__init__()
        self.add_module('objA', objA)

    def access_member(self):
        this = self
        self.objA.x = 2.0
        return self.objA.x

    def access_memeber_in_loop(self):
        this = self
        for i in range(int(0), int(1)):
            self.objA.x = 3.0
        return self.objA.x

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
obj_example_class = ExampleClass().to(DEVICE)
print(obj_example_class.class_method())
obj_scalar_class = ScalarClass(3.0).to(DEVICE)
print(obj_scalar_class.return_member_variable())
a = Vec(3.0, 4.0)
b = Vec(1.0, 0.0)
print(a.x)
print(a.y)
dot_ab = a.dot(b)
print(dot_ab)
c = a.scale(4)
print(c.x)
print(c.y)
pos0 = torch.tensor([0.0, 10.0], device=DEVICE)
vel0 = torch.tensor([1.0, 0.0], device=DEVICE)
gravity = torch.tensor([0.0, (-9.81)], device=DEVICE)
p = Particle(pos0, vel0, 9.0)
ke0 = p.kinetic_energy()
print(ke0)
p1 = p.step(gravity, 0.5)
print(p1.pos)
p2 = p1.step(gravity, 0.5)
print(p2.pos)
v = torch.as_tensor(torch.tensor([2.0, 3.4], device=DEVICE)).requires_grad_(True).to(DEVICE)
ke0_v = ke_wrt_vel(v)
print(ke0_v)
dKE_dv = compute_grad(lambda _dv: ke_wrt_vel(_dv), v)
print(dKE_dv)
vy0 = torch.tensor(3.0, requires_grad=True)
print(ke_vy(vy0))
print(compute_grad(lambda _dvy0: ke_vy(_dvy0), vy0))
x0 = torch.tensor(3.0, requires_grad=True)
print(norm_sq_wrt_x(x0))
print(compute_grad(lambda _dx0: norm_sq_wrt_x(_dx0), x0))
x1 = torch.tensor(5.0, requires_grad=True)
vec = Vec(x1, 4.0).to(DEVICE)
print(compute_grad(vec.norm_sq(), x1))
x1 = torch.tensor(5.0, requires_grad=True)
vec = Vec(x1, 4.0).to(DEVICE)
print(compute_grad(vec.x, x1))
obj_A = A(1.0).to(DEVICE)
obj_B = B(obj_A).to(DEVICE)
print(obj_B.access_member())
print(obj_B.access_memeber_in_loop())