import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def dist_3d(p1, p2):
    total = 0
    for i in range(int(0), int(3)):
        total = total + ((p1[int(i)] - p2[int(i)]) ** 2)
    return (total ** (1 / 2))

def F(q1, q2, x1, x2):
    return (((1 / ((4 * π) * ε0)) * ((q1 * q2) / (dist_3d(x1, x2) ** 2))) * ((x2 - x1) / dist_3d(x1, x2)))

def E(q1, x1, x2):
    return (((1 / ((4 * π) * ε0)) * (q1 / (dist_3d(x1, x2) ** 2))) * ((x2 - x1) / dist_3d(x1, x2)))

def dV_dt(V, Capa, Ress):
    return ((-V) / (Ress * Capa))

def rk4_step(V, Capa, Ress, dt):
    k1 = dV_dt(V, Capa, Ress)
    k2 = dV_dt((V + ((0.5 * dt) * k1)), Capa, Ress)
    k3 = dV_dt((V + ((0.5 * dt) * k2)), Capa, Ress)
    k4 = dV_dt((V + (dt * k3)), Capa, Ress)
    return (V + ((dt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))

def voltage_at(V, Capa, Ress, dt, steps):
    for i in range(int(0), int(steps)):
        V = rk4_step(V, Capa, Ress, dt)
    return V

def current_at(V, Capa, Ress, dt, steps):
    V = voltage_at(V, Capa, Ress, dt, steps)
    return (V / Ress)

# === Program ===
π, ε0 = 3.14159, 8.854e-12
coulomb_f = F(1, 1, torch.tensor([0, 0, 0], device=DEVICE), torch.tensor([1, 1, 1], device=DEVICE))
print(print(coulomb_f))
coulomb_e = E(1, torch.tensor([0, 0, 0], device=DEVICE), torch.tensor([1, 1, 1], device=DEVICE))
print(print(coulomb_e))
Capa, Ress, V0 = 1.0, 10.0, 10.0
dt = 0.01
steps = 200
V_final = voltage_at(V0, Capa, Ress, dt, steps)
print(print(V_final))
I_final = current_at(V0, Capa, Ress, dt, steps)
print(print(I_final))
V_exact = (V0 * torch.exp((((-dt) * steps) / (Ress * Capa)) if isinstance((((-dt) * steps) / (Ress * Capa)), torch.Tensor) else torch.tensor(float((((-dt) * steps) / (Ress * Capa))))))
print(print(V_exact))