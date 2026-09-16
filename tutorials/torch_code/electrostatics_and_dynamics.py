import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def dist_3d(p1, p2):
    return (((((p1[int(0)] - p2[int(0)]) ** 2) + ((p1[int(1)] - p2[int(1)]) ** 2)) + ((p1[int(2)] - p2[int(2)]) ** 2)) ** (1 / 2))

def F(q1, q2, x1, x2):
    return (((1 / ((4 * π) * ε0)) * ((q1 * q2) / (dist_3d(x1, x2) ** 2))) * ((x2 - x1) / dist_3d(x1, x2)))

def E(q1, x1, x2):
    return (((1 / ((4 * π) * ε0)) * (q1 / (dist_3d(x1, x2) ** 2))) * ((x2 - x1) / dist_3d(x1, x2)))

# === Classes ===
class VoltageSource(nn.Module):
    def __init__(self, Volt):
        super().__init__()
        self.Volt = torch.as_tensor(Volt).float()

    def voltage(self):
        this = self
        return self.Volt

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Resistor(nn.Module):
    def __init__(self, Ress):
        super().__init__()
        self.Ress = torch.as_tensor(Ress).float()

    def current(self, V):
        this = self
        V = torch.as_tensor(V, device=DEVICE).float()
        return (V / self.Ress)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Capacitor(nn.Module):
    def __init__(self, Capa):
        super().__init__()
        self.Capa = torch.as_tensor(Capa).float()

    def dV_dt(self, V, Ress):
        this = self
        V = torch.as_tensor(V, device=DEVICE).float()
        Ress = torch.as_tensor(Ress, device=DEVICE).float()
        return ((-V) / (Ress * self.Capa))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class RCCircuit(nn.Module):
    def __init__(self, source, resistor, capacitor):
        super().__init__()
        self.add_module('source', source)
        self.add_module('resistor', resistor)
        self.add_module('capacitor', capacitor)

    def rk4_step(self, V, dt):
        this = self
        V = torch.as_tensor(V, device=DEVICE).float()
        dt = torch.as_tensor(dt, device=DEVICE).float()
        k1 = self.capacitor.dV_dt(V, self.resistor.Ress)
        k2 = self.capacitor.dV_dt((V + ((0.5 * dt) * k1)), self.resistor.Ress)
        k3 = self.capacitor.dV_dt((V + ((0.5 * dt) * k2)), self.resistor.Ress)
        k4 = self.capacitor.dV_dt((V + (dt * k3)), self.resistor.Ress)
        return (V + ((dt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))

    def voltage_at(self, dt, steps):
        this = self
        dt = torch.as_tensor(dt, device=DEVICE).float()
        V = self.source.voltage()
        for i in range(int(0), int(steps)):
            V = self.rk4_step(V, dt)
        return V

    def current_at(self, dt, steps):
        this = self
        dt = torch.as_tensor(dt, device=DEVICE).float()
        V = self.voltage_at(dt, steps)
        return self.resistor.current(V)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
π, ε0 = 3.14159, 8.854e-12
coulomb_f = F(1, 1, torch.tensor([0, 0, 0], device=DEVICE), torch.tensor([1, 1, 1], device=DEVICE))
print(print(coulomb_f))
coulomb_e = E(1, torch.tensor([0, 0, 0], device=DEVICE), torch.tensor([1, 1, 1], device=DEVICE))
print(print(coulomb_e))
Capa, Ress, V0 = 1.0, 10.0, 10.0
dt = 0.01
steps = 200
source = VoltageSource(V0).to(DEVICE)
resistor = Resistor(Ress).to(DEVICE)
capacitor = Capacitor(Capa).to(DEVICE)
circuit = RCCircuit(source, resistor, capacitor).to(DEVICE)
V_final = circuit.voltage_at(dt, steps)
print(print(V_final))
I_final = circuit.current_at(dt, steps)
print(print(I_final))