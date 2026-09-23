import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def dist_3d(p1, p2):
    return (((((p1[int(0)] - p2[int(0)]) ** 2) + ((p1[int(1)] - p2[int(1)]) ** 2)) + ((p1[int(2)] - p2[int(2)]) ** 2)) ** (1 / 2))

def F(q1, q2, x1, x2):
    return (((1 / ((4 * π) * ε0)) * ((q1 * q2) / (dist_3d(x1, x2) ** 2))) * ((x2 - x1) / dist_3d(x1, x2)))

def E(q1, x1, x2):
    return (((1 / ((4 * π) * ε0)) * (q1 / (dist_3d(x1, x2) ** 2))) * ((x2 - x1) / dist_3d(x1, x2)))

def coulomb_graph(q, x):
    k = get_2d_array_num_rows(x)
    g = empty_graph(k)
    for i in range(int(0), int(k)):
        for j in range(int(0), int(k)):
            if i < j:
                g.add_weighted_edge((i * 1.0), (j * 1.0), ((((1 / ((4 * π) * ε0)) * q[int(i)]) * q[int(j)]) / dist_3d(x[int(i)], x[int(j)])))
    return g

def total_energy(g):
    k = g.num_vertices()
    acc = 0
    for u in range(int(0), int(k)):
        acc = acc + g.degree((u * 1.0))
    return (acc / 2)

def kcl_current(g, V):
    k = g.num_vertices()
    I = (V * 0.0)
    for u in range(int(0), int(k)):
        I[int(u)] = get_sum_of_1d_array((g.neighbors((u * 1.0)) * (V[int(u)] - V)))
    return I

def dV_dt(net, V):
    return (((-kcl_current(net.wires, V)) / net.capacitor.Capa) * net.free)

def rk4_step(net, V, dt, n=None):
    if n is None:
        n = int(V.shape[0])
    k1 = dV_dt(net, V)
    k2 = dV_dt(net, (V + ((0.5 * dt) * k1)))
    k3 = dV_dt(net, (V + ((0.5 * dt) * k2)))
    k4 = dV_dt(net, (V + (dt * k3)))
    return (V + ((dt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))

def voltage_at(net, V0, dt, steps):
    V = V0
    for i in range(int(0), int(steps)):
        V = rk4_step(net, V, dt)
    return V

def joule_power(net, V):
    return get_sum_of_1d_array((V * kcl_current(net.wires, V)))

def V_after(R_):
    wires = empty_graph(2)
    wires.add_weighted_edge(0.0, 1.0, (1 / R_))
    net = RCNetwork(wires, capacitor, torch.tensor([0.0, 1.0], device=DEVICE))
    V = voltage_at(net, torch.stack([torch.as_tensor(0.0), torch.as_tensor(V0)]), dt, steps)
    return V[int(1)]

def empty_graph(n_vertices):
    z = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n_vertices)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n_vertices)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    g = UndirectedGraph()
    g.adjacency = z
    return g

def get_sum_of_1d_array(x):
    total = 0
    for i in range(len(x)):
        total = total + x[int(i)]
    return total

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

# === Classes ===
class Resistor(nn.Module):
    def __init__(self, Ress):
        super().__init__()
        self.Ress = torch.as_tensor(Ress).float()

    def conductance(self):
        this = self
        return (1 / self.Ress)

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

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class RCNetwork(nn.Module):
    def __init__(self, wires, capacitor, free):
        super().__init__()
        self.add_module('wires', wires)
        self.add_module('capacitor', capacitor)
        self.free = torch.as_tensor(free).float()

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class UndirectedGraph(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.adjacency = None

    def num_vertices(self):
        this = self
        return (get_2d_array_num_rows(self.adjacency) * 1.0)

    def has_edge(self, u, v):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        v = torch.as_tensor(v, device=DEVICE).float()
        m = self.adjacency
        r = m[int(u)]
        return r[int(v)]

    def degree(self, u):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        m = self.adjacency
        r = m[int(u)]
        return get_sum_of_1d_array(r)

    def sq_degree_sum(self, s):
        this = self
        s = torch.as_tensor(s, device=DEVICE).float()
        m = self.adjacency
        k = get_2d_array_num_rows(m)
        acc = 0
        for i in range(int(0), int(k)):
            d = 0
            for j in range(int(0), int(k)):
                d = d + (s * m[int(i), int(j)])
            acc = acc + (d * d)
        return acc

    def neighbors(self, u):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        m = self.adjacency
        return m[int(u)]

    def add_weighted_edge(self, u, v, w):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        v = torch.as_tensor(v, device=DEVICE).float()
        w = torch.as_tensor(w, device=DEVICE).float()
        m = self.adjacency
        k = get_2d_array_num_rows(m)
        new_adj = torch.stack([torch.stack([m[int(a), int(b)] for _fi_b in range(int(k)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        new_adj[int(u), int(v)] = w
        new_adj[int(v), int(u)] = w
        self.adjacency = new_adj

    def add_edge(self, u, v):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        v = torch.as_tensor(v, device=DEVICE).float()
        self.add_weighted_edge(u, v, 1.0)

    def grow_adjacency(self, new_n):
        this = self
        old = self.adjacency
        result = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(new_n)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(new_n)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        m = get_2d_array_num_rows(old)
        for a in range(int(0), int(m)):
            for b in range(int(0), int(m)):
                result[int(a), int(b)] = old[int(a), int(b)]
        return result

    def add_vertex(self, new_n):
        this = self
        self.adjacency = self.grow_adjacency(new_n)

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
coulomb_f = F(1.0, 1.0, torch.tensor([0.0, 0.0, 0.0], device=DEVICE), torch.tensor([1.0, 1.0, 1.0], device=DEVICE))
print(print(coulomb_f))
coulomb_e = E(1.0, torch.tensor([0.0, 0.0, 0.0], device=DEVICE), torch.tensor([1.0, 1.0, 1.0], device=DEVICE))
print(print(coulomb_e))
q = torch.tensor([1.0, (-2.0), 1.0], device=DEVICE)
xs = torch.tensor([[(-1.0), 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)
cg = coulomb_graph(q, xs)
U_O = cg.degree(1.0)
print(print(U_O))
U = total_energy(cg)
print(print(U))
Capa, Ress, V0 = 1.0, 10.0, 10.0
dt = 0.01
steps = 200
capacitor = Capacitor(Capa).to(DEVICE)
resistor = Resistor(Ress).to(DEVICE)
rc_wires = empty_graph(2)
print(rc_wires.add_weighted_edge(0.0, 1.0, resistor.conductance()))
rc = RCNetwork(rc_wires, capacitor, torch.tensor([0.0, 1.0], device=DEVICE)).to(DEVICE)
V_rc = voltage_at(rc, torch.stack([torch.as_tensor(0.0), torch.as_tensor(V0)]), dt, steps)
print(print(V_rc))
I_final = resistor.current(V_rc[int(1)])
print(print(I_final))
ladder_wires = empty_graph(4)
print(ladder_wires.add_weighted_edge(0.0, 1.0, resistor.conductance()))
print(ladder_wires.add_weighted_edge(1.0, 2.0, resistor.conductance()))
print(ladder_wires.add_weighted_edge(2.0, 3.0, resistor.conductance()))
ladder = RCNetwork(ladder_wires, capacitor, torch.tensor([0.0, 1.0, 1.0, 1.0], device=DEVICE)).to(DEVICE)
V_ladder = voltage_at(ladder, torch.stack([torch.as_tensor(0.0), torch.as_tensor(V0), torch.as_tensor(V0), torch.as_tensor(V0)]), dt, steps)
print(print(V_ladder))
P_ladder = joule_power(ladder, V_ladder)
print(print(P_ladder))
R0 = torch.tensor(10.0, requires_grad=True)
print(print(V_after(R0)))
print(print(compute_grad(lambda _dR0: V_after(_dR0), R0)))