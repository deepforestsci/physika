import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def modulo(s, m):
    r = s
    d = (m * (2.0 ** (BITS - 1)))
    for k in range(int(0), int(BITS)):
        if r >= d:
            r = (r - d)
        d = (d / 2.0)
    return r

def floor(x):
    a = x
    if x < 0.0:
        a = (0.0 - x)
    r = a
    n = 0.0
    p = (2.0 ** (BITS - 1))
    for k in range(int(0), int(BITS)):
        if r >= p:
            r = (r - p)
            n = (n + p)
        p = (p / 2.0)
    result = n
    if x < 0.0:
        if r > 0.0:
            result = ((0.0 - n) - 1.0)
        else:
            result = (0.0 - n)
    return result

def hash_list(xs):
    h = 17.0
    for i in range(int(0), int(len(xs))):
        h = modulo(((h * 31.0) + xs[int(i)]), M)
    return h

def hash_step(h, x):
    return modulo(((h * 31.0) + x), M)

def bubble_sort(xs):
    k = len(xs)
    ys = torch.stack([xs[int(a)] for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    for i in range(int(0), int(k)):
        for j in range(int(0), int((k - 1))):
            if ys[int(j)] > ys[int((j + 1))]:
                t = (ys[int(j)] + 0.0)
                ys[int(j)] = ys[int((j + 1))]
                ys[int((j + 1))] = t
    return ys

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

def new_molecule(atomic_num, formal_charge):
    n_atoms = len(atomic_num)
    z = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n_atoms)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n_atoms)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    g = Molecule()
    g.adjacency = z
    g.atomic_num = atomic_num
    g.formal_charge = formal_charge
    return g

def degree(g, u):
    m = g.adjacency
    k = get_2d_array_num_rows(m)
    d = 0
    for v in range(int(0), int(k)):
        if m[int(u), int(v)] > 0.0:
            d = d + 1
    return d

def hydrogens(g, u):
    m = g.adjacency
    z = g.atomic_num
    k = get_2d_array_num_rows(m)
    h = 0
    for v in range(int(0), int(k)):
        if m[int(u), int(v)] > 0.0:
            if z[int(v)] == 1.0:
                h = h + 1
    return h

def aromatic(g, u):
    m = g.adjacency
    k = get_2d_array_num_rows(m)
    r = 0
    for v in range(int(0), int(k)):
        if m[int(u), int(v)] == 1.5:
            r = 1
    return r

def invariants(g):
    m = g.adjacency
    z = g.atomic_num
    c = g.formal_charge
    k = get_2d_array_num_rows(m)
    inv = torch.stack([torch.stack([(a * 0.0) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]]) for _fi_i in range(int(5)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    for a in range(int(0), int(k)):
        inv[int(0), int(a)] = z[int(a)]
        inv[int(1), int(a)] = degree(g, a)
        inv[int(2), int(a)] = c[int(a)]
        inv[int(3), int(a)] = hydrogens(g, a)
        inv[int(4), int(a)] = aromatic(g, a)
    return inv

def atom(g, u):
    inv = invariants(g)
    return inv[:, int(u)]

def atom_id(g, u):
    return hash_list(atom(g, u))

def initial_ids(g):
    inv = invariants(g)
    k = get_2d_array_num_rows(g.adjacency)
    new_ids = torch.stack([(a * 0.0) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    for a in range(int(0), int(k)):
        new_ids[int(a)] = hash_list(inv[:, int(a)])
    return new_ids

def update_ids(g, ids, r):
    m = g.adjacency
    k = get_2d_array_num_rows(m)
    new_ids = torch.stack([(a * 0.0) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    keys = torch.stack([(a * 0.0) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    for u in range(int(0), int(k)):
        for v in range(int(0), int(k)):
            keys[int(v)] = 0.0
            if m[int(u), int(v)] > 0.0:
                keys[int(v)] = (((2.0 * m[int(u), int(v)]) * M) + ids[int(v)])
        keys = bubble_sort(keys)
        h = 17.0
        h = hash_step(h, r)
        h = hash_step(h, ids[int(u)])
        for i in range(int(0), int(k)):
            if keys[int(i)] > 0.0:
                h = hash_step(h, keys[int(i)])
        new_ids[int(u)] = h
    return new_ids

def fingerprint(g, radius):
    z = g.atomic_num
    k = get_2d_array_num_rows(g.adjacency)
    fp = torch.stack([(b * 0.0) for _fi_b in range(int(N_BITS)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]])
    ids = initial_ids(g)
    for a in range(int(0), int(k)):
        if z[int(a)] > 1.0:
            fp[int(modulo(ids[int(a)], N_BITS))] = 1.0
    for r in range(int(0), int(radius)):
        ids = update_ids(g, ids, (r + 1.0))
        for a in range(int(0), int(k)):
            if z[int(a)] > 1.0:
                fp[int(modulo(ids[int(a)], N_BITS))] = 1.0
    return fp

def ecfp(g, diameter):
    return fingerprint(g, (diameter / 2.0))

def tanimoto(a, b, n=None):
    if n is None:
        n = int(a.shape[0])
    both = torch.sum((a * b))
    return (both / ((torch.sum(a) + torch.sum(b)) - both))

# === Classes ===
class Molecule(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.adjacency = None
        self.atomic_num = None
        self.formal_charge = None

    def num_atoms(self):
        this = self
        return (get_2d_array_num_rows(self.adjacency) * 1.0)

    def has_edge(self, u, v):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        v = torch.as_tensor(v, device=DEVICE).float()
        m = self.adjacency
        r = m[int(u)]
        return r[int(v)]

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

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
BITS = 32
M = 65521.0
N_BITS = 2048
CH4_atomic_num = torch.tensor([6, 1, 1, 1, 1], device=DEVICE)
CH4_formal_charge = torch.tensor([0, 0, 0, 0, 0], device=DEVICE)
CH4 = new_molecule(CH4_atomic_num, CH4_formal_charge)
print(CH4.add_edge(0.0, 1.0))
print(CH4.add_edge(0.0, 2.0))
print(CH4.add_edge(0.0, 3.0))
print(CH4.add_edge(0.0, 4.0))
print(invariants(CH4))
print(atom_id(CH4, 0.0))
print(atom_id(CH4, 1.0))
C6H6_atomic_num = torch.tensor([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1], device=DEVICE)
C6H6_formal_charge = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=DEVICE)
C6H6 = new_molecule(C6H6_atomic_num, C6H6_formal_charge)
print(C6H6.add_weighted_edge(0.0, 1.0, 1.5))
print(C6H6.add_weighted_edge(1.0, 2.0, 1.5))
print(C6H6.add_weighted_edge(2.0, 3.0, 1.5))
print(C6H6.add_weighted_edge(3.0, 4.0, 1.5))
print(C6H6.add_weighted_edge(4.0, 5.0, 1.5))
print(C6H6.add_weighted_edge(5.0, 0.0, 1.5))
print(C6H6.add_edge(0.0, 6.0))
print(C6H6.add_edge(1.0, 7.0))
print(C6H6.add_edge(2.0, 8.0))
print(C6H6.add_edge(3.0, 9.0))
print(C6H6.add_edge(4.0, 10.0))
print(C6H6.add_edge(5.0, 11.0))
print(invariants(C6H6))
print(atom_id(C6H6, 0.0))
print(atom_id(C6H6, 6.0))
C7H8_atomic_num = torch.tensor([6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], device=DEVICE)
C7H8_formal_charge = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=DEVICE)
C7H8 = new_molecule(C7H8_atomic_num, C7H8_formal_charge)
print(C7H8.add_weighted_edge(0.0, 1.0, 1.5))
print(C7H8.add_weighted_edge(1.0, 2.0, 1.5))
print(C7H8.add_weighted_edge(2.0, 3.0, 1.5))
print(C7H8.add_weighted_edge(3.0, 4.0, 1.5))
print(C7H8.add_weighted_edge(4.0, 5.0, 1.5))
print(C7H8.add_weighted_edge(5.0, 0.0, 1.5))
print(C7H8.add_edge(0.0, 6.0))
print(C7H8.add_edge(1.0, 7.0))
print(C7H8.add_edge(2.0, 8.0))
print(C7H8.add_edge(3.0, 9.0))
print(C7H8.add_edge(4.0, 10.0))
print(C7H8.add_edge(5.0, 11.0))
print(C7H8.add_edge(6.0, 12.0))
print(C7H8.add_edge(6.0, 13.0))
print(C7H8.add_edge(6.0, 14.0))
print(invariants(C7H8))
CH4_ids0 = initial_ids(CH4)
CH4_ids1 = update_ids(CH4, CH4_ids0, 1.0)
CH4_ids2 = update_ids(CH4, CH4_ids1, 2.0)
print(CH4_ids1)
print(CH4_ids2)
C6H6_ids0 = initial_ids(C6H6)
C6H6_ids1 = update_ids(C6H6, C6H6_ids0, 1.0)
C6H6_ids2 = update_ids(C6H6, C6H6_ids1, 2.0)
print(C6H6_ids1)
print(C6H6_ids2)
CH4_ecfp4 = ecfp(CH4, 4)
C6H6_ecfp4 = ecfp(C6H6, 4)
C7H8_ecfp4 = ecfp(C7H8, 4)
print(torch.sum(CH4_ecfp4 if isinstance(CH4_ecfp4, torch.Tensor) else torch.tensor(float(CH4_ecfp4))))
print(torch.sum(C6H6_ecfp4 if isinstance(C6H6_ecfp4, torch.Tensor) else torch.tensor(float(C6H6_ecfp4))))
print(torch.sum(C7H8_ecfp4 if isinstance(C7H8_ecfp4, torch.Tensor) else torch.tensor(float(C7H8_ecfp4))))
print(tanimoto(C6H6_ecfp4, C6H6_ecfp4))
print(tanimoto(C6H6_ecfp4, C7H8_ecfp4))
print(tanimoto(CH4_ecfp4, C6H6_ecfp4))
print(tanimoto(CH4_ecfp4, C7H8_ecfp4))