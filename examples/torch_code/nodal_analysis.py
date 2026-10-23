import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def zeros_1d(n):
    return torch.stack([(i * 0.0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def zeros_2d(n):
    return torch.stack([torch.stack([(i * 0.0) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def conductance_matrix(num_nodes, resistors, num_resistors):
    G = zeros_2d(num_nodes)
    n1 = 0.0
    n2 = 0.0
    g = 0.0
    for r in range(int(0), int(num_resistors)):
        n1 = resistors[int(r), int(0)]
        n2 = resistors[int(r), int(1)]
        g = (1.0 / resistors[int(r), int(2)])
        if n1 != 0.0:
            G[int((n1 - 1)), int((n1 - 1))] += g
        if n2 != 0.0:
            G[int((n2 - 1)), int((n2 - 1))] += g
        if n1 != 0.0:
            if n2 != 0.0:
                G[int((n1 - 1)), int((n2 - 1))] = (G[int((n1 - 1)), int((n2 - 1))] - g)
                G[int((n2 - 1)), int((n1 - 1))] = (G[int((n2 - 1)), int((n1 - 1))] - g)
    return G

def current_vector(num_nodes, sources, num_sources):
    I = zeros_1d(num_nodes)
    n1 = 0.0
    n2 = 0.0
    current = 0.0
    for s in range(int(0), int(num_sources)):
        n1 = sources[int(s), int(0)]
        n2 = sources[int(s), int(1)]
        current = sources[int(s), int(2)]
        if n1 != 0.0:
            I[int((n1 - 1))] = (I[int((n1 - 1))] - current)
        if n2 != 0.0:
            I[int((n2 - 1))] += current
    return I

def eye(n):
    E = zeros_2d(n)
    for i in range(int(0), int(n)):
        E[int(i), int(i)] = 1.0
    return E

def linear_solve(n, A, b):
    E = eye(n)
    for c in range(int(0), int(n)):
        b = torch.stack([(b[int(i)] - (((A[int(i), int(c)] - E[int(i), int(c)]) * b[int(c)]) / A[int(c), int(c)])) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
        A = torch.stack([torch.stack([(A[int(i), int(j)] - (((A[int(i), int(c)] - E[int(i), int(c)]) * A[int(c), int(j)]) / A[int(c), int(c)])) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return b

def nodal_analysis(num_nodes, resistors, num_resistors, sources, num_sources, k=None, m=None):
    if k is None:
        k = int(resistors.shape[0])
    if m is None:
        m = int(sources.shape[0])
    G = conductance_matrix(num_nodes, resistors, num_resistors)
    I = current_vector(num_nodes, sources, num_sources)
    return linear_solve(num_nodes, G, I)

def v1_of(Rs):
    res = torch.stack([torch.stack([torch.as_tensor(1, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(2, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(Rs[int(0)], dtype=torch.get_default_dtype(), device=DEVICE)]), torch.stack([torch.as_tensor(1, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(0, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(Rs[int(1)], dtype=torch.get_default_dtype(), device=DEVICE)]), torch.stack([torch.as_tensor(2, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(0, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(Rs[int(2)], dtype=torch.get_default_dtype(), device=DEVICE)])])
    v = nodal_analysis(2, res, 3, current_sources, 2)
    return v[int(0)]

def v1_of_r3(R3):
    res = torch.stack([torch.stack([torch.as_tensor(1, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(2, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(1000.0, dtype=torch.get_default_dtype(), device=DEVICE)]), torch.stack([torch.as_tensor(1, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(0, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(1000.0, dtype=torch.get_default_dtype(), device=DEVICE)]), torch.stack([torch.as_tensor(2, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(0, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(R3, dtype=torch.get_default_dtype(), device=DEVICE)])])
    v = nodal_analysis(2, res, 3, current_sources, 2)
    return v[int(0)]

# === Program ===
resistors = torch.tensor([[1, 2, 1000], [1, 0, 1000], [2, 0, 1000]], device=DEVICE)
current_sources = torch.tensor([[0, 1, 0.001], [0, 2, 0.001]], device=DEVICE)
voltages = nodal_analysis(2, resistors, 3, current_sources, 2, 3, 2)
print(voltages)
Rs = torch.as_tensor(torch.tensor([1000.0, 1000.0, 1000.0], device=DEVICE)).requires_grad_(True).to(DEVICE)
sensitivities = compute_grad(lambda _dRs: v1_of(_dRs), Rs)
print(sensitivities)
target = 1.2
R3 = torch.tensor(1000.0, requires_grad=True)
for it in range(int(0), int(6)):
    R3 = (R3 - ((v1_of_r3(R3) - target) / compute_grad(lambda _dR3: v1_of_r3(_dR3), R3)))
print(R3)
print(v1_of_r3(R3))