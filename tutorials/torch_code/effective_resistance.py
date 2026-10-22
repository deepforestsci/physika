import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def identity(size):
    eye = torch.zeros(int(size), int(size), device=DEVICE)
    for i in range(int(0), int(size)):
        eye[int(i), int(i)] = 1.0
    return eye

def conductance_matrix(n_nodes, edges, Rs):
    G = torch.zeros(int(n_nodes), int(n_nodes), device=DEVICE)
    n_edges = num_rows(edges)
    for k in range(int(0), int(n_edges)):
        u = edges[int(k), int(0)]
        v = edges[int(k), int(1)]
        g = (1.0 / Rs[int(k)])
        G[int(u), int(u)] = (G[int(u), int(u)] + g)
        G[int(v), int(v)] = (G[int(v), int(v)] + g)
        G[int(u), int(v)] = (G[int(u), int(v)] - g)
        G[int(v), int(u)] = (G[int(v), int(u)] - g)
    return G

def gauss_jordan_solve(A, b):
    size = num_rows(A)
    eye = identity(size)
    M = A
    x = b
    for k in range(int(0), int(size)):
        pivot = M[int(k), int(k)]
        r = torch.stack([(M[int(k), int(j)] / pivot) for _fi_j in range(int(size)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]])
        r_x = (x[int(k)] / pivot)
        c = torch.stack([(M[int(i), int(k)] - eye[int(i), int(k)]) for _fi_i in range(int(size)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
        M = torch.stack([torch.stack([(M[int(i), int(j)] - (c[int(i)] * r[int(j)])) for _fi_j in range(int(size)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(size)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
        x = torch.stack([(x[int(i)] - (c[int(i)] * r_x)) for _fi_i in range(int(size)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return x

def node_voltages(n_nodes, edges, Rs, a, b):
    G = conductance_matrix(n_nodes, edges, Rs)
    eye = identity(n_nodes)
    e_a = eye[int(a)]
    e_b = eye[int(b)]
    G_ground = torch.stack([torch.stack([(((G[int(i), int(j)] * (1.0 - e_b[int(i)])) * (1.0 - e_b[int(j)])) + (e_b[int(i)] * e_b[int(j)])) for _fi_j in range(int(n_nodes)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n_nodes)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    I_in = torch.stack([(e_a[int(i)] * (1.0 - e_b[int(i)])) for _fi_i in range(int(n_nodes)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return gauss_jordan_solve(G_ground, I_in)

def effective_resistance(n_nodes, edges, Rs, a, b):
    v = node_voltages(n_nodes, edges, Rs, a, b)
    return (v[int(a)] - v[int(b)])

def resistance_matrix(n_nodes, edges, Rs):
    R_eff = torch.zeros(int(n_nodes), int(n_nodes), device=DEVICE)
    for a in range(int(0), int(n_nodes)):
        for b in range(int((a + 1)), int(n_nodes)):
            r_ab = effective_resistance(n_nodes, edges, Rs, a, b)
            R_eff[int(a), int(b)] = r_ab
            R_eff[int(b), int(a)] = r_ab
    return R_eff

def branch_currents(n_nodes, edges, Rs, a, b):
    v = node_voltages(n_nodes, edges, Rs, a, b)
    n_edges = num_rows(edges)
    currents = torch.stack([((v[int(edges[int(k), int(0)])] - v[int(edges[int(k), int(1)])]) / Rs[int(k)]) for _fi_k in range(int(n_edges)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    return currents

# === Program ===
series_edges = torch.tensor([[0, 1], [1, 2], [2, 3]], device=DEVICE)
series_Rs = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
R_series = effective_resistance(4, series_edges, series_Rs, 0, 3)
print(print(R_series))
parallel_edges = torch.tensor([[0, 1], [0, 1]], device=DEVICE)
parallel_Rs = torch.tensor([2.0, 3.0], device=DEVICE)
R_parallel = effective_resistance(2, parallel_edges, parallel_Rs, 0, 1)
print(print(R_parallel))
bridge_edges = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [1, 2]], device=DEVICE)
balanced_Rs = torch.as_tensor(torch.tensor([1.0, 2.0, 2.0, 4.0, 5.0], device=DEVICE)).requires_grad_(True).to(DEVICE)
R_balanced = effective_resistance(4, bridge_edges, balanced_Rs, 0, 3)
print(print(R_balanced))
I_balanced = branch_currents(4, bridge_edges, balanced_Rs, 0, 3)
print(print(I_balanced))
unbalanced_Rs = torch.as_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE)).requires_grad_(True).to(DEVICE)
R_unbalanced = effective_resistance(4, bridge_edges, unbalanced_Rs, 0, 3)
print(print(R_unbalanced))
cube_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]], device=DEVICE)
cube_Rs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=DEVICE)
R_edge = effective_resistance(8, cube_edges, cube_Rs, 0, 1)
R_face = effective_resistance(8, cube_edges, cube_Rs, 0, 2)
R_body = effective_resistance(8, cube_edges, cube_Rs, 0, 6)
print(print(R_edge))
print(print(R_face))
print(print(R_body))
R_cube = resistance_matrix(8, cube_edges, cube_Rs)
print(print(R_cube))
dR_unbalanced = compute_grad(lambda _dunbalanced_Rs: effective_resistance(4, bridge_edges, _dunbalanced_Rs, 0, 3), unbalanced_Rs)
print(print(dR_unbalanced))
I_unbalanced = branch_currents(4, bridge_edges, unbalanced_Rs, 0, 3)
print(print((I_unbalanced * I_unbalanced)))
dR_balanced = compute_grad(lambda _dbalanced_Rs: effective_resistance(4, bridge_edges, _dbalanced_Rs, 0, 3), balanced_Rs)
print(print(dR_balanced))