import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def num_rows(x, k=None):
    if k is None:
        k = int(x.shape[0])
    return k

def row_sum(x, k=None):
    if k is None:
        k = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(x[int(i)]) for i in range(int(k))]).float())

def zeros_like(x, k=None):
    if k is None:
        k = int(x.shape[0])
    return torch.stack([torch.as_tensor((0.0 * x[int(i)])) for i in range(int(k))]).float()

def append(x, value, m=None):
    if m is None:
        m = int(x.shape[0])
    return torch.cat([x, torch.stack([torch.as_tensor(value)])])

def sq_degree_sum_of(x, s, k=None):
    if k is None:
        k = int(x.shape[0])
    return torch.sum(torch.stack([torch.as_tensor(((s * row_sum(x[int(i)], k)) * (s * row_sum(x[int(i)], k)))) for i in range(int(k))]).float())

def add_vertex_adjacency(x, k=None):
    if k is None:
        k = int(x.shape[0])
    padded = torch.stack([torch.as_tensor(append(x[int(i)], 0.0, k)) for i in range(int(k))])
    zero_row = zeros_like(padded[int(0)], ((lambda ih: ([ih := (ih + 1) for k in range(int(0))], ih)[1])(k) + 1))
    return torch.cat([padded, torch.stack([torch.as_tensor(zero_row)])])

# === Classes ===
class UndirectedGraph(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        self.adjacency = torch.as_tensor(adjacency).float()

    def num_vertices(self):
        this = self
        return num_rows(self.adjacency)

    def has_edge(self, u, v):
        this = self
        m = self.adjacency
        return m[int(u), int(v)]

    def neighbors(self, u):
        this = self
        m = self.adjacency
        return m[int(u)]

    def degree(self, u):
        this = self
        m = self.adjacency
        return row_sum(m[int(u)])

    def sq_degree_sum(self, s):
        this = self
        s = torch.as_tensor(s, device=DEVICE).float()
        return sq_degree_sum_of(self.adjacency, s)

    def add_weighted_edge(self, u, v, w):
        this = self
        w = torch.as_tensor(w, device=DEVICE).float()
        m = self.adjacency
        new_adj = torch.stack([(m[int(a)] + 0.0) for _fi_a in range(int(num_rows(m))) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        new_adj[int(u), int(v)] = w
        new_adj[int(v), int(u)] = w
        with torch.no_grad():
            self.adjacency.copy_(new_adj)

    def add_edge(self, u, v):
        this = self
        self.add_weighted_edge(u, v, 1.0)

# === Program ===
empty3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=DEVICE)
g = UndirectedGraph(empty3)
print(g.num_vertices())
print(g.add_edge(0, 1))
print(g.add_edge(1, 2))
print(g.neighbors(1))
print(g.degree(1))
print(g.has_edge(0, 2))
g = UndirectedGraph(add_vertex_adjacency(g.adjacency, 3))
print(g.add_edge(2, 3))
print(g.degree(3))
wg = UndirectedGraph(empty3)
print(wg.add_weighted_edge(0, 1, 2.0))
print(wg.add_weighted_edge(1, 2, 3.0))
s0 = torch.tensor(1.0, requires_grad=True)
print(wg.sq_degree_sum(s0))
print(compute_grad(wg.sq_degree_sum(s0), s0))