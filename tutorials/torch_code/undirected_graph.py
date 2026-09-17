import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
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

def empty_graph(n_vertices):
    z = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n_vertices)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n_vertices)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    g = UndirectedGraph()
    g.adjacency = z
    return g

# === Classes ===
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

# === Program ===
n0 = 3
g = empty_graph(n0)
print(g.num_vertices())
print(g.add_edge(0.0, 1.0))
print(g.add_edge(1.0, 2.0))
print(g.neighbors(1.0))
print(g.degree(1.0))
print(g.has_edge(0.0, 2.0))
n3 = 4
print(g.add_vertex(n3))
print(g.add_edge(2.0, 3.0))
print(g.degree(3.0))
wg = empty_graph(3)
print(wg.add_weighted_edge(0.0, 1.0, 2.0))
print(wg.add_weighted_edge(1.0, 2.0, 3.0))
s0 = torch.tensor(1.0, requires_grad=True)
print(wg.sq_degree_sum(s0))
print(compute_grad(wg.sq_degree_sum(s0), s0))