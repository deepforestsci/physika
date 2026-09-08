import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def empty_graph(n_vertices):
    z = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n_vertices)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n_vertices)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    g = UndirectedGraph()
    g.adjacency = z
    return g

# === Classes ===
class UndirectedGraph(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.learnable_params = []
        self.adjacency = None

    def num_vertices(self):
        this = self
        return (len(self.adjacency) * 1.0)

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
        return torch.sum(r if isinstance(r, torch.Tensor) else torch.tensor(float(r)))

    def neighbors(self, u):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        m = self.adjacency
        return m[int(u)]

    def add_edge(self, u, v):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        v = torch.as_tensor(v, device=DEVICE).float()
        m = self.adjacency
        k = len(m)
        new_adj = torch.stack([torch.stack([m[int(a), int(b)] for _fi_b in range(int(k)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        new_adj[int(u), int(v)] = 1.0
        new_adj[int(v), int(u)] = 1.0
        self.adjacency = new_adj

    def grow_adjacency(self, new_n):
        this = self
        old = self.adjacency
        result = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(new_n)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(new_n)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        m = len(old)
        for a in range(int(0), int(m)):
            for b in range(int(0), int(m)):
                result[int(a), int(b)] = old[int(a), int(b)]
        return result

    def add_vertex(self, new_n):
        this = self
        self.adjacency = self.grow_adjacency(new_n)

    def dfs(self, start):
        this = self
        start = torch.as_tensor(start, device=DEVICE).float()
        m = self.adjacency
        k = len(m)
        cap = ((k * k) + 1)
        visited = torch.stack([(a * 0.0) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        order = torch.stack([((a * 0.0) - 1.0) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        stack = torch.stack([(a * 0.0) for _fi_a in range(int(cap)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        sp = 0
        count = 0
        stack[int(sp)] = start
        sp = (sp + 1)
        for step in range(int(0), int(cap)):
            if sp > 0:
                sp = (sp - 1)
                v = (stack[int(sp)] * 1.0)
                if visited[int(v)] == 0.0:
                    visited[int(v)] = 1.0
                    order[int(count)] = (v * 1.0)
                    count = (count + 1)
                    for j in range(int(0), int(k)):
                        w = ((k - 1) - j)
                        if m[int(v), int(w)] == 1.0:
                            if visited[int(w)] == 0.0:
                                stack[int(sp)] = (w * 1.0)
                                sp = (sp + 1)
        return order

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

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
print(g.dfs(0.0))
print(g.dfs(2.0))
print(g.add_edge(0.0, 3.0))
print(g.dfs(0.0))
print(g.dfs(1.0))