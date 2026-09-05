import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def cycle_graph(n_nodes, w):
    values = torch.stack([(a * 1.0) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    adjacency = torch.stack([torch.stack([(((a + c) * 0.0) - 1.0) for _fi_c in range(int(2)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]]) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    weights = torch.stack([torch.stack([((a + c) * 0.0) for _fi_c in range(int(2)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]]) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    for a in range(int(0), int(n_nodes)):
        for s in range(int(0), int(2)):
            nb = (a - 1.0)
            if s > 0.5:
                nb = (a + 1.0)
            if nb < 0.0:
                nb = (n_nodes - 1.0)
            if nb > (n_nodes - 0.5):
                nb = 0.0
            adjacency[int(a), int(s)] = nb
            weights[int(a), int(s)] = w
    degrees = torch.stack([(2.0 + (a * 0.0)) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    return Graph(values, degrees, adjacency, weights)

def grid_graph(rows, cols):
    n_nodes = (rows * cols)
    values = torch.stack([(a * 1.0) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    adjacency = torch.stack([torch.stack([(((a + c) * 0.0) - 1.0) for _fi_c in range(int(4)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]]) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    weights = torch.stack([torch.stack([((a + c) * 0.0) for _fi_c in range(int(4)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]]) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    for r in range(int(0), int(rows)):
        for c in range(int(0), int(cols)):
            idx = ((r * cols) + c)
            if r > 0:
                adjacency[int(idx), int(0)] = (idx - cols)
                weights[int(idx), int(0)] = 1.0
            if r < (rows - 1):
                adjacency[int(idx), int(1)] = (idx + cols)
                weights[int(idx), int(1)] = 1.0
            if c > 0:
                adjacency[int(idx), int(2)] = (idx - 1.0)
                weights[int(idx), int(2)] = 1.0
            if c < (cols - 1):
                adjacency[int(idx), int(3)] = (idx + 1.0)
                weights[int(idx), int(3)] = 1.0
    degrees = torch.stack([(a * 0.0) for _fi_a in range(int(n_nodes)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    for a in range(int(0), int(n_nodes)):
        for c in range(int(0), int(4)):
            if adjacency[int(a), int(c)] >= 0.0:
                degrees[int(a)] += 1.0
    return Graph(values, degrees, adjacency, weights)

# === Classes ===
class Graph(nn.Module):
    def __init__(self, node_values, degrees, adjacency, edge_weights):
        super().__init__()
        self.node_values = torch.as_tensor(node_values).float()
        self.degrees = torch.as_tensor(degrees).float()
        self.adjacency = torch.as_tensor(adjacency).float()
        self.edge_weights = torch.as_tensor(edge_weights).float()
        self.learnable_params = [self.node_values, self.degrees, self.adjacency, self.edge_weights]

    def order(self):
        this = self
        return (len(self.node_values) * 1.0)

    def value_at(self, id):
        this = self
        id = torch.as_tensor(id, device=DEVICE).float()
        vals = self.node_values
        return vals[int(id)]

    def degree_of(self, id):
        this = self
        id = torch.as_tensor(id, device=DEVICE).float()
        degs = self.degrees
        return degs[int(id)]

    def size(self):
        this = self
        degs = self.degrees
        return (torch.sum(degs if isinstance(degs, torch.Tensor) else torch.tensor(float(degs))) / 2.0)

    def has_edge(self, a, b):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        b = torch.as_tensor(b, device=DEVICE).float()
        adj = self.adjacency
        row = adj[int(a)]
        w = len(row)
        result = 0.0
        for c in range(int(0), int(w)):
            if row[int(c)] == b:
                result = 1.0
        return result

    def weight_between(self, a, b):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        b = torch.as_tensor(b, device=DEVICE).float()
        adj = self.adjacency
        wts = self.edge_weights
        row = adj[int(a)]
        wrow = wts[int(a)]
        w = len(row)
        result = 0.0
        for c in range(int(0), int(w)):
            if row[int(c)] == b:
                result = wrow[int(c)]
        return result

    def node(self, id):
        this = self
        id = torch.as_tensor(id, device=DEVICE).float()
        degs = self.degrees
        adj = self.adjacency
        wts = self.edge_weights
        arow = adj[int(id)]
        wrow = wts[int(id)]
        return Node(self, id, degs[int(id)], arow, wrow)

    def adjacency_matrix(self):
        this = self
        adj = self.adjacency
        n = len(adj)
        row0 = adj[int(0)]
        w = len(row0)
        result = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        for a in range(int(0), int(n)):
            for c in range(int(0), int(w)):
                nb = adj[int(a), int(c)]
                if nb >= 0.0:
                    result[int(a), int(nb)] = 1.0
        return result

    def weight_matrix(self):
        this = self
        adj = self.adjacency
        wts = self.edge_weights
        n = len(adj)
        row0 = adj[int(0)]
        w = len(row0)
        result = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        for a in range(int(0), int(n)):
            for c in range(int(0), int(w)):
                nb = adj[int(a), int(c)]
                if nb >= 0.0:
                    result[int(a), int(nb)] = wts[int(a), int(c)]
        return result

    def degree_vector(self):
        this = self
        degs = self.degrees
        n = len(degs)
        out = torch.stack([(degs[int(a)] + (a * 0.0)) for _fi_a in range(int(n)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        return out

    def laplacian(self):
        this = self
        adj = self.adjacency
        degs = self.degrees
        n = len(adj)
        row0 = adj[int(0)]
        w = len(row0)
        result = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        for a in range(int(0), int(n)):
            result[int(a), int(a)] = degs[int(a)]
        for a in range(int(0), int(n)):
            for c in range(int(0), int(w)):
                nb = adj[int(a), int(c)]
                if nb >= 0.0:
                    result[int(a), int(nb)] = (result[int(a), int(nb)] - 1.0)
        return result

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Node(nn.Module):
    def __init__(self, graph, id, degree, neighbors, weights):
        super().__init__()
        self.add_module('graph', graph)
        self.id = torch.as_tensor(id).float()
        self.degree = torch.as_tensor(degree).float()
        self.neighbors = torch.as_tensor(neighbors).float()
        self.weights = torch.as_tensor(weights).float()
        self.learnable_params = [self.id, self.degree, self.neighbors, self.weights]

    def value(self):
        this = self
        return self.graph.value_at(self.id)

    def deg(self):
        this = self
        return self.degree

    def neighbor(self, slot):
        this = self
        slot = torch.as_tensor(slot, device=DEVICE).float()
        nbrs = self.neighbors
        return nbrs[int(slot)]

    def edge_weight(self, slot):
        this = self
        slot = torch.as_tensor(slot, device=DEVICE).float()
        w = self.weights
        return w[int(slot)]

    def weighted_degree(self):
        this = self
        nbrs = self.neighbors
        w = self.weights
        k = len(nbrs)
        total = 0.0
        for c in range(int(0), int(k)):
            if nbrs[int(c)] >= 0.0:
                total = total + w[int(c)]
        return total

    def is_isolated(self):
        this = self
        result = 0.0
        if self.degree == 0.0:
            result = 1.0
        return result

    def is_adjacent_to(self, b):
        this = self
        b = torch.as_tensor(b, device=DEVICE).float()
        nbrs = self.neighbors
        k = len(nbrs)
        result = 0.0
        for c in range(int(0), int(k)):
            if nbrs[int(c)] == b:
                result = 1.0
        return result

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
c4 = cycle_graph(4.0, 1.0)
print(c4.order())
print(c4.size())
print(c4.degree_of(0.0))
print(c4.has_edge(0.0, 1.0))
print(c4.has_edge(0.0, 2.0))
print(c4.adjacency_matrix())
print(c4.laplacian())
v0 = c4.node(0.0)
print(v0.value())
print(v0.deg())
print(v0.neighbor(0.0))
print(v0.neighbor(1.0))
print(v0.weighted_degree())
print(v0.is_isolated())
print(v0.is_adjacent_to(3.0))
print(v0.is_adjacent_to(2.0))
g = grid_graph(2.0, 3.0)
print(g.order())
print(g.size())
print(g.degree_of(1.0))
print(g.degree_of(0.0))
print(g.weight_between(0.0, 1.0))
print(g.weight_between(0.0, 5.0))
print(g.degree_vector())
print(g.weight_matrix())
print(g.laplacian())