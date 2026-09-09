import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def sq_dist(a, b):
    e0 = (a[int(0)] - b[int(0)])
    e1 = (a[int(1)] - b[int(1)])
    return ((e0 * e0) + (e1 * e1))

def nearest_of_two(point, C):
    d0 = sq_dist(point, C[int(0)])
    d1 = sq_dist(point, C[int(1)])
    label = 0.0
    if d1 < d0:
        label = 1.0
    return label

def assign_labels(X, C):
    return torch.stack([nearest_of_two(X[int(i)], C) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def cluster_mean(X, labels, target, fallback):
    s0 = 0.0
    s1 = 0.0
    cnt = 0.0
    for i in range(int(0), int(n)):
        if labels[int(i)] == target:
            s0 = s0 + X[int(i), int(0)]
            s1 = s1 + X[int(i), int(1)]
            cnt = cnt + 1.0
    return torch.where(torch.as_tensor(cnt > 0.0), torch.stack([torch.as_tensor((s0 / cnt)), torch.as_tensor((s1 / cnt))]), fallback)

def update_centroids(X, labels, C_old):
    c0 = cluster_mean(X, labels, 0.0, C_old[int(0)])
    c1 = cluster_mean(X, labels, 1.0, C_old[int(1)])
    return torch.stack([torch.as_tensor(c0), torch.as_tensor(c1)])

# === Program ===
n = 8
iters = 10
X = torch.tensor([[1.0, 1.0], [1.5, 2.0], [1.0, 2.5], [2.0, 1.5], [8.0, 8.0], [8.5, 9.0], [9.0, 8.5], [7.5, 9.5]], device=DEVICE)
C = torch.tensor([[1.0, 1.0], [8.0, 8.0]], device=DEVICE)
labels = torch.stack([(i * 0.0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for step in range(int(0), int(iters)):
    labels = assign_labels(X, C)
    C = update_centroids(X, labels, C)
print(print(labels))
print(print(C))