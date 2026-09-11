import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def absolute(a):
    return torch.sqrt((a * a) if isinstance((a * a), torch.Tensor) else torch.tensor(float((a * a))))

def get_sum_of_1d_array(x):
    total = 0
    for i in range(len(x)):
        total = total + x[int(i)]
    return total

def sq_dist(a, b):
    acc = 0.0
    for c in range(int(0), int(DIM)):
        acc = acc + ((a[int(c)] - b[int(c)]) * (a[int(c)] - b[int(c)]))
    return acc

def argmin_vec(v):
    av = absolute(v)
    best_j = 0.0
    best_v = av[int(0)]
    for j in range(int(0), int(K)):
        if av[int(j)] < best_v:
            best_v = av[int(j)]
            best_j = j
    return best_j

def assign_one(point, C, DIM=None, K=None):
    if DIM is None:
        DIM = int(point.shape[0])
    if K is None:
        K = int(C.shape[0])
    dists = torch.stack([torch.as_tensor(sq_dist(point, C[int(j)])) for j in range(int(K))]).float()
    return argmin_vec(dists)

def assign_labels(X, C, NPTS=None, DIM=None, K=None):
    if NPTS is None:
        NPTS = int(X.shape[0])
    if DIM is None:
        DIM = int(X.shape[1])
    if K is None:
        K = int(C.shape[0])
    return torch.stack([torch.as_tensor(assign_one(X[int(i)], C, DIM, K)) for i in range(int(NPTS))]).float()

def new_centroid(X, labels, target, fallback):
    sums = torch.stack([(c * 0.0) for _fi_c in range(int(DIM)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
    cnt = 0.0
    for i in range(int(0), int(NPTS)):
        if labels[int(i)] == target:
            sums = (sums + X[int(i)])
            cnt = cnt + 1.0
    return torch.where(torch.as_tensor(cnt > 0.0), (sums / cnt), fallback)

def update_centroids(X, labels, C_old):
    return torch.stack([new_centroid(X, labels, j, C_old[int(j)]) for _fi_j in range(int(K)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]])

def data_min(X):
    m = X[int(0)]
    for i in range(int(0), int(NPTS)):
        m = (((m + X[int(i)]) - absolute((m - X[int(i)]))) * 0.5)
    return m

def data_max(X):
    m = X[int(0)]
    for i in range(int(0), int(NPTS)):
        m = (((m + X[int(i)]) + absolute((m - X[int(i)]))) * 0.5)
    return m

def rand_centroid(lo, hi):
    s = torch.distributions.Uniform(0.0, 1.0).rsample((int(DIM),))
    return (lo + (s * (hi - lo)))

# === Program ===
SEED = 2
K = 2
DIM = 2
NPTS = 15
ITERS = 50
torch.manual_seed(int(SEED))
X = torch.tensor([[2.0, 2.2], [2.8, 2.9], [1.9, 3.1], [3.1, 2.0], [2.5, 2.6], [3.6, 3.1], [4.1, 2.7], [3.3, 3.4], [4.0, 3.6], [3.7, 2.9], [3.0, 4.0], [2.7, 3.9], [3.4, 4.2], [2.9, 3.5], [3.2, 3.7]], device=DEVICE)
lo_box = data_min(X)
hi_box = data_max(X)
C = torch.stack([rand_centroid(lo_box, hi_box) for _fi_j in range(int(K)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]])
prev_labels = torch.stack([((i * 0.0) - 1.0) for _fi_i in range(int(NPTS)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
labels = torch.stack([(i * 0.0) for _fi_i in range(int(NPTS)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
converged_at = (0.0 - 1.0)
for step in range(int(0), int(ITERS)):
    labels = assign_labels(X, C)
    moved = get_sum_of_1d_array(absolute((labels - prev_labels)))
    if moved == 0.0:
        if converged_at < 0.0:
            converged_at = step
    else:
        C = update_centroids(X, labels, C)
    prev_labels = labels
print(print(converged_at))
print(print(labels))
print(print(C))