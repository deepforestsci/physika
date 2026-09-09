import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def zero_1d(n):
    results = torch.stack([(i * 0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_2d(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_3d(a, b, c):
    results = torch.stack([torch.stack([torch.stack([(k * 0) for _fi_k in range(int(c)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(b)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(a)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def difference_matrix(r):
    results = zero_3d(num_points, num_points, 3)
    for i in range(int(0), int(num_points)):
        for j in range(int(0), int(num_points)):
            for k in range(int(0), int(3)):
                results[int(i), int(j), int(k)] = (r[int(i), int(k)] - r[int(j), int(k)])
    return results

def distance_matrix(rij):
    results = zero_2d(num_points, num_points)
    for i in range(int(0), int(num_points)):
        for j in range(int(0), int(num_points)):
            acc = 0
            for k in range(int(0), int(3)):
                acc = acc + (rij[int(i), int(j), int(k)] * rij[int(i), int(j), int(k)])
            results[int(i), int(j)] = torch.sqrt((acc + 1e-12) if isinstance((acc + 1e-12), torch.Tensor) else torch.tensor(float((acc + 1e-12))))
    return results

def ssp(x):
    return torch.log(((0.5 * torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) + 0.5) if isinstance(((0.5 * torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) + 0.5), torch.Tensor) else torch.tensor(float(((0.5 * torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) + 0.5))))

def Y0(rij):
    results = zero_3d(num_points, num_points, 1)
    for i in range(int(0), int(num_points)):
        for j in range(int(0), int(num_points)):
            results[int(i), int(j), int(0)] = 1.0
    return results

def Y1(rij):
    results = zero_3d(num_points, num_points, 3)
    for i in range(int(0), int(num_points)):
        for j in range(int(0), int(num_points)):
            x = rij[int(i), int(j), int(0)]
            y = rij[int(i), int(j), int(1)]
            z = rij[int(i), int(j), int(2)]
            r_norm = torch.sqrt(((((x * x) + (y * y)) + (z * z)) + 1e-12) if isinstance(((((x * x) + (y * y)) + (z * z)) + 1e-12), torch.Tensor) else torch.tensor(float(((((x * x) + (y * y)) + (z * z)) + 1e-12))))
            results[int(i), int(j), int(0)] = (x / r_norm)
            results[int(i), int(j), int(1)] = (y / r_norm)
            results[int(i), int(j), int(2)] = (z / r_norm)
    return results

def Y2(rij):
    results = zero_3d(num_points, num_points, 5)
    sqrt3 = torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0)))
    for i in range(int(0), int(num_points)):
        for j in range(int(0), int(num_points)):
            x = rij[int(i), int(j), int(0)]
            y = rij[int(i), int(j), int(1)]
            z = rij[int(i), int(j), int(2)]
            r2 = ((((x * x) + (y * y)) + (z * z)) + 1e-12)
            results[int(i), int(j), int(0)] = ((x * y) / r2)
            results[int(i), int(j), int(1)] = ((y * z) / r2)
            results[int(i), int(j), int(2)] = (((((2.0 * z) * z) - (x * x)) - (y * y)) / ((2.0 * sqrt3) * r2))
            results[int(i), int(j), int(3)] = ((z * x) / r2)
            results[int(i), int(j), int(4)] = (((x * x) - (y * y)) / (2.0 * r2))
    return results

def gaussian_rbf(dij, centers, gamma):
    results = zero_3d(num_points, num_points, rbf_count)
    for i in range(int(0), int(num_points)):
        for j in range(int(0), int(num_points)):
            for c in range(int(0), int(rbf_count)):
                diff = (dij[int(i), int(j)] - centers[int(c)])
                results[int(i), int(j), int(c)] = torch.exp((((-gamma) * diff) * diff) if isinstance((((-gamma) * diff) * diff), torch.Tensor) else torch.tensor(float((((-gamma) * diff) * diff))))
    return results

def to_column(x):
    n_x = len(x)
    results = zero_2d(n_x, 1)
    for i in range(int(0), int(n_x)):
        results[int(i), int(0)] = x[int(i)]
    return results

def ssp_col(x):
    return torch.log(((0.5 * torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) + 0.5) if isinstance(((0.5 * torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) + 0.5), torch.Tensor) else torch.tensor(float(((0.5 * torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) + 0.5))))

def radial_net(rbf_features, w1, b1, w2, b2):
    x_col = to_column(rbf_features)
    h_pre = ((w1 @ x_col) + b1)
    h = ssp_col(h_pre)
    out = ((w2 @ h) + b2)
    return out[int(0), int(0)]

def radial_field(rbf, w1, b1, w2, b2):
    results = zero_2d(num_points, num_points)
    for i in range(int(0), int(num_points)):
        for j in range(int(0), int(num_points)):
            efeat = rbf[int(i), int(j)]
            results[int(i), int(j)] = radial_net(efeat, w1, b1, w2, b2)
    return results

def tensor_product_reduce(cg, a, b, dim1, dim2, dim3):
    results = zero_1d(dim3)
    for k in range(int(0), int(dim3)):
        acc = 0
        for i in range(int(0), int(dim1)):
            for j in range(int(0), int(dim2)):
                acc = acc + ((cg[int(i), int(j), int(k)] * a[int(i)]) * b[int(j)])
        results[int(k)] = acc
    return results

def filter_00(masses, phi0):
    results = zero_1d(num_points)
    for a in range(int(0), int(num_points)):
        acc = torch.tensor([0.0], device=DEVICE)
        for b in range(int(0), int(num_points)):
            edge0 = torch.stack([torch.as_tensor(phi0[int(a), int(b)])])
            mass_b = torch.stack([torch.as_tensor(masses[int(b)])])
            contrib = tensor_product_reduce(cg_000, edge0, mass_b, 1, 1, 1)
            acc = torch.stack([torch.as_tensor((acc[int(0)] + contrib[int(0)]))])
        results[int(a)] = acc[int(0)]
    return results

def filter_22(masses, phi2, y2):
    results = zero_2d(num_points, 5)
    for a in range(int(0), int(num_points)):
        acc = zero_1d(5)
        for b in range(int(0), int(num_points)):
            edge2 = torch.stack([torch.as_tensor((phi2[int(a), int(b)] * y2[int(a), int(b), int(0)])), torch.as_tensor((phi2[int(a), int(b)] * y2[int(a), int(b), int(1)])), torch.as_tensor((phi2[int(a), int(b)] * y2[int(a), int(b), int(2)])), torch.as_tensor((phi2[int(a), int(b)] * y2[int(a), int(b), int(3)])), torch.as_tensor((phi2[int(a), int(b)] * y2[int(a), int(b), int(4)]))])
            mass_b = torch.stack([torch.as_tensor(masses[int(b)])])
            contrib = tensor_product_reduce(cg_202, edge2, mass_b, 5, 1, 5)
            for m in range(int(0), int(5)):
                acc[int(m)] = (acc[int(m)] + contrib[int(m)])
        for m in range(int(0), int(5)):
            results[int(a), int(m)] = acc[int(m)]
    return results

def matrix_from_0_2(out0, out2):
    results = zero_3d(num_points, 3, 3)
    sqrt3 = torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0)))
    for a in range(int(0), int(num_points)):
        d_xy = out2[int(a), int(0)]
        d_yz = out2[int(a), int(1)]
        d_z2 = out2[int(a), int(2)]
        d_zx = out2[int(a), int(3)]
        d_x2y2 = out2[int(a), int(4)]
        d_z2_scaled = (d_z2 / sqrt3)
        Mxx = (((0.0 - d_z2_scaled) + d_x2y2) + out0[int(a)])
        Myy = (((0.0 - d_z2_scaled) - d_x2y2) + out0[int(a)])
        Mzz = ((2.0 * d_z2_scaled) + out0[int(a)])
        results[int(a), int(0), int(0)] = Mxx
        results[int(a), int(0), int(1)] = d_xy
        results[int(a), int(0), int(2)] = d_zx
        results[int(a), int(1), int(0)] = d_xy
        results[int(a), int(1), int(1)] = Myy
        results[int(a), int(1), int(2)] = d_yz
        results[int(a), int(2), int(0)] = d_zx
        results[int(a), int(2), int(1)] = d_yz
        results[int(a), int(2), int(2)] = Mzz
    return results

def mse(pred, target):
    diff = (pred - target)
    result = (torch.sum((diff * diff) if isinstance((diff * diff), torch.Tensor) else torch.tensor(float((diff * diff)))) / 9.0)
    return result

def moi_tensor(points, masses, center_idx):
    cx = points[int(center_idx), int(0)]
    cy = points[int(center_idx), int(1)]
    cz = points[int(center_idx), int(2)]
    x = (points[:, int(0)] - cx)
    y = (points[:, int(1)] - cy)
    z = (points[:, int(2)] - cz)
    m = masses
    Ixx = torch.sum((((y * y) + (z * z)) * m) if isinstance((((y * y) + (z * z)) * m), torch.Tensor) else torch.tensor(float((((y * y) + (z * z)) * m))))
    Iyy = torch.sum((((x * x) + (z * z)) * m) if isinstance((((x * x) + (z * z)) * m), torch.Tensor) else torch.tensor(float((((x * x) + (z * z)) * m))))
    Izz = torch.sum((((x * x) + (y * y)) * m) if isinstance((((x * x) + (y * y)) * m), torch.Tensor) else torch.tensor(float((((x * x) + (y * y)) * m))))
    Ixy = torch.sum(((0.0 - (x * y)) * m) if isinstance(((0.0 - (x * y)) * m), torch.Tensor) else torch.tensor(float(((0.0 - (x * y)) * m))))
    Iyz = torch.sum(((0.0 - (y * z)) * m) if isinstance(((0.0 - (y * z)) * m), torch.Tensor) else torch.tensor(float(((0.0 - (y * z)) * m))))
    Ixz = torch.sum(((0.0 - (x * z)) * m) if isinstance(((0.0 - (x * z)) * m), torch.Tensor) else torch.tensor(float(((0.0 - (x * z)) * m))))
    moi = zero_2d(3, 3)
    moi[int(0), int(0)] = Ixx
    moi[int(1), int(1)] = Iyy
    moi[int(2), int(2)] = Izz
    moi[int(0), int(1)] = Ixy
    moi[int(1), int(0)] = Ixy
    moi[int(1), int(2)] = Iyz
    moi[int(2), int(1)] = Iyz
    moi[int(0), int(2)] = Ixz
    moi[int(2), int(0)] = Ixz
    return moi

def random_points(n, max_coord):
    pts = torch.stack([torch.distributions.Uniform((-max_coord), max_coord).rsample((int(3),)) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return pts

def random_masses(n, min_mass, max_mass):
    masses = torch.distributions.Uniform(min_mass, max_mass).rsample((int(n),))
    return masses

def transpose3x3(M):
    T = zero_2d(3, 3)
    T[int(0), int(0)] = M[int(0), int(0)]
    T[int(0), int(1)] = M[int(1), int(0)]
    T[int(0), int(2)] = M[int(2), int(0)]
    T[int(1), int(0)] = M[int(0), int(1)]
    T[int(1), int(1)] = M[int(1), int(1)]
    T[int(1), int(2)] = M[int(2), int(1)]
    T[int(2), int(0)] = M[int(0), int(2)]
    T[int(2), int(1)] = M[int(1), int(2)]
    T[int(2), int(2)] = M[int(2), int(2)]
    return T

def rotation_matrix_z(theta):
    c = torch.cos(theta if isinstance(theta, torch.Tensor) else torch.tensor(float(theta)))
    s = torch.sin(theta if isinstance(theta, torch.Tensor) else torch.tensor(float(theta)))
    Rmat = zero_2d(3, 3)
    Rmat[int(0), int(0)] = c
    Rmat[int(0), int(1)] = (0.0 - s)
    Rmat[int(0), int(2)] = 0.0
    Rmat[int(1), int(0)] = s
    Rmat[int(1), int(1)] = c
    Rmat[int(1), int(2)] = 0.0
    Rmat[int(2), int(0)] = 0.0
    Rmat[int(2), int(1)] = 0.0
    Rmat[int(2), int(2)] = 1.0
    return Rmat

def rotate_points(points, Rmat):
    RmatT = transpose3x3(Rmat)
    results = (points @ RmatT)
    return results

def rotate_matrix(Mmat, Rmat):
    RmatT = transpose3x3(Rmat)
    RM = (Rmat @ Mmat)
    result = (RM @ RmatT)
    return result

def max_abs_diff(Amat, Bmat):
    diff = (Amat - Bmat)
    sq = (diff * diff)
    worst = 0.0
    for i in range(int(0), int(3)):
        for j in range(int(0), int(3)):
            if sq[int(i), int(j)] > worst:
                worst = sq[int(i), int(j)]
    result = torch.sqrt(worst if isinstance(worst, torch.Tensor) else torch.tensor(float(worst)))
    return result

# === Classes ===
class MOIModel(nn.Module):
    def __init__(self, w1_0, b1_0, w2_0, b2_0, w1_2, b1_2, w2_2, b2_2):
        super().__init__()
        self.w1_0 = nn.Parameter(torch.as_tensor(w1_0))
        self.b1_0 = nn.Parameter(torch.as_tensor(b1_0))
        self.w2_0 = nn.Parameter(torch.as_tensor(w2_0))
        self.b2_0 = nn.Parameter(torch.as_tensor(b2_0))
        self.w1_2 = nn.Parameter(torch.as_tensor(w1_2))
        self.b1_2 = nn.Parameter(torch.as_tensor(b1_2))
        self.w2_2 = nn.Parameter(torch.as_tensor(w2_2))
        self.b2_2 = nn.Parameter(torch.as_tensor(b2_2))
        self.learnable_params = [self.w1_0, self.b1_0, self.w2_0, self.b2_0, self.w1_2, self.b1_2, self.w2_2, self.b2_2]

    def forward(self, points, masses, centers, gamma):
        this = self
        points = torch.as_tensor(points, device=DEVICE).float()
        masses = torch.as_tensor(masses, device=DEVICE).float()
        centers = torch.as_tensor(centers, device=DEVICE).float()
        gamma = torch.as_tensor(gamma, device=DEVICE).float()
        rij = difference_matrix(points)
        dij = distance_matrix(rij)
        y2 = Y2(rij)
        rbf = gaussian_rbf(dij, centers, gamma)
        phi0 = radial_field(rbf, self.w1_0, self.b1_0, self.w2_0, self.b2_0)
        phi2 = radial_field(rbf, self.w1_2, self.b1_2, self.w2_2, self.b2_2)
        out0 = filter_00(masses, phi0)
        out2 = filter_22(masses, phi2, y2)
        moi = matrix_from_0_2(out0, out2)
        return moi

    def loss_sample(self):
        this = self
        points = random_points(num_points, max_coord)
        masses = random_masses(num_points, min_mass, max_mass)
        masses[int(center_idx)] = 0.0
        target = moi_tensor(points, masses, center_idx)
        pred_full = self(points, masses, centers, gamma)
        pred = pred_full[int(center_idx)]
        result = mse(pred, target)
        return result

    def train(self, steps, lr):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        last_loss = 0
        for step in range(int(0), int(steps)):
            for rep in range(int(0), int(1)):
                current_loss = self.loss_sample()
                learnable_grads = compute_grad(current_loss, self.learnable_params)
                self.update(lr, learnable_grads)
                last_loss = current_loss
        return last_loss

    def evaluate(self):
        this = self
        total_loss = 0
        for s in range(int(0), int(eval_samples)):
            for rep in range(int(0), int(1)):
                current_loss = self.loss_sample()
                total_loss = (total_loss + current_loss)
        result = (total_loss / eval_samples)
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
num_points = 15
rbf_low = 0.0
rbf_high = 2.0
rbf_count = 30
rbf_spacing = ((rbf_high - rbf_low) / rbf_count)
centers = torch.stack([(rbf_low + (i * rbf_spacing)) for _fi_i in range(int(rbf_count)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
gamma = (1.0 / rbf_spacing)
hidden = 16
cg_000 = torch.tensor([[[1.0]]], device=DEVICE)
cg_202 = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 1.0]]], device=DEVICE)
center_idx = 0.0
max_coord = 0.5
min_mass = 0.5
max_mass = 2.0
w_init_low = (-0.2)
w_init_high = 0.2
w1_0 = torch.distributions.Uniform(w_init_low, w_init_high).rsample((int(16), int(30),))
b1_0 = zero_2d(16, 1)
w2_0 = torch.distributions.Uniform(w_init_low, w_init_high).rsample((int(1), int(16),))
b2_0 = zero_2d(1, 1)
w1_2 = torch.distributions.Uniform(w_init_low, w_init_high).rsample((int(16), int(30),))
b1_2 = zero_2d(16, 1)
w2_2 = torch.distributions.Uniform(w_init_low, w_init_high).rsample((int(1), int(16),))
b2_2 = zero_2d(1, 1)
moi_object = MOIModel(w1_0, b1_0, w2_0, b2_0, w1_2, b1_2, w2_2, b2_2).to(DEVICE)
eval_samples = 30
lr = 0.002
epochs = 1
loss_before = moi_object.evaluate()
print(print(loss_before))
final_loss = moi_object.train(epochs, lr)
print(print(final_loss))
avg_mse_loss = moi_object.evaluate()
print(print(avg_mse_loss))
test_points = random_points(num_points, max_coord)
test_masses = random_masses(num_points, min_mass, max_mass)
test_masses[int(center_idx)] = 0.0
rotation_angle = 0.9
Rmat = rotation_matrix_z(rotation_angle)
rotated_points = rotate_points(test_points, Rmat)
pred_orig_full = moi_object(test_points, test_masses, centers, gamma)
pred_orig = pred_orig_full[int(center_idx)]
pred_rot_full = moi_object(rotated_points, test_masses, centers, gamma)
pred_rot = pred_rot_full[int(center_idx)]
expected_rot = rotate_matrix(pred_orig, Rmat)
diff = max_abs_diff(pred_rot, expected_rot)
print(print(diff))
