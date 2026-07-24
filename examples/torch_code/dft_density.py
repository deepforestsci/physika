import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def mask_embed(values, active, n):
    out = ((0.0 * torch.arange(n)) * (1.0 + 0j))
    return torch.masked_scatter(out, active, values)

def orth(atoms, W):
    norm = torch.sqrt((atoms.volume() * torch.sum((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0) if isinstance((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0))))) if isinstance((atoms.volume() * torch.sum((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0) if isinstance((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0))))), torch.Tensor) else torch.tensor(float((atoms.volume() * torch.sum((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0) if isinstance((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0))))))))
    return (W / norm)

def get_n_total(atoms, Y):
    n_full = ((atoms.s1 * atoms.s2) * atoms.s3)
    Y_full = mask_embed(Y, atoms.active(), n_full)
    Yrs = op_I(atoms, Y_full)
    f0 = atoms.f[int(0)]
    return (f0 * (torch.abs(Yrs if isinstance(Yrs, torch.Tensor) else torch.tensor(float(Yrs))) ** 2.0))

def get_phi(atoms, n):
    n_c = (n * (1.0 + 0j))
    n_recip = op_J(atoms, n_c)
    return (((-4.0) * π) * op_Linv(atoms, op_O(atoms, n_recip)))

def flat_index(k):
    return torch.arange(k)

def axis_index(ms, axis_stride, axis_length):
    return torch.remainder(torch.floor((ms / axis_stride) if isinstance((ms / axis_stride), torch.Tensor) else torch.tensor(float((ms / axis_stride)))), axis_length)

def fold_freq(m, axis_length):
    return (m - (torch.gt(m, (axis_length / 2)) * axis_length))

def cell_volume(a):
    return ((a * a) * a)

def sample_coord(m, a, s):
    return (m * (a / s))

def recip_scale(a):
    return ((2 * π) / a)

def g_squared(n1, n2, n3, c):
    return ((c * c) * (((n1 * n1) + (n2 * n2)) + (n3 * n3)))

def active_mask(G2, ecut):
    return torch.le(G2, (2 * ecut))

def active_subset(G2, active):
    return torch.masked_select(G2, active)

def structure_factor(n1, n2, n3, c, px, py, pz):
    phase = (c * (((n1 * px) + (n2 * py)) + (n3 * pz)))
    return torch.exp(((-torch.tensor(1j)) * phase) if isinstance(((-torch.tensor(1j)) * phase), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * phase))))

def op_I(atoms, W):
    s1 = atoms.s1
    s2 = atoms.s2
    s3 = atoms.s3
    Ngrid = ((s1 * s2) * s3)
    active = atoms.active()
    if len(W) == len(active):
        reciprocal_grid = W
    else:
        reciprocal_grid = mask_embed(W, active, Ngrid)
    reciprocal_grid3d = torch.reshape(reciprocal_grid, (int(s1), int(s2), int(s3),))
    real_grid = torch.fft.ifftn(reciprocal_grid3d)
    return (torch.reshape(real_grid, (int(Ngrid),)) * Ngrid)

def op_J(atoms, W):
    s1 = atoms.s1
    s2 = atoms.s2
    s3 = atoms.s3
    Ngrid = ((s1 * s2) * s3)
    real_grid = torch.reshape(W, (int(s1), int(s2), int(s3),))
    reciprocal_grid = torch.fft.fftn(real_grid)
    return (torch.reshape(reciprocal_grid, (int(Ngrid),)) / Ngrid)

def op_O(atoms, W):
    return (atoms.volume() * W)

def op_Linv(atoms, W):
    G2 = atoms.g2()
    nonzero = (torch.gt(G2, 0.0) * 1.0)
    safe_G2 = (G2 + (1.0 - nonzero))
    return (((W / safe_G2) / (-atoms.volume())) * nonzero)

# === Classes ===
class Atoms(nn.Module):
    def __init__(self, a, ecut, s1, s2, s3, Natoms, px, py, pz, Nstate, Z_nuc, f):
        super().__init__()
        self.a = torch.as_tensor(a).float()
        self.ecut = torch.as_tensor(ecut).float()
        self.s1 = torch.as_tensor(s1).float() if isinstance(s1, (int, float, torch.Tensor)) else s1
        self.s2 = torch.as_tensor(s2).float() if isinstance(s2, (int, float, torch.Tensor)) else s2
        self.s3 = torch.as_tensor(s3).float() if isinstance(s3, (int, float, torch.Tensor)) else s3
        self.Natoms = torch.as_tensor(Natoms).float() if isinstance(Natoms, (int, float, torch.Tensor)) else Natoms
        self.px = torch.as_tensor(px).float()
        self.py = torch.as_tensor(py).float()
        self.pz = torch.as_tensor(pz).float()
        self.Nstate = torch.as_tensor(Nstate).float() if isinstance(Nstate, (int, float, torch.Tensor)) else Nstate
        self.Z_nuc = torch.as_tensor(Z_nuc).float()
        self.f = torch.as_tensor(f).float()
        self.learnable_params = [self.a, self.ecut, self.px, self.py, self.pz, self.Z_nuc, self.f]

    def volume(self):
        this = self
        return cell_volume(self.a)

    def freq_x(self):
        this = self
        ms = flat_index(((self.s1 * self.s2) * self.s3))
        return fold_freq(axis_index(ms, (self.s3 * self.s2), self.s1), self.s1)

    def freq_y(self):
        this = self
        ms = flat_index(((self.s1 * self.s2) * self.s3))
        return fold_freq(axis_index(ms, self.s3, self.s2), self.s2)

    def freq_z(self):
        this = self
        ms = flat_index(((self.s1 * self.s2) * self.s3))
        return fold_freq(axis_index(ms, 1, self.s3), self.s3)

    def coord_x(self):
        this = self
        ms = flat_index(((self.s1 * self.s2) * self.s3))
        return sample_coord(axis_index(ms, (self.s3 * self.s2), self.s1), self.a, self.s1)

    def coord_y(self):
        this = self
        ms = flat_index(((self.s1 * self.s2) * self.s3))
        return sample_coord(axis_index(ms, self.s3, self.s2), self.a, self.s2)

    def coord_z(self):
        this = self
        ms = flat_index(((self.s1 * self.s2) * self.s3))
        return sample_coord(axis_index(ms, 1, self.s3), self.a, self.s3)

    def gx(self):
        this = self
        return (recip_scale(self.a) * self.freq_x())

    def gy(self):
        this = self
        return (recip_scale(self.a) * self.freq_y())

    def gz(self):
        this = self
        return (recip_scale(self.a) * self.freq_z())

    def g2(self):
        this = self
        return g_squared(self.freq_x(), self.freq_y(), self.freq_z(), recip_scale(self.a))

    def active(self):
        this = self
        return active_mask(self.g2(), self.ecut)

    def g2c(self):
        this = self
        return active_subset(self.g2(), self.active())

    def sf(self):
        this = self
        n1 = self.freq_x()
        n2 = self.freq_y()
        n3 = self.freq_z()
        c = recip_scale(self.a)
        natoms = self.Natoms
        Sf = (structure_factor(n1, n2, n3, c, self.px[int(0)], self.py[int(0)], self.pz[int(0)]) * 0.0)
        for a in range(int(0), int(natoms)):
            Sf = (Sf + structure_factor(n1, n2, n3, c, self.px[int(a)], self.py[int(a)], self.pz[int(a)]))
        return Sf

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
π = 3.141592653589793
a = 16.0
ecut = 16.0
s = 60
Natoms = 1
Nstate = 1
px = torch.tensor([0.0], device=DEVICE)
py = torch.tensor([0.0], device=DEVICE)
pz = torch.tensor([0.0], device=DEVICE)
Z_nuc = torch.tensor([1.0], device=DEVICE)
f = torch.tensor([1.0], device=DEVICE)
H_atom = Atoms(a, ecut, s, s, s, Natoms, px, py, pz, Nstate, Z_nuc, f).to(DEVICE)
W0 = (torch.exp(((-0.5) * H_atom.g2c()) if isinstance(((-0.5) * H_atom.g2c()), torch.Tensor) else torch.tensor(float(((-0.5) * H_atom.g2c())))) * (1.0 + 0j))
Y0 = orth(H_atom, W0)
norm_check = (H_atom.volume() * torch.sum((torch.abs(Y0 if isinstance(Y0, torch.Tensor) else torch.tensor(float(Y0))) ** 2.0) if isinstance((torch.abs(Y0 if isinstance(Y0, torch.Tensor) else torch.tensor(float(Y0))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(Y0 if isinstance(Y0, torch.Tensor) else torch.tensor(float(Y0))) ** 2.0)))))
physika_print(norm_check)
n0 = get_n_total(H_atom, Y0)
dV = (H_atom.volume() / ((s * s) * s))
density_check = (torch.sum(n0 if isinstance(n0, torch.Tensor) else torch.tensor(float(n0))) * dV)
physika_print(density_check)
phi0 = get_phi(H_atom, n0)
physika_print(phi0[int(0)])
physika_print(phi0[int(1)])