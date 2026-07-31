import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def mask_embed(values, mask, n):
    empty = (torch.arange(n) * 0j)
    return torch.masked_scatter(empty, mask, values)

def op_O(atoms, W):
    return (atoms.volume() * W)

def op_O_mat(atoms, W):
    return (atoms.volume() * W)

def op_L(atoms, W):
    G2c = atoms.g2c()
    if len(W) == len(G2c):
        G2 = G2c
    else:
        G2 = atoms.g2()
    return (((-atoms.volume()) * G2) * W)

def op_L_mat(atoms, W):
    Npw = len(W)
    Nstate = len(W[int(0)])
    out = torch.reshape((torch.arange((Npw * Nstate)) * 0j), (int(Npw), int(Nstate),))
    for state in range(int(0), int(Nstate)):
        out[:, int(state)] = op_L(atoms, W[:, int(state)])
    return out

def op_Linv(atoms, W):
    G2 = atoms.g2()
    nonzero = (torch.gt(G2, 0.0) * 1.0)
    safe_G2 = (G2 + (1.0 - nonzero))
    return (((W / safe_G2) / (-atoms.volume())) * nonzero)

def op_Linv_mat(atoms, W):
    Npw = len(W)
    Nstate = len(W[int(0)])
    out = torch.reshape((torch.arange((Npw * Nstate)) * 0j), (int(Npw), int(Nstate),))
    for state in range(int(0), int(Nstate)):
        out[:, int(state)] = op_Linv(atoms, W[:, int(state)])
    return out

def op_J(atoms, W):
    s1 = atoms.s1
    s2 = atoms.s2
    s3 = atoms.s3
    Ngrid = ((s1 * s2) * s3)
    real_grid = torch.reshape(W, (int(s1), int(s2), int(s3),))
    reciprocal_grid = torch.fft.fftn(real_grid)
    return (torch.reshape(reciprocal_grid, (int(Ngrid),)) / Ngrid)

def op_J_mat(atoms, W):
    Ngrid = ((atoms.s1 * atoms.s2) * atoms.s3)
    Nstate = len(W[int(0)])
    out = torch.reshape((torch.arange((Ngrid * Nstate)) * 0j), (int(Ngrid), int(Nstate),))
    for state in range(int(0), int(Nstate)):
        out[:, int(state)] = op_J(atoms, W[:, int(state)])
    return out

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

def op_I_mat(atoms, W):
    Ngrid = ((atoms.s1 * atoms.s2) * atoms.s3)
    Nstate = len(W[int(0)])
    out = torch.reshape((torch.arange((Ngrid * Nstate)) * 0j), (int(Ngrid), int(Nstate),))
    for state in range(int(0), int(Nstate)):
        out[:, int(state)] = op_I(atoms, W[:, int(state)])
    return out

def op_Idag(atoms, W):
    Ngrid = ((atoms.s1 * atoms.s2) * atoms.s3)
    F = op_J(atoms, W)
    return (torch.masked_select(F, atoms.active()) * Ngrid)

def op_Idag_mat(atoms, W):
    Nactive = len(atoms.g2c())
    Nstate = len(W[int(0)])
    out = torch.reshape((torch.arange((Nactive * Nstate)) * 0j), (int(Nactive), int(Nstate),))
    for state in range(int(0), int(Nstate)):
        out[:, int(state)] = op_Idag(atoms, W[:, int(state)])
    return out

def op_Jdag(atoms, W):
    Ngrid = ((atoms.s1 * atoms.s2) * atoms.s3)
    return (op_I(atoms, W) / Ngrid)

def op_Jdag_mat(atoms, W):
    Ngrid = ((atoms.s1 * atoms.s2) * atoms.s3)
    return (op_I_mat(atoms, W) / Ngrid)

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
        return ((self.a * self.a) * self.a)

    def flat_index(self):
        this = self
        return torch.arange(((self.s1 * self.s2) * self.s3))

    def m1(self):
        this = self
        return torch.remainder(torch.floor((self.flat_index() / (self.s3 * self.s2)) if isinstance((self.flat_index() / (self.s3 * self.s2)), torch.Tensor) else torch.tensor(float((self.flat_index() / (self.s3 * self.s2))))), self.s1)

    def m2(self):
        this = self
        return torch.remainder(torch.floor((self.flat_index() / self.s3) if isinstance((self.flat_index() / self.s3), torch.Tensor) else torch.tensor(float((self.flat_index() / self.s3)))), self.s2)

    def m3(self):
        this = self
        return torch.remainder(self.flat_index(), self.s3)

    def fold_freq(self, m, axis_length):
        this = self
        m = torch.as_tensor(m, device=DEVICE).float()
        return (m - (torch.gt(m, (axis_length / 2)) * axis_length))

    def sample_coord(self, m, s):
        this = self
        m = torch.as_tensor(m, device=DEVICE).float()
        return (m * (self.a / s))

    def recip_scale(self):
        this = self
        return ((2 * π) / self.a)

    def structure_factor(self, n1, n2, n3, c, px, py, pz):
        this = self
        n1 = torch.as_tensor(n1, device=DEVICE).float()
        n2 = torch.as_tensor(n2, device=DEVICE).float()
        n3 = torch.as_tensor(n3, device=DEVICE).float()
        c = torch.as_tensor(c, device=DEVICE).float()
        px = torch.as_tensor(px, device=DEVICE).float()
        py = torch.as_tensor(py, device=DEVICE).float()
        pz = torch.as_tensor(pz, device=DEVICE).float()
        phase = (c * (((n1 * px) + (n2 * py)) + (n3 * pz)))
        return torch.exp(((-torch.tensor(1j)) * phase) if isinstance(((-torch.tensor(1j)) * phase), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * phase))))

    def freq_x(self):
        this = self
        return self.fold_freq(self.m1(), self.s1)

    def freq_y(self):
        this = self
        return self.fold_freq(self.m2(), self.s2)

    def freq_z(self):
        this = self
        return self.fold_freq(self.m3(), self.s3)

    def coord_x(self):
        this = self
        return self.sample_coord(self.m1(), self.s1)

    def coord_y(self):
        this = self
        return self.sample_coord(self.m2(), self.s2)

    def coord_z(self):
        this = self
        return self.sample_coord(self.m3(), self.s3)

    def gx(self):
        this = self
        return (self.recip_scale() * self.freq_x())

    def gy(self):
        this = self
        return (self.recip_scale() * self.freq_y())

    def gz(self):
        this = self
        return (self.recip_scale() * self.freq_z())

    def g2(self):
        this = self
        c = self.recip_scale()
        n1 = self.freq_x()
        n2 = self.freq_y()
        n3 = self.freq_z()
        return ((c * c) * (((n1 * n1) + (n2 * n2)) + (n3 * n3)))

    def active(self):
        this = self
        return torch.le(self.g2(), (2 * self.ecut))

    def g2c(self):
        this = self
        return torch.masked_select(self.g2(), self.active())

    def sf(self):
        this = self
        n1 = self.freq_x()
        n2 = self.freq_y()
        n3 = self.freq_z()
        c = self.recip_scale()
        natoms = self.Natoms
        Sf = (self.structure_factor(n1, n2, n3, c, self.px[int(0)], self.py[int(0)], self.pz[int(0)]) * 0.0)
        for a in range(int(0), int(natoms)):
            Sf = (Sf + self.structure_factor(n1, n2, n3, c, self.px[int(a)], self.py[int(a)], self.pz[int(a)]))
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
Sf = H_atom.sf()
physika_print(Sf[int(0)])
Sf_overlap = op_O(H_atom, Sf)
physika_print(Sf_overlap[int(0)])
Sf_laplacian = op_L(H_atom, Sf)
Sf_laplacian_inv = op_Linv(H_atom, Sf_laplacian)
physika_print(Sf_laplacian_inv[int(0)])
physika_print(Sf_laplacian_inv[int(1)])
Sf_forward = op_J(H_atom, Sf)
Sf_forward_inverse = op_I(H_atom, Sf_forward)
physika_print(Sf_forward_inverse[int(0)])
Sf_active = op_Idag(H_atom, Sf_forward_inverse)
Sf_adjoint = op_Jdag(H_atom, Sf_forward)
physika_print(Sf_adjoint[int(0)])