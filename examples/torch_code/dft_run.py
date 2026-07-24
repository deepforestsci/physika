import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
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

def op_O(atoms, W):
    return (atoms.volume() * W)

def op_L(atoms, W):
    G2c = atoms.g2c()
    if len(W) == len(G2c):
        G2 = G2c
    else:
        G2 = atoms.g2()
    return (((-atoms.volume()) * G2) * W)

def op_Linv(atoms, W):
    G2 = atoms.g2()
    nonzero = (torch.gt(G2, 0.0) * 1.0)
    safe_G2 = (G2 + (1.0 - nonzero))
    return (((W / safe_G2) / (-atoms.volume())) * nonzero)

def op_J(atoms, W):
    s1 = atoms.s1
    s2 = atoms.s2
    s3 = atoms.s3
    Ngrid = ((s1 * s2) * s3)
    real_grid = torch.reshape(W, (int(s1), int(s2), int(s3),))
    reciprocal_grid = torch.fft.fftn(real_grid)
    return (torch.reshape(reciprocal_grid, (int(Ngrid),)) / Ngrid)

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

def op_Jdag(atoms, W):
    Ngrid = ((atoms.s1 * atoms.s2) * atoms.s3)
    return (op_I(atoms, W) / Ngrid)

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

def mask_embed(values, active, n):
    out = ((0.0 * torch.arange(n)) * (1.0 + 0j))
    return torch.masked_scatter(out, active, values)

def coulomb(atoms):
    G2 = atoms.g2()
    nonzero = (torch.gt(G2, 0.0) * 1.0)
    safe_G2 = (G2 + (1.0 - nonzero))
    Vcoul = (((((-4.0) * π) * atoms.Z_nuc[int(0)]) / safe_G2) * nonzero)
    return op_J(atoms, (Vcoul * atoms.sf()))

def lda_x(n):
    cx = (((-3.0) / 4.0) * ((3.0 / (2.0 * π)) ** (2.0 / 3.0)))
    rs = ((3.0 / ((4.0 * π) * n)) ** (1.0 / 3.0))
    ex = (cx / rs)
    vx = ((4.0 / 3.0) * ex)
    return torch.stack([torch.as_tensor(ex), torch.as_tensor(vx)])

def lda_c_chachiyo(n):
    a = (-0.01554535)
    b = 20.4562557
    rs = ((3.0 / ((4.0 * π) * n)) ** (1.0 / 3.0))
    ec = (a * torch.log(((1.0 + (b / rs)) + (b / (rs ** 2.0))) if isinstance(((1.0 + (b / rs)) + (b / (rs ** 2.0))), torch.Tensor) else torch.tensor(float(((1.0 + (b / rs)) + (b / (rs ** 2.0)))))))
    vc = (ec + (((a * b) * (2.0 + rs)) / (3.0 * ((b + (b * rs)) + (rs ** 2.0)))))
    return torch.stack([torch.as_tensor(ec), torch.as_tensor(vc)])

def get_Ekin(atoms, Y):
    LY = op_L(atoms, Y)
    real_inner_product = torch.sum(((torch.real(Y if isinstance(Y, torch.Tensor) else torch.tensor(float(Y))) * torch.real(LY if isinstance(LY, torch.Tensor) else torch.tensor(float(LY)))) + (torch.real(((-torch.tensor(1j)) * Y) if isinstance(((-torch.tensor(1j)) * Y), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * Y)))) * torch.real(((-torch.tensor(1j)) * LY) if isinstance(((-torch.tensor(1j)) * LY), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * LY)))))) if isinstance(((torch.real(Y if isinstance(Y, torch.Tensor) else torch.tensor(float(Y))) * torch.real(LY if isinstance(LY, torch.Tensor) else torch.tensor(float(LY)))) + (torch.real(((-torch.tensor(1j)) * Y) if isinstance(((-torch.tensor(1j)) * Y), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * Y)))) * torch.real(((-torch.tensor(1j)) * LY) if isinstance(((-torch.tensor(1j)) * LY), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * LY)))))), torch.Tensor) else torch.tensor(float(((torch.real(Y if isinstance(Y, torch.Tensor) else torch.tensor(float(Y))) * torch.real(LY if isinstance(LY, torch.Tensor) else torch.tensor(float(LY)))) + (torch.real(((-torch.tensor(1j)) * Y) if isinstance(((-torch.tensor(1j)) * Y), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * Y)))) * torch.real(((-torch.tensor(1j)) * LY) if isinstance(((-torch.tensor(1j)) * LY), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * LY)))))))))
    f0 = atoms.f[int(0)]
    return (((-0.5) * f0) * real_inner_product)

def get_Ecoul(atoms, n, φ):
    phi_real = op_Jdag(atoms, op_O(atoms, φ))
    return (0.5 * torch.sum((n * torch.real(phi_real if isinstance(phi_real, torch.Tensor) else torch.tensor(float(phi_real)))) if isinstance((n * torch.real(phi_real if isinstance(phi_real, torch.Tensor) else torch.tensor(float(phi_real)))), torch.Tensor) else torch.tensor(float((n * torch.real(phi_real if isinstance(phi_real, torch.Tensor) else torch.tensor(float(phi_real))))))))

def get_Exc(atoms, n, exc):
    exc_recip = op_J(atoms, exc)
    exc_real = op_Jdag(atoms, op_O(atoms, exc_recip))
    return torch.sum((n * torch.real(exc_real if isinstance(exc_real, torch.Tensor) else torch.tensor(float(exc_real)))) if isinstance((n * torch.real(exc_real if isinstance(exc_real, torch.Tensor) else torch.tensor(float(exc_real)))), torch.Tensor) else torch.tensor(float((n * torch.real(exc_real if isinstance(exc_real, torch.Tensor) else torch.tensor(float(exc_real)))))))

def get_Een(n, ionic_potential):
    return torch.sum((n * torch.real(ionic_potential if isinstance(ionic_potential, torch.Tensor) else torch.tensor(float(ionic_potential)))) if isinstance((n * torch.real(ionic_potential if isinstance(ionic_potential, torch.Tensor) else torch.tensor(float(ionic_potential)))), torch.Tensor) else torch.tensor(float((n * torch.real(ionic_potential if isinstance(ionic_potential, torch.Tensor) else torch.tensor(float(ionic_potential)))))))

def get_Eewald(atoms, gcut, gamma):
    gexp = ((-1.0) * torch.log(gamma if isinstance(gamma, torch.Tensor) else torch.tensor(float(gamma))))
    ν = (0.5 * torch.sqrt(((gcut ** 2.0) / gexp) if isinstance(((gcut ** 2.0) / gexp), torch.Tensor) else torch.tensor(float(((gcut ** 2.0) / gexp)))))
    Ω = atoms.volume()
    natoms = atoms.Natoms
    sum_z2 = torch.sum((atoms.Z_nuc * atoms.Z_nuc) if isinstance((atoms.Z_nuc * atoms.Z_nuc), torch.Tensor) else torch.tensor(float((atoms.Z_nuc * atoms.Z_nuc))))
    total_z = torch.sum(atoms.Z_nuc if isinstance(atoms.Z_nuc, torch.Tensor) else torch.tensor(float(atoms.Z_nuc)))
    self_term = (((-ν) * sum_z2) / torch.sqrt(π if isinstance(π, torch.Tensor) else torch.tensor(float(π))))
    neutral_term = ((((-π) * total_z) * total_z) / ((2.0 * Ω) * (ν ** 2.0)))
    real_cutoff = (torch.sqrt((0.5 * gexp) if isinstance((0.5 * gexp), torch.Tensor) else torch.tensor(float((0.5 * gexp)))) / ν)
    max_image_index = torch.floor(((real_cutoff / atoms.a) + 2.0) if isinstance(((real_cutoff / atoms.a) + 2.0), torch.Tensor) else torch.tensor(float(((real_cutoff / atoms.a) + 2.0))))
    images_per_axis = ((2.0 * max_image_index) + 1.0)
    n_images = ((images_per_axis * images_per_axis) * images_per_axis)
    idx_r = torch.arange(n_images)
    di = (axis_index(idx_r, (images_per_axis * images_per_axis), images_per_axis) - max_image_index)
    dj = (axis_index(idx_r, images_per_axis, images_per_axis) - max_image_index)
    dk = (axis_index(idx_r, 1, images_per_axis) - max_image_index)
    Tx = (atoms.a * di)
    Ty = (atoms.a * dj)
    Tz = (atoms.a * dk)
    g_scale = recip_scale(atoms.a)
    max_g_index = torch.floor(((gcut / g_scale) + 2.0) if isinstance(((gcut / g_scale) + 2.0), torch.Tensor) else torch.tensor(float(((gcut / g_scale) + 2.0))))
    g_vectors_per_axis = ((2.0 * max_g_index) + 1.0)
    n_g_vectors = ((g_vectors_per_axis * g_vectors_per_axis) * g_vectors_per_axis)
    idx_g = torch.arange(n_g_vectors)
    gi = (axis_index(idx_g, (g_vectors_per_axis * g_vectors_per_axis), g_vectors_per_axis) - max_g_index)
    gj = (axis_index(idx_g, g_vectors_per_axis, g_vectors_per_axis) - max_g_index)
    gk = (axis_index(idx_g, 1, g_vectors_per_axis) - max_g_index)
    Gx = (g_scale * gi)
    Gy = (g_scale * gj)
    Gz = (g_scale * gk)
    G2 = (((Gx * Gx) + (Gy * Gy)) + (Gz * Gz))
    keep_g = (torch.gt(G2, 0.0) * 1.0)
    safe_G2 = (G2 + (1.0 - keep_g))
    g_weight = ((keep_g * torch.exp((((-0.25) * G2) / (ν ** 2.0)) if isinstance((((-0.25) * G2) / (ν ** 2.0)), torch.Tensor) else torch.tensor(float((((-0.25) * G2) / (ν ** 2.0)))))) / safe_G2)
    Eewald = (self_term + neutral_term)
    for ia in range(int(0), int(natoms)):
        for ja in range(int(0), int(natoms)):
            dpos_x = (atoms.px[int(ia)] - atoms.px[int(ja)])
            dpos_y = (atoms.py[int(ia)] - atoms.py[int(ja)])
            dpos_z = (atoms.pz[int(ia)] - atoms.pz[int(ja)])
            zizj = (atoms.Z_nuc[int(ia)] * atoms.Z_nuc[int(ja)])
            rx = (dpos_x - Tx)
            ry = (dpos_y - Ty)
            rz = (dpos_z - Tz)
            rmag = torch.sqrt((((rx * rx) + (ry * ry)) + (rz * rz)) if isinstance((((rx * rx) + (ry * ry)) + (rz * rz)), torch.Tensor) else torch.tensor(float((((rx * rx) + (ry * ry)) + (rz * rz)))))
            keep_r = (torch.gt(rmag, 0.0) * 1.0)
            safe_r = (rmag + (1.0 - keep_r))
            real_pair = ((0.5 * zizj) * torch.sum(((keep_r * torch.erfc((ν * rmag) if isinstance((ν * rmag), torch.Tensor) else torch.tensor(float((ν * rmag))))) / safe_r) if isinstance(((keep_r * torch.erfc((ν * rmag) if isinstance((ν * rmag), torch.Tensor) else torch.tensor(float((ν * rmag))))) / safe_r), torch.Tensor) else torch.tensor(float(((keep_r * torch.erfc((ν * rmag) if isinstance((ν * rmag), torch.Tensor) else torch.tensor(float((ν * rmag))))) / safe_r)))))
            gpos = (((Gx * dpos_x) + (Gy * dpos_y)) + (Gz * dpos_z))
            recip_pair = ((((2.0 * π) / Ω) * zizj) * torch.sum((g_weight * torch.cos(gpos if isinstance(gpos, torch.Tensor) else torch.tensor(float(gpos)))) if isinstance((g_weight * torch.cos(gpos if isinstance(gpos, torch.Tensor) else torch.tensor(float(gpos)))), torch.Tensor) else torch.tensor(float((g_weight * torch.cos(gpos if isinstance(gpos, torch.Tensor) else torch.tensor(float(gpos))))))))
            Eewald = ((Eewald + real_pair) + recip_pair)
    return Eewald

def init_W(atoms, seed):
    torch.manual_seed(int(seed))
    n_active = len(atoms.g2c())
    W = torch.distributions.Uniform(0.0, 1.0).rsample((int(n_active),))
    W = (W * (1.0 + 0j))
    return orth(atoms, W)

def energy_of_W(W, atoms, ionic_potential, Eewald):
    Y = orth(atoms, W)
    n = get_n_total(atoms, Y)
    φ = get_phi(atoms, n)
    n_c = ((n + 1e-10) * (1.0 + 0j))
    ex_vx = lda_x(n_c)
    ec_vc = lda_c_chachiyo(n_c)
    exc = (ex_vx[int(0)] + ec_vc[int(0)])
    Ekin = get_Ekin(atoms, Y)
    Ecoul = get_Ecoul(atoms, n, φ)
    Exc = get_Exc(atoms, n, exc)
    Een = get_Een(n, ionic_potential)
    return ((((Ekin + Ecoul) + Exc) + Een) + Eewald)

def sd(scf, Nit, beta, etol):
    atoms = scf.atoms
    ionic_potential = scf.ionic_potential
    Eewald = scf.Eewald
    W = scf.W
    g = (scf.W * 0.0)
    E_prev = 0.0
    E_cur = 0.0
    has_stepped = 0.0
    converged = 0.0
    active = 1.0
    for i in range(int(0), int(Nit)):
        E_cur = energy_of_W(W, atoms, ionic_potential, Eewald)
        converged = (converged + (((1.0 - converged) * has_stepped) * (torch.le(torch.abs((E_cur - E_prev) if isinstance((E_cur - E_prev), torch.Tensor) else torch.tensor(float((E_cur - E_prev)))), etol) * 1.0)))
        active = (1.0 - converged)
        g = compute_grad(lambda _dW: energy_of_W(_dW, atoms, ionic_potential, Eewald), W)
        W = (W - ((beta * active) * g))
        E_prev = E_cur
        has_stepped = 1.0
    return E_prev

def runSCF(atoms, Eewald, Nit, β, etol, seed):
    ionic_potential = coulomb(atoms)
    W0 = init_W(atoms, seed)
    Y0 = orth(atoms, W0)
    n0 = get_n_total(atoms, Y0)
    φ0 = get_phi(atoms, n0)
    n0_c = ((n0 + 1e-10) * (1.0 + 0j))
    ex_vx0 = lda_x(n0_c)
    ec_vc0 = lda_c_chachiyo(n0_c)
    exc0 = (ex_vx0[int(0)] + ec_vc0[int(0)])
    vxc0 = (ex_vx0[int(1)] + ec_vc0[int(1)])
    scf = SCF(atoms, ionic_potential, W0, Y0, n0, φ0, exc0, vxc0, Eewald)
    return sd(scf, Nit, β, etol)

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

class SCF(nn.Module):
    def __init__(self, atoms, ionic_potential, W, Y, n, φ, exc, vxc, Eewald):
        super().__init__()
        self.add_module('atoms', atoms)
        self.ionic_potential = nn.Parameter(torch.as_tensor(ionic_potential))
        self.W = nn.Parameter(torch.as_tensor(W))
        self.Y = nn.Parameter(torch.as_tensor(Y))
        self.n = nn.Parameter(torch.as_tensor(n))
        self.φ = nn.Parameter(torch.as_tensor(φ))
        self.exc = nn.Parameter(torch.as_tensor(exc))
        self.vxc = nn.Parameter(torch.as_tensor(vxc))
        self.Eewald = nn.Parameter(torch.as_tensor(Eewald))
        self.learnable_params = [self.ionic_potential, self.W, self.Y, self.n, self.φ, self.exc, self.vxc, self.Eewald]

    def forward(self, _unused):
        this = self
        _unused = torch.as_tensor(_unused, device=DEVICE).float()
        return _unused

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
gcut = 2.0
gamma = 1e-08
Nit = 1001
β = 1e-05
etol = 1e-06
seed = 1234
Natoms1 = 1
Nstate1 = 1
px1 = torch.tensor([0.0], device=DEVICE)
py1 = torch.tensor([0.0], device=DEVICE)
pz1 = torch.tensor([0.0], device=DEVICE)
Z_H = torch.tensor([1.0], device=DEVICE)
f_H = torch.tensor([1.0], device=DEVICE)
H_atom = Atoms(a, ecut, s, s, s, Natoms1, px1, py1, pz1, Nstate1, Z_H, f_H).to(DEVICE)
Eewald_H = get_Eewald(H_atom, gcut, gamma)
E_H = runSCF(H_atom, Eewald_H, Nit, β, etol, seed)
physika_print(E_H)
Z_He = torch.tensor([2.0], device=DEVICE)
f_He = torch.tensor([2.0], device=DEVICE)
He_atom = Atoms(a, ecut, s, s, s, Natoms1, px1, py1, pz1, Nstate1, Z_He, f_He).to(DEVICE)
Eewald_He = get_Eewald(He_atom, gcut, gamma)
E_He = runSCF(He_atom, Eewald_He, Nit, β, etol, seed)
physika_print(E_He)
Natoms2 = 2
Nstate2 = 1
px2 = torch.tensor([0.0, 1.4], device=DEVICE)
py2 = torch.tensor([0.0, 0.0], device=DEVICE)
pz2 = torch.tensor([0.0, 0.0], device=DEVICE)
Z_H2 = torch.tensor([1.0, 1.0], device=DEVICE)
f_H2 = torch.tensor([2.0], device=DEVICE)
H2_atom = Atoms(a, ecut, s, s, s, Natoms2, px2, py2, pz2, Nstate2, Z_H2, f_H2).to(DEVICE)
Eewald_H2 = get_Eewald(H2_atom, gcut, gamma)
E_H2 = runSCF(H2_atom, Eewald_H2, Nit, β, etol, seed)
physika_print(E_H2)