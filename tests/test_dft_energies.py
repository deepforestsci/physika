# tests/test_dft_energies.py
import math
import types
import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06

S = (2, 2, 2)
N = 8
A = 2.0
ECUT = 5.0
OMEGA = 8.0

ACTIVE = torch.tensor([True, True, True, False, True, False, False, False])
PI = 3.141592653589793
G2 = PI**2 * torch.tensor([0., 1., 1., 2., 1., 2., 2., 3.])
G2C = torch.masked_select(G2, ACTIVE)

W_ACT = torch.tensor([0 + 0j, 1 + 2j, -1 + 0j, 0 + 3j], dtype=torch.complex64)
N_TEST = torch.tensor([1., 2., 0.5, 3., 1.5, 0., 2.5, 1.], dtype=torch.float32)
PHI_TEST = torch.tensor([0 + 0j, 1 + 1j, 2 - 1j, 0 + 0j, -1 + 0j, 3 + 2j, 0 + 1j, 1 - 1j],
                        dtype=torch.complex64)
EXC_TEST = torch.tensor([0.1 + 0j, -0.2 + 0j, 0.05 + 0j, -0.3 + 0j, 0.15 + 0j,
                         -0.1 + 0j, 0.2 + 0j, -0.05 + 0j], dtype=torch.complex64)
ION_POT_TEST = torch.tensor([-1 + 0j, -0.5 + 0j, -2 + 0j, 0 + 0j, -1.5 + 0j,
                             -0.3 + 0j, -0.8 + 0j, -1.2 + 0j], dtype=torch.complex64)

H_A = 16.0
H_ECUT = 16.0
H_S = 4  # get_Eewald doesn't depend on grid resolution; keep it tiny


@pytest.fixture(scope="module")
def en_ns():
    """Execute examples/dft_energies.phyk and return its namespace."""
    return exec_phyk("dft_energies")


@pytest.fixture(scope="module")
def h_like_atoms(en_ns):
    return en_ns["Atoms"](A, ECUT, S[0], S[1], S[2], 0.0, 0.0, 0.0, 1, 1,
                          torch.tensor([1.0]), torch.tensor([1.0]))


def make_atoms(en_ns, f0=1.0, Z=1.0, a=H_A, ecut=H_ECUT, s=H_S):
    return en_ns["Atoms"](a, ecut, s, s, s, 0.0, 0.0, 0.0, 1, 1,
                          torch.tensor([Z]), torch.tensor([f0]))


class TestKineticEnergy:

    def test_matches_reference(self, en_ns, h_like_atoms):
        LY = -OMEGA * G2C * W_ACT
        real_inner = (W_ACT.real * LY.real + W_ACT.imag * LY.imag).sum()
        expected = -0.5 * 1.0 * real_inner
        out = en_ns["get_Ekin"](h_like_atoms, W_ACT)
        assert float(out) == pytest.approx(float(expected), rel=r_tol, abs=a_tol)

    def test_is_nonnegative(self, en_ns, h_like_atoms):
        # op_L is negative semi-definite (G2 >= 0), so Ekin = -0.5 f0 Re<Y|L|Y> >= 0.
        out = en_ns["get_Ekin"](h_like_atoms, W_ACT)
        assert float(out) >= -a_tol

    def test_scales_linearly_with_occupation(self, en_ns):
        atoms_f1 = en_ns["Atoms"](A, ECUT, S[0], S[1], S[2], 0.0, 0.0, 0.0, 1, 1,
                                  torch.tensor([1.0]), torch.tensor([1.0]))
        atoms_f2 = en_ns["Atoms"](A, ECUT, S[0], S[1], S[2], 0.0, 0.0, 0.0, 1, 1,
                                  torch.tensor([1.0]), torch.tensor([2.0]))
        e1 = float(en_ns["get_Ekin"](atoms_f1, W_ACT))
        e2 = float(en_ns["get_Ekin"](atoms_f2, W_ACT))
        assert e2 == pytest.approx(2.0 * e1, rel=r_tol)


class TestCoulombEnergy:

    def test_matches_reference(self, en_ns, h_like_atoms):
        Ov = OMEGA * PHI_TEST
        phi_real = torch.fft.ifftn(Ov.reshape(S)).reshape(-1)  # op_Jdag(op_O(phi, Ω))
        expected = 0.5 * (N_TEST * phi_real.real).sum()
        out = en_ns["get_Ecoul"](h_like_atoms, N_TEST, PHI_TEST)
        assert float(out) == pytest.approx(float(expected), rel=r_tol, abs=a_tol)

    def test_no_nan_or_inf(self, en_ns, h_like_atoms):
        out = en_ns["get_Ecoul"](h_like_atoms, N_TEST, PHI_TEST)
        assert math.isfinite(float(out))


class TestXCEnergy:

    def test_matches_reference(self, en_ns, h_like_atoms):
        exc_recip = torch.fft.fftn(EXC_TEST.reshape(S)).reshape(-1) / N          # op_J
        exc_real = torch.fft.ifftn((OMEGA * exc_recip).reshape(S)).reshape(-1)   # op_Jdag(op_O(...))
        expected = (N_TEST * exc_real.real).sum()
        out = en_ns["get_Exc"](h_like_atoms, N_TEST, EXC_TEST)
        assert float(out) == pytest.approx(float(expected), rel=r_tol, abs=a_tol)


class TestElectronIonEnergy:

    def test_matches_reference(self, en_ns):
        expected = (N_TEST * ION_POT_TEST.real).sum()
        out = en_ns["get_Een"](N_TEST, ION_POT_TEST)
        assert float(out) == pytest.approx(float(expected), rel=r_tol, abs=a_tol)

    def test_is_linear_in_potential(self, en_ns):
        out1 = float(en_ns["get_Een"](N_TEST, ION_POT_TEST))
        out2 = float(en_ns["get_Een"](N_TEST, 2.0 * ION_POT_TEST))
        assert out2 == pytest.approx(2.0 * out1, rel=r_tol)


def ref_eewald(a, Omega, Z, gcut, gamma):
    """Independent re-derivation of get_Eewald's single-nucleus formula."""
    gexp = -math.log(gamma)
    nu = 0.5 * math.sqrt(gcut**2 / gexp)
    zz = Z * Z
    self_term = -nu * zz / math.sqrt(math.pi)
    neutral_term = -math.pi * zz / (2.0 * Omega * nu**2)

    real_cutoff = math.sqrt(0.5 * gexp) / nu
    max_img = int(math.floor(real_cutoff / a + 2.0))
    real_term = 0.0
    for i in range(-max_img, max_img + 1):
        for j in range(-max_img, max_img + 1):
            for k in range(-max_img, max_img + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                r = a * math.sqrt(i * i + j * j + k * k)
                real_term += 0.5 * zz * math.erfc(nu * r) / r

    g_scale = 2.0 * math.pi / a
    max_g = int(math.floor(gcut / g_scale + 2.0))
    recip_term = 0.0
    for i in range(-max_g, max_g + 1):
        for j in range(-max_g, max_g + 1):
            for k in range(-max_g, max_g + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                G2v = (g_scale**2) * (i * i + j * j + k * k)
                recip_term += (2.0 * math.pi / Omega) * zz * math.exp(-0.25 * G2v / nu**2) / G2v

    return self_term + neutral_term + real_term + recip_term


class TestEwaldEnergy:

    def test_matches_reference(self, en_ns):
        atoms = make_atoms(en_ns)
        Omega = float(atoms.volume())
        expected = ref_eewald(H_A, Omega, 1.0, 2.0, 1e-8)
        out = en_ns["get_Eewald"](atoms, 2.0, 1e-8)
        assert float(out) == pytest.approx(expected, rel=1e-4)

    def test_no_nan_or_inf(self, en_ns):
        atoms = make_atoms(en_ns)
        out = en_ns["get_Eewald"](atoms, 2.0, 1e-8)
        assert math.isfinite(float(out))

    def test_scales_with_charge_squared(self, en_ns):
        # Every term (self/neutral/real/recip) is proportional to Z^2.
        atoms_z1 = make_atoms(en_ns, Z=1.0)
        atoms_z2 = make_atoms(en_ns, Z=2.0)
        e1 = float(en_ns["get_Eewald"](atoms_z1, 2.0, 1e-8))
        e2 = float(en_ns["get_Eewald"](atoms_z2, 2.0, 1e-8))
        assert e2 == pytest.approx(4.0 * e1, rel=1e-4)


class TestTotalEnergy:

    def test_sums_all_five_terms(self, en_ns, h_like_atoms):
        scf_stub = types.SimpleNamespace(
            atoms=h_like_atoms, Y=W_ACT, n=N_TEST, phi=PHI_TEST,
            exc=EXC_TEST, ionic_potential=ION_POT_TEST, Eewald=0.05)

        Ekin = float(en_ns["get_Ekin"](h_like_atoms, W_ACT))
        Ecoul = float(en_ns["get_Ecoul"](h_like_atoms, N_TEST, PHI_TEST))
        Exc = float(en_ns["get_Exc"](h_like_atoms, N_TEST, EXC_TEST))
        Een = float(en_ns["get_Een"](N_TEST, ION_POT_TEST))
        expected = Ekin + Ecoul + Exc + Een + 0.05

        out = en_ns["get_E"](scf_stub)
        assert float(out) == pytest.approx(expected, rel=r_tol, abs=a_tol)
