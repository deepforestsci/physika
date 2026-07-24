# tests/test_dft_density.py
import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06

S = (2, 2, 2)
N = 8
A = 2.0
ECUT = 5.0

ACTIVE = torch.tensor([True, True, True, False, True, False, False, False])
W_ACT = torch.tensor([0 + 0j, 1 + 2j, -1 + 0j, 0 + 3j], dtype=torch.complex64)

H_A = 16.0
H_ECUT = 16.0
H_S = 60


@pytest.fixture(scope="module")
def dens_ns():
    """Execute examples/dft_density.phyk and return its namespace."""
    return exec_phyk("dft_density")


@pytest.fixture(scope="module")
def h_like_atoms(dens_ns):
    return dens_ns["Atoms"](A, ECUT, S[0], S[1], S[2], 0.0, 0.0, 0.0, 1, 1,
                            torch.tensor([1.0]), torch.tensor([1.0]))


@pytest.fixture(scope="module")
def h_atom(dens_ns):
    return dens_ns["Atoms"](H_A, H_ECUT, H_S, H_S, H_S, 0.0, 0.0, 0.0, 1, 1,
                            torch.tensor([1.0]), torch.tensor([1.0]))


class TestMaskEmbed:

    def test_places_values_at_active_positions(self, dens_ns):
        values = torch.tensor([1 + 1j, 2 + 0j, 0 + 3j, -1 + 2j],
                              dtype=torch.complex64)
        out = dens_ns["mask_embed"](values, ACTIVE, N)
        expected = torch.zeros(N, dtype=torch.complex64)
        expected[ACTIVE] = values
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_zero_elsewhere(self, dens_ns):
        values = torch.tensor([1 + 1j, 2 + 0j, 0 + 3j, -1 + 2j],
                              dtype=torch.complex64)
        out = dens_ns["mask_embed"](values, ACTIVE, N)
        assert torch.allclose(out[~ACTIVE], torch.zeros(4, dtype=torch.complex64))


class TestOrth:

    def test_matches_reference(self, dens_ns, h_like_atoms):
        Omega = float(h_like_atoms.volume())
        out = dens_ns["orth"](h_like_atoms, W_ACT)
        expected = W_ACT / torch.sqrt(torch.tensor(Omega) * (W_ACT.abs() ** 2).sum())
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_normalizes_to_unity(self, dens_ns, h_like_atoms):
        Omega = float(h_like_atoms.volume())
        Y = dens_ns["orth"](h_like_atoms, W_ACT)
        norm = Omega * (Y.abs() ** 2).sum()
        assert float(norm) == pytest.approx(1.0, rel=1e-5)


class TestDensity:

    def test_matches_reference(self, dens_ns, h_like_atoms):
        Y = dens_ns["orth"](h_like_atoms, W_ACT)
        active = h_like_atoms.active()
        Y_full = torch.zeros(N, dtype=torch.complex64)
        Y_full[active] = Y
        Yrs = torch.fft.ifftn(Y_full.reshape(S)).reshape(-1) * N
        expected = 1.0 * Yrs.abs() ** 2  # f0 = 1.0
        out = dens_ns["get_n_total"](h_like_atoms, Y)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_is_nonnegative(self, dens_ns, h_like_atoms):
        Y = dens_ns["orth"](h_like_atoms, W_ACT)
        n = dens_ns["get_n_total"](h_like_atoms, Y)
        assert (n.real >= -a_tol).all()

    def test_integrates_to_occupation(self, dens_ns, h_atom):
        n_active = h_atom.g2c().shape[0]
        W0 = torch.rand(n_active, dtype=torch.float32).to(torch.complex64)
        Y0 = dens_ns["orth"](h_atom, W0)
        n0 = dens_ns["get_n_total"](h_atom, Y0)
        dV = float(h_atom.volume()) / (H_S ** 3)
        total_charge = float((n0 * dV).sum())
        assert total_charge == pytest.approx(1.0, rel=1e-3)


class TestHartreePotential:

    N_TEST = torch.tensor([1., 2., 0.5, 3., 1.5, 0., 2.5, 1.], dtype=torch.float32)

    def test_matches_reference(self, dens_ns, h_like_atoms):
        G2 = h_like_atoms.g2()
        Omega = float(h_like_atoms.volume())
        n_c = self.N_TEST.to(torch.complex64)
        n_recip = torch.fft.fftn(n_c.reshape(S)).reshape(-1) / N   # op_J
        Ov = Omega * n_recip                                        # op_O
        nonzero = G2 > 0
        safe_G2 = torch.where(nonzero, G2, torch.ones_like(G2))
        Linv = torch.where(nonzero, Ov / safe_G2 / (-Omega),
                           torch.zeros_like(Ov))                     # op_Linv
        expected = -4.0 * torch.pi * Linv

        out = dens_ns["get_phi"](h_like_atoms, self.N_TEST)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_dc_component_is_zero(self, dens_ns, h_like_atoms):
        out = dens_ns["get_phi"](h_like_atoms, self.N_TEST)
        assert out[0] == 0

    def test_no_nan_or_inf(self, dens_ns, h_like_atoms):
        out = dens_ns["get_phi"](h_like_atoms, self.N_TEST)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()
