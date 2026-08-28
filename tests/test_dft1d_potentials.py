import pytest
import torch
from tests.conftest import exec_phyk

RTOL = 1e-4
ATOL = 1e-5


def numerically_equivalent(actual, expected):
    """True when actual and expected match within tolerance."""
    a = actual.detach().double()
    e = torch.as_tensor(expected).double()
    return torch.allclose(a, e, rtol=RTOL, atol=ATOL)


@pytest.fixture(scope="module")
def pot_ns():
    """Namespace from executing examples/dft1d_potentials.phyk."""
    return exec_phyk("dft1d_potentials")


class TestHarmonicPotential:
    def test_is_x_squared(self, pot_ns):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = x * x
        assert numerically_equivalent(pot_ns["harmonic_potential"](x), expected)


class TestDiagEmbed:
    def test_places_vector_on_the_diagonal(self, pot_ns):
        v = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.eye(3)
        out = pot_ns["diag_embed"](v, mask)
        assert numerically_equivalent(out, torch.diag(v))

    def test_off_diagonal_mask_zeros_everything(self, pot_ns):
        v = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3, 3)
        out = pot_ns["diag_embed"](v, mask)
        assert numerically_equivalent(out, torch.zeros(3, 3))


class TestHartree:
    """Soft-Coulomb Hartree potential/energy on a 3-point grid, h = 1."""

    x = torch.tensor([-1.0, 0.0, 1.0])
    n = torch.tensor([0.2, 0.5, 0.3])
    eps = 0.2

    def test_potential_matches_pairwise_soft_coulomb_kernel(self, pot_ns):
        v_Ha = pot_ns["hartree_potential"](self.n, self.x, self.eps)
        expected = [1.0500340710298763, 1.5744694533375334, 1.2248458651324285]
        assert numerically_equivalent(v_Ha, expected)

    def test_energy_matches_half_n_dot_v(self, pot_ns):
        v_Ha = pot_ns["hartree_potential"](self.n, self.x, self.eps)
        E_Ha = pot_ns["hartree_energy"](self.n, v_Ha, 1.0)
        assert float(E_Ha) == pytest.approx(0.6823476502072352, rel=RTOL)

    def test_potential_matrix_is_symmetric_kernel(self, pot_ns):
        # Swapping which point holds the density should give the potential
        # felt at the swapped point (the soft-Coulomb kernel is symmetric).
        n_swapped = torch.tensor([0.3, 0.5, 0.2])
        v_a = pot_ns["hartree_potential"](self.n, self.x, self.eps)
        v_b = pot_ns["hartree_potential"](n_swapped, self.x, self.eps)
        assert numerically_equivalent(v_a, v_b.flip(0))
