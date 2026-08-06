import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06

PI = 3.141592653589793

# 2×2×2 periodic grid with a single hydrogen nucleus at the origin.
# a = 2 Bohr gives Ω = 8 Bohr^3.
S = (2, 2, 2)
N = 8
A = 2.0
ECUT = 5.0


@pytest.fixture(scope="module")
def pot_ns():
    """Execute examples/dft_potentials.phyk and return its namespace."""
    return exec_phyk("dft_potentials")


@pytest.fixture(scope="module")
def h_like_atoms(pot_ns):
    """Single H-like nucleus (Z=1) at the cell origin."""
    return pot_ns["Atoms"](A, ECUT, S[0], S[1], S[2], 1,
                           torch.tensor([0.0]), torch.tensor([0.0]),
                           torch.tensor([0.0]), 1,
                           torch.tensor([1.0]), torch.tensor([1.0]))


def make_atoms(pot_ns, Z):
    """Create a single atom with the given nuclear charge."""
    return pot_ns["Atoms"](A, ECUT, S[0], S[1], S[2], 1,
                           torch.tensor([0.0]), torch.tensor([0.0]),
                           torch.tensor([0.0]), 1,
                           torch.tensor([Z]), torch.tensor([1.0]))

def ref_coulomb(Z, G2, Sf):
    """Reference implementation of the Coulomb potential."""
    nonzero = G2 > 0
    safe_G2 = torch.where(nonzero, G2, torch.ones_like(G2))
    Vcoul = torch.where(nonzero, -4.0 * PI * Z / safe_G2, torch.zeros_like(G2))
    return torch.fft.fftn((Vcoul * Sf).reshape(S)).reshape(-1) / N


class TestCoulomb:

    def test_matches_reference(self, pot_ns, h_like_atoms):
        # Compare against an independent reference implementation.
        G2 = h_like_atoms.g2()
        Sf = h_like_atoms.sf()
        expected = ref_coulomb(1.0, G2, Sf)

        out = pot_ns["coulomb"](h_like_atoms)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_no_nan_or_inf(self, pot_ns, h_like_atoms):
        # The computed potential should contain only finite values.
        out = pot_ns["coulomb"](h_like_atoms)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()

    def test_scales_linearly_with_nuclear_charge(self, pot_ns):
        # Doubling the nuclear charge doubles the Coulomb potential.
        atoms_z1 = make_atoms(pot_ns, 1.0)
        atoms_z2 = make_atoms(pot_ns, 2.0)
        out_z1 = pot_ns["coulomb"](atoms_z1)
        out_z2 = pot_ns["coulomb"](atoms_z2)
        assert torch.allclose(out_z2, 2.0 * out_z1, rtol=r_tol, atol=a_tol)

    def test_output_shape(self, pot_ns, h_like_atoms):
        # Verify the output has one value per grid point.
        out = pot_ns["coulomb"](h_like_atoms)
        assert out.shape == (N, )
