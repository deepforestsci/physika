# tests/test_dft_scf.py
import math
import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06

S = (2, 2, 2)
A = 2.0
ECUT = 5.0

H_A = 16.0
H_ECUT = 16.0
H_S = 60


@pytest.fixture(scope="module")
def scf_ns():
    """Execute examples/dft_scf.phyk and return its namespace."""
    return exec_phyk("dft_scf")


@pytest.fixture(scope="module")
def h_like_atoms(scf_ns):
    return scf_ns["Atoms"](A, ECUT, S[0], S[1], S[2], 0.0, 0.0, 0.0, 1, 1,
                           torch.tensor([1.0]), torch.tensor([1.0]))


@pytest.fixture(scope="module")
def h_atom(scf_ns):
    return scf_ns["Atoms"](H_A, H_ECUT, H_S, H_S, H_S, 0.0, 0.0, 0.0, 1, 1,
                           torch.tensor([1.0]), torch.tensor([1.0]))


class TestInitW:

    def test_is_lowdin_normalized(self, scf_ns, h_like_atoms):
        W = scf_ns["init_W"](h_like_atoms, 42)
        Omega = float(h_like_atoms.volume())
        norm = Omega * (W.abs() ** 2).sum()
        assert float(norm) == pytest.approx(1.0, rel=1e-5)

    def test_reproducible_with_same_seed(self, scf_ns, h_like_atoms):
        W1 = scf_ns["init_W"](h_like_atoms, 42)
        W2 = scf_ns["init_W"](h_like_atoms, 42)
        assert torch.allclose(W1, W2, rtol=r_tol, atol=a_tol)

    def test_differs_with_different_seed(self, scf_ns, h_like_atoms):
        W1 = scf_ns["init_W"](h_like_atoms, 1)
        W2 = scf_ns["init_W"](h_like_atoms, 2)
        assert not torch.allclose(W1, W2, rtol=r_tol, atol=a_tol)


class TestEnergyOfW:

    def test_matches_manual_pipeline(self, scf_ns, h_like_atoms):
        W = scf_ns["init_W"](h_like_atoms, 7)
        ionic_potential = scf_ns["coulomb"](h_like_atoms)
        Eewald = 0.0

        Y = scf_ns["orth"](h_like_atoms, W)
        n = scf_ns["get_n_total"](h_like_atoms, Y)
        phi = scf_ns["get_phi"](h_like_atoms, n)
        n_c = (n + 1e-10) * (1.0 + 0j)
        ex_vx = scf_ns["lda_x"](n_c)
        ec_vc = scf_ns["lda_c_chachiyo"](n_c)
        exc = ex_vx[0] + ec_vc[0]
        Ekin = float(scf_ns["get_Ekin"](h_like_atoms, Y))
        Ecoul = float(scf_ns["get_Ecoul"](h_like_atoms, n, phi))
        Exc = float(scf_ns["get_Exc"](h_like_atoms, n, exc))
        Een = float(scf_ns["get_Een"](n, ionic_potential))
        expected = Ekin + Ecoul + Exc + Een + Eewald

        out = scf_ns["energy_of_W"](W, h_like_atoms, ionic_potential, Eewald)
        assert float(out) == pytest.approx(expected, rel=r_tol, abs=a_tol)


class TestSteepestDescent:

    def _scf_state(self, scf_ns, atoms, seed):
        ionic_potential = scf_ns["coulomb"](atoms)
        W0 = scf_ns["init_W"](atoms, seed)
        Y0 = scf_ns["orth"](atoms, W0)
        n0 = scf_ns["get_n_total"](atoms, Y0)
        phi0 = scf_ns["get_phi"](atoms, n0)
        n0_c = (n0 + 1e-10) * (1.0 + 0j)
        ex_vx0 = scf_ns["lda_x"](n0_c)
        ec_vc0 = scf_ns["lda_c_chachiyo"](n0_c)
        exc0 = ex_vx0[0] + ec_vc0[0]
        vxc0 = ex_vx0[1] + ec_vc0[1]
        Eewald = 0.0
        scf = scf_ns["SCF"](atoms, ionic_potential, W0, Y0, n0, phi0, exc0,
                            vxc0, Eewald)
        return scf, Eewald

    def test_reduces_energy(self, scf_ns, h_like_atoms):
        scf, Eewald = self._scf_state(scf_ns, h_like_atoms, 3)
        E0 = float(scf_ns["energy_of_W"](scf.W, h_like_atoms, scf.ionic_potential, Eewald))
        E_final = float(scf_ns["sd"](scf, 10, 1e-4, 1e-12))
        assert E_final <= E0 + a_tol

    def test_freezes_once_converged(self, scf_ns, h_like_atoms):
        scf_short, Eewald = self._scf_state(scf_ns, h_like_atoms, 5)
        scf_long, _ = self._scf_state(scf_ns, h_like_atoms, 5)
        e_short = float(scf_ns["sd"](scf_short, 3, 1e-4, 1e10))
        e_long = float(scf_ns["sd"](scf_long, 30, 1e-4, 1e10))
        assert e_short == pytest.approx(e_long, rel=1e-5)


class TestRunSCF:

    def test_smoke_h_like(self, scf_ns, h_like_atoms):
        out = scf_ns["runSCF"](h_like_atoms, 0.0, 5, 1e-4, 1e-8, 11)
        assert math.isfinite(float(out))
