import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06

PI = 3.141592653589793

# 2x2x2 grid; flat index = 4*m1 + 2*m2 + m3 (C-order).
S = (2, 2, 2)
N = 8

# H-atom parameters from SimpleDFT.jl examples.jl: a=16, ecut=16, s=60^3,
# Omega = 4096, Sf == 1 everywhere.
H_A = 16.0
H_ECUT = 16.0
H_S = 60
H_N_ACTIVE = 12533
H_OMEGA = 4096.0


@pytest.fixture(scope="module")
def atoms_ns():
    """Execute examples/dft_atoms.phyk and return its namespace."""
    return exec_phyk("dft_atoms")


@pytest.fixture(scope="module")
def h_atom(atoms_ns):
    """Atoms instance for the H-atom validation system (Natoms=1, Nstate=1)."""
    return atoms_ns["Atoms"](H_A, H_ECUT, H_S, H_S, H_S, 1,
                             torch.tensor([0.0]), torch.tensor([0.0]),
                             torch.tensor([0.0]), 1,
                             torch.tensor([1.0]), torch.tensor([1.0]))


class TestGridIndices:

    def test_flat_index_is_arange(self, atoms_ns):
        out = atoms_ns["flat_index"](N)
        assert torch.equal(out, torch.arange(N, dtype=out.dtype))

    def test_axis_index_unravels_c_order(self, atoms_ns):
        ms = atoms_ns["flat_index"](N)
        m1 = atoms_ns["axis_index"](ms, S[1] * S[2], S[0])
        m2 = atoms_ns["axis_index"](ms, S[2], S[1])
        m3 = atoms_ns["axis_index"](ms, 1, S[2])
        assert torch.equal(
            m1, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=m1.dtype))
        assert torch.equal(
            m2, torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=m2.dtype))
        assert torch.equal(
            m3, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=m3.dtype))

    def test_fold_freq_wraps_upper_half(self, atoms_ns):
        # q=4: q/2=2.0, only m>2 folds (m=3 -> 3-4=-1); m=2 stays put.
        m = torch.tensor([0., 1., 2., 3.])
        out = atoms_ns["fold_freq"](m, 4)
        assert torch.equal(out, torch.tensor([0., 1., 2., -1.]))


class TestCellGeometry:

    def test_cell_volume_is_a_cubed(self, atoms_ns):
        out = atoms_ns["cell_volume"](torch.tensor(2.0))
        assert out == pytest.approx(8.0)

    def test_sample_coord_spacing(self, atoms_ns):
        m = torch.tensor([0., 1., 2., 3.])
        out = atoms_ns["sample_coord"](m, torch.tensor(4.0), 4)
        assert torch.allclose(out,
                              torch.tensor([0., 1., 2., 3.]),
                              rtol=r_tol,
                              atol=a_tol)


class TestReciprocalSpace:

    def test_recip_scale(self, atoms_ns):
        out = atoms_ns["recip_scale"](torch.tensor(2 * PI))
        assert out == pytest.approx(1.0)

    def test_g_squared(self, atoms_ns):
        n1 = torch.tensor([1., 0.])
        n2 = torch.tensor([0., 1.])
        n3 = torch.tensor([0., 0.])
        out = atoms_ns["g_squared"](n1, n2, n3, torch.tensor(2.0))
        assert torch.allclose(out,
                              torch.tensor([4.0, 4.0]),
                              rtol=r_tol,
                              atol=a_tol)

    def test_active_mask_is_bool(self, atoms_ns):
        G2 = torch.tensor([0., 1., 2., 3.])
        out = atoms_ns["active_mask"](G2, torch.tensor(1.0))
        assert out.dtype == torch.bool
        assert torch.equal(out, torch.tensor([True, True, True, False]))

    def test_active_subset_selects_masked(self, atoms_ns):
        G2 = torch.tensor([0., 1., 2., 3.])
        mask = torch.tensor([True, False, True, False])
        out = atoms_ns["active_subset"](G2, mask)
        assert torch.equal(out, torch.tensor([0., 2.]))


class TestStructureFactor:

    def test_atom_at_origin_gives_all_ones(self, atoms_ns):
        n1 = torch.tensor([0., 1., -1.])
        n2 = torch.tensor([1., 0., 2.])
        n3 = torch.tensor([2., -1., 0.])
        out = atoms_ns["structure_factor"](n1, n2, n3, torch.tensor(1.0),
                                           torch.tensor(0.0),
                                           torch.tensor(0.0),
                                           torch.tensor(0.0))
        assert torch.allclose(out,
                              torch.ones_like(out),
                              rtol=r_tol,
                              atol=a_tol)

    def test_phase_matches_reference(self, atoms_ns):
        n1, n2, n3 = torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.0)
        cc, px, py, pz = 0.5, 1.0, 0.5, 0.0
        out = atoms_ns["structure_factor"](n1, n2, n3, torch.tensor(cc),
                                           torch.tensor(px), torch.tensor(py),
                                           torch.tensor(pz))
        phase = cc * (1.0 * px + 2.0 * py + 0.0 * pz)
        expected = torch.exp(torch.tensor(-1j * phase, dtype=torch.complex64))
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)


class TestHAtomBasis:
    """Validates the Atoms class against SimpleDFT.jl's H-atom system."""

    def test_grid_size(self, h_atom):
        assert h_atom.active().shape == (H_S**3, )

    def test_cell_volume(self, h_atom):
        assert float(h_atom.volume()) == pytest.approx(H_OMEGA)

    def test_active_count_matches_julia(self, h_atom):
        assert int(h_atom.active().sum()) == H_N_ACTIVE

    def test_active_is_bool_mask(self, h_atom):
        assert h_atom.active().dtype == torch.bool

    def test_g2c_keeps_active_only(self, h_atom):
        assert h_atom.g2c().shape == (H_N_ACTIVE, )

    def test_dc_component_is_zero(self, h_atom):
        assert float(h_atom.g2()[0]) == pytest.approx(0.0)

    def test_structure_factor_all_ones(self, h_atom):
        Sf = h_atom.sf()
        assert Sf.shape == (H_S**3, )
        assert torch.allclose(Sf, torch.ones_like(Sf), rtol=r_tol, atol=a_tol)

    def test_real_space_sampling_matches_grid(self, h_atom):
        # Max real-space coordinate along an axis is a*(s-1)/s.
        expected_max = H_A * (H_S - 1) / H_S
        assert float(h_atom.coord_x().max()) == pytest.approx(expected_max,
                                                              rel=1e-4)

    def test_g_vectors_consistent_with_g2(self, h_atom):
        gx, gy, gz = h_atom.gx(), h_atom.gy(), h_atom.gz()
        g2_from_components = gx * gx + gy * gy + gz * gz
        assert torch.allclose(g2_from_components,
                              h_atom.g2(),
                              rtol=r_tol,
                              atol=a_tol)

H2_A = 16.0
H2_S = 4


@pytest.fixture(scope="module")
def h2_atoms(atoms_ns):
    """Two H nuclei at the origin and (1.4, 0, 0)."""
    return atoms_ns["Atoms"](H2_A, H_ECUT, H2_S, H2_S, H2_S, 2,
                             torch.tensor([0.0, 1.4]), torch.tensor([0.0, 0.0]),
                             torch.tensor([0.0, 0.0]), 1,
                             torch.tensor([1.0, 1.0]), torch.tensor([2.0]))


class TestStructureFactorMultiAtom:
    """sf() must sum the single-atom phase kernel over every atom."""

    def test_two_atoms_sum_phases(self, atoms_ns, h2_atoms):
        Sf = h2_atoms.sf()
        n1, n2, n3 = h2_atoms.freq_x(), h2_atoms.freq_y(), h2_atoms.freq_z()
        c = torch.tensor(2 * PI / H2_A)
        z = torch.tensor(0.0)
        kern = atoms_ns["structure_factor"]
        expected = (kern(n1, n2, n3, c, z, z, z)
                    + kern(n1, n2, n3, c, torch.tensor(1.4), z, z))
        assert torch.allclose(Sf, expected, rtol=r_tol, atol=a_tol)

    def test_dc_equals_atom_count(self, h2_atoms):
        # Every atom contributes exp(0) = 1 at G = 0, so Sf[0] == Natoms.
        assert h2_atoms.sf()[0].real == pytest.approx(2.0)
