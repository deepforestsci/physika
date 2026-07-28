import pytest
import torch
from tests.conftest import exec_phyk

RTOL = 1e-5
ATOL = 1e-6
PI = 3.141592653589793


def _close(actual, expected):
    """allclose after casting both sides to double (dtype-agnostic)."""
    return torch.allclose(actual.detach().double(),
                          torch.as_tensor(expected, dtype=torch.double),
                          rtol=RTOL, atol=ATOL)


@pytest.fixture(scope="module")
def atoms_ns():
    """Namespace from executing examples/dft_atoms.phyk."""
    return exec_phyk("dft_atoms")


def make_atoms(ns, a, ecut, s1, s2, s3, px, py, pz, Z, f):
    """Construct an Atoms instance from plain Python lists."""
    return ns["Atoms"](a, ecut, s1, s2, s3, len(px),
                       torch.tensor(px), torch.tensor(py), torch.tensor(pz),
                       len(f), torch.tensor(Z), torch.tensor(f))


# --------------------------------------------------------------------------
# Tiny 2x2x2 cell: a=4, ecut=2, one atom at the origin.
# Flattened index unravels C-order as flat = 4*m1 + 2*m2 + m3.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny(atoms_ns):
    return make_atoms(atoms_ns, a=4.0, ecut=2.0, s1=2, s2=2, s3=2,
                      px=[0.0], py=[0.0], pz=[0.0], Z=[1.0], f=[1.0])


class TestGridIndexing:

    def test_flat_index_is_arange(self, tiny):
        assert tiny.flat_index().tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grid_indices_unravel_c_order(self, tiny):
        assert tiny.m1().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert tiny.m2().tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
        assert tiny.m3().tolist() == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_fold_freq_wraps_upper_half(self, tiny):
        folded = tiny.fold_freq(torch.tensor([0., 1., 2., 3.]), 4)
        assert folded.tolist() == [0, 1, 2, -1]

    def test_freqs_are_folded_grid_indices(self, tiny):
        assert tiny.freq_x().tolist() == tiny.m1().tolist()
        assert tiny.freq_y().tolist() == tiny.m2().tolist()
        assert tiny.freq_z().tolist() == tiny.m3().tolist()


class TestRealSpace:

    def test_cell_volume_is_a_cubed(self, tiny):
        assert float(tiny.volume()) == pytest.approx(64.0)  # 4^3

    def test_sample_coord_spacing(self, tiny):
        assert _close(tiny.sample_coord(torch.tensor([0., 1., 2., 3.]), 2),
                      [0., 2., 4., 6.])

    def test_coord_max_is_last_grid_point(self, tiny):
        assert float(tiny.coord_x().max()) == pytest.approx(2.0)


class TestReciprocalSpace:

    def test_recip_scale_is_two_pi_over_a(self, tiny):
        assert float(tiny.recip_scale()) == pytest.approx(2 * PI / 4)

    def test_g2_is_scaled_sum_of_squares(self, tiny):
        c2 = (2 * PI / 4) ** 2
        ss = [0, 1, 1, 2, 1, 2, 2, 3]  # per-point n1^2 + n2^2 + n3^2
        assert _close(tiny.g2(), [c2 * s for s in ss])

    def test_g2_dc_component_is_zero(self, tiny):
        assert float(tiny.g2()[0]) == pytest.approx(0.0)

    def test_g_components_reproduce_g2(self, tiny):
        gx, gy, gz = tiny.gx(), tiny.gy(), tiny.gz()
        assert _close(gx * gx + gy * gy + gz * gz, tiny.g2())

    def test_active_mask_selects_within_cutoff(self, tiny):
        active = tiny.active()
        assert active.dtype == torch.bool
        assert active.tolist() == [True, True, True, False,
                                   True, False, False, False]

    def test_g2c_keeps_active_entries(self, tiny):
        c2 = (2 * PI / 4) ** 2
        assert _close(tiny.g2c(), [0.0, c2, c2, c2])


class TestStructureFactor:

    def test_atom_at_origin_gives_all_ones(self, tiny):
        Sf = tiny.sf()
        assert Sf.shape == (8,)
        assert torch.allclose(Sf, torch.ones_like(Sf), rtol=RTOL, atol=ATOL)

    def test_phase_matches_reference(self, tiny):
        n1, n2, n3 = torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.0)
        c, px, py, pz = 0.5, 1.0, 0.5, 0.0
        out = tiny.structure_factor(n1, n2, n3, torch.tensor(c),
                                    torch.tensor(px), torch.tensor(py),
                                    torch.tensor(pz))
        phase = c * (1.0 * px + 2.0 * py + 0.0 * pz)
        expected = torch.exp(torch.tensor(-1j * phase, dtype=torch.complex64))
        assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# SimpleDFT.jl H-atom reference cell (examples.jl): a=16, ecut=16, s=60^3.
# --------------------------------------------------------------------------

H_A, H_ECUT, H_S = 16.0, 16.0, 60
H_OMEGA = 4096.0
H_N_ACTIVE = 12533


@pytest.fixture(scope="module")
def h_atom(atoms_ns):
    return make_atoms(atoms_ns, a=H_A, ecut=H_ECUT, s1=H_S, s2=H_S, s3=H_S,
                      px=[0.0], py=[0.0], pz=[0.0], Z=[1.0], f=[1.0])


class TestHAtomReference:
    """Validate the full basis against SimpleDFT.jl's published numbers."""

    def test_grid_size(self, h_atom):
        assert h_atom.g2().shape == (H_S ** 3,)

    def test_cell_volume(self, h_atom):
        assert float(h_atom.volume()) == pytest.approx(H_OMEGA)

    def test_active_count_matches_julia(self, h_atom):
        assert int(h_atom.active().sum()) == H_N_ACTIVE

    def test_g2c_length_matches_active_count(self, h_atom):
        assert h_atom.g2c().shape == (H_N_ACTIVE,)

    def test_dc_component_is_zero(self, h_atom):
        assert float(h_atom.g2()[0]) == pytest.approx(0.0)

    def test_structure_factor_all_ones(self, h_atom):
        Sf = h_atom.sf()
        assert torch.allclose(Sf, torch.ones_like(Sf), rtol=RTOL, atol=ATOL)

    def test_real_space_max_coord(self, h_atom):
        assert float(h_atom.coord_x().max()) == pytest.approx(
            H_A * (H_S - 1) / H_S, rel=1e-4)


# --------------------------------------------------------------------------
# Two-atom cell: the structure factor must sum the per-atom phase kernel.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2(atoms_ns):
    return make_atoms(atoms_ns, a=16.0, ecut=H_ECUT, s1=4, s2=4, s3=4,
                      px=[0.0, 1.4], py=[0.0, 0.0], pz=[0.0, 0.0],
                      Z=[1.0, 1.0], f=[2.0])


class TestMultiAtomStructureFactor:

    def test_sf_sums_per_atom_phases(self, h2):
        n1, n2, n3 = h2.freq_x(), h2.freq_y(), h2.freq_z()
        c = torch.tensor(2 * PI / 16.0)
        z = torch.tensor(0.0)
        expected = (h2.structure_factor(n1, n2, n3, c, z, z, z)
                    + h2.structure_factor(n1, n2, n3, c,
                                          torch.tensor(1.4), z, z))
        assert torch.allclose(h2.sf(), expected, rtol=RTOL, atol=ATOL)

    def test_dc_equals_atom_count(self, h2):
        assert h2.sf()[0].real == pytest.approx(2.0)
