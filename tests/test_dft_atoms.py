import pytest
import torch
from tests.conftest import exec_phyk

RTOL = 1e-5
ATOL = 1e-6
PI = 3.141592653589793


def numerically_equivalent(actual, expected):
    """True when actual and expected match within tolerance (real/complex)."""
    a = actual.detach()
    e = torch.as_tensor(expected)
    if a.is_complex() or e.is_complex():
        a, e = a.to(torch.complex128), e.to(torch.complex128)
    else:
        a, e = a.double(), e.double()
    return torch.allclose(a, e, rtol=RTOL, atol=ATOL)


@pytest.fixture(scope="module")
def atoms_ns():
    """Namespace from executing examples/dft_atoms.phyk."""
    return exec_phyk("dft_atoms")


def make_atoms(ns, a, ecut, s1, s2, s3, px, py, pz, Z, f):
    """Construct an Atoms instance from plain Python lists."""
    return ns["Atoms"](a, ecut, s1, s2, s3, len(px), torch.tensor(px),
                       torch.tensor(py), torch.tensor(pz), len(f),
                       torch.tensor(Z), torch.tensor(f))


"""Tiny 2x2x2 cell (a=4, ecut=2, one atom at origin);"""


@pytest.fixture(scope="module")
def tiny(atoms_ns):
    return make_atoms(atoms_ns,
                      a=4.0,
                      ecut=2.0,
                      s1=2,
                      s2=2,
                      s3=2,
                      px=[0.0],
                      py=[0.0],
                      pz=[0.0],
                      Z=[1.0],
                      f=[1.0])


class TestGridIndexing:
    """Flat index unravels into per-axis grid indices m1, m2, m3."""

    def test_flat_index_is_arange(self, tiny):
        # flat_index() returns 0..k-1
        assert tiny.flat_index().tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grid_indices_unravel_c_order(self, tiny):
        # m1 changes slowest (stride 4), m3 fastest (stride 1)
        assert tiny.m1().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert tiny.m2().tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
        assert tiny.m3().tolist() == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_fold_freq_wraps_upper_half(self, tiny):
        # indices strictly above s/2 fold negative.
        folded = tiny.fold_freq(torch.tensor([0., 1., 2., 3.]), 4)
        assert folded.tolist() == [0, 1, 2, -1]

    def test_freqs_are_folded_grid_indices(self, tiny):
        # on a size-2 axis nothing folds, so n_i == m_i
        assert tiny.freq_x().tolist() == tiny.m1().tolist()
        assert tiny.freq_y().tolist() == tiny.m2().tolist()
        assert tiny.freq_z().tolist() == tiny.m3().tolist()


class TestRealSpace:
    """Cell volume and real-space grid sampling."""

    def test_cell_volume_is_a_cubed(self, tiny):
        # volume = a^3
        assert float(tiny.volume()) == pytest.approx(64.0)

    def test_sample_coord_spacing(self, tiny):
        # spacing a/s = 4/2 = 2 per grid step
        assert numerically_equivalent(
            tiny.sample_coord(torch.tensor([0., 1., 2., 3.]), 2),
            [0., 2., 4., 6.])

    def test_coord_max_is_last_grid_point(self, tiny):
        # last grid point at a*(s-1)/s
        assert float(tiny.coord_x().max()) == pytest.approx(2.0)


class TestReciprocalSpace:
    """Reciprocal-space |G|^2, the cutoff mask, and the active basis."""

    def test_recip_scale_is_two_pi_over_a(self, tiny):
        # reciprocal scale = 2*pi/a
        assert float(tiny.recip_scale()) == pytest.approx(2 * PI / 4)

    def test_g2_is_scaled_sum_of_squares(self, tiny):
        # |G|^2 = c^2 * (n1^2 + n2^2 + n3^2), c = 2*pi/a
        c2 = (2 * PI / 4)**2
        ss = [0, 1, 1, 2, 1, 2, 2, 3]
        assert numerically_equivalent(tiny.g2(), [c2 * s for s in ss])

    def test_g2_dc_component_is_zero(self, tiny):
        # the G=0 term vanishes
        assert float(tiny.g2()[0]) == pytest.approx(0.0)

    def test_g_components_reproduce_g2(self, tiny):
        # gx^2 + gy^2 + gz^2 == g2
        gx, gy, gz = tiny.gx(), tiny.gy(), tiny.gz()
        assert numerically_equivalent(gx * gx + gy * gy + gz * gz, tiny.g2())

    def test_active_mask_selects_within_cutoff(self, tiny):
        # active where |G|^2 <= 2*ecut
        active = tiny.active()
        assert active.dtype == torch.bool
        assert active.tolist() == [
            True, True, True, False, True, False, False, False
        ]

    def test_g2c_keeps_active_entries(self, tiny):
        # g2c holds only the active |G|^2 values
        c2 = (2 * PI / 4)**2
        assert numerically_equivalent(tiny.g2c(), [0.0, c2, c2, c2])


class TestStructureFactor:
    """Single-atom structure factor, the phase kernel exp(-i G.r)."""

    def test_atom_at_origin_gives_all_ones(self, tiny):
        # atom at the origin -> Sf == 1 everywhere
        Sf = tiny.sf()
        assert Sf.shape == (8, )
        assert numerically_equivalent(Sf, torch.ones_like(Sf))

    def test_phase_matches_reference(self, tiny):
        # Sf == exp(-i * phase) for a shifted atom
        n1, n2, n3 = torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.0)
        c, px, py, pz = 0.5, 1.0, 0.5, 0.0
        out = tiny.structure_factor(n1, n2, n3, torch.tensor(c),
                                    torch.tensor(px), torch.tensor(py),
                                    torch.tensor(pz))
        phase = c * (1.0 * px + 2.0 * py + 0.0 * pz)
        expected = torch.exp(torch.tensor(-1j * phase, dtype=torch.complex64))
        assert numerically_equivalent(out, expected)


"""Folding cell (4x1x1, a=2*pi): upper-half frequencies fold negative."""


@pytest.fixture(scope="module")
def folding(atoms_ns):
    # length-4 axis folds; a=2*pi makes recip scale 1, so |G|^2 = n^2
    return make_atoms(atoms_ns,
                      a=2 * PI,
                      ecut=1.0,
                      s1=4,
                      s2=1,
                      s3=1,
                      px=[0.0],
                      py=[0.0],
                      pz=[0.0],
                      Z=[1.0],
                      f=[1.0])


class TestFolding:
    """Folding on a real grid feeds |G|^2 and the active mask."""

    def test_freq_folds_upper_half(self, folding):
        # the last grid point folds to -1, not 3
        assert folding.freq_x().tolist() == [0, 1, 2, -1]

    def test_folded_g2_uses_signed_frequency(self, folding):
        # |G|^2 squares the folded frequency: index 3 -> (-1)^2 = 1, not 3^2
        assert numerically_equivalent(folding.g2(), [0.0, 1.0, 4.0, 1.0])

    def test_active_count_with_folding(self, folding):
        # cutoff keeps 0,1,3; index 2 (|G|^2 = 4) is excluded
        assert folding.active().tolist() == [True, True, False, True]
        assert numerically_equivalent(folding.g2c(), [0.0, 1.0, 1.0])


"""Two-atom cell: the structure factor must sum the per-atom phase kernel."""


@pytest.fixture(scope="module")
def h2(atoms_ns):
    # two nuclei, at the origin and at (1.4, 0, 0)
    return make_atoms(atoms_ns,
                      a=16.0,
                      ecut=16.0,
                      s1=4,
                      s2=4,
                      s3=4,
                      px=[0.0, 1.4],
                      py=[0.0, 0.0],
                      pz=[0.0, 0.0],
                      Z=[1.0, 1.0],
                      f=[2.0])


class TestMultiAtomStructureFactor:
    """Structure factor sums the per-atom phase kernel."""

    def test_sf_sums_per_atom_phases(self, h2):
        # Sf equals the sum of each atom's phase kernel
        n1, n2, n3 = h2.freq_x(), h2.freq_y(), h2.freq_z()
        c = torch.tensor(2 * PI / 16.0)
        z = torch.tensor(0.0)
        expected = (
            h2.structure_factor(n1, n2, n3, c, z, z, z) +
            h2.structure_factor(n1, n2, n3, c, torch.tensor(1.4), z, z))
        assert numerically_equivalent(h2.sf(), expected)

    def test_dc_equals_atom_count(self, h2):
        # every atom contributes exp(0)=1 at G=0, so Sf[0] == Natoms
        assert h2.sf()[0].real == pytest.approx(2.0)
