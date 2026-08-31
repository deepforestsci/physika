import pytest
from tests.conftest import exec_phyk
import torch

r_tol = 1e-05
a_tol = 1e-06

# 2x2x2 grid, cubic cell a=2 -> Omega=8;
S = (2, 2, 2)
N = 8
A_SIDE = 2.0
ECUT = 5.0
OMEGA = 8.0
PI = 3.141592653589793


ACTIVE = torch.tensor([True, True, True, False, True, False, False, False])
G2 = PI**2 * torch.tensor([0., 1., 1., 2., 1., 2., 2., 3.])
G2C = torch.masked_select(G2, ACTIVE)

W_FULL = torch.tensor(
    [1 + 0j, 2 + 1j, 0 + 3j, -1 + 0j, 2 + 0j, 1 - 1j, 0 + 0j, 3 + 2j],
    dtype=torch.complex64)
W_ACT = torch.tensor([0 + 0j, 1 + 2j, -1 + 0j, 0 + 3j], dtype=torch.complex64)
W_MAT = torch.stack([W_FULL, W_FULL.flip(0)],
                    dim=1)  # 8 plane waves x 2 states


@pytest.fixture(scope="module")
def ops():
    """Execute examples/dft_operators.phyk and return its namespace."""
    return exec_phyk("dft_operators")


@pytest.fixture(scope="module")
def atoms(ops):
    """Atoms on the 2x2x2 smoke grid; reproduces G2/ACTIVE above."""
    return ops["Atoms"](A_SIDE, ECUT, 2, 2, 2, 1, torch.tensor([0.0]),
                        torch.tensor([0.0]), torch.tensor([0.0]), 1,
                        torch.tensor([1.0]), torch.tensor([1.0]))


class TestFixtureMatchesReference:
    """The hand-computed constants must be what Atoms actually builds."""

    def test_volume(self, atoms):
        # Omega = a^3 = 2^3
        assert atoms.volume().item() == pytest.approx(OMEGA)

    def test_g2(self, atoms):
        # |G|^2 on the full grid matches the hand-computed table
        assert torch.allclose(atoms.g2().float(), G2, rtol=r_tol, atol=a_tol)

    def test_active(self, atoms):
        # the cutoff keeps flat indices 0, 1, 2, 4
        assert torch.equal(atoms.active(), ACTIVE)

    def test_g2c(self, atoms):
        # g2c is |G|^2 restricted to the active set
        assert torch.allclose(atoms.g2c().float(), G2C, rtol=r_tol, atol=a_tol)


class TestOverlap:
    """Overlap operator: multiplication by the cell volume Omega."""

    def test_op_O_scales_by_volume(self, ops, atoms):
        # O(W) = Omega * W on every plane wave
        out = ops["op_O"](atoms, W_FULL)
        assert torch.allclose(out, OMEGA * W_FULL, rtol=r_tol, atol=a_tol)


class TestLaplacian:
    """Laplacian and its inverse in reciprocal space."""

    def test_op_L_active_uses_G2c(self, ops, atoms):
        # active-length input picks G2c, not the full G2
        out = ops["op_L"](atoms, W_ACT)
        assert torch.allclose(out,
                              -OMEGA * G2C * W_ACT,
                              rtol=r_tol,
                              atol=a_tol)

    def test_op_L_full_uses_G2(self, ops, atoms):
        # full-grid input picks the full G2
        out = ops["op_L"](atoms, W_FULL)
        assert torch.allclose(out,
                              -OMEGA * G2 * W_FULL,
                              rtol=r_tol,
                              atol=a_tol)

    def test_op_Linv_inverts_op_L_off_DC(self, ops, atoms):
        # Linv divides by the full G2, so it recovers W wherever G != 0
        out = ops["op_Linv"](atoms, ops["op_L"](atoms, W_FULL))
        expected = W_FULL.clone()
        expected[G2 == 0] = 0
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Linv_zeroes_DC(self, ops, atoms):
        # the G=0 mode has no inverse Laplacian; by convention it is 0
        assert ops["op_Linv"](atoms, W_FULL)[0] == 0

    def test_op_Linv_no_nan_or_inf(self, ops, atoms):
        # dividing by G2 must not blow up at the G=0 entry
        out = ops["op_Linv"](atoms, W_FULL)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()


class TestTransforms:
    """Transforms between real space and the full reciprocal grid."""

    def test_op_J_is_3d_fft(self, ops, atoms):
        # op_J is a 3-D fftn over the reshaped grid, divided by n
        out = ops["op_J"](atoms, W_FULL)
        expected = torch.fft.fftn(W_FULL.reshape(S)).reshape(-1) / N
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_I_is_3d_ifft(self, ops, atoms):
        # op_I is a 3-D ifftn over the reshaped grid, scaled by n
        out = ops["op_I"](atoms, W_FULL)
        expected = torch.fft.ifftn(W_FULL.reshape(S)).reshape(-1) * N
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    @pytest.mark.parametrize("vec", [
        [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j],
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
        [1 + 1j, 2 - 1j, 3j, -1 + 0j, 2 + 0j, 1 - 1j, 0j, 3 + 2j],
    ])
    def test_op_I_inverts_op_J(self, ops, atoms, vec):
        # op_J then op_I is the identity on the full grid
        W = torch.tensor(vec, dtype=torch.complex64)
        out = ops["op_I"](atoms, ops["op_J"](atoms, W))
        assert torch.allclose(out, W, rtol=r_tol, atol=a_tol)

    def test_op_J_dc_is_mean(self, ops, atoms):
        # the G=0 coefficient is the mean of the real-space values
        out = ops["op_J"](atoms, W_FULL)
        assert torch.allclose(out[0], W_FULL.mean(), rtol=r_tol, atol=a_tol)


class TestEmbed:
    """op_I on active-basis input scatters back to the full grid first."""

    def test_op_I_active_returns_full_grid(self, ops, atoms):
        # active-basis input still comes back on the full grid
        assert ops["op_I"](atoms, W_ACT).shape == (N, )

    def test_op_I_active_equals_op_I_of_embedded(self, ops, atoms):
        # embedding by hand first gives the same result
        scaffold = torch.zeros(N, dtype=torch.complex64).masked_scatter(
            ACTIVE, W_ACT)
        assert torch.allclose(ops["op_I"](atoms, W_ACT),
                              ops["op_I"](atoms, scaffold),
                              rtol=r_tol,
                              atol=a_tol)

    def test_mask_embed_roundtrips_through_mask_select(self, ops):
        # selecting the active entries back returns the input
        out = ops["mask_embed"](W_ACT, ACTIVE, N)
        assert torch.equal(torch.masked_select(out, ACTIVE), W_ACT)


class TestAdjoints:
    """Adjoint transforms op_Idag and op_Jdag."""

    def test_op_Idag_gathers_active(self, ops, atoms):
        # Idag is a forward fftn restricted to the active set
        out = ops["op_Idag"](atoms, W_FULL)
        expected = torch.masked_select(
            torch.fft.fftn(W_FULL.reshape(S)).reshape(-1), ACTIVE)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Idag_length(self, ops, atoms):
        # output lives on the active basis: 4 plane waves here
        assert ops["op_Idag"](atoms, W_FULL).shape == (4, )

    def test_op_Jdag_matches_reference(self, ops, atoms):
        # Jdag reduces to a plain ifftn with no 1/n factor
        out = ops["op_Jdag"](atoms, W_FULL)
        expected = torch.fft.ifftn(W_FULL.reshape(S)).reshape(-1)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Jdag_is_adjoint_of_op_J(self, ops, atoms):
        # <op_J(a), b> == <a, op_Jdag(b)> for the standard inner product
        a = W_FULL
        b = torch.tensor([2 + 1j, 0j, 1 - 1j, 3 + 0j, 0j, 1 + 1j, 2j, 1 + 0j],
                         dtype=torch.complex64)
        lhs = (ops["op_J"](atoms, a).conj() * b).sum()
        rhs = (a.conj() * ops["op_Jdag"](atoms, b)).sum()
        assert torch.allclose(lhs, rhs, rtol=r_tol, atol=a_tol)


class TestMatrixForms:
    """Matrices are (plane waves x states): each column is one state."""

    def test_op_J_mat_is_columnwise_op_J(self, ops, atoms):
        # op_J_mat applies op_J to each column independently
        out = ops["op_J_mat"](atoms, W_MAT)
        cols = [W_MAT[:, i] for i in range(W_MAT.shape[1])]
        expected = torch.stack(
            [torch.fft.fftn(c.reshape(S)).reshape(-1) / N for c in cols],
            dim=1)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_I_mat_inverts_op_J_mat(self, ops, atoms):
        # the matrix forms round-trip like the vector ones
        out = ops["op_I_mat"](atoms, ops["op_J_mat"](atoms, W_MAT))
        assert torch.allclose(out, W_MAT, rtol=r_tol, atol=a_tol)

    def test_op_L_mat_matches_columnwise(self, ops, atoms):
        # op_L_mat scales each column by -Omega * G2c
        W2 = torch.stack([W_ACT, 2 * W_ACT], dim=1)  # 4 x 2
        out = ops["op_L_mat"](atoms, W2)
        expected = -OMEGA * G2C.unsqueeze(1) * W2  # G2C down each column
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Linv_mat_matches_columnwise(self, ops, atoms):
        # each column matches the vector op_Linv
        out = ops["op_Linv_mat"](atoms, W_MAT)
        assert torch.allclose(out[:, 0],
                              ops["op_Linv"](atoms, W_MAT[:, 0]),
                              rtol=r_tol,
                              atol=a_tol)

    def test_op_Idag_mat_shape_and_values(self, ops, atoms):
        # op_Idag_mat applies op_Idag to each column
        out = ops["op_Idag_mat"](atoms, W_MAT)
        col0 = ops["op_Idag"](atoms, W_MAT[:, 0])
        assert out.shape == (4, 2)  # (active plane waves x states)
        assert torch.allclose(out[:, 0], col0, rtol=r_tol, atol=a_tol)
