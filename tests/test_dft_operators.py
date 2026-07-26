import pytest
from tests.conftest import exec_phyk
import torch

r_tol = 1e-05
a_tol = 1e-06

# 2x2x2 grid, cubic cell a=2 -> Omega=8; flat index = 4*m1 + 2*m2 + m3.
S = (2, 2, 2)
N = 8
A_SIDE = 2.0
ECUT = 5.0
OMEGA = 8.0
PI = 3.141592653589793

# active set for ecut=5: G2 = pi^2*(m1^2+m2^2+m3^2) <= 10 at flat
# indices 0, 1, 2, 4.
ACTIVE = torch.tensor([True, True, True, False, True, False, False, False])
G2 = PI**2 * torch.tensor([0., 1., 1., 2., 1., 2., 2., 3.])
G2C = torch.masked_select(G2, ACTIVE)

W_FULL = torch.tensor(
    [1 + 0j, 2 + 1j, 0 + 3j, -1 + 0j, 2 + 0j, 1 - 1j, 0 + 0j, 3 + 2j],
    dtype=torch.complex64)
W_ACT = torch.tensor([0 + 0j, 1 + 2j, -1 + 0j, 0 + 3j], dtype=torch.complex64)
W_MAT = torch.stack([W_FULL, W_FULL.flip(0)])  # 2 states x 8


@pytest.fixture(scope="module")
def ops():
    """Execute examples/dft_operators.phyk and return its namespace."""
    return exec_phyk("dft_operators")


@pytest.fixture(scope="module")
def atoms(ops):
    """Atoms on the 2x2x2 smoke grid; reproduces G2/ACTIVE above."""
    return ops["Atoms"](A_SIDE, ECUT, 2, 2, 2, 0.0, 0.0, 0.0, 1, 1,
                        torch.tensor([1.0]), torch.tensor([1.0]))


class TestFixtureMatchesReference:
    """The hand-computed constants must be what Atoms actually builds."""

    def test_volume(self, atoms):
        assert atoms.volume().item() == pytest.approx(OMEGA)

    def test_g2(self, atoms):
        assert torch.allclose(atoms.g2().float(), G2, rtol=r_tol, atol=a_tol)

    def test_active(self, atoms):
        assert torch.equal(atoms.active(), ACTIVE)

    def test_g2c(self, atoms):
        assert torch.allclose(atoms.g2c().float(), G2C, rtol=r_tol, atol=a_tol)


class TestOverlap:

    def test_op_O_scales_by_volume(self, ops, atoms):
        out = ops["op_O"](atoms, W_FULL)
        assert torch.allclose(out, OMEGA * W_FULL, rtol=r_tol, atol=a_tol)


class TestLaplacian:

    def test_op_L_active_uses_G2c(self, ops, atoms):
        out = ops["op_L"](atoms, W_ACT)
        assert torch.allclose(out,
                              -OMEGA * G2C * W_ACT,
                              rtol=r_tol,
                              atol=a_tol)

    def test_op_L_full_uses_G2(self, ops, atoms):
        out = ops["op_L"](atoms, W_FULL)
        assert torch.allclose(out,
                              -OMEGA * G2 * W_FULL,
                              rtol=r_tol,
                              atol=a_tol)

    def test_op_Linv_inverts_op_L_off_DC(self, ops, atoms):
        # op_Linv always divides by the full G2, as in SimpleDFT, so it takes
        # full-grid input only. Linv(L(W)) recovers W wherever G != 0.
        out = ops["op_Linv"](atoms, ops["op_L"](atoms, W_FULL))
        expected = W_FULL.clone()
        expected[G2 == 0] = 0
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Linv_zeroes_DC(self, ops, atoms):
        assert ops["op_Linv"](atoms, W_FULL)[0] == 0

    def test_op_Linv_no_nan_or_inf(self, ops, atoms):
        out = ops["op_Linv"](atoms, W_FULL)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()


class TestTransforms:

    def test_op_J_is_3d_fft(self, ops, atoms):
        # op_J must equal fftn on the reshaped grid, divided by n; a 1-D
        # fft of the flat vector gives different numbers.
        out = ops["op_J"](atoms, W_FULL)
        expected = torch.fft.fftn(W_FULL.reshape(S)).reshape(-1) / N
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_I_is_3d_ifft(self, ops, atoms):
        out = ops["op_I"](atoms, W_FULL)
        expected = torch.fft.ifftn(W_FULL.reshape(S)).reshape(-1) * N
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    @pytest.mark.parametrize("vec", [
        [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j],
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
        [1 + 1j, 2 - 1j, 3j, -1 + 0j, 2 + 0j, 1 - 1j, 0j, 3 + 2j],
    ])
    def test_op_I_inverts_op_J(self, ops, atoms, vec):
        W = torch.tensor(vec, dtype=torch.complex64)
        out = ops["op_I"](atoms, ops["op_J"](atoms, W))
        assert torch.allclose(out, W, rtol=r_tol, atol=a_tol)

    def test_op_J_dc_is_mean(self, ops, atoms):
        out = ops["op_J"](atoms, W_FULL)
        assert torch.allclose(out[0], W_FULL.mean(), rtol=r_tol, atol=a_tol)


class TestEmbed:
    """op_I on active-basis input scatters back to the full grid first."""

    def test_op_I_active_returns_full_grid(self, ops, atoms):
        assert ops["op_I"](atoms, W_ACT).shape == (N, )

    def test_op_I_active_equals_op_I_of_embedded(self, ops, atoms):
        scaffold = torch.zeros(N, dtype=torch.complex64).masked_scatter(
            ACTIVE, W_ACT)
        assert torch.allclose(ops["op_I"](atoms, W_ACT),
                              ops["op_I"](atoms, scaffold),
                              rtol=r_tol,
                              atol=a_tol)

    def test_mask_embed_roundtrips_through_mask_select(self, ops):
        out = ops["mask_embed"](W_ACT, ACTIVE, N)
        assert torch.equal(torch.masked_select(out, ACTIVE), W_ACT)


class TestAdjoints:

    def test_op_Idag_gathers_active(self, ops, atoms):
        out = ops["op_Idag"](atoms, W_FULL)
        expected = torch.masked_select(
            torch.fft.fftn(W_FULL.reshape(S)).reshape(-1), ACTIVE)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Idag_length(self, ops, atoms):
        assert ops["op_Idag"](atoms, W_FULL).shape == (4, )

    def test_op_Jdag_matches_reference(self, ops, atoms):
        out = ops["op_Jdag"](atoms, W_FULL)
        expected = torch.fft.ifftn(W_FULL.reshape(S)).reshape(-1)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Jdag_is_adjoint_of_op_J(self, ops, atoms):
        # <op_J(a), b> == <a, op_Jdag(b)> for the standard inner product.
        a = W_FULL
        b = torch.tensor([2 + 1j, 0j, 1 - 1j, 3 + 0j, 0j, 1 + 1j, 2j, 1 + 0j],
                         dtype=torch.complex64)
        lhs = (ops["op_J"](atoms, a).conj() * b).sum()
        rhs = (a.conj() * ops["op_Jdag"](atoms, b)).sum()
        assert torch.allclose(lhs, rhs, rtol=r_tol, atol=a_tol)


class TestMatrixForms:

    def test_op_J_mat_is_rowwise_op_J(self, ops, atoms):
        out = ops["op_J_mat"](atoms, W_MAT)
        expected = torch.stack(
            [torch.fft.fftn(row.reshape(S)).reshape(-1) / N for row in W_MAT])
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_I_mat_inverts_op_J_mat(self, ops, atoms):
        out = ops["op_I_mat"](atoms, ops["op_J_mat"](atoms, W_MAT))
        assert torch.allclose(out, W_MAT, rtol=r_tol, atol=a_tol)

    def test_op_L_mat_matches_rowwise(self, ops, atoms):
        W2 = torch.stack([W_ACT, 2 * W_ACT])
        out = ops["op_L_mat"](atoms, W2)
        assert torch.allclose(out, -OMEGA * G2C * W2, rtol=r_tol, atol=a_tol)

    def test_op_Linv_mat_matches_rowwise(self, ops, atoms):
        out = ops["op_Linv_mat"](atoms, W_MAT)
        assert torch.allclose(out[0],
                              ops["op_Linv"](atoms, W_MAT[0]),
                              rtol=r_tol,
                              atol=a_tol)

    def test_op_Idag_mat_shape_and_values(self, ops, atoms):
        out = ops["op_Idag_mat"](atoms, W_MAT)
        row0 = ops["op_Idag"](atoms, W_MAT[0])
        assert out.shape == (2, 4)
        assert torch.allclose(out[0], row0, rtol=r_tol, atol=a_tol)
