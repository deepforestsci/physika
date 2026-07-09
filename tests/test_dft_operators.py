import pytest
from tests.conftest import exec_phyk
import torch

r_tol = 1e-05
a_tol = 1e-06

# 2x2x2 grid; flat index = 4*m1 + 2*m2 + m3.
S = (2, 2, 2)
N = 8
OMEGA = 8.0
PI = 3.141592653589793

# active set for ecut=5: G2 = pi^2*(m1^2+m2^2+m3^2) <= 10 at flat
# indices 0, 1, 2, 4.
ACTIVE = torch.tensor(
    [True, True, True, False, True, False, False, False])
G2 = PI**2 * torch.tensor([0., 1., 1., 2., 1., 2., 2., 3.])
G2C = torch.masked_select(G2, ACTIVE)

W_FULL = torch.tensor(
    [1 + 0j, 2 + 1j, 0 + 3j, -1 + 0j, 2 + 0j, 1 - 1j, 0 + 0j, 3 + 2j],
    dtype=torch.complex64)
W_ACT = torch.tensor([0 + 0j, 1 + 2j, -1 + 0j, 0 + 3j],
                     dtype=torch.complex64)
W_MAT = torch.stack([W_FULL, W_FULL.flip(0)])  # 2 states x 8

@pytest.fixture(scope="module")
def ops():
    """Execute examples/dft_operators.phyk and return its namespace."""
    return exec_phyk("dft_operators")


class TestOverlap:

    def test_op_O_scales_by_volume(self, ops):
        out = ops["op_O"](W_FULL, OMEGA)
        assert torch.allclose(out, OMEGA * W_FULL, rtol=r_tol, atol=a_tol)


class TestLaplacian:

    def test_op_L_matches_reference(self, ops):
        out = ops["op_L"](W_ACT, G2C, OMEGA)
        expected = -OMEGA * G2C * W_ACT
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Linv_inverts_op_L_off_DC(self, ops):
        # Linv(L(W)) recovers W wherever G != 0; the DC entry is zeroed.
        out = ops["op_Linv"](ops["op_L"](W_ACT, G2C, OMEGA), G2C, OMEGA)
        expected = W_ACT.clone()
        expected[G2C == 0] = 0
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Linv_zeroes_DC(self, ops):
        out = ops["op_Linv"](W_ACT, G2C, OMEGA)
        assert out[0] == 0

    def test_op_Linv_no_nan_or_inf(self, ops):
        out = ops["op_Linv"](W_ACT, G2C, OMEGA)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()


class TestTransforms:

    def test_op_J_is_3d_fft(self, ops):
        # op_J must equal fftn on the reshaped grid, divided by n; a 1-D
        # fft of the flat vector gives different numbers.
        out = ops["op_J"](W_FULL, *S)
        expected = torch.fft.fftn(W_FULL.reshape(S)).reshape(-1) / N
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_I_is_3d_ifft(self, ops):
        out = ops["op_I"](W_FULL, *S)
        expected = torch.fft.ifftn(W_FULL.reshape(S)).reshape(-1) * N
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    @pytest.mark.parametrize("vec", [
        [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j],
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
        [1 + 1j, 2 - 1j, 3j, -1 + 0j, 2 + 0j, 1 - 1j, 0j, 3 + 2j],
    ])
    def test_op_I_inverts_op_J(self, ops, vec):
        W = torch.tensor(vec, dtype=torch.complex64)
        out = ops["op_I"](ops["op_J"](W, *S), *S)
        assert torch.allclose(out, W, rtol=r_tol, atol=a_tol)

    def test_op_J_dc_is_mean(self, ops):
        out = ops["op_J"](W_FULL, *S)
        assert torch.allclose(out[0], W_FULL.mean(), rtol=r_tol, atol=a_tol)


class TestAdjoints:

    def test_op_Idag_gathers_active(self, ops):
        out = ops["op_Idag"](W_FULL, ACTIVE, *S)
        expected = torch.masked_select(
            torch.fft.fftn(W_FULL.reshape(S)).reshape(-1), ACTIVE)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Idag_length(self, ops):
        assert ops["op_Idag"](W_FULL, ACTIVE, *S).shape == (4, )

    def test_op_Jdag_matches_reference(self, ops):
        out = ops["op_Jdag"](W_FULL, *S)
        expected = torch.fft.ifftn(W_FULL.reshape(S)).reshape(-1)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_Jdag_is_adjoint_of_op_J(self, ops):
        # <op_J(a), b> == <a, op_Jdag(b)> for the standard inner product.
        a = W_FULL
        b = torch.tensor([2 + 1j, 0j, 1 - 1j, 3 + 0j, 0j, 1 + 1j, 2j, 1 + 0j],
                         dtype=torch.complex64)
        lhs = (ops["op_J"](a, *S).conj() * b).sum()
        rhs = (a.conj() * ops["op_Jdag"](b, *S)).sum()
        assert torch.allclose(lhs, rhs, rtol=r_tol, atol=a_tol)


class TestMatrixForms:

    def test_op_J_mat_is_rowwise_op_J(self, ops):
        out = ops["op_J_mat"](W_MAT, *S)
        expected = torch.stack(
            [torch.fft.fftn(row.reshape(S)).reshape(-1) / N
             for row in W_MAT])
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_op_I_mat_inverts_op_J_mat(self, ops):
        out = ops["op_I_mat"](ops["op_J_mat"](W_MAT, *S), *S)
        assert torch.allclose(out, W_MAT, rtol=r_tol, atol=a_tol)

    def test_op_L_mat_matches_rowwise(self, ops):
        W2 = torch.stack([W_ACT, 2 * W_ACT])
        out = ops["op_L_mat"](W2, G2C, OMEGA)
        assert torch.allclose(out, -OMEGA * G2C * W2, rtol=r_tol, atol=a_tol)

    def test_op_Idag_mat_shape_and_values(self, ops):
        out = ops["op_Idag_mat"](W_MAT, ACTIVE, *S)
        row0 = ops["op_Idag"](W_MAT[0], ACTIVE, *S)
        assert out.shape == (2, 4)
        assert torch.allclose(out[0], row0, rtol=r_tol, atol=a_tol)
