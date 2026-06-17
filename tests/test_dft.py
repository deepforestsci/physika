import pytest
from tests.conftest import exec_phyk
import torch

r_tol = 1e-05
a_tol = 1e-06


@pytest.fixture(scope="module")
def fft_ns():
    """Execute examples/fft_dft_operators.phyk and return its namespace."""
    return exec_phyk("fft_dft_operators")


class TestDFTTransforms:
    """Tests for the plane-wave DFT operators op_J / op_I (fft / ifft + reshape)."""

    @pytest.mark.parametrize("var", ["field", "spectrum", "recovered"])
    def test_shape(self, fft_ns, var):
        assert fft_ns[var].shape == (8, )

    @pytest.mark.parametrize("var", ["field", "spectrum", "recovered"])
    def test_dtype(self, fft_ns, var):
        assert fft_ns[var].dtype == torch.complex64

    def test_roundtrip_recovers_field(self, fft_ns):
        # op_I is the exact inverse of op_J: op_I(op_J(W)) == W.
        assert torch.allclose(fft_ns["recovered"], fft_ns["field"],
                              rtol=r_tol, atol=a_tol)

    def test_opJ_is_genuine_3d_transform(self, fft_ns):
        # The decisive test: op_J must equal a true 3-D FFT of the field
        # reshaped onto the 2x2x2 grid, divided by n.  A 1-D FFT of the flat
        # vector would give different numbers -> this proves reshape works.
        field = fft_ns["field"]
        n = field.numel()
        expected = torch.fft.fftn(field.reshape(2, 2, 2)).reshape(-1) / n
        assert torch.allclose(fft_ns["spectrum"], expected,
                              rtol=r_tol, atol=a_tol)

    def test_opJ_dc_is_mean(self, fft_ns):
        # Because op_J divides by n, bin 0 (DC) is the mean of the field.
        field = fft_ns["field"]
        assert torch.allclose(fft_ns["spectrum"][0], field.sum() / field.numel(),
                              rtol=r_tol, atol=a_tol)

    @pytest.mark.parametrize("vec", [
        [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j],
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
        [1 + 1j, 2 - 1j, 3j, -1 + 0j, 2 + 0j, 1 - 1j, 0j, 3 + 2j],
    ])
    def test_opI_inverts_opJ(self, fft_ns, vec):
        op_J = fft_ns["op_J"]
        op_I = fft_ns["op_I"]
        W = torch.tensor(vec, dtype=torch.complex64)
        out = op_I(op_J(W, 2, 2, 2), 2, 2, 2)
        assert torch.allclose(out, W, rtol=r_tol, atol=a_tol)
