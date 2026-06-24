import pytest
from tests.conftest import exec_phyk
import torch

r_tol = 1e-04
a_tol = 1e-06


@pytest.fixture(scope="module")
def fft_ns():
    """Execute examples/example_fft.phyk and return its namespace."""
    return exec_phyk("example_fft")


class TestStandardFFT:
    """fft / fft2 / fftn and their inverses."""

    def test_shape(self, fft_ns):
        assert fft_ns["X"].shape == (4, )
        assert fft_ns["X2"].shape == (2, 4)
        assert fft_ns["X3"].shape == (2, 2, 4)

    def test_dtype(self, fft_ns):
        assert fft_ns["X"].dtype == torch.complex64
        assert fft_ns["X2"].dtype == torch.complex64
        assert fft_ns["X3"].dtype == torch.complex64

    def test_matches_torch(self, fft_ns):
        assert torch.allclose(fft_ns["X"],
                              torch.fft.fft(fft_ns["x"]),
                              rtol=r_tol,
                              atol=a_tol)
        assert torch.allclose(fft_ns["X2"],
                              torch.fft.fft2(fft_ns["x2"]),
                              rtol=r_tol,
                              atol=a_tol)
        assert torch.allclose(fft_ns["X3"],
                              torch.fft.fftn(fft_ns["x3"]),
                              rtol=r_tol,
                              atol=a_tol)

    def test_roundtrip(self, fft_ns):
        # ifft inverts fft back to the original signal (cast to compare).
        assert torch.allclose(torch.fft.ifft(fft_ns["X"]),
                              fft_ns["x"],
                              rtol=r_tol,
                              atol=a_tol)
        assert torch.allclose(torch.fft.ifft2(fft_ns["X2"]),
                              fft_ns["x2"].to(torch.complex64),
                              rtol=r_tol,
                              atol=a_tol)
        assert torch.allclose(torch.fft.ifftn(fft_ns["X3"]),
                              fft_ns["x3"].to(torch.complex64),
                              rtol=r_tol,
                              atol=a_tol)


class TestRealFFT:
    """rfft / rfft2 / rfftn: real input, half spectrum on the last axis."""

    def test_shape(self, fft_ns):
        # last axis n becomes n // 2 + 1
        assert fft_ns["Xr"].shape == (5, )
        assert fft_ns["Xr2"].shape == (2, 3)
        assert fft_ns["Xr3"].shape == (2, 2, 3)

    def test_dtype(self, fft_ns):
        assert fft_ns["Xr"].dtype == torch.complex64
        assert fft_ns["Xr2"].dtype == torch.complex64
        assert fft_ns["Xr3"].dtype == torch.complex64

    def test_matches_torch(self, fft_ns):
        assert torch.allclose(fft_ns["Xr"],
                              torch.fft.rfft(fft_ns["xr"]),
                              rtol=r_tol,
                              atol=a_tol)
        assert torch.allclose(fft_ns["Xr2"],
                              torch.fft.rfft2(fft_ns["xr2"]),
                              rtol=r_tol,
                              atol=a_tol)
        assert torch.allclose(fft_ns["Xr3"],
                              torch.fft.rfftn(fft_ns["xr3"]),
                              rtol=r_tol,
                              atol=a_tol)

    def test_roundtrip_recovers_real_input(self, fft_ns):
        # irfft returns a real tensor matching the original real signal.
        rec = torch.fft.irfft(fft_ns["Xr"])
        assert rec.dtype == torch.float32
        assert torch.allclose(rec, fft_ns["xr"], rtol=r_tol, atol=a_tol)
        assert torch.allclose(torch.fft.irfft2(fft_ns["Xr2"]),
                              fft_ns["xr2"],
                              rtol=r_tol,
                              atol=a_tol)
        assert torch.allclose(torch.fft.irfftn(fft_ns["Xr3"]),
                              fft_ns["xr3"],
                              rtol=r_tol,
                              atol=a_tol)


class TestDifferentiableFFT:
    """grad flows back through fft to the real input signal."""

    def test_grad_shape_and_dtype(self, fft_ns):
        g = fft_ns["g"]
        assert g.shape == (4, )
        assert g.dtype == torch.float32

    def test_grad_is_finite(self, fft_ns):
        assert torch.isfinite(fft_ns["g"]).all()

    def test_grad_matches_torch(self, fft_ns):
        s = fft_ns["s"].detach().clone().requires_grad_(True)
        t = fft_ns["t"]
        diff = torch.fft.fft(s) - t
        loss = (torch.abs(diff) * torch.abs(diff)).sum()
        loss.backward()
        assert torch.allclose(fft_ns["g"], s.grad, rtol=r_tol, atol=a_tol)
