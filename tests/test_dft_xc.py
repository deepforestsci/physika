import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06

PI = 3.141592653589793

# Representative electron densities used in the exchange-correlation tests.
N_TEST = torch.tensor([0.5 + 0j, 1.0 + 0j, 2.0 + 0j, 4.0 + 0j],
                      dtype=torch.complex64)


@pytest.fixture(scope="module")
def xc_ns():
    """Execute examples/dft_xc.phyk"""
    return exec_phyk("dft_xc")


def ref_wigner_seitz_radius(n):
    """Reference implementation of the Wigner-Seitz radius."""
    return (3.0 / (4.0 * PI * n))**(1.0 / 3.0)


class TestSlaterExchange:

    def test_matches_reference(self, xc_ns):
        # Compare against an independent reference implementation.
        rs = ref_wigner_seitz_radius(N_TEST)
        cx = (-3.0 / 4.0) * (3.0 / (2.0 * PI))**(2.0 / 3.0)
        ex_expected = cx / rs
        vx_expected = (4.0 / 3.0) * ex_expected

        out = xc_ns["lda_x"](N_TEST)
        assert torch.allclose(out[0], ex_expected, rtol=r_tol, atol=a_tol)
        assert torch.allclose(out[1], vx_expected, rtol=r_tol, atol=a_tol)

    def test_potential_is_four_thirds_energy(self, xc_ns):
        # Verify the Slater relation vx = 4/3 ex.
        ex, vx = xc_ns["lda_x"](N_TEST)
        assert torch.allclose(vx, (4.0 / 3.0) * ex, rtol=r_tol, atol=a_tol)

    def test_energy_density_is_negative(self, xc_ns):
        # Exchange energy is negative for positive electron densities.
        ex, _ = xc_ns["lda_x"](N_TEST)
        assert (ex.real < 0).all()

    def test_magnitude_grows_with_density(self, xc_ns):
        # The magnitude of the exchange energy increases with density.
        ex, _ = xc_ns["lda_x"](N_TEST)
        assert (ex.real[1:] < ex.real[:-1]).all()

    def test_output_shape(self, xc_ns):
        # Verify the output contains ex and vx for each density value.
        out = xc_ns["lda_x"](N_TEST)
        assert out.shape == (2, N_TEST.shape[0])


class TestChachiyoCorrelation:

    def test_matches_reference(self, xc_ns):
        # Compare against an independent reference implementation.
        rs = ref_wigner_seitz_radius(N_TEST)
        a = -0.01554535
        b = 20.4562557
        ec_expected = a * torch.log(1.0 + b / rs + b / rs**2.0)
        vc_expected = ec_expected + a * b * (2.0 +
                                             rs) / (3.0 *
                                                    (b + b * rs + rs**2.0))

        out = xc_ns["lda_c_chachiyo"](N_TEST)
        assert torch.allclose(out[0], ec_expected, rtol=r_tol, atol=a_tol)
        assert torch.allclose(out[1], vc_expected, rtol=r_tol, atol=a_tol)

    def test_energy_density_is_negative(self, xc_ns):
        # Correlation energy is negative for positive electron densities.
        ec, _ = xc_ns["lda_c_chachiyo"](N_TEST)
        assert (ec.real < 0).all()
