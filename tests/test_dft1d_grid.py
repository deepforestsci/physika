import pytest
import torch
from tests.conftest import exec_phyk

RTOL = 1e-4
ATOL = 1e-5


def numerically_equivalent(actual, expected):
    """True when actual and expected match within tolerance."""
    a = actual.detach().double()
    e = torch.as_tensor(expected).double()
    return torch.allclose(a, e, rtol=RTOL, atol=ATOL)


@pytest.fixture(scope="module")
def grid_ns():
    """Namespace from executing examples/dft1d_grid.phyk."""
    return exec_phyk("dft1d_grid")


@pytest.fixture(scope="module")
def tiny(grid_ns):
    """5-point grid on [-2, 2] (h = 1), the same one used in the .phyk example."""
    return grid_ns["Grid1D"](2.0, 5)


class TestSpacingAndCoordinates:
    def test_h_is_two_L_over_n_minus_1(self, tiny):
        assert float(tiny.h()) == pytest.approx(1.0)

    def test_x_is_uniformly_spaced(self, tiny):
        assert numerically_equivalent(tiny.x(), [-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_other_box_size(self, grid_ns):
        grid = grid_ns["Grid1D"](1.0, 4)
        # h = 2*1 / (4-1) = 2/3
        assert float(grid.h()) == pytest.approx(2.0 / 3.0)
        assert numerically_equivalent(grid.x(), [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])


class TestIdentity:
    def test_identity_is_the_identity_matrix(self, tiny):
        expected = torch.eye(5, dtype=torch.float64)
        assert numerically_equivalent(tiny.identity(), expected)


class TestLaplacian:
    def test_matches_tridiagonal_stencil(self, tiny):
        # -2/h^2 on the diagonal, 1/h^2 on the first off-diagonals, h = 1.
        expected = torch.tensor([
            [-2.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, -2.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, -2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, -2.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, -2.0],
        ])
        assert numerically_equivalent(tiny.laplacian(), expected)

    def test_scales_with_h_squared(self, grid_ns):
        # A box twice as wide with the same point count halves h ->
        # quarters the Laplacian's magnitude.
        small = grid_ns["Grid1D"](2.0, 5)
        big = grid_ns["Grid1D"](4.0, 5)
        ratio = (big.laplacian() / small.laplacian())[0][0]
        assert float(ratio) == pytest.approx(0.25, rel=1e-4)

    def test_symmetric(self, tiny):
        D2 = tiny.laplacian()
        assert numerically_equivalent(D2, D2.T)


class TestKinetic:
    def test_is_half_negative_laplacian(self, tiny):
        assert numerically_equivalent(tiny.kinetic(), -0.5 * tiny.laplacian())
