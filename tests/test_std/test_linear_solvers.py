from tests.conftest import exec_phyk, run_phyk, type_errors
import pytest
import torch

r_tol = 1e-02


@pytest.fixture(scope="module")
def linear_solvers_ns():
    """
    Execute Std/linear_solvers.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("linear_solvers")


class TestGaussianElimination:
    """Test suites for ``gaussian_solve`` method"""

    def test_basic_import(self):
        # Test importing gaussian_solver gives no errors.
        src = ("from Std.linear_solvers import gaussian_solve\n")
        errors = type_errors(src)
        assert errors == []

    def test_gaussian_correctness(self, linear_solvers_ns):
        # Test correctness of gaussian_solve outputs.

        gaussian_solve = linear_solvers_ns["gaussian_solve"]

        # Simple 2x2 matrix
        A = torch.tensor([[2, 1], [1, 3]], dtype=torch.float32)
        b = torch.tensor([5, 8], dtype=torch.float32)
        res = gaussian_solve(A, b)
        expected = torch.tensor([1.4, 2.2], dtype=torch.float32)
        assert torch.allclose(res, expected, atol=r_tol)

        # Simple 3x3 matrix
        A = torch.tensor([[1, 2, 1], [3, 1, -1], [2, -1, 1]],
                         dtype=torch.float32)
        b = torch.tensor([8, 2, 3], dtype=torch.float32)
        res = gaussian_solve(A, b)
        expected = torch.tensor([1, 2, 3], dtype=torch.float32)
        assert torch.allclose(res, expected, atol=r_tol)

        # Simple 4x4 matrix
        A = torch.tensor(
            [[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]],
            dtype=torch.float32)
        b = torch.tensor([1, 0, 0, 1], dtype=torch.float32)
        res = gaussian_solve(A, b)
        expected = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        assert torch.allclose(res, expected, atol=r_tol)

        # Diagonal matrix
        A = torch.tensor([[2, 0, 0], [0, 3, 0], [0, 0, 4]],
                         dtype=torch.float32)
        b = torch.tensor([4, 9, 16], dtype=torch.float32)
        res = gaussian_solve(A, b)
        expected = torch.tensor([2, 3, 4], dtype=torch.float32)
        assert torch.allclose(res, expected, atol=r_tol)

        # Partial pivoting example
        A = torch.tensor([[0.001, 1, 1], [1, 1, 1], [1, 1, 2]],
                         dtype=torch.float32)
        b = torch.tensor([2.001, 3, 4], dtype=torch.float32)
        res = gaussian_solve(A, b)
        expected = torch.tensor([1, 1, 1], dtype=torch.float32)
        assert torch.allclose(res, expected, atol=r_tol)

    def test_gaussian_solve_differentiability(self):
        # Test differerentiability of gaussian_solve.
        src = (
            "from Std.linear_solvers import gaussian_solve, get_2d_array_num_cols, get_2d_array_num_rows, get_1d_array_length, zero_2d_array, zero_1d_array\n"  # noqa
            "def f(alpha: ℝ): ℝ:\n"
            "    A: ℝ[3, 3] = [\n"
            "        [1, 2, 1],\n"
            "        [3, 1, -1],\n"
            "        [2, -1, 1]\n"
            "    ]\n"
            "    b: ℝ[3] = [8 * alpha, 2, 3]\n"
            "    ans: ℝ[3] = gaussian_solve(A, b)\n"
            "    return ans[0]**2 + ans[1]**2 + ans[2]**2\n"
            "\n"
            "alpha: ℝ = 1.0\n"
            "results: ℝ = grad(f, alpha)\n")
        ns = run_phyk(src, "physika/Std/linear_solvers.phyk")
        res = ns["results"]
        expected = torch.tensor([26.6667])
        assert torch.allclose(res, expected, atol=r_tol)
