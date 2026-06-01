import pytest
from tests.conftest import exec_phyk
import torch

r_tol = 1e-02


@pytest.fixture(scope="module")
def for_ns():
    """
    Execute for.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("for")


class TestProgramLevelAssignment:
    """
    Tests to check shape and values of 1D, 2D and 3D arrays
    in program/top level code
    """

    def test_1d_assign_by_index(self, for_ns):
        assert torch.equal(for_ns["sample_1d_array"],
                           torch.tensor([0.0, 2.0, 4.0]))

    def test_2d_assign_by_index(self, for_ns):
        assert torch.equal(for_ns["sample_2d_array"],
                           torch.tensor([[0.0, 2.0], [0.0, 2.0]]))

    def test_3d_assign_by_index(self, for_ns):
        assert torch.equal(
            for_ns["sample_3d_array"],
            torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]]))


class TestFunctionLevelAssignment:
    """
    Tests to check shape and values of 1D, 2D and 3D arrays
    in function level code
    """

    def test_1d_function_assign(self, for_ns):
        assert torch.equal(for_ns["arr1d"], torch.tensor([0.0, 2.0, 4.0]))

    def test_2d_function_assign(self, for_ns):
        assert torch.equal(for_ns["arr2d"],
                           torch.tensor([[0.0, 2.0], [0.0, 2.0]]))

    def test_3d_function_assign(self, for_ns):
        assert torch.equal(
            for_ns["arr3d"],
            torch.tensor([[[0.0, 1.0], [1.0, 2.0]], [[2.0, 3.0], [3.0, 4.0]]]))


class TestFunctionLevelImplicitForLoops:
    """
    Tests to check correctness of implicit for loops.
    """

    @pytest.mark.parametrize(
        "x_val, y_val, expected",
        [
            (
                [1, 2, 3, 4],
                [0, 5, 6, 7],
                [
                    [0, 5, 6, 7],
                    [0, 10, 12, 14],
                    [0, 15, 18, 21],
                    [0, 20, 24, 28],
                ],
            ),
            (
                [1, 2],
                [3, 4],
                [
                    [3, 4],
                    [6, 8],
                ],
            ),
        ],
    )
    def test_implicit_equals(self, for_ns, x_val, y_val, expected):
        """Test for implicit for loop with equals"""
        f = for_ns["outer_product"]
        x = torch.tensor(x_val)
        y = torch.tensor(y_val)

        results = f(x, y)
        expected = torch.tensor(expected)

        assert torch.allclose(results, expected, rtol=r_tol)

    @pytest.mark.parametrize(
        "u_val, v_val, expected",
        [
            (
                [1, 2, 3],
                [1, 2, 3],
                [
                    [1, 2, 3],
                    [2, 4, 6],
                    [3, 6, 9],
                ],
            ),
        ],
    )
    def test_implicit_plus_eq(self, for_ns, u_val, v_val, expected):
        """Test for implicit for loop with plus-equals"""
        f = for_ns["outer_accum"]
        u = torch.tensor(u_val)
        v = torch.tensor(v_val)

        results = f(u, v)
        expected = torch.tensor(expected)

        assert torch.allclose(results, expected, rtol=r_tol)
