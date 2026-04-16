import pytest
from conftest import exec_phyk
import torch


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
