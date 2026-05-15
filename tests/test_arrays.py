import pytest
from tests.conftest import exec_phyk


@pytest.fixture(scope="module")
def arrays_ns():
    """
    Execute example_arrays.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_arrays")


class TestNDIndexing:
    """
    Correctness tests for ND array indexing using example_arrays.phyk.
    """

    def test_2d_example(self, arrays_ns):
        """
        2D example: u0[i, j] on a 4x4 matrix
        u0: R[4, 4] = [
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.75, 0.75, 0.00],
        [0.00, 0.75, 0.75, 10.00],
        [0.00, 0.00, 0.00, 0.00]
        ]
        """
        ref_array = [
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.75, 0.75, 0.00],
            [0.00, 0.75, 0.75, 10.00],
            [0.00, 0.00, 0.00, 0.00],
        ]

        assert arrays_ns["u0"].shape == (4, 4)
        for i in range(4):
            for j in range(4):
                assert float(arrays_ns["u0"][i, j]) == ref_array[i][j]

        # test physika indexing syntax u[i, j] from .phyk source
        assert float(arrays_ns["u00"]) == 0.00
        assert float(arrays_ns["u01"]) == 0.00
        assert float(arrays_ns["u02"]) == 0.00
        assert float(arrays_ns["u03"]) == 0.00

        assert float(arrays_ns["u10"]) == 0.00
        assert float(arrays_ns["u11"]) == 0.75
        assert float(arrays_ns["u12"]) == 0.75
        assert float(arrays_ns["u13"]) == 0.00

        assert float(arrays_ns["u20"]) == 0.00
        assert float(arrays_ns["u21"]) == 0.75
        assert float(arrays_ns["u22"]) == 0.75
        assert float(arrays_ns["u23"]) == 10.00

        assert float(arrays_ns["u30"]) == 0.00
        assert float(arrays_ns["u31"]) == 0.00
        assert float(arrays_ns["u32"]) == 0.00
        assert float(arrays_ns["u33"]) == 0.00

    def test_3d_array_indexing(self, arrays_ns):
        """
        3D array: T[i, j, k] on a 2x3x4 tensor with values from 1 to 24.
        """
        import torch
        expected = torch.arange(1, 25, dtype=torch.int64).reshape(2, 3, 4)
        assert arrays_ns["T"].shape == expected.shape
        assert torch.allclose(arrays_ns["T"], expected)

        assert float(arrays_ns["T"][0, 0, 0]) == 1.0
        assert float(arrays_ns["T"][1, 2, 3]) == 24.0
        assert float(arrays_ns["T"][0, 1, 2]) == 7.0

        # test physika indexing syntax T[i, j, k] from .phyk source
        assert torch.equal(
            arrays_ns["T0"],
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0]]))
        assert torch.equal(arrays_ns["T12"],
                           torch.tensor([21.0, 22.0, 23.0, 24.0]))
        assert float(arrays_ns["T000"]) == 1.0
        assert float(arrays_ns["T123"]) == 24.0
        assert float(arrays_ns["T012"]) == 7.0


class TestNdIndexAssign:
    """
    Correctness test for Nd array index assignment
    """

    # Program level
    def test_1d_program_level(self, arrays_ns):
        assert arrays_ns["prog_1d"].shape == (2, )
        assert int(arrays_ns["prog_1d"][0]) == 1
        assert int(arrays_ns["prog_1d"][1]) == 2

    def test_2d_program_level(self, arrays_ns):
        assert arrays_ns["prog_2d"].shape == (2, 2)
        assert int(arrays_ns["prog_2d"][0, 0]) == 1
        assert int(arrays_ns["prog_2d"][0, 1]) == 1
        assert int(arrays_ns["prog_2d"][1, 0]) == 1
        assert int(arrays_ns["prog_2d"][1, 1]) == 2

    def test_3d_program_level(self, arrays_ns):
        assert arrays_ns["prog_3d"].shape == (2, 2, 2)
        assert int(arrays_ns["prog_3d"][0, 0, 0]) == 1
        assert int(arrays_ns["prog_3d"][0, 0, 1]) == 1
        assert int(arrays_ns["prog_3d"][0, 1, 0]) == 1
        assert int(arrays_ns["prog_3d"][0, 1, 1]) == 1
        assert int(arrays_ns["prog_3d"][1, 0, 0]) == 1
        assert int(arrays_ns["prog_3d"][1, 0, 1]) == 1
        assert int(arrays_ns["prog_3d"][1, 1, 0]) == 1
        assert int(arrays_ns["prog_3d"][1, 1, 1]) == 2

    # Function level
    def test_1d_function_level(self, arrays_ns):
        assert arrays_ns["func_1d"].shape == (2, )
        assert int(arrays_ns["func_1d"][0]) == 1
        assert int(arrays_ns["func_1d"][1]) == 3

    def test_2d_function_level(self, arrays_ns):
        assert arrays_ns["func_2d"].shape == (2, 2)
        assert int(arrays_ns["func_2d"][0, 0]) == 1
        assert int(arrays_ns["func_2d"][0, 1]) == 1
        assert int(arrays_ns["func_2d"][1, 0]) == 1
        assert int(arrays_ns["func_2d"][1, 1]) == 3

    def test_3d_function_level(self, arrays_ns):
        assert arrays_ns["func_3d"].shape == (2, 2, 2)
        assert int(arrays_ns["func_3d"][0, 0, 0]) == 1
        assert int(arrays_ns["func_3d"][0, 0, 1]) == 1
        assert int(arrays_ns["func_3d"][0, 1, 0]) == 1
        assert int(arrays_ns["func_3d"][0, 1, 1]) == 1
        assert int(arrays_ns["func_3d"][1, 0, 0]) == 1
        assert int(arrays_ns["func_3d"][1, 0, 1]) == 1
        assert int(arrays_ns["func_3d"][1, 1, 0]) == 1
        assert int(arrays_ns["func_3d"][1, 1, 1]) == 3
