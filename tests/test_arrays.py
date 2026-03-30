from pathlib import Path

import pytest

from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast
from physika.codegen import from_ast_to_torch

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def exec_phyk(stem: str) -> dict:
    """
    Helper function to execute a .phyk file and return the resulting namespace
    ``ns`` dict.
    """
    source = (EXAMPLES_DIR / f"{stem}.phyk").read_text()
    symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = parser.parse(source, lexer=lexer)
    unified = build_unified_ast(program_ast, symbol_table)
    code = from_ast_to_torch(unified, print_code=False)
    ns: dict = {}
    exec(code, ns)
    return ns


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
        expected = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
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
