import pytest
from tests.conftest import exec_phyk
from physika.codegen import from_ast_to_torch
from physika.utils.ast_utils import build_unified_ast
from physika.parser import parser, symbol_table
from physika.lexer import lexer
from physika.runtime import DEVICE


def parse_source_to_ast(source: str, source_path=None) -> dict:
    """Run lexer/parser and build_unified_ast on a Physika source string."""
    symbol_table.clear()
    lexer.lexer.lineno = 1  # reset PLY line counter for deterministic output
    program_ast = parser.parse(source, lexer=lexer)

    return build_unified_ast(program_ast, symbol_table)


@pytest.fixture(scope="module")
def gpu_ns():
    """
    Execute example_gpu_support.phyk, build unified AST,
    execute; return namespace.
    """
    return exec_phyk("example_gpu_support")


class TestGPUsupport:
    """Tests for ``examples/example_gpu_support.phyk`` file"""

    def test_device_type_correctness(self, gpu_ns):
        # Verify correct DEVICE type
        assert gpu_ns["x_tensor"].device.type == DEVICE.type
        assert gpu_ns["x_matrix"].device.type == DEVICE.type
        assert gpu_ns["results"].device.type == DEVICE.type

    def test_tensor_device(self):
        # Verify generated pytorch code for simple tensor declaration with
        # DEVICE support
        physika_source = ("x_tensor: R[5] = [1.0, 2.0, 3.0, 4.0, 5.0]")
        ast = parse_source_to_ast(physika_source)
        torch_code = from_ast_to_torch(ast)
        assert ("x_tensor = torch.tensor("
                "[1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE)" in torch_code)

    def test_class_device(self):
        # Verify generated pytorch code for class declaration with DEVICE
        # support
        physika_source = ("class MatrixMultiply(x: R):\n"
                          "     def λ(A: ℝ[m, k], B: ℝ[k, n]) → ℝ[m, n]:\n"
                          "         return A @ B\n"
                          "obj: MatrixMultiply = MatrixMultiply(1.0)")
        ast = parse_source_to_ast(physika_source)
        torch_code = from_ast_to_torch(ast)
        assert "A = torch.as_tensor(A, device=DEVICE).float()" in torch_code
        assert "B = torch.as_tensor(B, device=DEVICE).float()" in torch_code
        assert "obj = MatrixMultiply(1.0).to(DEVICE)" in torch_code
