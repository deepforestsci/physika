import pytest
from tests.conftest import exec_phyk


@pytest.fixture(scope="module")
def modulo_ns():
    """
    Execute modulo.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("modulo")


class TestModulo:
    """Tests for verification of results of modulo example."""

    def test_program_level(self, modulo_ns):
        """Verification of Program level operations."""
        a: int = 17
        b: int = 5
        assert modulo_ns["mod_literals"] == 17 % 5
        assert modulo_ns["mod_variables"] == a % b
        assert modulo_ns["mod_float"] == 7.5 % 2
        assert modulo_ns["mod_negative"] == -7 % 3
        assert modulo_ns["mod_precedence"] == 2 + 7 % 4 * 3

    def test_function_level(self, modulo_ns):
        """Verification of Function level operations."""
        import torch
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert modulo_ns["func_scalar"] == 2
        assert modulo_ns["func_twice"] == 0
        assert torch.equal(modulo_ns["func_tensor"],
                           torch.tensor([10, 12, 15, 17], device=DEVICE) % 4)
        assert modulo_ns["func_angle"] == 7.0 % 6.283185307179586

    def test_inside_loop(self, modulo_ns):
        """Verification of Inside loop operations."""
        import torch
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert torch.equal(
            modulo_ns["loop_comprehension"],
            torch.tensor([10, 11, 12, 13, 14, 15], device=DEVICE) % 4)
        assert modulo_ns["loop_count_odd"] == 3

    def test_class_level(self, modulo_ns):
        """Verification of Class level operations."""
        import torch
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert modulo_ns["class_wrap"] == 27 % 12
        assert torch.equal(modulo_ns["class_wrap_all"],
                           torch.tensor([5, 13, 25], device=DEVICE) % 12)

    def test_scaler_and_tensor(self, modulo_ns):
        """Verification of Scaler or Tensor operations."""
        import torch
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        s: int = 10
        t: torch.Tensor = torch.tensor([10, 12, 15, 17], device=DEVICE)
        u: torch.Tensor = torch.tensor([3, 5, 7, 4], device=DEVICE)
        A: torch.Tensor = torch.tensor([[7, 8, 9], [10, 11, 12]],
                                       device=DEVICE)
        B: torch.Tensor = torch.tensor([[2, 3, 4], [5, 6, 7]], device=DEVICE)
        assert modulo_ns["scalar_mod_scalar"] == s % 3
        assert torch.equal(modulo_ns["tensor_mod_scalar"], t % 4)
        assert torch.equal(modulo_ns["tensor_mod_tensor"], t % u)
        assert torch.equal(modulo_ns["matrix_mod_scalar"], A % 5)
        assert torch.equal(modulo_ns["matrix_mod_matrix"], A % B)

    def test_differentiation(self, modulo_ns):
        """Verification of differentiation operations."""
        import torch
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert modulo_ns["grad_dividend"] == torch.tensor(1.0, device=DEVICE)
