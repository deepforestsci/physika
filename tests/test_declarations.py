import pytest
import torch
from tests.conftest import exec_phyk
from physika.lexer import lexer


def first_type_value(src):
    """Return the value of the first TYPE token lexed from src."""
    lexer.lexer.lineno = 1
    lexer.input(src)
    while True:
        tok = lexer.token()
        if tok is None:
            return None
        if tok.type == "TYPE":
            return tok.value


class TestTypeNotations:
    """Each Unicode / LaTeX / ASCII notation lexes to its type."""

    def test_real_forms(self):
        assert first_type_value("ℝ") == "ℝ"
        assert first_type_value(r"\mathbb{R}") == "ℝ"
        assert first_type_value(r"\R") == "ℝ"
        assert first_type_value("R") == "ℝ"

    def test_integer_forms(self):
        assert first_type_value("ℤ") == "ℤ"
        assert first_type_value(r"\mathbb{Z}") == "ℤ"
        assert first_type_value(r"\Z") == "ℤ"
        assert first_type_value("Z") == "ℤ"

    def test_natural_forms(self):
        assert first_type_value("ℕ") == "ℕ"
        assert first_type_value(r"\mathbb{N}") == "ℕ"
        assert first_type_value(r"\N") == "ℕ"
        assert first_type_value("N") == "ℕ"

    def test_complex_forms(self):
        assert first_type_value("ℂ") == "ℂ"
        assert first_type_value(r"\mathbb{C}") == "ℂ"


@pytest.fixture(scope="module")
def decl_ns():
    """Execute example_declarations.phyk; return the namespace."""
    return exec_phyk("example_declarations")


class TestRealDeclarations:
    """Tests for ℝ scalar and array declarations."""

    def test_real_scalars(self, decl_ns):
        assert float(decl_ns["r_unicode"]) == 3.14
        assert float(decl_ns["r_mathbb"]) == 2.5
        assert float(decl_ns["r_macro"]) == 1.5
        assert float(decl_ns["r_ascii"]) == 0.5

    def test_real_arrays(self, decl_ns):
        assert torch.allclose(decl_ns["r_vector"],
                              torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(decl_ns["r_matrix"],
                              torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


class TestIntegerDeclarations:
    """Tests for ℤ scalar and array declarations."""

    def test_integer_scalars(self, decl_ns):
        assert float(decl_ns["z_unicode"]) == 10.0
        assert float(decl_ns["z_mathbb"]) == -4.0
        assert float(decl_ns["z_macro"]) == 7.0
        assert float(decl_ns["z_ascii"]) == 42.0

    def test_integer_array(self, decl_ns):
        assert torch.allclose(decl_ns["z_vector"].float(),
                              torch.tensor([1.0, 2.0, 3.0]))


class TestNaturalDeclarations:
    """Tests for ℕ scalar and array declarations."""

    def test_natural_scalars(self, decl_ns):
        assert float(decl_ns["n_unicode"]) == 5.0
        assert float(decl_ns["n_mathbb"]) == 8.0
        assert float(decl_ns["n_macro"]) == 3.0
        assert float(decl_ns["n_ascii"]) == 1.0

    def test_natural_array(self, decl_ns):
        assert torch.allclose(decl_ns["n_vector"].float(),
                              torch.tensor([0.0, 1.0, 2.0, 3.0]))


class TestComplexDeclarations:
    """Tests for ℂ scalar and array declarations."""

    def test_complex_scalars(self, decl_ns):
        assert complex(decl_ns["c_unicode"]) == complex(3, 1)
        assert complex(decl_ns["c_mathbb"]) == complex(5, 3)

    def test_complex_array(self, decl_ns):
        assert torch.allclose(decl_ns["c_vector"],
                              torch.tensor([1 + 2j, 3 + 4j]))
