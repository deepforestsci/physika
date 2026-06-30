import pytest
from tests.conftest import exec_phyk
import torch


@pytest.fixture(scope="module")
def numeric_ns():
    """
    Execute greek_letter_and_scientific_notation.phyk, build unified AST,
    execute; return namespace.
    """
    return exec_phyk("greek_letter_and_scientific_notation")


class TestScientifiNotation:
    """Tests for scientific notation `e`"""

    def test_scientific_notation_values(self, numeric_ns):
        # Test scientific notation declarations
        assert float(numeric_ns["x"]) == 1e5
        assert float(numeric_ns["y"]) == 3e5
        assert float(numeric_ns["z"]) == 4e5


class TestGreekLetters:
    """Tests for Greek letter variables"""

    def test_basic_declarations(self, numeric_ns):
        # Test greek letters variable declarations
        assert float(numeric_ns["α"]) == float(1.0)
        assert float(numeric_ns["β"]) == float(2.0)
        assert float(numeric_ns["σ"]) == 5.6704e-8
        assert float(numeric_ns["ψ"]) == float(0.5)
        assert torch.equal(numeric_ns["greek_letters_array"],
                           torch.tensor([1.0, 2.0]))

    def test_numeric_operations(self, numeric_ns):
        # Test greek letters operations
        assert float(numeric_ns["results"]) == pytest.approx(3.0)

    def test_gradients(self, numeric_ns):
        # Test greek letters gradients correctness
        assert float(numeric_ns["grad_μ"]) == pytest.approx(4.0)

    def test_if_else(self, numeric_ns):
        # Test greek letters if-else
        assert float(numeric_ns["result_if"]) == pytest.approx(2.0)

    def test_for_loop(self, numeric_ns):
        # Test greek letters for loop
        assert float(numeric_ns["sum_Ω"]) == pytest.approx(1.5)
