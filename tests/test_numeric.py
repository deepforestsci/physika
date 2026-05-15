import pytest
from tests.conftest import exec_phyk


@pytest.fixture(scope="module")
def numeric_ns():
    """
    Execute example_arrays.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_numeric_types")


class TestIntegerType:
    """Tests for integer type ℤ declarations and arithmetic."""

    def test_integer_value(self, numeric_ns):
        assert float(numeric_ns["a"]) == 10.0

    def test_integer_addition(self, numeric_ns):
        assert float(numeric_ns["z_add"]) == 13.0


class TestRealType:
    """Tests for integer type ℤ declarations and arithmetic."""

    def test_real_value(self, numeric_ns):
        assert float(numeric_ns["x"]) == 3.14

    def test_real_multiplication(self, numeric_ns):
        assert float(numeric_ns["r_mul"]) == 3.14 * 2


class TestMixedTypes:
    """Tests for expressions mixing ℤ and ℝ types."""

    def test_mixed_values(self, numeric_ns):
        assert float(numeric_ns["z_number"]) == 1.0
        assert float(numeric_ns["r_number"]) == 2.0

    def test_mixed_multiplication(self, numeric_ns):
        assert float(numeric_ns["result"]) == 2.0


class TestNegativeValues:
    """Tests for negative ℤ and ℝ value declarations."""

    def test_negative_integer(self, numeric_ns):
        assert float(numeric_ns["neg_int"]) == -7.0

    def test_negative_real(self, numeric_ns):
        assert abs(float(numeric_ns["neg_float"]) - (-3.14)) < 1e-5

    def test_negative_array(self, numeric_ns):
        import torch
        assert torch.allclose(numeric_ns["neg_array"],
                              torch.tensor([-1.0, -2.0, -3.0]))
