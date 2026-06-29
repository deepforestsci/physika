import pytest
from tests.conftest import exec_phyk
import torch
from physika.runtime import compute_grad

r_tol = 1e-02


@pytest.fixture(scope="module")
def numeric_ns():
    """
    Execute example_arrays.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_complex")


class TestComplexType:
    """Tests for complex type ℂ declarations"""

    def test_complex_value(self, numeric_ns):
        assert complex(numeric_ns["x"]) == 3 + 1j

    def test_complex_add(self, numeric_ns):
        assert complex(numeric_ns["complex_add"]) == 8 + 4j

    def test_complex_mul(self, numeric_ns):
        assert complex(numeric_ns["complex_mul"]) == 12 + 14j

    @pytest.mark.parametrize("x_val, expected_value", [(3 + 1j, 3.1623),
                                                       (5 + 3j, 5.8310)])
    def test_complex_function(self, numeric_ns, x_val, expected_value):
        f = numeric_ns["magnitude"]
        x = torch.tensor(x_val, dtype=torch.complex64)
        true_values = f(x)
        print("this are true values -> ", true_values)
        assert torch.allclose(true_values,
                              torch.tensor(expected_value),
                              rtol=r_tol)

    def test_array_imag(self, numeric_ns):
        assert torch.allclose(
            numeric_ns["array_imag"],
            torch.tensor([1j, 2j, 3j], dtype=torch.complex64))

    def test_array_complex(self, numeric_ns):
        assert torch.allclose(
            numeric_ns["array_complex"],
            torch.tensor([1 + 9j, 7 + 2j, 3 + 5j], dtype=torch.complex64))

    def test_array_nested_complex(self, numeric_ns):
        assert torch.allclose(
            numeric_ns["nested_complex"],
            torch.tensor([[1 + 2j, 3 + 4j], [4 + 9j, 7 + 2j]],
                         dtype=torch.complex64))


class TestComplexGradients:
    """Tests for gradients correctness for complex type `ℂ`"""

    def test_complex_scalar_grad(self, numeric_ns):
        # Test case for scalar complex grad correctness
        f = numeric_ns["f"]
        x = 1 + 3j
        x_tensor = torch.tensor(x, dtype=torch.complex64)
        expected = compute_grad(f, x_tensor)
        assert torch.allclose(
            numeric_ns["scalar_grad"],
            expected,
            rtol=r_tol,
        )

    def test_complex_tensor_grad(self, numeric_ns):
        # Test case for tensor complex grad correctness
        expected = torch.tensor(
            [2 + 4j, 6 + 2j],
            dtype=torch.complex64,
        )
        assert torch.allclose(
            numeric_ns["tensor_grad"],
            expected,
            rtol=r_tol,
        )

    def test_complex_grad_in_class(self, numeric_ns):
        # Test case for complex grad inside class
        expected = torch.tensor(
            [2 + 4j],
            dtype=torch.complex64,
        )
        assert torch.allclose(
            numeric_ns["class_grad"],
            expected,
            rtol=r_tol,
        )
