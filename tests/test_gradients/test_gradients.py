import math
from pathlib import Path

import pytest
import torch

from physika.codegen import from_ast_to_torch
from gradient_checker import numerical_gradient
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.runtime import compute_grad
from physika.utils.ast_utils import build_unified_ast

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
r_tol = 1e-02


# Helper
def compile(phyk_name: str) -> dict:
    """Parse a .phyk file and return the executed namespace."""
    src = (EXAMPLES_DIR / f"{phyk_name}.phyk").read_text()
    symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = parser.parse(src, lexer=lexer)
    unified = build_unified_ast(program_ast, symbol_table)
    code = from_ast_to_torch(unified, print_code=False)
    name_space: dict = {}
    exec(code, name_space)
    return name_space


class TestDiffIfElse:

    @pytest.fixture()
    def name_space(self):
        """Call examples/diff_ifelse.phyk file"""
        return compile("diff_ifelse")

    @pytest.mark.parametrize(
        "x_val, expected_grad",
        [
            # x > 0: f(x) = x**2  -> f'(x) = 2x
            (3.0, 6.0),
            (1.0, 2.0),
            (0.5, 1.0),
            # x <=0: f(x) = −x  -> f'(x) = −1
            (-2.0, -1.0),
            (-1.0, -1.0),
            (-0.5, -1.0),
        ])
    def test_physika_matches_analytical(self, name_space, x_val,
                                        expected_grad):
        """Physika grad matches the analytical derivative"""
        f = name_space["f"]
        x = torch.tensor(x_val)
        physika_grad = compute_grad(f, x)
        assert abs(physika_grad - expected_grad) < r_tol

    @pytest.mark.parametrize("x_val", [3.0, 1.0, 0.5, -2.0, -1.0, -0.5])
    def test_physika_matches_numerical(self, name_space, x_val):
        """Physika grad matches numerical gradient"""
        f = name_space["f"]
        x = torch.tensor(x_val)

        physika_grad = compute_grad(f, x)
        num_grad = numerical_gradient(f, x)[0]
        assert abs(physika_grad - num_grad) < r_tol


class TestDiffIfCosSin:
    """Gradients of f(x) = cos(x) if x>0 else sin(x).

    f'(x):
        x > 0: f'(x) = -sin(x)
        x ≤ 0: f'(x) =  cos(x)
    """

    @pytest.fixture()
    def name_space(self):
        """Call examples/diff_sincos.phyk file"""
        return compile("diff_sincos")

    @pytest.mark.parametrize(
        "x_val, expected_grad",
        [
            # x <= 0: f'(x) = cos(x)
            (-1.5, math.cos(-1.5)),
            (-0.5, math.cos(-0.5)),
            # x > 0: f'(x) = -sin(x)
            (0.5, -math.sin(0.5)),
            (1.5, -math.sin(1.5)),
            (3.14, -math.sin(3.14)),
        ])
    def test_physika_matches_analytical(self, name_space, x_val,
                                        expected_grad):
        """physika grad matches the analytical derivative"""
        f = name_space["f"]
        x = torch.tensor(x_val)
        physika_grad = compute_grad(f, x)
        assert abs(physika_grad - expected_grad) < r_tol

    @pytest.mark.parametrize("x_val", [-0.5, 0.5, 1.5])
    def test_physika_matches_numerical(self, name_space, x_val):
        """Test for numerical and physika gradient solutions"""
        f = name_space["f"]
        x = torch.tensor(x_val)
        physika_grad = compute_grad(f, x)
        num_grad = numerical_gradient(f, x)[0]
        assert abs(physika_grad - num_grad) < r_tol


class TestDiffThreshold:
    """
    L(t) = 3*(t-0.75)**2+0.1 if t>0.5 else (t**2)+2.0.

    L'(t):
        t > 0.5 : L'(t) = 6*(t - 0.75)
        t ≤ 0.5: L'(t) = 2*t
    """

    @pytest.fixture()
    def name_space(self):
        """Call examples/diff_threshold.phyk file"""
        return compile("diff_threshold")

    @pytest.mark.parametrize(
        "t_val, expected_grad",
        [
            # t > 0.5: L'(t) = 6*(t - 0.75)
            (0.9, 6 * (0.9 - 0.75)),
            (0.6, 6 * (0.6 - 0.75)),
            (0.75, 0.0),
            # t ≤ 0.5: L'(t) = 2*t
            (0.3, 2 * 0.3),
            (0.1, 2 * 0.1),
            (-0.5, 2 * -0.5),
        ])
    def test_physika_matches_analytical(self, name_space, t_val,
                                        expected_grad):
        """Physika matches the analytical derivative."""
        f = name_space["L"]
        t = torch.tensor(t_val)
        physika_grad = compute_grad(f, t)
        assert abs(physika_grad - expected_grad) < r_tol

    @pytest.mark.parametrize("t_val", [0.9, 0.6, -0.5])
    def test_physika_matches_numerical(self, name_space, t_val):
        """Physika grads matches numerical gradients."""
        f = name_space["L"]
        t = torch.tensor(t_val)
        physika_grad = compute_grad(f, t)
        num_grad = numerical_gradient(f, t)[0]
        assert abs(physika_grad - num_grad) < r_tol


class TestDiffIfElseClasses:
    """Gradients of PiecewiseNet.

    PiecewiseNet implements
    forward(x) = x**2 if x > 0 else -x.
        x > 0:  forward'(x) = 2x
        x ≤ 0:  forward'(x) = -1
    """

    @pytest.fixture()
    def name_space(self):
        return compile("if_else_contexts")

    @pytest.mark.parametrize(
        "x_val, expected_grad",
        [
            # x > 0: forward'(x) = 2x
            (2.0, 4.0),
            (1.0, 2.0),
            (0.5, 1.0),
            # x ≤ 0: forward'(x) = -1
            (-1.5, -1.0),
            (-1.0, -1.0),
            (-0.5, -1.0),
        ])
    def test_class_matches_analytical(self, name_space, x_val, expected_grad):
        """Class ifelse gradient matches the analytical derivative."""
        net = name_space["net"]
        x = torch.tensor(x_val)
        physika_grad = compute_grad(net, x)
        assert abs(physika_grad - expected_grad) < r_tol

    @pytest.mark.parametrize("x_val", [2.0, 1.0, -1.5, -0.5])
    def test_class_matches_numerical(self, name_space, x_val):
        """Class ifelse gradient matches the numerical gradient."""
        net = name_space["net"]
        x = torch.tensor(x_val)
        physika_grad = compute_grad(net, x)
        num_grad = numerical_gradient(net, x)[0]
        assert abs(physika_grad - num_grad) < r_tol


class TestDiffFor:
    """
    Tests for differentiable for-loops.
    """

    @pytest.fixture()
    def ns(self):
        return compile("diff_for")

    @pytest.mark.parametrize("s_val, expected", [
        (2.0, 6.0),
        (0.0, 6.0),
        (-1.0, 6.0),
    ])
    def test_sum_for_expr_analytical(self, ns, s_val, expected):
        f = ns["sum_for_expr"]
        g = compute_grad(f, torch.tensor(s_val))
        assert abs(float(g) - expected) < r_tol

    @pytest.mark.parametrize("s_val", [2.0, 0.0, -1.0])
    def test_sum_for_expr_numerical(self, ns, s_val):
        f = ns["sum_for_expr"]
        x = torch.tensor([s_val])
        g = compute_grad(f, torch.tensor(s_val))
        ng = numerical_gradient(lambda v: f(v[0]), x, h=1e-3)
        assert abs(float(g) - float(ng[0])) < r_tol

    @pytest.mark.parametrize("s_val, expected", [
        (1.0, 10.0),
        (3.0, 10.0),
        (-2.0, 10.0),
    ])
    def test_dot_with_arr_analytical(self, ns, s_val, expected):
        f = ns["dot_with_arr"]
        g = compute_grad(f, torch.tensor(s_val))
        assert abs(float(g) - expected) < r_tol

    @pytest.mark.parametrize("s_val", [1.0, 3.0, -2.0])
    def test_dot_with_arr_numerical(self, ns, s_val):
        f = ns["dot_with_arr"]
        x = torch.tensor([s_val])
        g = compute_grad(f, torch.tensor(s_val))
        ng = numerical_gradient(lambda v: f(v[0]), x, h=1e-3)
        assert abs(float(g) - float(ng[0])) < r_tol

    @pytest.mark.parametrize("s_val, expected", [
        (1.0, 10.0),
        (0.0, 10.0),
        (-1.0, 10.0),
    ])
    def test_matmul_scale_analytical(self, ns, s_val, expected):
        f = ns["matmul_scale"]
        g = compute_grad(f, torch.tensor(s_val))
        assert abs(float(g) - expected) < r_tol

    @pytest.mark.parametrize("s_val", [1.0, 0.0, -1.0])
    def test_matmul_scale_numerical(self, ns, s_val):
        f = ns["matmul_scale"]
        x = torch.tensor([s_val])
        g = compute_grad(f, torch.tensor(s_val))
        ng = numerical_gradient(lambda v: f(v[0]), x, h=1e-3)
        assert abs(float(g) - float(ng[0])) < r_tol

    @pytest.mark.parametrize("s_val, expected", [
        (1.0, 495.0),
        (0.0, 495.0),
        (-1.0, 495.0),
    ])
    def test_nested_sum_analytical(self, ns, s_val, expected):
        f = ns["nested_sum"]
        g = compute_grad(f, torch.tensor(s_val))
        assert abs(float(g) - expected) < r_tol

    @pytest.mark.parametrize("x_val, expected", [
        (2.0, [1.0, 2.0, 3.0]),
        (0.0, [1.0, 2.0, 3.0]),
        (-1.0, [1.0, 2.0, 3.0]),
    ])
    def test_scale_vec_jacobian_analytical(self, ns, x_val, expected):
        f = ns["scale_vec"]
        g = compute_grad(f, torch.tensor(x_val))
        for gi, ei in zip(g.tolist(), expected):
            assert abs(gi - ei) < r_tol

    @pytest.mark.parametrize("x_val", [2.0, 1.0, -0.5])
    def test_scale_vec_jacobian_numerical(self, ns, x_val):
        f = ns["scale_vec"]
        x = torch.tensor(x_val)
        h = 1e-4
        jac_num = (f(x + h) - f(x - h)) / (2 * h)
        g = compute_grad(f, x)
        for gi, ngi in zip(g.tolist(), jac_num.tolist()):
            assert abs(gi - ngi) < r_tol

    @pytest.mark.parametrize("x_val, expected", [
        (3.0, [6.0, 12.0, 18.0, 24.0]),
        (1.0, [2.0, 4.0, 6.0, 8.0]),
        (-2.0, [-4.0, -8.0, -12.0, -16.0]),
    ])
    def test_sq_vec_jacobian_analytical(self, ns, x_val, expected):
        f = ns["sq_vec"]
        g = compute_grad(f, torch.tensor(x_val))
        for gi, ei in zip(g.tolist(), expected):
            assert abs(gi - ei) < r_tol

    @pytest.mark.parametrize("x_val", [3.0, 1.0, -2.0])
    def test_sq_vec_jacobian_numerical(self, ns, x_val):
        f = ns["sq_vec"]
        x = torch.tensor(x_val)
        h = torch.tensor(1e-3)
        jac_num = (f(x + h) - f(x - h)) / (2 * h)
        g = compute_grad(f, x)
        for gi, ngi in zip(g.tolist(), jac_num.tolist()):
            assert abs(gi - ngi) < r_tol

    @pytest.mark.parametrize("x_val", [0.5, 1.0, -0.3])
    def test_cos_freqs_jacobian_analytical(self, ns, x_val):
        f = ns["cos_freqs"]
        g = compute_grad(f, torch.tensor(x_val))
        expected = [-(k + 1) * math.sin((k + 1) * x_val) for k in range(4)]
        for gi, ei in zip(g.tolist(), expected):
            assert abs(gi - ei) < r_tol

    @pytest.mark.parametrize("x_val", [0.5, 1.0, -0.3])
    def test_cos_freqs_jacobian_numerical(self, ns, x_val):
        f = ns["cos_freqs"]
        x = torch.tensor(x_val)
        h = torch.tensor(1e-4)
        jac_num = (f(x + h) - f(x - h)) / (2 * h)
        g = compute_grad(f, x)
        for gi, ngi in zip(g.tolist(), jac_num.tolist()):
            assert abs(gi - ngi) < r_tol

    @pytest.mark.parametrize("x_vals, expected_diag", [
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        ([0.5, 1.0, 2.0], [1.0, 2.0, 4.0]),
    ])
    def test_elementwise_sq_jacobian_analytical(self, ns, x_vals,
                                                expected_diag):
        f = ns["elementwise_sq"]
        x = torch.tensor(x_vals)
        jac = compute_grad(f, x)  # shape R[3, 3]
        assert jac.shape == (3, 3)
        for i, ei in enumerate(expected_diag):
            assert abs(jac[i, i].item() - ei) < r_tol
            for j in range(3):
                if j != i:
                    assert abs(jac[i, j].item()) < r_tol


class TestGradFunction:
    """Gradient calculations inside function statements"""

    @pytest.fixture()
    def name_space(self):
        """Call examples/example_check_gradients.phyk file"""
        return compile("example_check_gradients")

    @pytest.mark.parametrize("x_val, expected_grad", [
        ([1.0], [2.0]),
        ([3.0], [6.0]),
    ])
    def test_function_matches_analytical(self, name_space, x_val,
                                         expected_grad):
        """function gradient matches the analytical derivative."""
        f = name_space["f"]
        x = torch.tensor(x_val, requires_grad=True)
        physika_grad = f(x)
        expected = torch.tensor(expected_grad)
        assert torch.allclose(physika_grad, expected, rtol=r_tol)

    @pytest.mark.parametrize("x_val", [[5.0]])
    def test_function_matches_numerical(self, name_space, x_val):
        """function gradient matches the numerical gradient."""
        f = name_space["f"]
        x = torch.tensor(x_val, requires_grad=True)
        physika_grad = f(x)

        def y_func(x_input):
            return (x_input[0]**2.0).sum()

        num_grad = numerical_gradient(y_func, x)[0]
        assert torch.allclose(physika_grad, num_grad, rtol=r_tol)
