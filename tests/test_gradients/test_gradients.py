import math
from pathlib import Path

import numpy as np
import pytest
import torch

from codegen import from_ast_to_torch
from gradient_checker import numerical_gradient
from lexer import lexer
from parser import parser, symbol_table
from runtime import compute_grad
from utils.ast_utils import build_unified_ast


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

    @pytest.mark.parametrize("x_val, expected_grad", [
        # x > 0: f(x) = x**2  -> f'(x) = 2x
        (3.0,  6.0),
        (1.0,  2.0),
        (0.5,  1.0),
        # x <=0: f(x) = −x  -> f'(x) = −1
        (-2.0, -1.0),
        (-1.0, -1.0),
        (-0.5, -1.0),
    ])
    def test_physika_matches_analytical(self, name_space, x_val, expected_grad):
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
        num_grad = numerical_gradient(f, np.array([x_val]))[0]
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

    @pytest.mark.parametrize("x_val, expected_grad", [
        # x <= 0: f'(x) = cos(x)
        (-1.5, math.cos(-1.5)),
        (-0.5, math.cos(-0.5)),
        # x > 0: f'(x) = -sin(x)
        ( 0.5, -math.sin(0.5)),
        ( 1.5, -math.sin(1.5)),
        ( 3.14, -math.sin(3.14)),
    ])
    def test_physika_matches_analytical(self, name_space, x_val, expected_grad):
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
        num_grad = numerical_gradient(f, np.array([x_val]))[0]
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

    @pytest.mark.parametrize("t_val, expected_grad", [
        # t > 0.5: L'(t) = 6*(t - 0.75)
        (0.9,  6 * (0.9  - 0.75)),
        (0.6,  6 * (0.6  - 0.75)),
        (0.75, 0.0),
        # t ≤ 0.5: L'(t) = 2*t
        (0.3,  2 * 0.3),
        (0.1,  2 * 0.1),
        (-0.5, 2 * -0.5),
    ])
    def test_physika_matches_analytical(self, name_space, t_val, expected_grad):
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
        num_grad = numerical_gradient(f, np.array([t_val]))[0]
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

    @pytest.mark.parametrize("x_val, expected_grad", [
        # x > 0: forward'(x) = 2x
        (2.0,  4.0),
        (1.0,  2.0),
        (0.5,  1.0),
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
        num_grad = numerical_gradient(net, np.array([x_val]))[0]
        assert abs(physika_grad - num_grad) < r_tol
