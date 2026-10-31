from fractions import Fraction

import pytest

from physika.core.expr import App, BVar, Const, Lit
from physika.core.inductive import mk_builtin_env
from physika.units import unit_to_dimvec_cic
from physika.utils.cic_utils.inductive_utils import int_lit
from physika.solver import solver
from physika.solver.solver import solve_bindings, solve_block, solve_equations
from physika.solver.solver_candidate import (arith, solve_candidate,
                                             unsupported_reason)
from physika.solver.solver_dims import (add_terms, check_dimensions, dim_env,
                                        dim_str, has_units, is_number,
                                        same_dim)
from physika.solver.solver_linear import (eq_row, frac, gauss_jordan, lin,
                                          linear_system, solve_linear,
                                          verify_linear_row)
from physika.solver.solver_poly import (deriv_call, divisors, inline_call,
                                        numeric_roots, p_add, p_deriv, p_eval,
                                        p_mul, p_pow, poly_of, poly_roots,
                                        poly_str, rational_roots, solve_poly,
                                        verify_deriv, verify_poly_root,
                                        wrt_name)
from physika.solver.solver_chain import (chain_side, fresh_definition,
                                         fresh_linear_pattern, fresh_pattern,
                                         is_var, solve_chain, ssqrt,
                                         sym_solve_poly)
from physika.solver.solver_sym_linear import (is_zero, solve_sym_linear,
                                              sym_gauss_jordan,
                                              sym_linear_rows)
from physika.solver.sym_utils import (
    NotInRingFragment, ast_to_sym, call_key, chain_rule, check_ring_int,
    contains_var, defs_rule, deriv_args, eval_numeric, expand_rule, int_exp,
    is_neg, is_num, lit_or_expr, rewrite, rewrite_once, ring_degree, sadd,
    sdiv, smul, sneg, sp_add, sp_mul, ssub, substitute_chain, substitute_defs,
    sym_app, sym_deriv, sym_expand_calls, sym_inline_calls, sym_poly_of,
    verify_identity, verify_ring_identity)
from physika.solver.utils import (KIND_SYMBOL, KINDS, eval_const, fmt_num,
                                  free_vars, in_domain, int_sum, kind_and_unit,
                                  reduces_true, to_expr)


def var(n):
    """
    Helper function to build a variable node.
    """
    return ("var", n)


def num(n):
    """
    Helper function to build a number node
    """
    return ("num", n)


def eq(lhs, rhs):
    """
    Helper function to build an equation node.
    """
    return ("cond_eq", lhs, rhs)


def run_block(variables, kind, conds):
    """
    Helper function to run ``solve_block`` without a candidate.

    Returns solved lines and the bindings.
    """
    bindings = {}
    node = ("solve_decl", variables, kind, None, conds, 1)
    return solve_block(node, mk_builtin_env(), {"program": []},
                       bindings), bindings


# Verify a quadriatic equation with cnaidate
# x * x == x + 12, roots 4 and -3
QUADRATIC = eq(("mul", var("x"), var("x")), ("add", var("x"), num(12)))

# m : ℝ ← [kg] = 2.0
#  a : ℝ ← [m, s**-2] = 3.0
UNIT_AST = {
    "program": [("unit_decl", "m", "ℝ", [("kg", 1)], num(2.0), 1),
                ("unit_decl", "a", "ℝ", [("m", 1), ("s", -2)], num(3.0), 2)]
}
NEWTON = {"kg": 1, "m": 1, "s": -2}


class TestSolver:
    """
    Tests for ``solve_block``, ``solve_equations`` and ``solve_bindings``.
    """

    def test_solve_block_candidate(self):
        """
        A candidate is verified by the kernel and no binding is added
        for a candidate verification.
        """
        env, bindings = mk_builtin_env(), {}
        node = ("solve_decl", ["x"], "ℕ", [num(4)], [QUADRATIC], 1)
        assert solve_block(node, env, {"program": []}, bindings) == [
            "Line 1: solve x : ℕ = 4  ✓ verified"
        ]
        assert bindings == {}
        node = ("solve_decl", ["x"], "ℤ", [num(-3)], [QUADRATIC], 2)
        assert solve_block(node, env, {"program": []}) == [
            "Line 2: solve x : ℤ = -3  ✓ verified"
        ]

    def test_solve_block_rejects(self):
        """
        Unsupported types and non-equation constraints should exit early
        """
        env = mk_builtin_env()
        node = ("solve_decl", ["x"], "ℂ", None, [QUADRATIC], 1)
        assert solve_block(node, env, {"program": []}) == [
            "Line 1: solve x: only ℕ / ℤ / ℝ unknowns are supported "
            "(got 'ℂ')"
        ]
        node = ("solve_decl", ["x"], "ℕ", None, [("cond_lt", var("x"), num(3))
                                                 ], 2)
        assert solve_block(node, env, {"program": []}) == [
            "Line 2: solve x: every constraint must be an equation "
            "`lhs == rhs`"
        ]

    def test_solve_block_dimensions(self):
        """
        A unit mismatch stops before the candidate check."
        """
        node = ("solve_decl", ["F"], ("unit_typed", "ℝ", [("kg", 1)]),
                [num(6)], [eq(var("F"), ("mul", var("m"), var("a")))], 3)
        assert solve_block(node, mk_builtin_env(), UNIT_AST) == [
            "Line 3: solve F: the equation is dimensionally inconsistent: "
            "[kg] vs [kg·m·s⁻²]"
        ]

    def test_solve_block_where_block(self):
        """
        The ``where:`` block form unpacks its locals and equation.
        """
        block = ("solve_block", [("solve_local", "k", "ℤ", num(2))],
                 eq(("mul", var("k"), var("x")), num(6)), None)
        node = ("solve_decl", ["x"], "ℤ", [num(3)], block, 1)
        out = solve_block(node, mk_builtin_env(), {"program": []})
        assert out == ["Line 1: solve x: free variable 'k'"]

    def test_solve_block_paths(self):
        """
        Test solve block paths without a candidate.
        """
        x_plus_y = ("add", var("x"), var("y"))
        x_minus_y = ("sub", var("x"), var("y"))
        # path 2, linear system
        out, b = run_block(["x", "y"], "ℤ",
                           [eq(x_plus_y, num(10)),
                            eq(x_minus_y, num(4))])
        assert out == [
            "Line 1: solve x, y : ℤ  →  x = 7, y = 3   "
            "(kernel-verified)"
        ]
        assert b == {"x": num(7), "y": num(3)}
        out, _ = run_block(
            ["x", "y"], "ℤ",
            [eq(x_plus_y, num(1)), eq(x_plus_y, num(2))])
        assert "not a uniquely-solvable linear system" in out[0]
        # path 3, polynomial, two roots so no binding
        out, b = run_block(["x"], "ℤ", [QUADRATIC])
        assert out == ["Line 1: solve x : ℤ  →  x = 4, -3   (kernel-verified)"]
        assert b == {}
        # path 4, symbolic chain
        kx = ("mul", var("k"), var("x"))
        out, b = run_block(["x"], "ℝ", [eq(kx, var("F"))])
        assert out[-1].startswith("Line 1: solve x : ℝ  →  x = (F / k)")
        assert b == {"x": ("div", var("F"), var("k"))}
        # path 4b, symbolic linear system
        out, b = run_block(
            ["x", "y"], "ℝ",
            [eq(("mul", num(2), var("x")), var("s")),
             eq(var("y"), kx)])
        assert "x = (s / 2), y = (k * (s / 2))" in out[-1]
        assert set(b) == {"x", "y"}
        # no path applies
        cos_x = eq(("call", "cos", [var("x")]), num(1))
        assert run_block(["x"], "ℝ", [cos_x])[0] == [
            "Line 1: solve x : ℝ  ✗ the equation is neither linear nor a "
            "polynomial in x"
        ]

    def test_solve_equations(self):
        """Reports every ``solve`` statement in order, skips the rest."""
        ast = {
            "program": [("decl", "y", "ℝ", num(1), 1),
                        ("solve_decl", ["x"], "ℕ", [num(4)], [QUADRATIC], 2),
                        ("solve_decl", ["x"], "ℕ", [num(5)], [QUADRATIC], 3)]
        }
        assert solve_equations(ast, mk_builtin_env()) == [
            "Line 2: solve x : ℕ = 4  ✓ verified",
            "Line 3: solve x : ℕ = 5  ✗ not a root"
        ]
        assert solve_equations({"program": []}, mk_builtin_env()) == []

    def test_solve_bindings(self, monkeypatch):
        """Bindings that use names with no value are dropped."""
        ast = {
            "program": [("solve_decl", ["x"], "ℕ", [num(4)], [QUADRATIC], 1)]
        }
        assert solve_bindings(ast, mk_builtin_env()) == {}

        def fake_solve_block(node, env, unified_ast, bindings=None):
            bindings.update({"x": var("y"), "y": num(1), "z": var("w")})
            return []

        monkeypatch.setattr(solver, "solve_block", fake_solve_block)
        assert solve_bindings(ast, mk_builtin_env()) == {
            "x": var("y"),
            "y": num(1)
        }


class TestSolverCandidate:
    """
    Tests for ``arith``, ``solve_candidate`` and ``unsupported_reason``.
    """

    def test_arith(self):
        """
        Builds ``Nat.*`` / ``Int.*`` terms with unknown as BVar(0).
        """
        nat_add = App(App(Const("Nat.add", ()), BVar(0)), Lit(2))
        assert arith(("add", var("x"), num(2)), "x", False) == nat_add
        int_neg = App(Const("Int.neg", ()), int_lit(3))
        assert arith(("neg", num(3)), "x", True) == int_neg

        mul = Const("Nat.mul", ())
        cube = App(App(mul, App(App(mul, BVar(0)), BVar(0))), BVar(0))
        assert arith(("pow", var("x"), num(3)), "x", False) == cube
        assert arith(("pow", var("x"), num(0)), "x", True) == int_lit(1)

    def test_arith_errors(self):
        """Constructs outside the ℕ or ℤ fragment raise with a reason."""
        cases = [
            (("var", "y"), False, "free variable 'y'"),
            (num(-1), False, "negative literal in a ℕ equation"),
            (("neg", var("x")), False, "negation in a ℕ equation"),
            (("pow", var("x"), var("x")), True, "non-literal exponent"),
            (("pow", var("x"), num(-1)), True, "negative exponent"),
            (("div", var("x"), num(2)), True, "operator 'div'"),
            ("x", True, "non-tuple node 'x'"),
        ]
        for expr, is_int, msg in cases:
            try:
                arith(expr, "x", is_int)
            except Exception as exc:
                assert str(exc) == msg
            else:
                raise AssertionError(f"no error for {expr!r}")

    def test_solve_candidate(self):
        """Kernel verdict, plus the ℝ and out-of-fragment messages."""
        env = mk_builtin_env()
        assert solve_candidate(["x"], "nat", [num(4)], [QUADRATIC], env,
                               "Line 1: solve x",
                               "ℕ") == ["Line 1: solve x : ℕ = 4  ✓ verified"]
        assert solve_candidate(
            ["x"], "int", [num(5)], [QUADRATIC], env, "Line 1: solve x",
            "ℤ") == ["Line 1: solve x : ℤ = 5  ✗ not a root"]
        assert solve_candidate(
            ["x"], "real", [num(4)], [QUADRATIC], env, "Line 1: solve x",
            "ℝ") == [
                "Line 1: solve x: the verify form (`= c`) needs an ℕ / ℤ "
                "unknown"
            ]
        assert solve_candidate(
            ["x"], "nat", [num(-4)], [QUADRATIC], env, "Line 1: solve x",
            "ℕ") == ["Line 1: solve x: negative literal in a ℕ equation"]

    def test_unsupported_reason(self):
        """Names the unsupported construct, ``None`` when supported."""
        assert unsupported_reason([QUADRATIC], "x", True) is None
        div = eq(("div", num(6), var("x")), num(2))
        assert unsupported_reason([div], "x", True) == "operator 'div'"


class TestSolverDims:
    """
    Tests for the dimensional analysis of ``solve`` statements.
    """

    def test_dim_solver(self):
        """
        Tests for dimensional solvers functions and chceker.
        """
        assert is_number(("mul", num(2), ("neg", num(3))))
        assert not is_number(("mul", num(2), var("x")))
        assert not is_number("x")

        expr = ("sub", ("add", var("a"), ("neg", var("b"))), var("c"))
        assert add_terms(expr) == [var("a"), var("b"), var("c")]
        assert add_terms(var("a")) == [var("a")]

        env = mk_builtin_env()
        newton = unit_to_dimvec_cic(NEWTON, env)
        assert same_dim(newton, unit_to_dimvec_cic(dict(NEWTON), env), env)
        assert not same_dim(newton, unit_to_dimvec_cic({"kg": 1}, env), env)

        env = mk_builtin_env()
        assert dim_str(unit_to_dimvec_cic(NEWTON, env), env) == "kg·m·s⁻²"

        env = mk_builtin_env()
        block = [("solve_local", "k", ("unit_typed", "ℝ", [("s", 1)]), num(1)),
                 ("solve_local", "c", "ℝ", num(1))]
        dims, func_dims = dim_env(["F"], NEWTON, block, UNIT_AST, env)
        assert set(dims) == {"m", "a", "k", "F"}
        assert dim_str(dims["k"], env) == "s"
        assert dim_str(dims["F"], env) == "kg·m·s⁻²"
        assert func_dims == {}
        # an unknown with no declared unit is dimensionless
        dims, _ = dim_env(["x"], {}, [], {"program": []}, env)
        assert dim_str(dims["x"], env) == "1"

        unit_param = ("unit_typed", "ℝ", [("kg", 1)])
        assert has_units({"kg": 1}, [], {"program": []})
        assert has_units({}, [], UNIT_AST)
        assert has_units({}, [("solve_local", "k", unit_param, num(1))],
                         {"program": []})
        assert has_units({}, [], {
            "program": [],
            "functions": {
                "f": {
                    "params": [("p", unit_param)]
                }
            }
        })
        assert not has_units({}, [("solve_local", "k", "ℝ", num(1))],
                             {"program": [("decl", "y", "ℝ", num(1), 1)]})

    def test_check_dimensions(self):
        """
        Check dimensions of a solve lblock statemetn.
        """
        env = mk_builtin_env()
        tag = "Line 3: solve F"
        f_eq_ma = eq(var("F"), ("mul", var("m"), var("a")))
        # no units anywhere: nothing to check
        assert check_dimensions(["x"], {}, [QUADRATIC], [], None,
                                {"program": []}, env, tag) == ([], True)
        assert check_dimensions(
            ["F"], NEWTON, [f_eq_ma], [], None, UNIT_AST, env,
            tag) == ([f"{tag}: dimensions consistent  ([kg·m·s⁻²])"], True)
        assert check_dimensions(
            ["F"], {"kg": 1}, [f_eq_ma], [], None, UNIT_AST, env, tag) == ([
                f"{tag}: the equation is dimensionally inconsistent: "
                "[kg] vs [kg·m·s⁻²]"
            ], False)
        # a bare number takes the other side's unit
        assert check_dimensions(["F"], NEWTON, [eq(var("F"), num(6))], [],
                                None, UNIT_AST, env, tag)[1]
        # exp of a dimensioned quantity is not well-formed
        bad = eq(("call", "exp", [var("m")]), num(2))
        lines, ok = check_dimensions(["F"], NEWTON, [f_eq_ma, bad], [], None,
                                     UNIT_AST, env, tag)
        assert not ok
        assert lines == [
            f"{tag}: equation 2 is not dimensionally "
            "well-formed: exp() requires a dimensionless "
            "argument"
        ]


class TestSolverUtils:
    """
    Tests for the helpers shared by the solver.
    """

    def test_kinds(self):
        """Check type kins and units declared in a solve block"""
        assert KIND_SYMBOL == {"nat": "ℕ", "int": "ℤ", "real": "ℝ"}
        assert set(KINDS) == {"ℕ", "ℤ", "ℝ"}

        assert kind_and_unit("ℤ") == ("int", {})
        assert kind_and_unit("ℂ") == (None, {})
        assert kind_and_unit(("unit_typed", "ℝ", [("m", 1)])) == ("real", {
            "m": 1
        })
        assert kind_and_unit(("tensor", [(2, "invariant")])) == (None, {})

    def test_free_vars(self):
        """Test free variables should be catched properly."""
        expr = ("add", var("x"), ("call", "f", [var("k"), num(2)]))
        assert free_vars(expr, {"x"}) == {"k"}
        assert free_vars(expr, set()) == {"x", "k"}
        assert free_vars(num(1), set()) == set()

    def test_eval_const(self):
        """Evalaute const values in a solve block"""
        assert eval_const(("div", num(1), num(3)), {}) == Fraction(1, 3)
        assert eval_const(("pow", var("a"), num(2)), {"a": 3}) == 9
        assert eval_const(("neg", num(2.5)), {}) == -2.5

    def test_to_expr(self):
        """Tests for getting the correct expression for a const or variable"""
        assert to_expr(Fraction(1, 3)) == ("div", num(1), num(3))
        assert to_expr(Fraction(4)) == num(4)
        assert to_expr(2.5) == num(2.5)
        assert to_expr(var("x")) == var("x")

    def test_fmt_num(self):
        """Tets that the crreect format for a numbewr is presented"""
        assert fmt_num(Fraction(4)) == "4"
        assert fmt_num(Fraction(1, 3)) == "1/3"
        assert fmt_num(2.0) == "2"
        assert fmt_num(1 / 3) == "0.333333"
        assert fmt_num(7) == "7"

    def test_in_domain(self):
        """Checks the value is in the corrrect domain"""
        assert in_domain(Fraction(-2), "int")
        assert not in_domain(Fraction(-2), "nat")
        assert not in_domain(Fraction(1, 2), "int")
        assert not in_domain(2.0, "nat")
        assert in_domain(2.5, "real")

    def test_int_sum(self):
        """Tetss the fast path for int value is correct"""
        env = mk_builtin_env()
        assert int_sum([]) == int_lit(0)
        assert int_sum([5]) == int_lit(5)
        total = App(App(Const("Int.eqb", ()), int_sum([3, -4, 10])),
                    int_lit(9))
        assert reduces_true(total, env)

    def test_reduces_true(self):
        """Checks the correct boolean value is presented."""
        env = mk_builtin_env()

        def eqb(a, b):
            return App(App(Const("Nat.eqb", ()), Lit(a)), Lit(b))

        assert reduces_true(eqb(4, 4), env)
        assert not reduces_true(eqb(3, 5), env)
        # a stuck term is not verified
        assert not reduces_true(Const("unknown", ()), env)


class TestSolverLinear:
    """
    Tests for ``lin``, ``frac``, ``eq_row``, ``gauss_jordan``,
    ``verify_linear_row``, ``linear_system`` and ``solve_linear``.
    """

    def test_lin(self):
        """Linear forms, constants from scope, and non-linear terms."""
        expr = ("sub", ("mul", num(2), var("x")), var("c"))
        assert lin(expr, ["x"], {"c": Fraction(4)}) == {
            "x": Fraction(2),
            None: Fraction(-4)
        }
        assert lin(("div", var("x"), num(2)), ["x"]) == {"x": Fraction(1, 2)}
        assert lin(("mul", var("x"), var("y")), ["x", "y"]) is None
        assert lin(("pow", var("x"), num(2)), ["x"]) is None
        assert lin(var("k"), ["x"]) is None

    def test_rows(self):
        """``frac`` and ``eq_row`` give exact ``lhs - rhs`` rows."""
        assert (frac(3), frac(2.0), frac(0.25)) == (Fraction(3), Fraction(2),
                                                    Fraction(1, 4))
        assert eq_row(eq(var("x"), num(4)), ["x"]) == {
            "x": Fraction(1),
            None: Fraction(-4)
        }

    def test_gauss_jordan(self):
        """Unique, singular and non-square systems."""
        one = Fraction(1)
        rows = [{
            "x": one,
            "y": one,
            None: Fraction(-10)
        }, {
            "x": one,
            "y": -one,
            None: Fraction(-4)
        }]
        assert gauss_jordan(rows, ["x", "y"]) == {
            "x": Fraction(7),
            "y": Fraction(3)
        }
        assert gauss_jordan([rows[0], rows[0]], ["x", "y"]) is None
        assert gauss_jordan(rows[:1], ["x", "y"]) is None

    def test_verify_linear_row(self):
        """The kernel checks a row with rational coefficients."""
        row = {"x": Fraction(2), None: Fraction(-1)}
        env = mk_builtin_env()
        assert verify_linear_row(row, {"x": Fraction(1, 2)}, env)
        assert not verify_linear_row(row, {"x": Fraction(1)}, env)

    def test_linear_system(self):
        """Other names become auxiliary unknowns, one equation in one
        unknown is left to the polynomial path."""
        eq_k = eq(("add", var("x"), var("k")), num(3))
        _, aux, allvars = linear_system([eq_k], ["x"], {})
        assert (aux, allvars) == (["k"], ["x", "k"])
        assert linear_system([eq_k], ["x"], {"k": 1}) is None
        assert linear_system([eq(("mul", var("x"), var("x")), num(1))],
                             ["x", "y"], {}) is None

    def test_solve_linear(self):
        """Report lines for ℤ, rational and ℕ solutions."""
        env = mk_builtin_env()
        conds = [
            eq(("add", var("x"), var("y")), num(10)),
            eq(("sub", var("x"), var("y")), num(5))
        ]
        system = linear_system(conds, ["x", "y"], {})
        assert solve_linear(system, ["x", "y"], "real", env, "T", "ℝ") == [
            "T : ℝ  →  x = 15/2, y = 5/2   (kernel-verified)"
        ]
        assert "no ℤ solution" in solve_linear(system, ["x", "y"], "int", env,
                                               "T", "ℤ")[0]


class TestSolverPoly:
    """
    Tests for the polynomial helpers and ``solve_poly``.
    """

    def test_poly_arithmetic(self):
        """
        Test helper funcctions ``p_add``, ``p_mul``, ``p_pow``,
        ``p_deriv`` and ``p_eval``.
        """
        x1 = {1: Fraction(1), 0: Fraction(1)}  # x + 1
        assert p_add(x1, {1: Fraction(1)}, sub=True) == {0: Fraction(1)}
        assert p_mul(x1, x1) == p_pow(x1, 2) == {
            2: Fraction(1),
            1: Fraction(2),
            0: Fraction(1)
        }
        assert p_deriv({2: Fraction(3), 0: Fraction(7)}) == {1: Fraction(6)}
        assert p_eval(x1, Fraction(2)) == 3

    def test_poly_of(self):
        """Scope constants, ``deriv``, ``sqrt``, inlined functions and
        non-polynomials."""
        square = ("pow", ("sub", var("x"), var("a")), num(2))
        assert poly_of(square, "x", {"a": 3}, {}, {}) == {
            2: Fraction(1),
            1: Fraction(-6),
            0: Fraction(9)
        }
        assert poly_of(("call", "deriv", [square]), "x", {"a": 3}, {}, {}) == {
            1: Fraction(2),
            0: Fraction(-6)
        }
        assert poly_of(("call", "sqrt", [num(4)]), "x", {}, {}, {}) == {0: 2.0}
        assert poly_of(("call", "sqrt", [num(-4)]), "x", {}, {}, {}) is None
        assert poly_of(("div", var("x"), num(0)), "x", {}, {}, {}) is None
        assert poly_of(("div", num(1), var("x")), "x", {}, {}, {}) is None
        funcs = {
            "double": {
                "params": [("y", "ℝ")],
                "statements": [],
                "body": ("mul", num(2), var("y"))
            }
        }
        assert inline_call("double", [var("x")], "x", {}, funcs, {}) == {
            1: Fraction(2)
        }
        assert wrt_name([var("f"), var("t")]) == "t"

    def test_roots(self):
        """Closed form, rational root theorem and numeric roots."""
        assert divisors(-6) == [1, 2, 3, 6]
        cubic = {
            3: Fraction(2),
            2: Fraction(-5),
            1: Fraction(1),
            0: Fraction(2)
        }
        assert rational_roots(cubic) == [
            Fraction(-1, 2), Fraction(1),
            Fraction(2)
        ]
        assert numeric_roots({
            3: Fraction(1),
            0: Fraction(-2)
        }) == [1.2599210499]
        # x (x - 2)(x - 3), the factored out root 0 is not returned
        assert poly_roots({
            3: Fraction(1),
            2: Fraction(-5),
            1: Fraction(6)
        }) == [Fraction(3), Fraction(2)]
        assert poly_roots({2: Fraction(1), 0: Fraction(1)}) == []

    def test_verify(self):
        """Kernel checks of a root and of a derivative."""
        env = mk_builtin_env()
        poly = {2: Fraction(1, 3), 0: Fraction(-3)}
        assert verify_poly_root(poly, Fraction(3), env)
        assert not verify_poly_root(poly, Fraction(2), env)
        f = {2: Fraction(3), 0: Fraction(7)}
        assert verify_deriv(f, {1: Fraction(6)}, env)
        assert not verify_deriv(f, {1: Fraction(3)}, env)

    def test_report_helpers(self):
        """`Test for helpers for report results in solver block
        `deriv_call`` and ``poly_str``.
        """
        f = ("pow", var("t"), num(2))
        assert deriv_call(
            ("add", ("call", "deriv", [f, var("t")]), num(1))) == (f, "t")
        assert deriv_call(var("t")) is None
        assert poly_str({
            2: Fraction(3),
            1: Fraction(1),
            0: Fraction(-7)
        }, "x") == "3·x^2 + x - 7"

    def test_solve_poly(self):
        """Roots, ``deriv`` lines, trivial and rootless equations."""
        env = mk_builtin_env()
        out = solve_poly("x", "nat", {}, [QUADRATIC], {}, {}, env, "T", "ℕ")
        assert out == ["T : ℕ  →  x = 4   (kernel-verified)"]
        bindings = {}
        cube = ("call", "deriv", [("pow", var("x"), num(3))])
        out = solve_poly("x", "real", {"m": 1}, [eq(cube, num(12))], {}, {},
                         env, "T", "ℝ", bindings)
        assert out == [
            "T: d/dx [ x^3 ]  =  3·x^2   (power rule, "
            "kernel-checked)", "T : ℝ  →  x = 2, -2  [m]   (kernel-verified)"
        ]
        assert bindings == {}
        assert solve_poly("x", "int", {}, [eq(var("x"), var("x"))], {}, {},
                          env, "T",
                          "ℤ") == ["T : ℤ  →  any x (equation is trivial)"]
        sq = eq(("pow", var("x"), num(2)), num(-1))
        assert solve_poly("x", "int", {}, [sq], {}, {}, env, "T",
                          "ℤ") == ["T : ℤ  ✗ no ℤ root"]
        assert solve_poly("x", "int", {}, [eq(var("k"), num(1))], {}, {}, env,
                          "T", "ℤ") is None


class TestSolverChain:
    """
    Tests for the chain helpers and ``solve_chain``.
    """

    def test_patterns(self):
        """New names and powers are definitions, the unknown is not."""
        square = ("pow", var("q"), num(2))
        assert is_var(var("q")) and not is_var(num(1))
        assert fresh_pattern(var("q"), {}, "x") == "q"
        assert fresh_pattern(square, {}, "x") == square
        assert fresh_pattern(var("x"), {}, "x") is None
        assert fresh_pattern(var("q"), {"q": num(1)}, "x") is None
        value = ("add", var("a"), var("b"))
        assert fresh_definition(var("q"), value, var("q"), value, {},
                                "x") == ("q", value)
        loop = ("add", var("q"), num(1))
        assert fresh_definition(var("q"), loop, var("q"), loop, {},
                                "x") is None
        side = ("mul", num(2), var("alpha"))
        assert fresh_linear_pattern(side, var("pi"), side, var("pi"), {},
                                    "x", {}) == ("alpha", ("div", var("pi"),
                                                           num(2)))
        # ħ * ω: neither bare factor is the coefficient
        side = ("mul", var("h"), var("w"))
        assert fresh_linear_pattern(side, var("E"), side, var("E"), {}, "x",
                                    {}) is None

    def test_sym_solve_poly(self):
        """Linear and quadratic roots, ``None`` for other degrees."""
        assert ssqrt(Fraction(9, 4)) == Fraction(3, 2)
        assert ssqrt(var("d")) == ("call", "sqrt", [var("d")])
        assert sym_solve_poly({
            1: var("k"),
            0: var("F")
        }) == [("div", ("neg", var("F")), var("k"))]
        assert sym_solve_poly({
            2: Fraction(1),
            0: Fraction(-4)
        }) == [Fraction(2), Fraction(-2)]
        assert sym_solve_poly({3: Fraction(1), 0: Fraction(1)}) is None
        assert sym_solve_poly({1: Fraction(0), 0: Fraction(1)}) is None

    def test_solve_chain(self):
        """Definitions, a checked claim and the final solve."""
        a, b, q, y = var("a"), var("b"), var("q"), var("y")
        conds = [
            eq(q, ("add", a, b)),
            eq(("sub", q, b), a),
            eq(("mul", y, q), a)
        ]
        bindings = {}
        out = solve_chain(conds, "y", {}, {}, "T", "ℝ", bindings)
        assert out[0] == "T: q := (a + b)   (definition)"
        assert out[1].startswith("T: step 2:") and "✓ verified" in out[1]
        assert out[-1].startswith("T : ℝ  →  y = (a / (a + b))")
        assert bindings == {"y": ("div", a, ("add", a, b))}
        deriv_t = ("call", "deriv", [("mul", var("v"), var("t")), var("t")])
        assert chain_side(deriv_t, {"v": num(3), "t": num(0)}, {}) == num(3)
        assert solve_chain([], "y", {}, {}, "T", "ℝ") is None
        assert solve_chain([eq(("call", "cos", [y]), a)], "y", {}, {}, "T",
                           "ℝ") is None


class TestSolverSymLinear:
    """
    Tests for symbolic solve of linear systems and helpers.
    """

    def test_rows_and_elimination(self):
        """Vrrifes correct symbolic rows, cross terms operations."""
        assert is_zero(Fraction(0)) and not is_zero(var("k"))
        kx = eq(("mul", var("k"), var("x")), var("F"))
        assert sym_linear_rows([kx], ["x"], {}, {}) == [[var("k"), var("F")]]
        xy = eq(("mul", var("x"), var("y")), num(1))
        assert sym_linear_rows([xy, xy], ["x", "y"], {}, {}) is None
        zero, one = Fraction(0), Fraction(1)
        assert sym_gauss_jordan([[Fraction(2), zero,
                                  var("s")],
                                 [zero, one,
                                  var("t")]]) == [("div", var("s"), num(2)),
                                                  var("t")]
        assert sym_gauss_jordan([[zero, zero, one], [zero, zero, one]]) is None

    def test_solve_sym_linear(self):
        """Report lines, bindings and linear ystems are not square."""
        conds = [
            eq(("mul", num(2), var("x")), var("s")),
            eq(var("y"), ("mul", var("k"), var("x")))
        ]
        bindings = {}
        out = solve_sym_linear(conds, ["x", "y"], {}, {}, "T", "ℝ", {"m": 1},
                               bindings)
        assert out[1] == "T: symbolic in k, s"
        assert "x = (s / 2), y = (k * (s / 2))  [m]" in out[2]
        assert bindings == {
            "x": ("div", var("s"), num(2)),
            "y": ("mul", var("k"), ("div", var("s"), num(2)))
        }
        assert solve_sym_linear(conds[:1], ["x", "y"], {}, {}, "T", "ℝ",
                                {}) is None


class TestSymUtils:
    """
    Tests for the symbolic helpers in ``sym_utils``.
    """

    def test_coefficient_arithmetic(self):
        """Numbers stay exact in symbolic expresisons and
        0, 1, -1 and negations simplify."""
        a, b = var("a"), var("b")
        assert is_num(Fraction(1, 2)) and not is_num(a)
        assert is_neg(("neg", a)) and not is_neg(a)
        assert sadd(Fraction(1, 2), 1) == Fraction(3, 2)
        assert sadd(0, a) == a
        assert ssub(0, a) == ("neg", a)
        assert sneg(sneg(a)) == a
        assert smul(-1, a) == ("neg", a)
        assert smul(("neg", a), b) == ("neg", ("mul", a, b))
        assert smul(0, a) == 0
        assert sdiv(a, 1) == a and sdiv(1, 4) == Fraction(1, 4)
        assert sdiv(("neg", a), ("neg", b)) == ("div", a, b)
        assert sp_add({1: Fraction(2)}, {1: var("k")}) == {
            1: ("add", num(2), var("k"))
        }
        assert sp_mul({1: Fraction(1)}, {
            1: var("k"),
            0: Fraction(3)
        }) == {
            2: var("k"),
            1: Fraction(3)
        }

    def test_substitution(self):
        """Definitions, chain keys, ``deriv`` and function inlining."""
        cos = ("call", "cos", [var("θ")])
        assert contains_var(cos, "θ") and not contains_var(cos, "x")
        assert lit_or_expr(num(3)) == Fraction(3)
        d = ("call", "deriv", [("mul", var("a"), var("t")), var("t")])
        assert deriv_args(d) == (("mul", var("a"), var("t")), "t")
        assert deriv_args(("call", "deriv", [var("f")])) is None
        defs = {"a": ("mul", num(2), var("b")), "b": var("c")}
        assert substitute_defs(var("a"), defs) == ("mul", num(2), var("c"))
        assert rewrite_once(("neg", var("a")), defs_rule,
                            defs) == (("neg", defs["a"]), True)
        assert rewrite(var("a"), defs_rule, defs, 1) == defs["a"]
        square = ("pow", var("q"), num(2))
        assert chain_rule(square, {square: var("s")}) == (var("s"), True)
        # the t of deriv(f, t) is kept
        assert substitute_chain(d, {"t": num(0)}) == d
        assert expand_rule(d, None) == (var("a"), True)
        assert sym_expand_calls(
            ("add", d, var("x"))) == ("add", var("a"), var("x"))
        funcs = {
            "sq": {
                "params": [("y", "ℝ")],
                "statements": [],
                "body": ("mul", var("y"), var("y"))
            }
        }
        assert sym_inline_calls(("call", "sq", [var("a")]),
                                funcs) == ("mul", var("a"), var("a"))

    def test_poly_and_deriv(self):
        """Symbolic polynomials, exponents and derivatives."""
        cos = ("call", "cos", [var("θ")])
        assert sym_poly_of(("add", ("mul", cos, var("x")), num(1)), "x") == {
            1: cos,
            0: Fraction(1)
        }
        assert sym_poly_of(("div", num(1), var("x")), "x") is None
        assert int_exp(("neg", num(3))) == -3 and int_exp(var("k")) is None
        two_t = ("mul", num(2), var("t"))
        assert sym_deriv(("call", "sin", [two_t]),
                         "t") == ("mul", ("call", "cos", [two_t]), num(2))
        assert sym_deriv(("call", "tan", [var("t")]), "t") is None

    def test_eval_numeric(self):
        """Floats, complex numbers, functions and errors."""
        subs = {"a": Fraction(3, 2)}
        assert eval_numeric(("mul", var("a"), ("call", "sqrt", [num(4)])),
                            subs) == 3.0
        assert eval_numeric(("mul", ("imaginary", ), ("imaginary", )),
                            {}) == -1
        funcs = {
            "sq": {
                "params": [("y", "ℝ")],
                "statements": [],
                "body": ("mul", var("y"), var("y"))
            }
        }
        assert eval_numeric(("call", "sq", [var("a")]), subs, funcs) == 2.25
        with pytest.raises(ValueError):
            eval_numeric(("call", "unknown", [num(1)]), {})

    def test_ring_translation(self):
        """Atoms, degrees and ``Sym`` terms of the ring fragment."""
        check_ring_int(Fraction(4))
        with pytest.raises(NotInRingFragment):
            check_ring_int(2.5)
        cos = ("call", "cos", [var("θ")])
        assert call_key(cos) == ("cos", (("var", "θ"), ))
        with pytest.raises(NotInRingFragment):
            call_key(("call", "cos", [("add", var("a"), var("b"))]))
        atoms = {}
        assert ring_degree(("mul", ("pow", var("x"), num(2)), cos), atoms) == 3
        assert atoms == {"x": 0, call_key(cos): 1}
        with pytest.raises(NotInRingFragment):
            ring_degree(("div", var("x"), var("y")), {})
        assert sym_app("atom", Lit(0)) == App(Const("Sym.atom", ()), Lit(0))
        assert ast_to_sym(("neg", var("x")),
                          {"x": 0}) == sym_app("neg", sym_app("atom", Lit(0)))

    def test_verify(self):
        """Kernel ring check and rando instances checker."""
        a, b = var("a"), var("b")
        assert verify_ring_identity(("add", a, b), ("add", b, a)) is True
        assert verify_ring_identity(("add", a, b), a) is False
        assert verify_ring_identity(("div", a, b), a) is None
        sin2 = ("pow", ("call", "sin", [var("t")]), num(2))
        cos2 = ("pow", ("call", "cos", [var("t")]), num(2))
        assert verify_identity(sin2, ("sub", num(1), cos2), ["t"]) is True
        assert verify_identity(sin2, cos2, ["t"]) is False
        # root of x² == a substituted for x
        assert verify_identity(("pow", var("x"), num(2)),
                               a, ["a"],
                               unknown="x",
                               root=("call", "sqrt", [a])) is True
        assert verify_identity(("call", "sqrt", [("neg", a)]), a,
                               ["a"]) is None
