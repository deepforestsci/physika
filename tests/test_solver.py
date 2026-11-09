from fractions import Fraction

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
