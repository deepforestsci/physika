"""
Dimensional analysis in the CIC kernel: ``Int`` arithmetic reduction,
componentwise ``DimVec`` arithmetic, the ``x: ℝ ← [unit] = expr``
declaration check, and the ``def f(x: ℝ ← [u], …): ℝ ← [r]`` function
return-dimension check (``check_func_dims``).
"""
import pytest

from physika.core.expr import App, Const
from physika.core.inductive import mk_builtin_env
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.reduction import is_def_eq
from physika.utils.cic_utils.inductive_utils import (
    int_lit,
    mk_dim_vec_literal,
    read_dim_vec_literal,
)
from physika.units import (
    UnitError,
    check_unit_decl_cic,
    dim_analysis,
    dim_sqrt_cic,
    expr_unit_to_dimvec_cic,
    infer_unit,
    local_annotation,
    param_dim_spec,
    parse_unit_ast,
    record_func_dim,
    sig_unit_spec,
    unit_div,
    unit_mul,
    unit_pow,
    unit_to_dim_const_name,
    unit_to_dim_vec_exponents,
    unit_to_dimvec_cic,
    update_unit_cic_env,
    update_unit_env,
)

from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast
from physika.core.elab.elab import Elab
from physika.units import check_func_dims


@pytest.fixture(scope="module")
def env():
    """
    Helper function to create a CIC environment with builtin inductive types.
    """
    return mk_builtin_env()


def Int(k):
    """A ``Int`` CIC term for an int ``k``."""
    return int_lit(k)


def elab_cic(src):
    """
    Helper function to elaborate a source code and run dimensional analysis
    pass and returns CIC elaboration result with any units errors.
    """
    symbol_table.clear()
    nodes = parser.parse(src, lexer=lexer)
    u = build_unified_ast([s for s in nodes], symbol_table)
    elab = Elab(mk_builtin_env())
    res = elab.elaborate(u)
    errs = list(res.get("errors", []))
    func_dims: dict = {}
    for fname, fdef in u.get("functions", {}).items():
        check_func_dims(fname, fdef, elab.state.env, errs.append, func_dims)
    res["errors"] = errs
    return res


def elab_program(src):
    """
    Helper function to parse and elaborate a physika file (src) and return
    ``(unified_ast, cic_env)`` for running dimensional analysis.
    """
    symbol_table.clear()
    lexer.lexer.lineno = 1
    nodes = parser.parse(src, lexer=lexer)
    u = build_unified_ast([s for s in nodes], symbol_table)
    elab = Elab(mk_builtin_env())
    elab.elaborate(u)
    return u, elab.state.env


class TestUnitAlgebra:
    """
    Tests for ``unit_div`` and ``unit_pow`` operations over exponent arithmetic
    ``Unit`` dict.
    """

    @pytest.mark.parametrize("u1, u2, expected", [
        ({
            "m": 1
        }, {
            "s": 1
        }, {
            "m": 1,
            "s": -1
        }),
        ({
            "kg": 1
        }, {
            "kg": 1
        }, {}),
        ({
            "m": 2,
            "s": -1
        }, {
            "m": 1,
            "s": -1
        }, {
            "m": 1
        }),
    ])
    def test_div(self, u1, u2, expected):
        """
        Tests for ``unit_div``.
        """
        assert unit_div(u1, u2) == expected

    @pytest.mark.parametrize("u, exp, expected", [
        ({
            "m": 1,
            "s": -2
        }, 3, {
            "m": 3,
            "s": -6
        }),
        ({
            "m": 2
        }, 0, {}),
        ({
            "m": 1,
            "s": -1
        }, -1, {
            "m": -1,
            "s": 1
        }),
    ])
    def test_pow(self, u, exp, expected):
        """
        Tests for ``unit_pow``.
        """
        assert unit_pow(u, exp) == expected


class TestParseUnitAst:
    """
    Checks parser unit declaration to ``Unit`` conversion.
    """

    def test_unit_pairs_to_dict(self):
        """
        Verifies different test cases from parsed unit declaration to Unit
        dict.
        """
        assert parse_unit_ast([("m", 1), ("s", -2)]) == {"m": 1, "s": -2}

        # Should reduce to dimensionless
        assert parse_unit_ast([("m", 1), ("m", -1)]) == {}


class TestUnitToDimConstName:
    """
    Test from ``Unit`` dict to Dim Const name conversion for registering in
    CIC.
    """

    def test_dim_const(self):
        """
        Check proper conversion of Unit to Dim Const.
        """
        assert unit_to_dim_const_name({}) == "dim.one"
        # existing units
        assert unit_to_dim_const_name({"s": -2, "kg": 1}) == "dim.kg+1_s-2"
        # order should not mattter
        assert unit_to_dim_const_name({
            "m": 1,
            "kg": 1
        }) == unit_to_dim_const_name({
            "kg": 1,
            "m": 1
        })  # noqa: E501


class TestUnitToDimVecExponents:
    """
    Check Unit to DimVec conversion. DimVec should alaways be a 7 length
    vector. If a quantity is dimensionless, we should have a zero array
    length 7.
    """

    def test_si_unit_to_exponent_list(self):
        """
        Test cases for Unit to DimVec conversion.
        """
        assert unit_to_dim_vec_exponents({
            "kg": 1,
            "s": -2
        }) == [1, 0, -2, 0, 0, 0, 0]  # noqa: E501
        # dimensionless case
        assert unit_to_dim_vec_exponents({}) == [0, 0, 0, 0, 0, 0, 0]


class TestUnitToDimvecCic:
    """
    Test for checking ``DimVec`` CIC elaboration.
    """

    def test_si_unit_is_reducible(self, env):
        lctx, mctx = LocalContext(), MetaVarContext()
        v = unit_to_dimvec_cic({"kg": 1, "s": -2}, env)
        assert read_dim_vec_literal(v, env, lctx,
                                    mctx) == [1, 0, -2, 0, 0, 0,
                                              0]  # noqa: E501

        # case for custom dim
        c = unit_to_dimvec_cic({"eV": 1}, env)
        assert isinstance(c, Const) and c.name == "dim.eV+1"
        assert env.constants.get("dim.eV+1") is not None


class TestSignatureSpecParsers:
    """
    ``sig_unit_spec`` / ``local_annotation`` / ``param_dim_spec`` should
    read a dimension bracket off a typespec.
    """

    def test_sig_unit_spec_si_and_custom(self):
        """
        Check if parsed units are converted to dictionary.
        """
        assert sig_unit_spec([("kg", 1), ("s", -2)]) == {"kg": 1, "s": -2}
        assert sig_unit_spec([("eV", 1)]) == {"eV": 1}

    def test_local_annotation_reads_unit_typed_only(self):
        """
        Test if a unit declaration AST is converted propert to Unit
        """
        assert local_annotation(("unit_typed", "ℝ", [("m", 1),
                                                     ("s", -1)])) == {
                                                         "m": 1,
                                                         "s": -1
                                                     }
        assert local_annotation("ℝ") is None

    def test_param_dim_dimensionless(self):
        """
        Check a dimensionless expression returns an empty Unit.
        """
        assert param_dim_spec("ℝ") == {}
        assert param_dim_spec(("tensor", [("n", "invariant")])) == {}


class TestRecordFuncDim:
    """
    Verify ``record_func_dim`` stores a verified function's return dimension
    into ``func_dims`` (CIC term) and ``func_sigs`` a ``Unit``."""

    def test_records_term_and_concrete_unit(self, env):
        """
        record_func_dim should store both in func_sigs a Unit and at
        ``func_dims`` a verified CIC term.

        """
        force = mk_dim_vec_literal([1, 1, -2, 0, 0, 0, 0])  # kg*m*s^2
        func_dims, func_sigs = {}, {}
        record_func_dim("net_force", force, [{}, {}], env, func_dims,
                        func_sigs)
        assert func_sigs["net_force"] == \
            ([{}, {}], {"kg": 1, "m": 1, "s": -2})
        assert func_dims["net_force"] is force

        # non supported SI unit is not stored
        opaque = unit_to_dimvec_cic({"eV": 1}, env)
        func_dims, func_sigs = {}, {}
        record_func_dim("energy", opaque, [{}], env, func_dims, func_sigs)
        assert func_sigs["energy"] == ([{}], None)


class TestIntArithmetic:
    """Checks ``Int.neg`` / ``Int.add`` / ``Int.sub`` / ``Int.mul`` reduce via
    ``Int.rec`` to the constructor."""

    @pytest.mark.parametrize("a, b, expected", [
        (2, 3, 5),
        (-2, -3, -5),
        (2, -1, 1),
        (-5, 2, -3),
        (0, 0, 0),
    ])
    def test_add(self, env, a, b, expected):
        """
        Tests for verifying correct reduction of Int.add operation.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        s = App(App(Const("Int.add", ()), Int(a)), Int(b))
        assert is_def_eq(s, Int(expected), env, lctx, mctx)[0]

    @pytest.mark.parametrize("a, b, expected", [
        (5, 3, 2),
        (2, 5, -3),
        (-1, -1, 0),
        (3, -4, 7),
    ])
    def test_sub(self, env, a, b, expected):
        """
        Tests for verifying correct reduction of Int.sub operation.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        s = App(App(Const("Int.sub", ()), Int(a)), Int(b))
        assert is_def_eq(s, Int(expected), env, lctx, mctx)[0]

    @pytest.mark.parametrize("a, b, expected", [
        (2, 3, 6),
        (-2, 3, -6),
        (2, -3, -6),
        (-2, -4, 8),
        (0, 5, 0),
    ])
    def test_mul(self, env, a, b, expected):
        """
        Tests for verifying correct reduction of Int.mul operation.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        s = App(App(Const("Int.mul", ()), Int(a)), Int(b))
        assert is_def_eq(s, Int(expected), env, lctx, mctx)[0]

    @pytest.mark.parametrize("a, expected", [(3, -3), (-2, 2), (0, 0)])
    def test_neg(self, env, a, expected):
        """
        Tests for verifying correct reduction of Int.neg operation.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        s = App(Const("Int.neg", ()), Int(a))
        assert is_def_eq(s, Int(expected), env, lctx, mctx)[0]


class TestDimArithmetic:
    """
    Verify ``dim_mul`` / ``dim_div`` / ``dim_pow`` computes componentwise on
    ``Vec Int 7`` with kernel.
    """

    def test_dim_mul_is_componentwise_add(self, env):
        """
        ``dim_mul`` should compute correctly addition resulting unit operation
        across dimensions.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        kg = mk_dim_vec_literal([1, 0, 0, 0, 0, 0, 0])
        accel = mk_dim_vec_literal([0, 1, -2, 0, 0, 0, 0])
        force = App(App(Const("dim_mul", ()), kg), accel)
        assert read_dim_vec_literal(force, env, lctx,
                                    mctx) == [1, 1, -2, 0, 0, 0, 0]

    def test_dim_div_is_componentwise_sub(self, env):
        """
        ``dim_div`` should compute correctly substraction resulting unit
        operation across dimensions.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        force = mk_dim_vec_literal([1, 1, -2, 0, 0, 0, 0])
        accel = mk_dim_vec_literal([0, 1, -2, 0, 0, 0, 0])
        mass = App(App(Const("dim_div", ()), force), accel)
        assert read_dim_vec_literal(mass, env, lctx,
                                    mctx) == [1, 0, 0, 0, 0, 0, 0]

    def test_dim_pow_scales(self, env):
        """
        ``dim_pow`` should compute correctly multiplication resulting unit
        operation across dimensions.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        accel = mk_dim_vec_literal([0, 1, -2, 0, 0, 0, 0])
        sq = App(App(Const("dim_pow", ()), accel), int_lit(2))
        inv = App(App(Const("dim_pow", ()), accel), int_lit(-1))
        assert read_dim_vec_literal(sq, env, lctx,
                                    mctx) == [0, 2, -4, 0, 0, 0, 0]
        assert read_dim_vec_literal(inv, env, lctx,
                                    mctx) == [0, -1, 2, 0, 0, 0, 0]

    def test_kernel_unit_multiplication(self, env):
        """
        Testing that kernel verified resulting term matches manually mutiplying
        terms.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        u1, u2 = {"kg": 1, "s": -1}, {"m": 1, "s": -1}
        py = unit_mul(u1, u2)
        d1 = mk_dim_vec_literal([1, 0, -1, 0, 0, 0, 0])
        d2 = mk_dim_vec_literal([0, 1, -1, 0, 0, 0, 0])
        kernel = App(App(Const("dim_mul", ()), d1), d2)
        exps = read_dim_vec_literal(kernel, env, lctx, mctx)
        from physika.utils.cic_utils.inductive_utils import SI_BASE_UNITS
        kernel_unit = {b: e for b, e in zip(SI_BASE_UNITS, exps) if e != 0}
        assert kernel_unit == py

    def test_dim_sqrt_cic(self, env):
        """
        ``dim_sqrt_cic`` halves even exponents and gives ``None`` for an
        odd case.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        gh = unit_to_dimvec_cic({"m": 2, "s": -2}, env)
        assert read_dim_vec_literal(dim_sqrt_cic(gh, env), env, lctx,
                                    mctx) == [0, 1, -1, 0, 0, 0, 0]
        assert dim_sqrt_cic(unit_to_dimvec_cic({"m": 1}, env), env) is None

    def test_expr_dims_sqrt_and_complex(self, env):
        """
        ``sqrt`` takes the square root of the dimension, ``i`` and complex
        literals are dimensionless and ``real``/``imag`` keep the
        argument's dimension.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        units = {"z": {"m": 1}, "h": {"m": 2, "s": -2}}
        dims = {k: unit_to_dimvec_cic(u, env) for k, u in units.items()}
        for expr, expected in ((("call", "sqrt", [
            ("var", "h")
        ]), [0, 1, -1, 0, 0, 0,
             0]), (("complex", 2j), [0] * 7), (("call", "imag", [
                 ("var", "z")
             ]), [0, 1, 0, 0, 0, 0, 0])):
            d = expr_unit_to_dimvec_cic(expr, dims, env)
            assert read_dim_vec_literal(d, env, lctx, mctx) == expected
        assert expr_unit_to_dimvec_cic(("call", "sqrt", [("var", "z")]), dims,
                                       env) is None
        with pytest.raises(UnitError):
            expr_unit_to_dimvec_cic(("call", "exp", [("var", "z")]), dims, env)
        assert infer_unit(("imaginary", ), units) == {}
        assert infer_unit(("call", "real", [("var", "z")]), units) == {"m": 1}


class TestUnitDeclCheck:
    """``check_unit_decl_cic`` kernel verified ``x: ℝ ← [unit] = expr``."""

    def check_unit_decl_update_env(self, env, program):
        """
        Helper function for checking a unit declaration from parsed AST and
        updating environment with stored unit declarations so far.
        """
        errs, ue, uce = [], {}, {}
        for stmt in program:
            if stmt[0] == "unit_decl":
                check_unit_decl_cic(stmt, ue, uce, env, errs.append)
            update_unit_env(stmt, ue)
            update_unit_cic_env(stmt, uce, env)
        return errs

    def test_consistent_units_pass(self, env):
        """
        Test dimensional analysis unit declaration and iference from CIC.
        """
        program = [
            ("unit_decl", "m", "ℝ", [("kg", 1)], ("num", 5.0), 1),
            ("unit_decl", "a", "ℝ", [("m", 1), ("s", -2)], ("num", 2.0), 2),
            ("unit_decl", "F", "ℝ", [("m", 1), ("kg", 1), ("s", -2)],
             ("mul", ("var", "m"), ("var", "a")), 3),
        ]
        assert self.check_unit_decl_update_env(env, program) == []

    def test_inconsistent_units_flagged(self, env):
        """
        Test units declared for a incorrect result is catched properly. Checks
        CIC infers properly a badly declared unit.
        """
        program = [
            ("unit_decl", "m", "ℝ", [("kg", 1)], ("num", 5.0), 1),
            ("unit_decl", "a", "ℝ", [("m", 1), ("s", -2)], ("num", 2.0), 2),
            ("unit_decl", "Fbad", "ℝ", [("m", 1), ("s", -2)],
             ("mul", ("var", "m"), ("var", "a")), 3),
        ]
        errs = self.check_unit_decl_update_env(env, program)
        assert len(errs) == 1
        assert errs[
            0] == "Line 3: unit mismatch for 'Fbad': declared [m·s⁻²] but expression has unit [kg·m·s⁻²]."  # noqa: E501

        program = [
            ("unit_decl", "x", "ℝ", [("m", 1)], ("num", 1.0), 1),
            ("unit_decl", "y", "ℝ", [("s", 1)], ("num", 1.0), 2),
            ("unit_decl", "z", "ℝ", [("m", 1)], ("add", ("var", "x"),
                                                 ("var", "y")), 3),
        ]
        errs = self.check_unit_decl_update_env(env, program)

        assert len(errs) == 1
        assert errs[
            0] == "Line 3: unit error in 'z': dimension mismatch in addition"

    def test_division_cancels_to_expected_unit(self, env):
        """
        Check a division operation also substract dimenion units.
        """
        program = [
            ("unit_decl", "d", "ℝ", [("m", 1)], ("num", 9.0), 1),
            ("unit_decl", "v", "ℝ", [("m", 1), ("s", -1)], ("num", 3.0), 2),
            ("unit_decl", "t", "ℝ", [("s", 1)], ("div", ("var", "d"),
                                                 ("var", "v")), 3),
        ]
        assert self.check_unit_decl_update_env(env, program) == []


class TestDimensionedSignatureParsing:
    """
    Testing ``← [unit]`` syntax on params and the return parses.
    An unannotated ``ℝ`` in a dimension signature is just represents a
    dimensionless quantity.
    """

    def test_unannotated_paramm_is_not_an_error(self):
        """
        Test for a "uncomplete" unit annotation on a parameter is not an error.
        """
        res = elab_cic("def f(x: ℝ ← [m], t: ℝ): ℝ ← [m]:\n    return x * t\n")
        assert res.get("errors", []) == []

    def test_dimensionless_parses(self):
        """
        An explicit dimensionless bracket should also parse ``[]``.
        """
        res = elab_cic("def scale(x: ℝ ← [m], k: ℝ ← []): ℝ ← [m]:\n"
                       "    return x * k\n")
        assert "scale" in (res.get("resolved_bodies") or {})


class TestQuantityReturnCheck:
    """
    Verify ``check_func_dims`` uses ``dim_mul`` or ``dim_div``
    through a function's body that uses dimension units and ``is_def_eq``
    checks the result against the return annotation.
    """

    def test_correct_division_dimension_verifies(self):
        """
        A fucntion body that return a dimension operation should match
        with its return type.
        """
        res = elab_cic("def speed(d: ℝ ← [m], t: ℝ ← [s]): ℝ ← [m, s**-1]:\n"
                       "    return d / t\n")
        assert res.get("errors") == []
        assert "speed" in (res.get("resolved_bodies") or {})

        # wrong return dimension type should be catched
        res = elab_cic("def speed(d: ℝ ← [m], t: ℝ ← [s]): ℝ ← [m, s]:\n"
                       "    return d / t\n")
        assert res.get(
            "errors"
        )[0] == "In function 'speed': return declares dimension [m·s] but the body has dimension [m·s⁻¹]."  # noqa: E501


class TestBodyLocalThreading:
    """
    A function body's intermediate expression have
    their dimensions inferred.
    """

    def test_multi_step_body_verifies(self):
        """
        Verifies intermediate expression inside a function body are correctly
        inferred and errors catched.
        """
        res = elab_cic("def speed(d: ℝ ← [m], t: ℝ ← [s]): ℝ ← [m, s**-1]:\n"
                       "    speed = d / t\n"
                       "    return speed\n")
        assert res.get("errors") == []

        # body carried [kg²·m·s⁻²] but correct should be [kg·m²·s⁻²]
        res = elab_cic("def nt(a: ℝ ← [m], g: ℝ ← [kg, m, s**-2], "
                       "rho: ℝ ← [kg, m**-3]): ℝ ← [kg, m**2, s**-2]:\n"
                       "    arm = a * a\n"
                       "    force = rho * g * a\n"
                       "    return force * arm\n")
        assert res.get(
            "errors"
        )[0] == "In function 'nt': return declares dimension [m²·kg·s⁻²] but the body has dimension [kg²·m·s⁻²]."  # noqa: E501


class TestCustomDimension:
    """
    A custom unit (outside defined SI) is registered and checked for unit
    consistency but is not verified by CIC kernel.
    """

    def test_custom_dim(self):
        """
        An array carrying [eV] should be able to unit checked proerly.
        """
        res = elab_cic("def scale(v: ℝ[n] ← [eV], k: ℝ): ℝ ← [eV]:\n"
                       "    return sum(for i: ℕ(n) → v[i] * k)\n")
        assert res.get("errors") == []

        # An error in return typ should be catched.
        res = elab_cic("def bad(v: ℝ[n] ← [eV], k: ℝ ← [s]): ℝ ← [eV]:\n"
                       "    return sum(for i: ℕ(n) → v[i] * k)\n")
        print(res.get("errors"))
        assert res.get(
            "errors"
        )[0] == "In function 'bad': return declares dimension [eV] but the body has dimension [eV·s]."  # noqa: E501


class TestDimAnalysis:
    """
    ``dim_analysis`` over a unified AST.
    """

    def test_dimensionless_program(self):
        """
        A program with no unit annotation returns no errors.
        """
        u, e = elab_program("x: ℝ = 1.0\ny: ℝ = x + 2.0\n")

        assert dim_analysis(u, e, {}) == []

    def test_ast_wrong_unit_decl(self):
        """
        ``force`` matches ``kg·(m·s⁻²)`` and passes while ``wrong`` declares
        ``[s]`` for the same product should report an error.
        """
        ast = {
            "functions": {},
            "program":
            [("unit_decl", "mass", "ℝ", [("kg", 1)], ("num", 5.0), 1),
             ("unit_decl", "accel", "ℝ", [("m", 1),
                                          ("s", -2)], ("num", 2.0), 2),
             ("unit_decl", "force", "ℝ", [("kg", 1), ("m", 1), ("s", -2)],
              ("mul", ("var", "mass"), ("var", "accel")), 3),
             ("unit_decl", "wrong", "ℝ", [("s", 1)], ("mul", ("var", "mass"),
                                                      ("var", "accel")), 4)]
        }
        errs = dim_analysis(ast, mk_builtin_env(), {})

        assert len(errs) == 1
        assert errs[
            0] == "Line 4: unit mismatch for 'wrong': declared [s] but expression has unit [kg·m·s⁻²]."  # noqa: E501

    def test_func_sigs_dimension_annotated_function(self):
        """
        A CIC elaborated fucntion records ``(param specs, return unit)`` in
        ``func_sigs``.
        """
        u, e = elab_program(
            "def speed(d: ℝ ← [m], t: ℝ ← [s]): ℝ ← [m, s**-1]:\n"
            "    return d / t\n")
        func_sigs: dict = {}

        assert dim_analysis(u, e, func_sigs) == []
        assert func_sigs["speed"] == ([{"m": 1}, {"s": 1}], {"m": 1, "s": -1})

    def test_wrong_function_return_dimension_is_reported(self):
        """
        Declaring ``ℝ ← [m, s]`` for a body that computes ``m·s⁻¹`` is reported
        as an error message.
        """
        u, e = elab_program("def speed(d: ℝ ← [m], t: ℝ ← [s]): ℝ ← [m, s]:\n"
                            "    return d / t\n")
        errs = dim_analysis(u, e, {})

        assert len(errs) == 1
        assert errs[
            0] == "In function 'speed': return declares dimension [m·s] but the body has dimension [m·s⁻¹]."  # noqa: E501
