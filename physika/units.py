from typing import Any, Callable, Dict, List, Optional, Tuple

from physika.core.expr import App, Const, Expr
from physika.core.environment import ConstantInfo, Environment

Unit = Dict[str, int]  # unit name keys, integer exponent values

# It might be better to handle these with Nat/Int inductive types.
EXP_SUPER: Dict[int, str] = {
    1: '',
    2: '²',
    3: '³',
    4: '⁴',
    5: '⁵',
    6: '⁶',
    7: '⁷',
    8: '⁸',
    9: '⁹',
    -1: '⁻¹',
    -2: '⁻²',
    -3: '⁻³',
    -4: '⁻⁴',
    -5: '⁻⁵',
    -6: '⁻⁶',
    -7: '⁻⁷',
    -8: '⁻⁸',
    -9: '⁻⁹',
}


class UnitError(Exception):
    pass


def unit_mul(u1: Unit, u2: Unit) -> Unit:
    """
    Multiply two dimensional units (exponent addition).

    Parameters
    ----------
    u1: Unit
        First term to multiply.
    u2 : Unit
        Second term to multiply.

    Examples
    --------
    >>> from physika.units import unit_mul
    >>> unit_mul({'m': 1, 's': -2}, {'s': 1})
    {'m': 1, 's': -1}
    >>> unit_mul({'m': 1}, {'m': -1})
    {}
    """
    result = dict(u1)
    for base, exp in u2.items():
        result[base] = result.get(base, 0) + exp
    return {k: v for k, v in result.items() if v != 0}


def unit_div(u1: Unit, u2: Unit) -> Unit:
    """
    Divide two units (exponent substraction).

    Parameters
    ----------
    u1 : Unit
        The numerator unit.
    u2 : Unit
        The denominator unit.

    Examples
    --------
    >>> from physika.units import unit_div
    >>> unit_div({'m': 1}, {'s': 1})
    {'m': 1, 's': -1}
    >>> unit_div({'kg': 1}, {'kg': 1})
    {}
    """
    result = dict(u1)
    for base, exp in u2.items():
        result[base] = result.get(base, 0) - exp
    return {k: v for k, v in result.items() if v != 0}


def unit_pow(u: Unit, exp: int) -> Unit:
    """
    Raise a unit to an integer power.

    Parameters
    ----------
    u : Unit
        Base unit.
    exp : int
        Value to raise the unit to, truncated at zero.

    Examples
    --------
    >>> from physika.units import unit_pow
    >>> unit_pow({'m': 1, 's': -2}, 2)
    {'m': 2, 's': -4}
    >>> unit_pow({'m': 2}, 0)
    {}
    """
    n = int(exp)
    return {base: v * n for base, v in u.items() if v * n != 0}


def unit_to_str(u: Unit) -> str:
    """
    Format a Unit dictionary as superscript string with dot product between
    dimensions (e.g. ``'kg·m·s⁻²'``).

    Parameters
    ----------
    u : Unit
        Unit to format.

    Examples
    --------
    >>> from physika.units import unit_to_str
    >>> unit_to_str({'kg': 1, 'm': 1, 's': -2})
    'kg·m·s⁻²'
    >>> unit_to_str({'kg': 1})
    'kg'
    >>> unit_to_str({})
    '1'
    """
    if not u:
        return "1"
    parts = []
    # positive exponents first
    for base in sorted(u, key=lambda k: (-u[k], k)):
        exp = u[base]
        suffix = EXP_SUPER.get(exp, f'^{exp}')
        parts.append(f"{base}{suffix}")
    return '·'.join(parts)


def parse_unit_ast(unit_ast: List[Tuple[str, int]]) -> Unit:
    """
    Convert parsed list containing dimension units to Unit object.

    Parameters
    ----------
    unit_ast : List
        List of ``(base, exponent)`` pairs from a unit declaration AST.

    Examples
    --------
    >>> from physika.units import parse_unit_ast
    >>> parse_unit_ast([('kg', 1)])
    {'kg': 1}
    >>> parse_unit_ast([('m', 1), ('s', -2)])
    {'m': 1, 's': -2}
    >>> parse_unit_ast([('m', 1), ('m', -1)])
    {}
    """
    result: Unit = {}
    for base, exp in unit_ast:
        if exp != 0:
            result[base] = result.get(base, 0) + int(exp)
    return {k: v for k, v in result.items() if v != 0}


def infer_unit(expr: Any,
               unit_env: Dict[str, Unit],
               func_sigs: Optional[Dict[str, Any]] = None) -> Optional[Unit]:
    """
    Infer ``Unit`` of a Physika expression by walking it and computing
    dimensional arithmetic.

    Walks AST and for each tags applies a different rule to compute a resulting
    unit. Returns ``None`` if the unit cannot be determined, for example and
    unbound variable. This step is no CIC bases (no kernel verified) and serves
    to fill in holes (as MVar) in CIC elaboration for the actual CIC trusted
    kernel pass.

    Parameters
    ----------
    expr : ASTNode
        Physika expression tuple.
    unit_env : dict
        Variable name are keys and unit values.
    func_sigs : dict
        ``name`` are keys and (param specs, return spec) values
        for function signatures.

    Examples
    --------
    >>> from physika.units import infer_unit
    >>> infer_unit(("num", 2.0), {})
    {}
    >>> infer_unit(("var", "x"), {"x": {"m": 1}})
    {'m': 1}
    >>> infer_unit(("mul", ("var", "f"), ("var", "t")),
    ...            {"f": {"kg": 1, "m": 1, "s": -2}, "t": {"s": 1}})
    {'kg': 1, 'm': 1, 's': -1}
    >>> infer_unit(("var", "unbound"), {}) is None
    True
    """
    fs = func_sigs or {}
    if not isinstance(expr, tuple):
        return None
    op = expr[0]
    if op == "num":
        return {}
    if op in ("imaginary", "complex"):
        return {}
    if op == "var":
        return unit_env.get(expr[1])
    if op == "neg":
        return infer_unit(expr[1], unit_env, fs)
    if op in ("index", "chain_index", "indexN") and isinstance(expr[1], str):
        return unit_env.get(expr[1])
    if op == "for_expr":
        return infer_unit(expr[3], unit_env, fs)
    if op in ("add", "sub"):
        u1 = infer_unit(expr[1], unit_env, fs)
        u2 = infer_unit(expr[2], unit_env, fs)
        if u1 is None:
            return u2
        if u2 is None:
            return u1
        if u1 != u2:
            what = "addition" if op == "add" else "subtraction"
            raise UnitError(f"unit mismatch in {what}: "
                            f"[{unit_to_str(u1)}] vs [{unit_to_str(u2)}]")
        return u1
    if op == "mul":
        u1 = infer_unit(expr[1], unit_env, fs)
        u2 = infer_unit(expr[2], unit_env, fs)
        return None if u1 is None or u2 is None else unit_mul(u1, u2)
    if op == "div":
        u1 = infer_unit(expr[1], unit_env, fs)
        u2 = infer_unit(expr[2], unit_env, fs)
        return None if u1 is None or u2 is None else unit_div(u1, u2)
    if op == "pow":
        u = infer_unit(expr[1], unit_env, fs)
        if u is None:
            return None
        if isinstance(expr[2], tuple) and expr[2][0] == "num":
            return unit_pow(u, expr[2][1])
        return None
    if op == "call":
        fname, args = expr[1], expr[2]
        if fname in ("sum", "abs", "real", "imag") and len(args) == 1:
            # projections that keep the argument's dimension: |x| keeps
            # x's unit, and real(z)/imag(z) keep z's unit too (splitting
            # a complex quantity into its two real components doesn't
            # change what physical dimension either component carries).
            return infer_unit(args[0], unit_env, fs)
        if fname == "sqrt" and len(args) == 1:
            u = infer_unit(args[0], unit_env, fs)
            if u is None:
                return None
            if all(x % 2 == 0 for x in u.values()):
                return {b: x // 2 for b, x in u.items() if x != 0}
            return None
        if fname in ("exp", "log", "sin", "cos", "tan") and len(args) == 1:
            u = infer_unit(args[0], unit_env, fs)
            if u:
                raise UnitError(
                    f"{fname}() requires a dimensionless argument, "
                    f"got [{unit_to_str(u)}]")
            return {}
        if fname in fs:
            # declared/inferred return unit
            return fs[fname][1]
    return None


def dim_sqrt_cic(d: Expr, env: Environment) -> Optional[Expr]:
    """
    ``DimVec`` of ``sqrt`` applied to a quantity of dimension *d*.

    ``DimVec`` exponents are integers, so ``dim_pow`` cannot take ``1/2``
    directly.  The candidate ``h`` is computed and accepted only when
    the kernel proves ``dim_pow h 2 ≡ d``.

    Examples
    --------
    >>> from physika.units import dim_sqrt_cic, unit_to_dimvec_cic
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     read_dim_vec_literal)
    >>> env = mk_builtin_env()
    >>> gh = unit_to_dimvec_cic({"m": 2, "s": -2}, env)   # g·h
    >>> read_dim_vec_literal(dim_sqrt_cic(gh, env), env, LocalContext(),
    ...                      MetaVarContext())             # m·s⁻¹
    [0, 1, -1, 0, 0, 0, 0]
    >>> dim_sqrt_cic(unit_to_dimvec_cic({"m": 1}, env), env) is None
    True
    """
    from physika.core.local_context import LocalContext
    from physika.core.metavar import MetaVarContext
    from physika.core.reduction import is_def_eq
    from physika.utils.cic_utils.inductive_utils import (
        int_lit,
        mk_dim_vec_literal,
        read_dim_vec_literal,
    )

    lctx, mctx = LocalContext(), MetaVarContext()
    exps = read_dim_vec_literal(d, env, lctx, mctx)
    if exps is None or any(e % 2 for e in exps):
        return None
    half = mk_dim_vec_literal([e // 2 for e in exps])
    squared = App(App(Const("dim_pow", ()), half), int_lit(2))
    ok, _ = is_def_eq(squared, d, env, lctx, mctx, allow_assign=False)
    return half if ok else None


def expr_unit_to_dimvec_cic(
        expr: Any,
        unit_cic_env: Dict[str, Expr],
        env: Environment,
        func_dims: Optional[Dict[str, Expr]] = None) -> Optional[Expr]:
    """
    Build a CIC term for ``expr`` unit. This CIC term is ``DimVec``
    which is represented by ``Vec Int 7`` inductive type. Each integer in
    this ``Vec`` is an exponent for supported SI units
    (``kg, m, s, A, K, mol, cd``). Once a ``DimVec`` is created for ``expr``,
    a trusted kernel pass will compare it against the declared
    return type.

    Parameters
    ----------
    expr : ASTNode
        Physika expression tuple.
    unit_cic_env : dict
        Variable name keys, ``DimVec`` CIC expression values.
    env : Environment
        CIC environment holding ``DimVec`` / ``dim_mul`` / ``dim_div`` /
        ``dim_pow`` (from ``mk_builtin_env``).
    func_dims : dict, optional
        Function name keys, return ``DimVec`` expression values.

    Examples
    --------
    >>> from physika.units import (
    ...     expr_unit_to_dimvec_cic, unit_to_dimvec_cic)
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     read_dim_vec_literal)
    >>> env = mk_builtin_env()
    >>> lctx, mctx = LocalContext(), MetaVarContext()
    >>> cenv = {"f": unit_to_dimvec_cic({"kg": 1, "m": 1, "s": -2}, env),
    ...         "t": unit_to_dimvec_cic({"s": 1}, env)}
    >>> term = expr_unit_to_dimvec_cic(     # unreduced App(dim_mul, …)
    ...     ("mul", ("var", "f"), ("var", "t")), cenv, env)
    >>> read_dim_vec_literal(term, env, lctx, mctx)   # kg*m*s^2 · s
    [1, 1, -1, 0, 0, 0, 0]
    >>> expr_unit_to_dimvec_cic(("var", "unbound"), {}, env) is None
    True
    """
    from physika.core.local_context import LocalContext
    from physika.core.metavar import MetaVarContext
    from physika.core.reduction import is_def_eq
    from physika.utils.cic_utils.inductive_utils import (
        SI_BASE_UNITS,
        int_lit,
        mk_dim_vec_literal,
    )

    fd = func_dims or {}
    dim_one = mk_dim_vec_literal([0] * len(SI_BASE_UNITS))

    if not isinstance(expr, tuple):
        return None
    op = expr[0]
    if op == "num":
        return dim_one
    if op in ("imaginary", "complex"):
        return dim_one
    if op == "var":
        return unit_cic_env.get(expr[1])
    if op == "neg":
        return expr_unit_to_dimvec_cic(expr[1], unit_cic_env, env, fd)
    if op in ("index", "chain_index", "indexN") and isinstance(expr[1], str):
        return unit_cic_env.get(expr[1])
    if op == "for_expr":
        return expr_unit_to_dimvec_cic(expr[3], unit_cic_env, env, fd)
    if op in ("add", "sub"):
        d1 = expr_unit_to_dimvec_cic(expr[1], unit_cic_env, env, fd)
        d2 = expr_unit_to_dimvec_cic(expr[2], unit_cic_env, env, fd)
        if d1 is None:
            return d2
        if d2 is None:
            return d1
        ok, _ = is_def_eq(d1,
                          d2,
                          env,
                          LocalContext(),
                          MetaVarContext(),
                          allow_assign=False)
        if not ok:
            what = "addition" if op == "add" else "subtraction"
            raise UnitError(f"dimension mismatch in {what}")
        return d1
    if op == "mul":
        d1 = expr_unit_to_dimvec_cic(expr[1], unit_cic_env, env, fd)
        d2 = expr_unit_to_dimvec_cic(expr[2], unit_cic_env, env, fd)
        if d1 is None or d2 is None:
            return None
        return App(App(Const("dim_mul", ()), d1), d2)
    if op == "div":
        d1 = expr_unit_to_dimvec_cic(expr[1], unit_cic_env, env, fd)
        d2 = expr_unit_to_dimvec_cic(expr[2], unit_cic_env, env, fd)
        if d1 is None or d2 is None:
            return None
        return App(App(Const("dim_div", ()), d1), d2)
    if op == "pow":
        d = expr_unit_to_dimvec_cic(expr[1], unit_cic_env, env, fd)
        if d is None:
            return None
        if (isinstance(expr[2], tuple) and expr[2][0] == "num"
                and float(expr[2][1]).is_integer()):
            return App(App(Const("dim_pow", ()), d), int_lit(int(expr[2][1])))
        return None
    if op == "call":
        fname, args = expr[1], expr[2]
        if fname in ("sum", "abs", "real", "imag") and len(args) == 1:
            return expr_unit_to_dimvec_cic(args[0], unit_cic_env, env, fd)
        if (fname in ("sqrt", "exp", "log", "sin", "cos", "tan")
                and len(args) == 1):
            d = expr_unit_to_dimvec_cic(args[0], unit_cic_env, env, fd)
            if d is None:
                return None
            dimensionless, _ = is_def_eq(d,
                                         Const("dim.one", ()),
                                         env,
                                         LocalContext(),
                                         MetaVarContext(),
                                         allow_assign=False)
            if dimensionless:
                return dim_one
            if fname == "sqrt":
                return dim_sqrt_cic(d, env)
            raise UnitError(f"{fname}() requires a dimensionless argument")
        # a call to a function whose return dimension is known
        if fname in fd:
            return fd[fname]
    return None


def update_unit_env(stmt: Any,
                    unit_env: Dict[str, Unit],
                    func_sigs: Optional[Dict[str, Any]] = None) -> None:
    """
    Record a ``stmt`` variable in ``unit_env`` with its unit stored as a
    ``Unit`` dict.

    A ``unit_decl`` takes the declared units in bracket. A
    ``decl`` or ``assign`` node infers the unit of its right hand side.

    Parameters
    ----------
    stmt : ASTNode
        Program statement tuple.
    unit_env : dict
        Variable name keys, ``Unit`` values.
    func_sigs : dict, optional
        Function signatures passed to ``infer_unit``

    Examples
    --------
    >>> from physika.units import update_unit_env
    >>> env: Dict[str, Unit] = {}
    >>> decl = ("unit_decl", "m", "ℝ", [("kg", 1)], ("num", 1.0), 1)
    >>> update_unit_env(decl, env)
    >>> unit_to_str(env['m'])
    'kg'
    >>> update_unit_env(("decl", "m", "ℝ", ("num", 2.0), 2), env)
    >>> env['m']
    {}
    """
    if stmt is None or not isinstance(stmt, tuple):
        return
    op = stmt[0]
    if op == "unit_decl":
        unit_env[stmt[1]] = parse_unit_ast(stmt[3])
    elif op == "decl":
        try:
            inferred = infer_unit(stmt[3], unit_env, func_sigs)
        except UnitError:
            inferred = None
        if inferred is not None:
            unit_env[stmt[1]] = inferred
    elif op == "assign":
        try:
            inferred = infer_unit(stmt[2], unit_env, func_sigs)
        except UnitError:
            inferred = None
        if inferred is not None:
            unit_env[stmt[1]] = inferred


def update_unit_cic_env(stmt: Any, unit_cic_env: Dict[str, Expr],
                        env: Environment) -> None:
    """
    Record a variable in ``unit_cic_env`` and its unit as a ``DimVec`` CIC
    term. This function is for registering a CIC dimension units term while
    kernel verification runs in a separate pass.

    Parameters
    ----------
    stmt : ASTNode
        Program statement tuple for ``unit_decl``, ``decl``, and ``assign``
        nodes.
    unit_cic_env : dict
        Variable name keys and ``DimVec`` CIC terms are the values.
    env : Environment
        CIC environment with solved variables so far.

    Examples
    --------
    >>> from physika.units import update_unit_cic_env
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     read_dim_vec_literal)
    >>> env = mk_builtin_env()
    >>> cenv = {}
    >>> update_unit_cic_env(
    ...     ("unit_decl", "m", "ℝ", [("kg", 1)], ("num", 1.0), 1), cenv, env)
    >>> update_unit_cic_env(
    ...     ("unit_decl", "a", "ℝ", [("m", 1), ("s", -2)], ("num", 9.8), 2),
    ...     cenv, env)
    >>> update_unit_cic_env(
    ...     ("decl", "F", "ℝ", ("mul", ("var", "m"), ("var", "a")), 3),
    ...     cenv, env)
    >>> read_dim_vec_literal(   # F = m * a  reduces to  kg·m·s⁻²
    ...     cenv["F"], env, LocalContext(), MetaVarContext())
    [1, 1, -2, 0, 0, 0, 0]
    """
    if stmt is None or not isinstance(stmt, tuple):
        return
    op = stmt[0]
    if op == "unit_decl":
        unit_cic_env[stmt[1]] = unit_to_dimvec_cic(parse_unit_ast(stmt[3]),
                                                   env)
    elif op == "decl":
        try:
            d = expr_unit_to_dimvec_cic(stmt[3], unit_cic_env, env)
        except UnitError:
            d = None
        if d is not None:
            unit_cic_env[stmt[1]] = d
    elif op == "assign":
        try:
            d = expr_unit_to_dimvec_cic(stmt[2], unit_cic_env, env)
        except UnitError:
            d = None
        if d is not None:
            unit_cic_env[stmt[1]] = d


def unit_to_dim_const_name(unit: Unit) -> str:
    """
    Return CIC constant (Const) name for a dimension.

    ``(base, exponent)`` pairs so the same physical dimension maps to the
    same ``Const`` name starting with ``dim.``.

    Parameters
    ----------
    unit : Unit
        Unit and dimension dictionary.

    Examples
    --------
    >>> from physika.units import unit_to_dim_const_name
    >>> unit_to_dim_const_name({})
    'dim.one'
    >>> unit_to_dim_const_name({'kg': 1})
    'dim.kg+1'
    >>> unit_to_dim_const_name({'kg': 1, 'm': 1, 's': -2})
    'dim.kg+1_m+1_s-2'
    """
    if not unit:
        return "dim.one"
    parts = sorted(unit.items())
    return "dim." + "_".join(f"{b}{e:+d}" for b, e in parts)


def unit_to_dim_vec_exponents(unit: Unit) -> Optional[list]:
    """
    Convert a ``Unit`` object to a list of element exponent in
    order (``kg, m, s, A, K, mol, cd``), or ``None`` if
    unit name is not in supported list.

    Parameters
    ----------
    unit : Unit
        Unit object to convert dimension to list of exponents.

    Examples
    --------
    >>> from physika.units import unit_to_dim_vec_exponents
    >>> unit_to_dim_vec_exponents({'kg': 1, 's': -2})
    [1, 0, -2, 0, 0, 0, 0]
    >>> unit_to_dim_vec_exponents({'furlong': 1}) is None
    True
    """
    from physika.utils.cic_utils.inductive_utils import SI_BASE_UNITS
    if any(base not in SI_BASE_UNITS for base in unit):
        return None
    return [unit.get(base, 0) for base in SI_BASE_UNITS]


def unit_to_dimvec_cic(unit: Unit, env: Environment) -> Expr:
    """
    Elaborate a ``Unit`` to ``DimVec`` CIC expression.

    An SI unit becomes a genuine ``Vec Int 7`` literal that inductive
    rules can be applied to and the kernel can reduce.

    Currently, a unit with a non-SI base instead gets a canonical opaque
    named constant, registered in ``env``. This could be addressed by
    supporting user inductive types.

    Parameters
    ----------
    unit : Unit
        Dimension units to elaborate.
    env : Environment
        CIC environmet.

    Examples
    --------
    >>> from physika.units import unit_to_dimvec_cic
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     read_dim_vec_literal)
    >>> env = mk_builtin_env()
    >>> v = unit_to_dimvec_cic({'kg': 1, 's': -2}, env)
    >>> read_dim_vec_literal(v, env, LocalContext(), MetaVarContext())
    [1, 0, -2, 0, 0, 0, 0]
    """
    from physika.utils.cic_utils.inductive_utils import mk_dim_vec_literal
    exponents = unit_to_dim_vec_exponents(unit)
    if exponents is not None:
        return mk_dim_vec_literal(exponents)
    DIM_VEC = Const("DimVec", ())
    dim_name = unit_to_dim_const_name(unit)
    if env.constants.get(dim_name) is None:
        env.add_constant(ConstantInfo(dim_name, (), DIM_VEC, None))
    return Const(dim_name, ())


def check_unit_decl_cic(
    stmt: Any,
    unit_env: Dict[str, Unit],
    unit_cic_env: Dict[str, Expr],
    cic_env: Environment,
    add_error: Callable[[str], None],
) -> None:
    """
    Verify a unit declaratin node matches with inferred unit using kernel
    verification (defintional equality).

    Parameters
    ----------
    stmt : tuple
        A ``("unit_decl", name, type, unit_ast, expr, lineno)`` node.
    unit_env : dict
        Variable name → ``Unit`` mapping; used only to format the error
        message, never for the verdict.
    unit_cic_env : dict
        Variable name → DimVec CIC term (from ``update_unit_cic_env``),
        for looking up the variables the RHS references.
    cic_env : Environment
        CIC environment (from ``mk_builtin_env``).
    add_error : callable
        Receives a plain-text error string on a unit mismatch.

    Examples
    --------
    >>> from physika.units import check_unit_decl_cic, unit_to_dimvec_cic
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()
    >>> cenv = {"m": unit_to_dimvec_cic({"kg": 1}, env),
    ...         "a": unit_to_dimvec_cic({"m": 1, "s": -2}, env)}
    >>> errs = []
    >>> mul_ma = ("mul", ("var", "m"), ("var", "a"))
    >>> # F : ℝ ← [kg, m, s**-2] = m * a
    >>> check_unit_decl_cic(
    ...     ("unit_decl", "F", "ℝ", [("kg", 1), ("m", 1), ("s", -2)],
    ...      mul_ma, 3), {}, cenv, env, errs.append)
    >>> errs
    []
    >>> # declared [kg] instead should report an erro
    >>> check_unit_decl_cic(
    ...     ("unit_decl", "F", "ℝ", [("kg", 1)], mul_ma, 4),
    ...     {}, cenv, env, errs.append)
    >>> "declared [kg] but expression has unit [kg" in errs[-1]
    True
    """
    from physika.core.local_context import LocalContext
    from physika.core.metavar import MetaVarContext
    from physika.core.reduction import is_def_eq
    from physika.utils.cic_utils.inductive_utils import (
        SI_BASE_UNITS,
        read_dim_vec_literal,
    )

    name, unit_ast, expr, lineno = stmt[1], stmt[3], stmt[4], stmt[5]
    annotated = parse_unit_ast(unit_ast)
    lctx, mctx = LocalContext(), MetaVarContext()

    try:
        dimvec_inf = expr_unit_to_dimvec_cic(expr, unit_cic_env, cic_env)
    except UnitError as e:
        add_error(f"Line {lineno}: unit error in '{name}': {e}")
        return

    if dimvec_inf is None:
        # Unit cannt be inferred
        return

    # Data type with no registered dimension
    is_dimensionless, _ = is_def_eq(dimvec_inf,
                                    Const("dim.one", ()),
                                    cic_env,
                                    lctx,
                                    mctx,
                                    allow_assign=False)
    if is_dimensionless:
        return

    t_ann = App(Const("Quantity", ()), unit_to_dimvec_cic(annotated, cic_env))
    t_inf = App(Const("Quantity", ()), dimvec_inf)

    ok, _ = is_def_eq(t_ann, t_inf, cic_env, lctx, mctx, allow_assign=False)
    if not ok:
        exps = read_dim_vec_literal(dimvec_inf, cic_env, lctx, mctx)
        if exps is not None:
            inferred_str = unit_to_str({
                b: e
                for b, e in zip(SI_BASE_UNITS, exps) if e != 0
            })
        else:
            # Fallback to non-CI unit inference
            inferred = infer_unit(expr, unit_env)
            if inferred:
                inferred_str = unit_to_str(inferred)
            else:
                "?"
        add_error(f"Line {lineno}: unit mismatch for '{name}': "
                  f"declared [{unit_to_str(annotated)}] "
                  f"but expression has unit [{inferred_str}].")


def sig_unit_spec(unit_ast: list) -> Unit:
    """
    Parse a dimension bracket signature into a ``Unit`` dictionary.

    If a unit declaration is in current SI, then is kernel checked.
    Else, a non-SI identifier (e.g. ``eV``) is a custom dimension, checked with
    ``Unit`` operations, not reduced by the kernel.

    Parameters
    ----------
    unit_ast : list
        ``(base, exponent)`` pairs from a unit declaration AST.

    Examples
    --------
    >>> from physika.units import sig_unit_spec
    >>> sig_unit_spec([('kg', 1), ('s', -2)])
    {'kg': 1, 's': -2}
    >>> sig_unit_spec([('eV', 1)])
    {'eV': 1}
    """
    spec: Unit = {}
    for base, exp in unit_ast:
        spec[base] = spec.get(base, 0) + int(exp)
    return {k: v for k, v in spec.items() if v != 0}


def local_annotation(ts: Any) -> Optional[Unit]:
    """
    Unit of a body local's type annotation, or ``None``.

    Parameters
    ----------
    ts : typespec
        A ``body_decl``'s type spec; a ``("unit_typed", base, unit_ast)``
        tuple yields ``sig_unit_spec(unit_ast)``, anything else ``None``.

    Examples
    --------
    >>> from physika.units import local_annotation
    >>> local_annotation(("unit_typed", "ℝ", [("m", 1), ("s", -1)]))
    {'m': 1, 's': -1}
    >>> local_annotation("ℝ") is None
    True
    """
    if isinstance(ts, tuple) and ts and ts[0] == "unit_typed":
        return sig_unit_spec(ts[2])
    return None


def thread_locals(statements: list, curr_env: Dict[str, Unit], fs: Dict[str,
                                                                        Any],
                  fname: str, add_error: Callable[[str], None]) -> bool:
    """
    Walk statements and infer each ``body_assign`` or
    ``body_decl`` local's unit with ``infer_unit``. Then, record infer type
    in ``curr_env``, and check any annotated local against its declared unit.

    Parameters
    ----------
    statements : list
        Function body's statement tuples.
    curr_env : dict
        Variable name to ``Unit`` mapping.
    fs : dict
        Function signatures.
    fname : str
        Function name for  adding error messages.
    add_error : callable
        Function that recieve a error message to be appended.


    Examples
    --------
    >>> from physika.units import thread_locals
    >>> curr_env = {"v": {"m": 1, "s": -1}}
    >>> errs = []
    >>> stmts = [("body_assign", "ke",
    ...           ("mul", ("var", "v"), ("var", "v")))]
    >>> thread_locals(stmts, curr_env, {}, "f", errs.append)
    False
    >>> curr_env["ke"]
    {'m': 2, 's': -2}
    >>> bad = [("body_decl", "x", ("unit_typed", "ℝ", [("kg", 1)]),
    ...        ("var", "v"))]
    >>> thread_locals(bad, curr_env, {}, "f", errs.append)
    True
    >>> "declares dimension [kg]" in errs[-1]
    True
    """
    for stmt in statements:
        if not isinstance(stmt, tuple) or not stmt:
            continue
        tag = stmt[0]
        if tag == "body_assign":
            nm, rhs = stmt[1], stmt[2]
            ann = None
        elif tag == "body_decl":
            nm, ann, rhs = stmt[1], local_annotation(stmt[2]), stmt[3]
        else:
            continue
        try:
            u = infer_unit(rhs, curr_env, fs)
        except UnitError as e:
            add_error(f"In function '{fname}': {e}")
            return True
        if ann is not None:
            if u is not None and u != ann:
                add_error(
                    f"In function '{fname}': local '{nm}' declares dimension "
                    f"[{unit_to_str(ann)}] but its value has dimension "
                    f"[{unit_to_str(u)}].")
                return True
            curr_env[nm] = ann
        elif u is not None:
            curr_env[nm] = u
    return False


def thread_locals_cic(statements: list, dim_env: Dict[str, Expr],
                      cic_env: Environment, func_dims: Dict[str,
                                                            Expr], fname: str,
                      add_error: Callable[[str],
                                          None], lctx: Any, mctx: Any) -> bool:
    """
    Add each body unit declaraiton into ``dim_env``
    as its unreduced ``DimVec`` CIC term. Any
    annotated local is checked against its declared unit with kernel's
    ``is_def_eq``.

    Parameters
    ----------
    statements : list
        Function body's statement tuples.
    dim_env : dict
        Variable to ``DimVec`` CIC term.
    cic_env : Environment
        CIC environment with dim constants.
    func_dims : dict
        Function name to return ``DimVec`` term.
    fname : str
        Function name for adding error messages
    add_error : callable
        Callable function that recieve a error message to be appended.
    lctx, mctx
        Local and metavariable contexts that is used for checking
        ``is_def_eq``.


    Examples
    --------
    >>> from physika.units import thread_locals_cic, unit_to_dimvec_cic
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     read_dim_vec_literal)
    >>> env = mk_builtin_env()
    >>> lctx, mctx = LocalContext(), MetaVarContext()
    >>> dim_env = {"v": unit_to_dimvec_cic({"m": 1, "s": -1}, env)}
    >>> stmts = [("body_assign", "ke",
    ...           ("mul", ("var", "v"), ("var", "v")))]
    >>> thread_locals_cic(
    ...     stmts, dim_env, env, {}, "f", [].append, lctx, mctx)
    False
    >>> read_dim_vec_literal(dim_env["ke"], env, lctx, mctx)
    [0, 2, -2, 0, 0, 0, 0]
    """
    from physika.core.reduction import is_def_eq
    for stmt in statements:
        if not isinstance(stmt, tuple) or not stmt:
            continue
        tag = stmt[0]
        if tag == "body_assign":
            nm, rhs = stmt[1], stmt[2]
            ann = None
        elif tag == "body_decl":
            nm, ann, rhs = stmt[1], local_annotation(stmt[2]), stmt[3]
        else:
            continue
        try:
            d = expr_unit_to_dimvec_cic(rhs, dim_env, cic_env, func_dims)
        except UnitError as e:
            add_error(f"In function '{fname}': {e}")
            return True
        if ann is not None:
            ann_dv = unit_to_dimvec_cic(ann, cic_env)
            if d is not None:
                ok, _ = is_def_eq(d,
                                  ann_dv,
                                  cic_env,
                                  lctx,
                                  mctx,
                                  allow_assign=False)
                if not ok:
                    add_error(
                        f"In function '{fname}': local '{nm}' declares "
                        f"dimension [{unit_to_str(ann)}] but its value has "
                        f"a different dimension.")
                    return True
            dim_env[nm] = ann_dv
        elif d is not None:
            dim_env[nm] = d
    return False


def param_dim_spec(ts: Any) -> Optional[Unit]:
    """
    Dimension spec for a function parameter type.

    Annotated ``Unit`` for a parameter (``{}`` when
    unannotated). Returns ``None`` for a non real parameter (``ℕ`` / ``ℤ``
    / ``ℂ`` / or class instance), which carries no dimension.

    Parameters
    ----------
    ts : typespec
        The parameter's type spec.

    Examples
    --------
    >>> from physika.units import param_dim_spec
    >>> param_dim_spec("ℝ")
    {}
    >>> param_dim_spec(("unit_typed", "ℝ", [("kg", 1)]))
    {'kg': 1}
    >>> param_dim_spec("ℕ") is None
    True
    """
    if isinstance(ts, tuple) and ts and ts[0] == "unit_typed":
        return sig_unit_spec(ts[2])
    # past the guard, ts is not unit-typed, so it is already the base
    if ts == "ℝ" or (isinstance(ts, tuple) and ts and ts[0] == "tensor"):
        return {}
    return None


def record_func_dim(
    name: str,
    dimvec: Expr,
    param_specs: List[Optional[Unit]],
    cic_env: Environment,
    func_dims: Dict[str, Expr],
    func_sigs: Dict[str, Any],
) -> None:
    """
    Record a verified function's return dimension in function's dimensions and
    signature disctionaries.

    ``func_dims[name]`` registers ``DimVec`` CIC term and ``func_sigs[name]``
    gets ``(param_specs, return_spec)`` with the return as ``Unit``.

    Parameters
    ----------
    name : str
        Function's name.
    dimvec : Expr
        Function's return ``DimVec`` CIC term (possibly unreduced).
    param_specs : list of (Unit or None)
        Parameter dimension specs.
    cic_env : Environment
        CIC environment.
    func_dims : dict
        Function name to ``DimVec`` term for return mapping.
    func_sigs : dict
        Function name to ``(param specs, return spec)`` as ``Unit`` dictionary.

    Examples
    --------
    >>> from physika.units import record_func_dim
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     mk_dim_vec_literal)
    >>> env = mk_builtin_env()
    >>> fd, fs = {}, {}
    >>> force = mk_dim_vec_literal([1, 1, -2, 0, 0, 0, 0])   # kg*m*s^2
    >>> record_func_dim("net_force", force, [{}, {}], env, fd, fs)
    >>> fs["net_force"]
    ([{}, {}], {'kg': 1, 'm': 1, 's': -2})
    >>> "net_force" in fd
    True
    """
    from physika.core.local_context import LocalContext
    from physika.core.metavar import MetaVarContext
    from physika.utils.cic_utils.inductive_utils import (
        SI_BASE_UNITS,
        read_dim_vec_literal,
    )

    lctx, mctx = LocalContext(), MetaVarContext()
    func_dims[name] = dimvec
    exps = read_dim_vec_literal(dimvec, cic_env, lctx, mctx)
    concrete = ({
        b: e
        for b, e in zip(SI_BASE_UNITS, exps) if e != 0
    } if exps is not None else None)
    func_sigs[name] = (param_specs, concrete)


def check_func_dims(
    name: str,
    func_def: dict,
    cic_env: Environment,
    add_error: Callable[[str], None],
    func_dims: Dict[str, Expr],
    func_sigs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Check that a function body that carries a dimensional unit is correct with
    respect to its return dimension.

    Parameters
    ----------
    name : str
        Function's name.
    func_def : dict
        Function's AST entry.
    cic_env : Environment
        CIC environment.
    add_error : callable
        Callable function that recieve a error message to be appended.
    func_dims : dict
        Name to return ``DimVec`` CIC term dictionary.
    func_sigs : dict, optional
        Name to ``(param specs, return spec)`` mapping as ``Unit``.
    """
    from physika.core.local_context import LocalContext
    from physika.core.metavar import MetaVarContext
    from physika.core.reduction import is_def_eq
    from physika.utils.cic_utils.inductive_utils import (
        SI_BASE_UNITS,
        mk_dim_vec_literal,
        read_dim_vec_literal,
    )

    params = func_def.get("params", [])
    return_type = func_def.get("return_type", "ℝ")
    body = func_def.get("body")
    if body is None:
        return

    fs = func_sigs if func_sigs is not None else {}
    statements = func_def.get("statements", [])
    param_specs = [param_dim_spec(ts) for _, ts in params]
    ret_annotated = (isinstance(return_type, tuple) and return_type
                     and return_type[0] == "unit_typed")
    ret_spec = sig_unit_spec(return_type[2]) if ret_annotated else None

    local_annotations = [
        local_annotation(s[2]) for s in statements
        if isinstance(s, tuple) and s and s[0] == "body_decl"
    ]
    # check if declared units have non SI supported units
    has_custom = any(s is not None and any(b not in SI_BASE_UNITS for b in s)
                     for s in param_specs + [ret_spec] + local_annotations)

    # Custom dimension uses `Unit` operations
    if has_custom:
        curr_env: Dict[str, Unit] = {
            pname: (spec if spec is not None else {})
            for (pname, _), spec in zip(params, param_specs)
        }
        if thread_locals(statements, curr_env, fs, name, add_error):
            return
        try:
            body_unit = infer_unit(body, curr_env, fs)
        except UnitError as e:
            add_error(f"In function '{name}': {e}")
            return
        if (ret_spec is not None and body_unit is not None
                and body_unit != ret_spec):
            add_error(f"In function '{name}': return declares dimension "
                      f"[{unit_to_str(ret_spec)}] but the body has dimension "
                      f"[{unit_to_str(body_unit)}].")
            return
        fs[name] = (param_specs, ret_spec if ret_annotated else body_unit)
        return

    # SI untis are kernel verified via `is_def_eq` over Vec Int 7
    dim_env = {}
    for (pname, _), spec in zip(params, param_specs):
        if spec is None:
            continue
        dim_env[pname] = (unit_to_dimvec_cic(spec, cic_env) if spec else Const(
            "dim.one", ()))

    lctx, mctx = LocalContext(), MetaVarContext()
    if thread_locals_cic(statements, dim_env, cic_env, func_dims, name,
                         add_error, lctx, mctx):
        return
    try:
        body_dim = expr_unit_to_dimvec_cic(body, dim_env, cic_env, func_dims)
    except UnitError as e:
        add_error(f"In function '{name}': {e}")
        return

    if not ret_annotated:
        if body_dim is not None:
            exps = read_dim_vec_literal(body_dim, cic_env, lctx, mctx)
            if exps is not None and any(exps):
                record_func_dim(name, mk_dim_vec_literal(exps), param_specs,
                                cic_env, func_dims, fs)
        return

    ret_dim = unit_to_dimvec_cic(ret_spec or {}, cic_env)
    if body_dim is None:
        return  # cannot verify body's dimension

    ok, _ = is_def_eq(body_dim,
                      ret_dim,
                      cic_env,
                      lctx,
                      mctx,
                      allow_assign=False)
    if ok:
        record_func_dim(name, ret_dim, param_specs, cic_env, func_dims, fs)
        return

    got = read_dim_vec_literal(body_dim, cic_env, lctx, mctx)
    got_str = (unit_to_str({
        b: e
        for b, e in zip(SI_BASE_UNITS, got) if e != 0
    }) if got is not None else "?")
    add_error(f"In function '{name}': return declares dimension "
              f"[{unit_to_str(ret_spec or {})}] but the body has dimension "
              f"[{got_str}].")


def dim_analysis(unified_ast: Dict[str, Any], cic_env: Environment,
                 func_sigs: Dict[str, Any]) -> List[str]:
    """
    Run dimensional analysis pass from a parsed AST.

    Physika kernel checks unit declaration statemts ``x : ℝ ← [unit] = expr``
    and each function that contain dimensional units. If there are no type
    annotations for units, the the program is condered dimensionless and runs
    without checking dimensional analysis.

    Parameters
    ----------
    unified_ast : dict
        Unified AST parsed from source code.
    cic_env : Environment
        CIC environment the program was already elaborated.
    func_sigs : dict
        Function name to ``(param specs, return unit)`` filled as dimensional
        units are resolved.

    Examples
    --------
    >>> from physika.units import dim_analysis  # noqa: E501
    >>> from physika.core.inductive import mk_builtin_env
    >>> ast = {"functions": {}, "program": [
    ...     ("unit_decl", "mass", "ℝ", [("kg", 1)], ("num", 5.0), 1),
    ...     ("unit_decl", "accel", "ℝ", [("m", 1), ("s", -2)],
    ...      ("num", 2.0), 2),
    ...     ("unit_decl", "force", "ℝ", [("kg", 1), ("m", 1), ("s", -2)],
    ...      ("mul", ("var", "mass"), ("var", "accel")), 3),
    ...     ("unit_decl", "wrong", "ℝ", [("s", 1)],
    ...      ("mul", ("var", "mass"), ("var", "accel")), 4)]}
    >>> errs = dim_analysis(ast, mk_builtin_env(), {})
    >>> len(errs)
    1
    >>> errs[0]
    "Line 4: unit mismatch for 'wrong': declared [s] but expression has unit [kg·m·s⁻²]."
    """
    funcs = unified_ast.get("functions", {})
    program = unified_ast.get("program", [])

    has_units = (
        # a parameter carries a unit annotation
        any(
            isinstance(ts, tuple) and ts and ts[0] == "unit_typed"
            for fd in funcs.values() for _, ts in fd.get("params", []))
        # a return type carries a unit annotation
        or any(
            isinstance(fd.get("return_type"), tuple)
            and fd["return_type"][0] == "unit_typed" for fd in funcs.values())
        # a top-level unit declaration exists
        or any(
            isinstance(s, tuple) and s and s[0] == "unit_decl"
            for s in program))
    if not has_units:
        return []

    errors: List[str] = []
    func_dims: Dict[str, Expr] = {}
    for fname, fdef in funcs.items():
        check_func_dims(fname, fdef, cic_env, errors.append, func_dims,
                        func_sigs)

    unit_env: Dict[str, Unit] = {}
    unit_cic_env: Dict[str, Expr] = {}
    for stmt in program:
        if isinstance(stmt, tuple) and stmt and stmt[0] == "unit_decl":
            check_unit_decl_cic(stmt, unit_env, unit_cic_env, cic_env,
                                errors.append)
        update_unit_env(stmt, unit_env, func_sigs)
        update_unit_cic_env(stmt, unit_cic_env, cic_env)
    return errors
