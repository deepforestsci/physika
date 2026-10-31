from fractions import Fraction
from math import lcm
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physika.core.expr import App, Const
from physika.core.environment import Environment
from physika.utils.cic_utils.inductive_utils import int_lit
from physika.units import unit_to_str
from physika.solver.utils import (DERIV_NAMES, fmt_num, fraction_isqrt,
                                  in_domain, int_sum, reduces_true, to_expr)

# Polynomial data type of one unknown as {degree: coeff} as fractions
Poly = Dict[int, Any]


def p_add(a: Poly, b: Poly, sub: bool = False) -> Poly:
    """
    Returns ``a + b``, or ``a - b`` when *sub* is true.

    Parameters
    ----------
    a : Poly
        Left operand.
    b : Poly
        Right operand.
    sub : bool, optional
        Subtract *b* instead of adding it.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import p_add
    >>> p_add({1: Fraction(2), 0: Fraction(1)}, {1: Fraction(2)}, sub=True)
    {0: Fraction(1, 1)}
    """
    out = dict(a)
    for d, c in b.items():
        out[d] = out.get(d, 0) + (-c if sub else c)
    return {d: c for d, c in out.items() if c != 0}


def p_mul(a: Poly, b: Poly) -> Poly:
    """
    Returns ``a * b``.

    Parameters
    ----------
    a : Poly
        Left operand.
    b : Poly
        Right operand.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import p_mul
    >>> x_plus_1 = {1: Fraction(1), 0: Fraction(1)}
    >>> p_mul(x_plus_1, x_plus_1)
    {2: Fraction(1, 1), 1: Fraction(2, 1), 0: Fraction(1, 1)}
    """
    out: Poly = {}
    for da, ca in a.items():
        for db, cb in b.items():
            out[da + db] = out.get(da + db, 0) + ca * cb
    return {d: c for d, c in out.items() if c != 0}


def p_pow(p: Poly, k: int) -> Poly:
    """
    Returns ``p ** k`` for ``k >= 0``.

    Parameters
    ----------
    p : Poly
        Base.
    k : int
        Non-negative exponent.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import p_pow
    >>> p_pow({1: Fraction(2)}, 3)
    {3: Fraction(8, 1)}
    """
    out: Poly = {0: Fraction(1)}
    for _ in range(k):
        out = p_mul(out, p)
    return out


def p_deriv(p: Poly) -> Poly:
    """
    Returns the derivative of ``p`` with the power rule.

    Parameters
    ----------
    p : Poly
        Polynomial to differentiate.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import p_deriv
    >>> p_deriv({2: Fraction(3), 1: Fraction(1), 0: Fraction(7)})
    {1: Fraction(6, 1), 0: Fraction(1, 1)}
    """
    return {d - 1: c * d for d, c in p.items() if d >= 1}


def p_eval(p: Poly, x: Any) -> Any:
    """
    Returns ``p`` evaluated at ``x``.

    Parameters
    ----------
    p : Poly
        Polynomial to evaluate.
    x : Fraction or float
        Value of the unknown.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import p_eval
    >>> p_eval({2: Fraction(1), 0: Fraction(-3)}, Fraction(4))
    Fraction(13, 1)
    """
    return sum((c * x**d for d, c in p.items()), Fraction(0))


def wrt_name(args: List[Any]) -> Optional[str]:
    """
    Returns the variable name of ``deriv(f, t)``'s second argument, or
    ``None`` for ``deriv(f)``.

    Parameters
    ----------
    args : list[AST]
        Arguments of a ``deriv`` call.

    Examples
    --------
    >>> from physika.solver.solver_poly import wrt_name
    >>> variable = lambda n: ("var", n)
    >>> wrt_name([variable("f"), variable("t")]), wrt_name([variable("f")])
    ('t', None)
    """
    if (len(args) == 2 and isinstance(args[1], tuple) and args[1][0] == "var"):
        return args[1][1]
    return None


def poly_of(expr: Any, unknown: str, scope: Dict[str, Any], functions: dict,
            locs: Dict[str, Poly]) -> Optional[Poly]:
    """
    Returns *expr* as a polynomial for `unknown`, or ``None`` if it is not
    a polynomial with numeric coefficients.
    Constants that are in *scope* are substituted. Calls to ``functions`` are
    evaluated inline and ``deriv(...)`` is
    expanded with the power rule.

    Parameters
    ----------
    expr : AST
        Physika expression.
    unknown : str
        Name of the unknown.
    scope : dict
        Values of the constants already known.
    functions : dict
        Function definitions, by name.
    locs : dict
        Polynomials of the parameters and locals of an inlined function.

    Examples
    --------
    >>> from physika.solver.solver_poly import poly_of
    >>> variable = lambda n: ("var", n)
    >>> square = ("pow", ("sub", variable("x"), variable("a")), ("num", 2))
    >>> poly_of(square, "x", {"a": 3}, {}, {})
    {2: Fraction(1, 1), 1: Fraction(-6, 1), 0: Fraction(9, 1)}
    >>> poly_of(("call", "deriv", [square]), "x", {"a": 3}, {}, {})
    {1: Fraction(2, 1), 0: Fraction(-6, 1)}
    """
    if not isinstance(expr, tuple):
        return None
    op = expr[0]
    if op == "num":
        v = expr[1]
        return {0: v if isinstance(v, float) else Fraction(int(v))}
    if op == "var":
        if expr[1] == unknown:
            return {1: Fraction(1)}
        if expr[1] in locs:
            return dict(locs[expr[1]])
        if expr[1] in scope:
            return {0: scope[expr[1]]}
        return None
    if op == "neg":
        p = poly_of(expr[1], unknown, scope, functions, locs)
        return None if p is None else p_mul(p, {0: -1})
    if op in ("add", "sub", "mul", "div"):
        a = poly_of(expr[1], unknown, scope, functions, locs)
        b = poly_of(expr[2], unknown, scope, functions, locs)
        if a is None or b is None:
            return None
        if op == "mul":
            return p_mul(a, b)
        if op == "div":
            # only division by a non-zero constant stays a polynomial
            if set(b) != {0} or b[0] == 0:
                return None
            return p_mul(a, {0: 1 / b[0]})
        return p_add(a, b, op == "sub")
    if op == "pow":
        base = poly_of(expr[1], unknown, scope, functions, locs)
        if base is None or not (isinstance(expr[2], tuple)
                                and expr[2][0] == "num"):
            return None
        return p_pow(base, int(expr[2][1]))
    if op != "call":
        return None
    fname, args = expr[1], expr[2]
    if fname == "sqrt" and len(args) == 1:
        inner = poly_of(args[0], unknown, scope, functions, locs)
        if inner is None or set(inner) != {0} or inner[0] < 0:
            return None
        return {0: float(inner[0])**0.5}
    if fname in DERIV_NAMES and len(args) in (1, 2):
        wrt = wrt_name(args) or unknown
        if wrt == unknown:
            inner = poly_of(args[0], unknown, scope, functions, locs)
            return None if inner is None else p_deriv(inner)
        # d/dt of a known constant t: differentiate in t, then evaluate at
        # the value of t
        if wrt not in scope:
            return None
        fp = poly_of(args[0], wrt, scope, functions, {})
        return None if fp is None else {0: p_eval(p_deriv(fp), scope[wrt])}
    if fname in functions:
        return inline_call(fname, args, unknown, scope, functions, locs)
    return None


def inline_call(fname: str, args: List[Any], unknown: str, scope: Dict[str,
                                                                       Any],
                functions: dict, outer: Dict[str, Poly]) -> Optional[Poly]:
    """
    Returns the call ``fname(args)`` as a polynomial for `unknown`, by
    replacing the parameters with the arguments in the function's locals
    and body.

    Parameters
    ----------
    fname : str
        Name of the called function.
    args : list[AST]
        Arguments of the call.
    unknown : str
        Name of the unknown.
    scope : dict
        Values of the constants already known.
    functions : dict
        Function definitions, by name.
    outer : dict
        Polynomials of the caller's locals, used for the arguments.

    Examples
    --------
    >>> from physika.solver.solver_poly import inline_call
    >>> variable = lambda n: ("var", n)
    >>> funcs = {"double": {"params": [("y", "ℝ")], "statements": [],
    ...                     "body": ("mul", ("num", 2), variable("y"))}}
    >>> inline_call("double", [variable("x")], "x", {}, funcs, {})
    {1: Fraction(2, 1)}
    """
    fdef = functions[fname]
    params = [p[0] for p in fdef.get("params", [])]
    if len(args) != len(params):
        return None
    locs: Dict[str, Poly] = {}
    for pname, arg in zip(params, args):
        ap = poly_of(arg, unknown, scope, functions, outer)
        if ap is None:
            return None
        locs[pname] = ap
    for stmt in fdef.get("statements", []):
        if isinstance(stmt, tuple) and stmt[0] in ("body_assign", "body_decl"):
            p = poly_of(stmt[-1], unknown, scope, functions, locs)
            if p is None:
                return None
            locs[stmt[1]] = p
    return poly_of(fdef.get("body"), unknown, scope, functions, locs)


def divisors(n: int) -> List[int]:
    """
    Returns the positive divisors of ``n``.

    Parameters
    ----------
    n : int
        Integer whose divisors are returned.

    Examples
    --------
    >>> from physika.solver.solver_poly import divisors
    >>> divisors(-6)
    [1, 2, 3, 6]
    """
    n = abs(n)
    return [d for d in range(1, n + 1) if n % d == 0]


def rational_roots(reduced: Poly) -> List[Fraction]:
    """
    Returns the rational roots of ``reduced == 0`` in increasing order,
    with the rational root theorem: a root ``p/q`` has ``p`` dividing the
    constant term and ``q`` the leading coefficient. *reduced* has a
    non-zero constant term, and no roots are found when a coefficient is a
    float.

    Parameters
    ----------
    reduced : Poly
        Polynomial with a non-zero constant term.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import rational_roots
    >>> # (x - 1)(x - 2)(2x + 1) = 2x³ - 5x² + x + 2
    >>> cubic = {3: Fraction(2), 2: Fraction(-5), 1: Fraction(1),
    ...          0: Fraction(2)}
    >>> rational_roots(cubic)
    [Fraction(-1, 2), Fraction(1, 1), Fraction(2, 1)]
    """
    if not all(isinstance(c, Fraction) for c in reduced.values()):
        return []
    m = lcm(*(c.denominator for c in reduced.values()))
    c0, cn = int(reduced[0] * m), int(reduced[max(reduced)] * m)
    cands = {
        Fraction(s * p, q)
        for p in divisors(c0)
        for q in divisors(cn)
        for s in (1, -1)
    }
    return sorted(r for r in cands if p_eval(reduced, r) == 0)


def numeric_roots(reduced: Poly) -> List[float]:
    """
    Returns the real roots of ``reduced == 0`` computed by ``numpy.roots``,
    rounded to 10 decimals.

    Parameters
    ----------
    reduced : Poly
        Polynomial with a non-zero constant term.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import numeric_roots
    >>> numeric_roots({3: Fraction(1), 0: Fraction(-2)})  # x³ = 2
    [1.2599210499]
    """
    deg = max(reduced)
    coeffs = [float(reduced.get(deg - i, 0)) for i in range(deg + 1)]
    return sorted({
        round(float(z.real), 10)
        for z in np.roots(coeffs) if abs(z.imag) < 1e-9
    })


def poly_roots(poly: Poly) -> List[Any]:
    """
    Returns the non-zero roots of ``poly == 0``. The lowest power of the
    unknown is factored out first (its root is 0). Degree ≤ 2 uses the
    closed form, exact when the discriminant is a rational square, and
    higher degrees use ``rational_roots``, else ``numeric_roots``.

    Parameters
    ----------
    poly : Poly
        Non-zero polynomial, ``lhs - rhs`` of the equation.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import poly_roots
    >>> # x³ - 5x² + 6x = x (x - 2)(x - 3)
    >>> poly_roots({3: Fraction(1), 2: Fraction(-5), 1: Fraction(6)})
    [Fraction(3, 1), Fraction(2, 1)]
    >>> poly_roots({2: Fraction(1), 0: Fraction(-2)})  # x² = 2
    [1.4142135623730951, -1.4142135623730951]
    """
    low = min(poly)
    reduced = {d - low: c for d, c in poly.items()}
    deg = max(reduced)
    if deg == 1:
        return [-reduced[0] / reduced[1]]
    if deg == 2:
        a2, a1, a0 = reduced[2], reduced.get(1, 0), reduced[0]
        exact = all(isinstance(c, Fraction) for c in reduced.values())
        # exact square root of the discriminant, else the float one
        rd: Any = fraction_isqrt(a1 * a1 - 4 * a2 * a0) if exact else None
        if rd is None:
            a2, a1, a0 = float(a2), float(a1), float(a0)
            disc = a1 * a1 - 4 * a2 * a0
            if disc < 0:
                return []
            rd = disc**0.5
        roots = [(-a1 + rd) / (2 * a2)]
        return roots + [(-a1 - rd) / (2 * a2)] if rd else roots
    if deg >= 3:
        return rational_roots(reduced) or numeric_roots(reduced)
    return []


def verify_poly_root(poly: Poly, root: Fraction, env: Environment) -> bool:
    """
    CIC kernel checks that ``poly`` is 0 at the rational ``root``.
    Denominators are cleared to get an integer identity, then
    ``Int.eqb sum 0`` must reduce to ``Bool.true``. ``False`` when a
    coefficient is a float.

    Parameters
    ----------
    poly : Poly
        Polynomial, ``lhs - rhs`` of the equation.
    root : Fraction
        Root to check.
    env : Environment
        Environment with the ``Int.eqb`` definition.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.solver.solver_poly import verify_poly_root
    >>> poly = {2: Fraction(1, 3), 0: Fraction(-3)}  # x²/3 - 3 == 0
    >>> env = mk_builtin_env()
    >>> verify_poly_root(poly, Fraction(3), env)
    True
    >>> verify_poly_root(poly, Fraction(2), env)
    False
    """
    if not all(isinstance(c, Fraction) for c in poly.values()):
        return False
    # c * (p/q)^d * q^deg * m is an integer for every term
    deg, p, q = max(poly), root.numerator, root.denominator
    m = lcm(*(c.denominator for c in poly.values()))
    terms = [int(c * m * p**d * q**(deg - d)) for d, c in poly.items()]
    check = App(App(Const("Int.eqb", ()), int_sum(terms)), int_lit(0))
    return reduces_true(check, env)


def verify_deriv(f: Poly, g: Poly, env: Environment) -> bool:
    """
    CIC kernel checks that ``g`` is the derivative of ``f``, one
    coefficient at a time: ``g[d] == (d + 1) * f[d + 1]``. ``False`` when
    a coefficient is a float.

    Parameters
    ----------
    f : Poly
        Polynomial being differentiated.
    g : Poly
        Claimed derivative of *f*.
    env : Environment
        Environment with the ``Int.eqb`` definition.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.solver.solver_poly import verify_deriv
    >>> f = {2: Fraction(3), 0: Fraction(7)}  # 3x² + 7
    >>> env = mk_builtin_env()
    >>> verify_deriv(f, {1: Fraction(6)}, env)
    True
    >>> verify_deriv(f, {1: Fraction(3)}, env)
    False
    """
    for d in set(g) | {d - 1 for d in f if d >= 1}:
        gd, fd = g.get(d, Fraction(0)), f.get(d + 1, Fraction(0))
        if not (isinstance(gd, Fraction) and isinstance(fd, Fraction)):
            return False
        m = gd.denominator * fd.denominator
        check = App(App(Const("Int.eqb", ()), int_lit(int(gd * m))),
                    int_lit(int((d + 1) * fd * m)))
        if not reduces_true(check, env):
            return False
    return True


def deriv_call(expr: Any) -> Optional[Tuple[Any, Optional[str]]]:
    """
    Returns ``(argument, wrt)`` of the first ``deriv(...)`` call in
    *expr*, where *wrt* is ``None`` for ``deriv(f)``.

    Parameters
    ----------
    expr : AST
        Physika expression to search.

    Examples
    --------
    >>> from physika.solver.solver_poly import deriv_call
    >>> variable = lambda n: ("var", n)
    >>> f = ("pow", variable("t"), ("num", 2))
    >>> deriv_call(("add", ("call", "deriv", [f, variable("t")]),
    ...             ("num", 1)))
    (('pow', ('var', 't'), ('num', 2)), 't')
    """
    if not isinstance(expr, (list, tuple)):
        return None
    if (isinstance(expr, tuple) and expr and expr[0] == "call"
            and expr[1] in DERIV_NAMES and expr[2]):
        return expr[2][0], wrt_name(expr[2])
    for sub in expr:
        found = deriv_call(sub)
        if found is not None:
            return found
    return None


def poly_str(p: Poly, var: str) -> str:
    """
    Returns ``p`` as a string for report lines.

    Parameters
    ----------
    p : Poly
        Polynomial to show.
    var : str
        Name of the unknown.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_poly import poly_str
    >>> poly_str({2: Fraction(3), 1: Fraction(1), 0: Fraction(-7)}, "x")
    '3·x^2 + x - 7'
    """
    parts = []
    for d in sorted(p, reverse=True):
        power = var if d == 1 else f"{var}^{d}"
        if d == 0:
            parts.append(fmt_num(p[d]))
        else:
            parts.append(power if p[d] == 1 else f"{fmt_num(p[d])}·{power}")
    return " + ".join(parts).replace("+ -", "- ") or "0"


def solve_poly(
        var: str,
        kind: str,
        unit: dict,
        conds: List[tuple],
        scope: Dict[str, Any],
        funcs: dict,
        env: Environment,
        tag: str,
        tname: str,
        bindings: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
    """
    Solves an equation that is a polynomial in one unknown with
    numeric coefficients. Each ``deriv(...)`` gets a report line checked
    with ``verify_deriv``, and rational roots are kernel-checked with
    ``verify_poly_root``. Returns the report lines, or ``None`` when a side
    is not a polynomial. A root is added to `bindings` if is sucesfully
    verified.

    Parameters
    ----------
    var : str
        Name of the unknown.
    kind : str
        ``"nat"``, ``"int"`` or ``"real"``.
    unit : dict
        Declared unit of the unknown.
    conds : list[AST]
        ``("cond_eq", lhs, rhs)`` equations, only the first one is solved.
    scope : dict
        Values of the constants already known.
    funcs : dict
        Function definitions, by name.
    env : Environment
        Environment with the ``Int.eqb`` definition.
    tag : str
        Prefix for every report line.
    tname : str
        Type symbol shown in the report line.
    bindings : dict, optional
        Solved unknowns, filled in place.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.solver.solver_poly import solve_poly
    >>> variable = lambda n: ("var", n)
    >>> # x² - 5x + 6 == 0
    >>> lhs = ("add", ("sub", ("pow", variable("x"), ("num", 2)),
    ...                ("mul", ("num", 5), variable("x"))), ("num", 6))
    >>> eq = ("cond_eq", lhs, ("num", 0))
    >>> solve_poly("x", "int", {}, [eq], {}, {}, mk_builtin_env(),
    ...            "Line 1: solve x", "ℤ")
    ['Line 1: solve x : ℤ  →  x = 3, 2   (kernel-verified)']
    """
    lhs, rhs = conds[0][1], conds[0][2]
    pl = poly_of(lhs, var, scope, funcs, {})
    pr = poly_of(rhs, var, scope, funcs, {})
    if pl is None or pr is None:
        return None
    poly = p_add(pl, pr, sub=True)
    if not poly:
        return [f"{tag} : {tname}  →  any {var} (equation is trivial)"]

    lines: List[str] = []
    for side in (lhs, rhs):
        call = deriv_call(side)
        if call is None:
            continue
        wrt = call[1] or var
        fp = poly_of(call[0], wrt, scope, funcs, {})
        if fp is None:
            continue
        gp = p_deriv(fp)
        mark = "kernel-checked" if verify_deriv(fp, gp, env) else "symbolic"
        at = (f"  at {wrt} = {fmt_num(scope[wrt])}"
              if wrt != var and wrt in scope else "")
        lines.append(f"{tag}: d/d{wrt} [ {poly_str(fp, wrt)} ]  =  "
                     f"{poly_str(gp, wrt)}{at}   (power rule, {mark})")

    roots = [r for r in poly_roots(poly) if in_domain(r, kind)]
    # the factored out lowest power gives the root 0
    trivial = min(poly) > 0 and in_domain(0, kind)
    unit_s = f"  [{unit_to_str(unit)}]" if unit else ""
    if not roots:
        if trivial:
            return lines + [f"{tag} : {tname}  →  {var} = 0{unit_s}"]
        return lines + [f"{tag} : {tname}  ✗ no {tname} root"]
    if not all(isinstance(r, Fraction) for r in roots):
        degree = "linear" if max(poly) <= 1 else f"degree {max(poly)}"
        method = f"{degree} root, numeric"
    elif all(verify_poly_root(poly, r, env) for r in roots):
        method = "kernel-verified"
    else:
        method = ("exact rational root; kernel cert skipped "
                  "(cleared coefficients exceed the reducer's range)")
    if bindings is not None and len(roots) == 1 and not trivial:
        bindings[var] = to_expr(roots[0])
    body = ", ".join(fmt_num(r) for r in roots)
    extra = "   (plus the trivial root 0)" if trivial else ""
    return lines + [
        f"{tag} : {tname}  →  {var} = {body}{unit_s}"
        f"   ({method}){extra}"
    ]
