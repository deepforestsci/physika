from fractions import Fraction
from math import lcm
from typing import Any, Dict, List, Optional, Tuple

from physika.core.expr import App, Const
from physika.core.environment import Environment
from physika.utils.cic_utils.inductive_utils import int_lit
from physika.solver.utils import (fmt_num, free_vars, in_domain, int_sum,
                                  reduces_true, to_expr)

# Data type for expressing linear solutions and checking with mypy
Lin = Dict[Optional[str], Fraction]


def lin(expr: Any,
        unknowns: List[str],
        scope: Optional[Dict[str, Any]] = None) -> Optional[Lin]:
    """
    Linear form of ``expr`` over ``unknowns``, or ``None`` if it is not
    linear. The form maps each unknown to its coefficient and the key
    ``None`` to the constant term, so ``2*x + 3*y - 4`` is
    ``{"x": 2, "y": 3, None: -4}``.

    A ``var`` that is not an unknown but is a key of ``scope`` contributes
    its numeric value as a constant.

    Parameters
    ----------
    expr : AST
        Physika expression.
    unknowns : list[str]
        Names of the unknowns.
    scope : dict, optional
        Values of the constants already known.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_linear import lin
    >>> variable = lambda n: ("var", n)
    >>> two_x = ("mul", ("num", 2), variable("x"))
    >>> three_y = ("mul", ("num", 3), variable("y"))
    >>> expr = ("sub", ("add", two_x, three_y), variable("c"))
    >>> lin(expr, ["x", "y"], {"c": Fraction(4)})
    {'x': Fraction(2, 1), 'y': Fraction(3, 1), None: Fraction(-4, 1)}
    >>> lin(("mul", variable("x"), variable("y")), ["x", "y"]) is None
    True
    """
    scope = scope or {}
    if not isinstance(expr, tuple):
        return None
    op = expr[0]
    # a number is a constant term
    if op == "num":
        return {None: frac(expr[1])}
    # an unknown has coefficient 1
    if op == "var":
        if expr[1] in unknowns:
            return {expr[1]: Fraction(1)}
        # a known constant is a constant term
        if expr[1] in scope:
            return {None: frac(scope[expr[1]])}
        return None
    # negates every coefficient
    if op == "neg":
        inner = lin(expr[1], unknowns, scope)
        return None if inner is None else {k: -v for k, v in inner.items()}
    # a + or - b adds or subtracts the coefficients key by key
    if op in ("add", "sub"):
        a, b = lin(expr[1], unknowns, scope), lin(expr[2], unknowns, scope)
        if a is None or b is None:
            return None
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, Fraction(0)) + (v if op == "add" else -v)
        return out
    # a * b is linear only when one side is a constant
    # scales every coefficient of the other side
    if op == "mul":
        a, b = lin(expr[1], unknowns, scope), lin(expr[2], unknowns, scope)
        if a is None or b is None:
            return None
        if set(a) == {None}:
            return {k: v * a[None] for k, v in b.items()}
        if set(b) == {None}:
            return {k: v * b[None] for k, v in a.items()}
        return None
    # a / b is linear only when b is a non zero constant
    if op == "div":
        a, b = lin(expr[1], unknowns, scope), lin(expr[2], unknowns, scope)
        if a is None or b is None or set(b) != {None} or b[None] == 0:
            return None
        return {k: v / b[None] for k, v in a.items()}
    # a ** n needs a number exponent
    if op == "pow":
        base = lin(expr[1], unknowns, scope)
        if base is None or not (isinstance(expr[2], tuple)
                                and expr[2][0] == "num"):
            return None
        exp_k = int(expr[2][1])
        if exp_k == 1:
            return base
        if set(base) == {None}:
            return {None: base[None]**exp_k}
        return None
    return None


def frac(v: Any) -> Fraction:
    """
    Returns a number as an exact ``Fraction``. A non-integer float is
    rounded to the closest fraction with denominator up to 10⁹.

    Parameters
    ----------
    v : int, float or Fraction
        Number to convert.

    Examples
    --------
    >>> from physika.solver.solver_linear import frac
    >>> frac(3), frac(2.0), frac(0.25)
    (Fraction(3, 1), Fraction(2, 1), Fraction(1, 4))
    """
    if isinstance(v, Fraction):
        return v
    return Fraction(v).limit_denominator(10**9)


def eq_row(cond: tuple,
           unknowns: List[str],
           scope: Optional[Dict[str, Any]] = None) -> Optional[Lin]:
    """
    Returns ``lhs - rhs`` as a linear form, which must
    equal 0 (from cond).

    Returns ``None`` when a side is not linear.

    Parameters
    ----------
    cond : AST
        ``("cond_eq", lhs, rhs)`` equation.
    unknowns : list[str]
        Names of the unknowns.
    scope : dict, optional
        Values of the constants already known.

    Examples
    --------
    >>> from physika.solver.solver_linear import eq_row
    >>> variable = lambda n: ("var", n)
    >>> eq_row(("cond_eq", variable("x"), ("num", 4)), ["x"])
    {'x': Fraction(1, 1), None: Fraction(-4, 1)}
    """
    return lin(("sub", cond[1], cond[2]), unknowns, scope)


def gauss_jordan(rows: List[Lin],
                 unknowns: List[str]) -> Optional[Dict[str, Fraction]]:
    """
    Exact Gauss–Jordan elimination on ``rows`` (each one
    ``sum coeff * var + const == 0``). Returns the unique solution, or
    ``None`` when the system is not square or is singular.

    Parameters
    ----------
    rows : list[Lin]
        Linear forms from ``eq_row``.
    unknowns : list[str]
        Names of the unknowns, one per row.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_linear import gauss_jordan
    >>> # x + y - 10 == 0, x - y - 4 == 0
    >>> rows = [{"x": Fraction(1), "y": Fraction(1), None: Fraction(-10)},
    ...         {"x": Fraction(1), "y": Fraction(-1), None: Fraction(-4)}]
    >>> gauss_jordan(rows, ["x", "y"])
    {'x': Fraction(7, 1), 'y': Fraction(3, 1)}
    """
    n = len(unknowns)
    if len(rows) != n:
        return None
    zero = Fraction(0)
    # augmented matrix [A | b] of A x = b, where b is -const
    mat = [[r.get(v, zero) for v in unknowns] + [-r.get(None, zero)]
           for r in rows]
    for col in range(n):
        piv = next((r for r in range(col, n) if mat[r][col] != 0), None)
        if piv is None:
            return None
        mat[col], mat[piv] = mat[piv], mat[col]
        mat[col] = [x / mat[col][col] for x in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[col])]
    return {v: mat[i][n] for i, v in enumerate(unknowns)}


def verify_linear_row(row: Lin, sol: Dict[str, Fraction],
                      env: Environment) -> bool:
    """
    CIC kernel checks that ``row`` is 0 at ``sol``. Denominators are
    cleared to get an integer identity, then ``Int.eqb sum 0`` must reduce
    to ``Bool.true``.

    Parameters
    ----------
    row : Lin
        Linear form from ``eq_row``.
    sol : dict
        Value of every unknown in *row*.
    env : Environment
        Environment that contains ``Int.eqb`` definition.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.solver.solver_linear import verify_linear_row
    >>> row = {"x": Fraction(2), None: Fraction(-1)}  # 2x - 1 == 0
    >>> env = mk_builtin_env()
    >>> verify_linear_row(row, {"x": Fraction(1, 2)}, env)
    True
    >>> verify_linear_row(row, {"x": Fraction(1)}, env)
    False
    """
    m = lcm(*(v.denominator for v in sol.values()),
            *(c.denominator for c in row.values()))
    terms = [int(c * m * (1 if k is None else sol[k])) for k, c in row.items()]
    check = App(App(Const("Int.eqb", ()), int_sum(terms)), int_lit(0))
    return reduces_true(check, env)


def linear_system(
        conds: List[tuple], variables: List[str],
        scope: Dict[str,
                    Any]) -> Optional[Tuple[List[Lin], List[str], List[str]]]:
    """
    Returns ``(rows, aux, vars)`` when the equations form a linear
    system, else ``None``. A name that is not an unknown or in
    *scope* becomes an auxiliary unknown (*aux*). This aux is then solved
    and reported if eliminated during row operations.

    A single equation in one unknown is left to the
    polynomial path.

    Parameters
    ----------
    conds : list[AST]
        ``("cond_eq", lhs, rhs)`` equations.
    variables : list[str]
        Names of the unknowns.
    scope : dict
        Values of the constants already known.

    Examples
    --------
    >>> from physika.solver.solver_linear import linear_system
    >>> variable = lambda n: ("var", n)
    >>> eq = ("cond_eq", ("add", variable("x"), variable("k")), ("num", 3))
    >>> rows, aux, all_vars = linear_system([eq], ["x"], {})
    >>> aux, all_vars
    (['k'], ['x', 'k'])
    >>> linear_system([eq], ["x"], {"k": 1}) is None
    True
    """
    known = set(scope) | set(variables)
    aux = sorted(set().union(*(free_vars(c, known) for c in conds)))
    allvars = list(variables) + aux
    rows = [
        r for r in (eq_row(c, allvars, scope) for c in conds) if r is not None
    ]
    if not conds or len(rows) != len(conds):
        return None
    return (rows, aux, allvars) if aux or len(variables) > 1 else None


def solve_linear(
        system: Tuple[List[Lin], List[str], List[str]],
        variables: List[str],
        kind: str,
        env: Environment,
        tag: str,
        tname: str,
        bindings: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
    """
    Solves a ``linear_system`` result with gauss jordan method and
    kernel verifies each equation with ``verify_linear_row``.
    Returns a solution if successful, or ``None`` when there is no unique
    solution.
    A verified solution is added to `bindings` for codegen.

    Parameters
    ----------
    system : tuple
        ``(rows, aux, all_vars)`` from ``linear_system``.
    variables : list[str]
        Names of the unknowns.
    kind : str
        ``"nat"``, ``"int"`` or ``"real"``.
    env : Environment
        CIC environment.
    tag : str
        Prefix for the report line.
    tname : str
        Type symbol shown in the report line.
    bindings : dict, optional
        Solved unknowns, filled in place.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.solver.solver_linear import linear_system, solve_linear
    >>> variable = lambda n: ("var", n)
    >>> eqs = [("cond_eq", ("add", variable("x"), variable("y")),
    ...         ("num", 10)),
    ...        ("cond_eq", ("sub", variable("x"), variable("y")),
    ...         ("num", 4))]
    >>> system = linear_system(eqs, ["x", "y"], {})
    >>> solve_linear(system, ["x", "y"], "int", mk_builtin_env(),
    ...              "Line 1: solve x, y", "ℤ")
    ['Line 1: solve x, y : ℤ  →  x = 7, y = 3   (kernel-verified)']
    """
    rows, aux, allvars = system
    sol = gauss_jordan(rows, allvars)
    if sol is None:
        return None
    body = ", ".join(f"{v} = {fmt_num(sol[v])}" for v in variables)
    if not all(in_domain(sol[v], kind) for v in variables):
        return [
            f"{tag} : {tname}  ✗ no {tname} solution "
            f"(unique rational solution: {body})"
        ]
    ok = all(verify_linear_row(r, sol, env) for r in rows)
    note = ""
    if kind == "int" and any(v.denominator != 1 for v in sol.values()):
        note = "  (rational solution)"
    if aux:
        elim = ", ".join(f"{v} = {fmt_num(sol[v])}" for v in aux)
        note += f"   [eliminated: {elim}]"
    if ok and bindings is not None:
        bindings.update({v: to_expr(sol[v]) for v in variables})
    verdict = "kernel-verified" if ok else "UNVERIFIED"
    return [f"{tag} : {tname}  →  {body}   ({verdict}){note}"]
