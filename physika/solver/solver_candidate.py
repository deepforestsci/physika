from typing import Any, List, Optional

from physika.core.expr import App, BVar, Const, Lit
from physika.core.environment import Environment
from physika.utils.cic_utils.expr_utils import instantiate1
from physika.utils.cic_utils.inductive_utils import int_lit
from physika.solver.utils import eval_const, reduces_true


def arith(expr: Any, var: str, is_int: bool) -> Any:
    """
    CIC term for `expr` with the single unknown `var` as ``BVar(0)``.

    Maps ``num``/``var``/``neg``/``add``/``sub``/``mul`` and
    literal-exponent ``pow`` onto ``Nat.*`` or ``Int.*``.  Leaving `var` as a
    bound variable lets ``solve_candidate`` plug in the candidate with
    ``instantiate1(term, cand)``.

    Used by ``solve_candidate`` three times per statement. On the
    equation's lhs, its rhs, and the candidate itself. The candidate cannot
    mention a variable.

    Parameters
    ----------
    expr : tuple
        Physika-AST arithmetic expression.
    var : str
        Name of the unknown.
    is_int : bool
        ``True`` -> ``Int.*`` / ``int_lit``; ``False`` -> ``Nat.*`` / ``Lit``.


    Examples
    --------
    >>> from physika.solver.solver_candidate import arith
    >>> arith(("add", ("var", "x"), ("num", 2)), "x", False) # noqa: E501
    App(func=App(func=Const(name='Nat.add', levels=()), arg=BVar(idx=0)), arg=Lit(val=2))
    """
    prefix = "Int" if is_int else "Nat"
    if not isinstance(expr, tuple):
        raise Exception(f"non-tuple node {expr!r}")
    op = expr[0]
    if op == "num":
        n = int(expr[1])
        if not is_int and n < 0:
            raise Exception("negative literal in a ℕ equation")
        return int_lit(n) if is_int else Lit(n)
    if op == "var":
        if expr[1] != var:
            raise Exception(f"free variable {expr[1]!r}")
        return BVar(0)
    if op == "neg":
        if not is_int:
            raise Exception("negation in a ℕ equation")
        return App(Const("Int.neg", ()), arith(expr[1], var, is_int))
    if op in ("add", "sub", "mul"):
        return App(
            App(Const(f"{prefix}.{op}", ()), arith(expr[1], var, is_int)),
            arith(expr[2], var, is_int))
    if op == "pow":
        if not (isinstance(expr[2], tuple) and expr[2][0] == "num"):
            raise Exception("non-literal exponent")
        k = int(expr[2][1])
        if k < 0:
            raise Exception("negative exponent")
        if k == 0:
            return int_lit(1) if is_int else Lit(1)
        base = arith(expr[1], var, is_int)
        acc = base
        for _ in range(k - 1):
            acc = App(App(Const(f"{prefix}.mul", ()), acc), base)
        return acc
    raise Exception(f"operator {op!r}")


def solve_candidate(variables: List[str], kind: str, candidates: List[Any],
                    conds: List[tuple], env: Environment, tag: str,
                    tname: str) -> List[str]:
    """
    Report ``✓ verified`` when a cancidate is succesfully verified and
    when the kernel reduces ``eqb (lhs c) (rhs c)`` to ``Bool.true``,
    else ``✗ not a root``. Called by ``solver.solve_block`` when the
    ``solve_decl`` has a candidate (its 4th slot is not ``None``).

    Parameters
    ----------
    variables : list of str
        Declared unknowns. Only ``variables[0]`` is used.
    kind : str
        Valid type kinds are Nat, Int or Real.
    candidates : list
        ``[c]`` AST of the value after ``=`` symbol.
    conds : list of tuple
        ``[("cond_eq", lhs, rhs)]``. Only the first is checked.
    env : Environment
        CIC environment.
    tag : str
        Report prefix (``"Line 3: solve x"``) and type symbol (``"ℕ"``).


    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.solver.solver_candidate import solve_candidate
    >>> eq = ("cond_eq", ("add", ("var", "x"), ("num", 2)), ("num", 5))
    >>> solve_candidate(["x"], "nat", [("num", 3)], [eq],
    ...                 mk_builtin_env(), "Line 1: solve x", "ℕ")
    ['Line 1: solve x : ℕ = 3  ✓ verified']
    >>> solve_candidate(["x"], "nat", [("num", 4)], [eq],
    ...                 mk_builtin_env(), "Line 1: solve x", "ℕ")
    ['Line 1: solve x : ℕ = 4  ✗ not a root']
    """
    if kind == "real":
        return [f"{tag}: the verify form (`= c`) needs an ℕ / ℤ unknown"]
    is_int = kind == "int"
    var = variables[0]
    cond = conds[0]
    try:
        lhs = arith(cond[1], var, is_int)
        rhs = arith(cond[2], var, is_int)
        cand = arith(candidates[0], var, is_int)
    except Exception as exc:
        return [f"{tag}: {exc}"]

    cand_val = eval_const(candidates[0], {})
    shown: Any = cand_val if cand_val is not None else "candidate"
    name = "Int.eqb" if is_int else "Nat.eqb"
    check = App(App(Const(name, ()), instantiate1(lhs, cand)),
                instantiate1(rhs, cand))
    mark = "✓ verified" if reduces_true(check, env) else "✗ not a root"
    return [f"{tag} : {tname} = {shown}  {mark}"]


def unsupported_reason(conds: List[tuple], var: str,
                       is_int: bool) -> Optional[str]:
    """
    Why the first equation falls outside ``arith``'s ℕ / ℤ fragment
    (e.g. ``"operator 'div'"``), or ``None`` if it is inside.

    Examples
    --------
    >>> from physika.solver.solver_candidate import unsupported_reason
    >>> eq = ("cond_eq", ("div", ("num", 6), ("var", "x")), ("num", 2))
    >>> unsupported_reason([eq], "x", True)
    "operator 'div'"
    """
    try:
        arith(conds[0][1], var, is_int)
        arith(conds[0][2], var, is_int)
    except Exception as exc:
        return str(exc)
    return None
