"""
Helpers shared across the solver's paths: ``solve`` type specs, AST
walks, constant evaluation, number formatting and kernel checks.
"""
import math
import operator
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from physika.core.expr import App, Const, Expr
from physika.core.environment import Environment
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.reduction import whnf
from physika.units import sig_unit_spec
from physika.utils.cic_utils.inductive_utils import int_lit

# Unknown's type (the parser only produces ℕ / ℤ / ℝ) -> solver kind
KINDS = {"ℕ": "nat", "ℤ": "int", "ℝ": "real"}
# Solver kind -> type symbol shown in report lines
KIND_SYMBOL = {kind: symbol for symbol, kind in KINDS.items()}

# Names of the formal derivative inside a `where` equation
DERIV_NAMES = {"deriv", "d_dx", "∂", "D"}

# Declaration tag -> position of its right hand side
_RHS_INDEX = {"unit_decl": 4, "decl": 3, "solve_local": 3, "assign": 2}

_BINOPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv
}
_BINOP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}


def kind_and_unit(type_spec: Any) -> Tuple[Optional[str], dict]:
    """
    Returns the kind (``"nat"``, ``"int"``, ``"real"`` or ``None`` when
    unsupported) and the ``← [unit]`` of a ``solve`` unknown's type.

    Parameters
    ----------
    type_spec : str or tuple
        ``"ℤ"``, or ``("unit_typed", "ℝ", unit_ast)`` with a unit.

    Examples
    --------
    >>> from physika.solver.utils import kind_and_unit
    >>> kind_and_unit("ℤ"), kind_and_unit("ℂ")
    (('int', {}), (None, {}))
    >>> kind_and_unit(("unit_typed", "ℝ", [("m", 1), ("s", -2)]))
    ('real', {'m': 1, 's': -2})
    """
    if isinstance(type_spec, tuple) and type_spec[0] == "unit_typed":
        return KINDS.get(type_spec[1]), sig_unit_spec(type_spec[2])
    return (KINDS.get(type_spec) if isinstance(type_spec, str) else None), {}


def free_vars(expr: Any, exclude: set) -> set:
    """
    Returns the variable names in expr that are not in exclude.

    Parameters
    ----------
    expr : AST or list
        Physika expressions as list.
    exclude : set
        Names to ignore.

    Examples
    --------
    >>> from physika.solver.utils import free_vars
    >>> variable = lambda n: ("var", n)
    >>> free_vars(("add", variable("x"), variable("k")), {"x"})
    {'k'}
    """
    if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "var":
        return set() if expr[1] in exclude else {expr[1]}
    if isinstance(expr, (tuple, list)):
        return set().union(*(free_vars(s, exclude) for s in expr))
    return set()


def eval_const(expr: Any, scope: Dict[str, Any]) -> Optional[Any]:
    """
    Evaluates a constant arithmetic expression. Integers stay exact
    ``Fraction`` values, a float or ``sqrt`` gives a float. Returns
    ``None`` when *expr* is not a constant.

    Parameters
    ----------
    expr : AST
        Physika expression.
    scope : dict
        Values of the variables already known.

    Examples
    --------
    >>> from physika.solver.utils import eval_const
    >>> eval_const(("div", ("num", 1), ("num", 3)), {})
    Fraction(1, 3)
    >>> eval_const(("add", ("var", "a"), ("num", 1)), {"a": 2})
    Fraction(3, 1)
    >>> eval_const(("var", "b"), {}) is None
    True
    """
    if not isinstance(expr, tuple):
        return None
    op = expr[0]
    if op == "num":
        v = expr[1]
        return v if isinstance(v, float) else Fraction(int(v))
    if op == "var":
        return scope.get(expr[1])
    if op == "neg":
        v = eval_const(expr[1], scope)
        return None if v is None else -v
    if op in _BINOPS or op == "pow":
        a, b = eval_const(expr[1], scope), eval_const(expr[2], scope)
        if a is None or b is None:
            return None
        return a**int(b) if op == "pow" else _BINOPS[op](a, b)
    if op == "call" and expr[1] == "sqrt" and len(expr[2]) == 1:
        v = eval_const(expr[2][0], scope)
        return None if v is None else float(v)**0.5
    return None


def _name_and_rhs(stmt: Any) -> Optional[Tuple[str, Any]]:
    """
    Returns ``(name, rhs)`` of a declaration statement, ``None`` for any
    other statement.

    Parameters
    ----------
    stmt : AST
        Top-level statement or ``where:`` block local.

    Examples
    --------
    >>> from physika.solver.utils import _name_and_rhs
    >>> _name_and_rhs(("decl", "a", "ℝ", ("num", 2), 1))
    ('a', ('num', 2))
    >>> _name_and_rhs(("print", ("var", "a"))) is None
    True
    """
    if isinstance(stmt, tuple) and stmt and stmt[0] in _RHS_INDEX:
        return stmt[1], stmt[_RHS_INDEX[stmt[0]]]
    return None


def eval_scope(program: List[Any],
               scope: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluates the constant declarations of `program` into `scope` in place.
    Declarations that are not constants are skipped.

    Parameters
    ----------
    program : list
        Top-level statements or ``where:`` block locals.
    scope : dict, optional
        Values to extend.

    Examples
    --------
    >>> from physika.solver.utils import eval_scope
    >>> eval_scope([("decl", "a", "ℝ", ("num", 2), 1),
    ...             ("decl", "b", "ℝ", ("mul", ("var", "a"), ("num", 3)), 2)])
    {'a': Fraction(2, 1), 'b': Fraction(6, 1)}
    """
    scope = {} if scope is None else scope
    for stmt in program:
        decl = _name_and_rhs(stmt)
        if decl is not None:
            val = eval_const(decl[1], scope)
            if val is not None:
                scope[decl[0]] = val
    return scope


def raw_defs(program: List[Any], block_locals: List[Any]) -> Dict[str, Any]:
    """
    Returns the right hand side of every declaration, numeric or symbolic,
    by name. Block locals have prefernce over top-level declarations.

    Examples
    --------
    >>> from physika.solver.utils import raw_defs
    >>> raw_defs([("decl", "a", "ℝ", ("var", "t"), 1)], [])
    {'a': ('var', 't')}
    """
    decls = (_name_and_rhs(s) for s in list(program) + list(block_locals))
    return {d[0]: d[1] for d in decls if d is not None}


def to_expr(c: Any) -> Any:
    """
    Returns a coefficient (number or AST) as a Physika expression.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.utils import to_expr
    >>> to_expr(Fraction(1, 3))
    ('div', ('num', 1), ('num', 3))
    """
    if isinstance(c, tuple):
        return c
    if isinstance(c, Fraction) and c.denominator != 1:
        return ("div", ("num", c.numerator), ("num", c.denominator))
    if isinstance(c, Fraction):
        return ("num", c.numerator)
    return ("num", c)


def fraction_isqrt(x: Fraction) -> Optional[Fraction]:
    """
    Exact square root of a ``Fraction``, ``None`` when it is not exact.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.utils import fraction_isqrt
    >>> fraction_isqrt(Fraction(9, 4)), fraction_isqrt(Fraction(2))
    (Fraction(3, 2), None)
    """
    if x < 0:
        return None
    n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
    if n * n == x.numerator and d * d == x.denominator:
        return Fraction(n, d)
    return None


def fmt_num(v: Any) -> str:
    """
    Number as shown in report lines.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.utils import fmt_num
    >>> fmt_num(Fraction(4)), fmt_num(Fraction(1, 3)), fmt_num(2.0)
    ('4', '1/3', '2')
    """
    if isinstance(v, Fraction) and v.denominator == 1:
        return str(v.numerator)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:.6g}"
    return str(v)


def expr_str(e: Any) -> str:
    """
    Physika expression as a string, for report lines.

    Examples
    --------
    >>> from physika.solver.utils import expr_str
    >>> expr_str(("mul", ("num", 2), ("call", "sqrt", [("var", "x")])))
    '(2 * sqrt(x))'
    """
    if not isinstance(e, tuple):
        return str(e)
    op = e[0]
    if op == "num":
        return fmt_num(e[1])
    if op == "var":
        return str(e[1])
    if op == "imaginary":
        return "i"
    if op == "complex":
        return str(e[1])
    if op == "neg":
        return f"-{expr_str(e[1])}"
    if op in _BINOP_SYMBOLS:
        return f"({expr_str(e[1])} {_BINOP_SYMBOLS[op]} {expr_str(e[2])})"
    if op == "call":
        return f"{e[1]}({', '.join(expr_str(a) for a in e[2])})"
    return str(e)


def in_domain(r: Any, kind: Optional[str]) -> bool:
    """
    Checks if root *r* is a valid value for a ℕ / ℤ / ℝ unknown.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.utils import in_domain
    >>> in_domain(Fraction(-2), "int"), in_domain(Fraction(-2), "nat")
    (True, False)
    """
    if kind == "real":
        return isinstance(r, (int, float, Fraction))
    if isinstance(r, float) or Fraction(r).denominator != 1:
        return False
    return kind != "nat" or r >= 0


def int_sum(values: List[int]) -> Expr:
    """
    ``Int.add`` chain of integer literals, ``0`` when empty.

    Examples
    --------
    >>> from physika.solver.utils import int_sum
    >>> int_sum([]) == int_lit(0)
    True
    """
    if not values:
        return int_lit(0)
    acc = int_lit(values[0])
    for v in values[1:]:
        acc = App(App(Const("Int.add", ()), acc), int_lit(v))
    return acc


def reduces_true(term: Expr, env: Environment) -> bool:
    """
    Checks if the CIC kernel reduces *term* to ``Bool.true``. A stuck term
    or a ``RecursionError`` counts as not verified. The recursion limit is
    raised while reducing, since ``Nat`` / ``Int`` arithmetic recurses
    deeply.

    Parameters
    ----------
    term : Expr
        Closed CIC term of type ``Bool`` (``Nat.eqb a b``, ``Int.eqb a b``).
    env : Environment
        Environment with the ``Nat.eqb`` / ``Int.eqb`` definitions.

    Examples
    --------
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.solver.utils import reduces_true
    >>> env = mk_builtin_env()
    >>> eqb = lambda a, b: App(App(Const("Nat.eqb", ()), Lit(a)), Lit(b))
    >>> reduces_true(eqb(4, 4), env), reduces_true(eqb(3, 5), env)
    (True, False)
    """
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20000))
    try:
        w = whnf(term, env, LocalContext(), MetaVarContext())
    except RecursionError:
        return False
    finally:
        sys.setrecursionlimit(old)
    return isinstance(w, Const) and w.name == "Bool.true"
