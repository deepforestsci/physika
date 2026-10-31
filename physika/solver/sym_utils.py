import cmath
import math
import operator
import random
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

from physika.core.expr import App, Const, Expr, Lit
from physika.core.inductive import mk_builtin_env
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.reduction import whnf
from physika.utils.cic_utils.inductive_utils import int_lit
from physika.utils.cic_utils.sym_ring import monomial_basis, reg_sym_ops
from physika.solver.utils import DERIV_NAMES, to_expr

# Largest monomial basis the Sym ring check builds
MAX_SYM_BASIS = 40  # larger identities use the random-instance check instead.

ARITH = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "pow": operator.pow
}
MATH_UNARY: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log,
    "abs": abs
}
# complex argument versions
CMATH_UNARY: Dict[str, Callable[..., Any]] = {
    "sqrt": cmath.sqrt,
    "sin": cmath.sin,
    "cos": cmath.cos,
    "exp": cmath.exp,
    "log": cmath.log
}


def is_num(c: Any) -> bool:
    """
    Checks if a coefficient is a number and not an AST.

    Parameters
    ----------
    c : Fraction, int, float or AST
        Coefficient.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.sym_utils import is_num
    >>> is_num(Fraction(1, 2)), is_num(("var", "a"))
    (True, False)
    """
    return isinstance(c, (Fraction, int, float))


def is_neg(c: Any) -> bool:
    """
    Checks if a coefficient is a negated AST ``("neg", e)``.

    Parameters
    ----------
    c : Fraction, int, float or AST
        Coefficient.

    Examples
    --------
    >>> from physika.solver.sym_utils import is_neg
    >>> is_neg(("neg", ("var", "a"))), is_neg(("var", "a"))
    (True, False)
    """
    return isinstance(c, tuple) and len(c) == 2 and c[0] == "neg"


def sadd(a: Any, b: Any) -> Any:
    """
    Returns ``a + b`` for coefficients that are numbers or ASTs. Numbers
    are added exactly and a zero term is dropped.

    Parameters
    ----------
    a : Fraction, int, float or AST
        Left operand.
    b : Fraction, int, float or AST
        Right operand.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.sym_utils import sadd
    >>> sadd(Fraction(1, 2), 1), sadd(0, ("var", "a"))
    (Fraction(3, 2), ('var', 'a'))
    >>> sadd(("var", "a"), 2)
    ('add', ('var', 'a'), ('num', 2))
    """
    if is_num(a) and is_num(b):
        return Fraction(a) + Fraction(b)
    if is_num(a) and a == 0:
        return to_expr(b)
    if is_num(b) and b == 0:
        return to_expr(a)
    return ("add", to_expr(a), to_expr(b))


def ssub(a: Any, b: Any) -> Any:
    """
    Returns ``a - b`` for coefficients that are numbers or ASTs.

    Parameters
    ----------
    a : Fraction, int, float or AST
        Left operand.
    b : Fraction, int, float or AST
        Right operand.

    Examples
    --------
    >>> from physika.solver.sym_utils import ssub
    >>> ssub(0, ("var", "a"))
    ('neg', ('var', 'a'))
    """
    if is_num(a) and is_num(b):
        return Fraction(a) - Fraction(b)
    if is_num(b) and b == 0:
        return to_expr(a)
    if is_num(a) and a == 0:
        return sneg(b)
    return ("sub", to_expr(a), to_expr(b))


def sneg(a: Any) -> Any:
    """
    Returns ``-a`` for a coefficient that is a number or an AST, with
    ``--a`` simplified to ``a``.

    Parameters
    ----------
    a : Fraction, int, float or AST
        Operand.

    Examples
    --------
    >>> from physika.solver.sym_utils import sneg
    >>> sneg(sneg(("var", "a")))
    ('var', 'a')
    """
    if is_num(a):
        return -Fraction(a)
    if is_neg(a):
        return a[1]
    return ("neg", to_expr(a))


def smul(a: Any, b: Any) -> Any:
    """
    Returns ``a * b`` for coefficients that are numbers or ASTs. Factors
    0, 1 and -1 are simplified and a negation is moved outside.

    Parameters
    ----------
    a : Fraction, int, float or AST
        Left operand.
    b : Fraction, int, float or AST
        Right operand.

    Examples
    --------
    >>> from physika.solver.sym_utils import smul
    >>> variable = lambda n: ("var", n)
    >>> smul(-1, variable("a"))
    ('neg', ('var', 'a'))
    >>> smul(("neg", variable("a")), variable("b"))
    ('neg', ('mul', ('var', 'a'), ('var', 'b')))
    """
    if is_num(a) and is_num(b):
        return Fraction(a) * Fraction(b)
    if (is_num(a) and a == 0) or (is_num(b) and b == 0):
        return Fraction(0)
    if is_num(a) and a in (1, -1):
        return to_expr(b) if a == 1 else sneg(b)
    if is_num(b) and b in (1, -1):
        return to_expr(a) if b == 1 else sneg(a)
    if is_neg(a):
        return sneg(smul(a[1], b))
    if is_neg(b):
        return sneg(smul(a, b[1]))
    return ("mul", to_expr(a), to_expr(b))


def sdiv(a: Any, b: Any) -> Any:
    """
    Returns ``a / b`` for coefficients that are numbers or ASTs, with
    ``(-p) / (-q)`` simplified to ``p / q``.

    Parameters
    ----------
    a : Fraction, int, float or AST
        Dividend.
    b : Fraction, int, float or AST
        Divisor.

    Examples
    --------
    >>> from physika.solver.sym_utils import sdiv
    >>> sdiv(("var", "a"), 1), sdiv(1, 4)
    (('var', 'a'), Fraction(1, 4))
    """
    if is_num(a) and is_num(b):
        return Fraction(a) / Fraction(b)
    if is_num(b) and b == 1:
        return to_expr(a)
    if is_neg(a) and is_neg(b):
        return sdiv(a[1], b[1])
    return ("div", to_expr(a), to_expr(b))


# Symbolic polynomial in one unknown as {degree: coeff}
SPoly = Dict[int, Any]


def sp_add(a: SPoly, b: SPoly, sub: bool = False) -> SPoly:
    """
    Returns ``a + b``, or ``a - b`` when *sub* is true.

    Parameters
    ----------
    a : SPoly
        Left operand.
    b : SPoly
        Right operand.
    sub : bool, optional
        Subtract *b* instead of adding it.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.sym_utils import sp_add
    >>> sp_add({1: Fraction(2)}, {1: ("var", "k")})
    {1: ('add', ('num', 2), ('var', 'k'))}
    """
    out = dict(a)
    for d, c in b.items():
        prev = out.get(d, Fraction(0))
        out[d] = ssub(prev, c) if sub else sadd(prev, c)
    return out


def sp_mul(a: SPoly, b: SPoly) -> SPoly:
    """
    Returns ``a * b``.

    Parameters
    ----------
    a : SPoly
        Left operand.
    b : SPoly
        Right operand.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.sym_utils import sp_mul
    >>> sp_mul({1: Fraction(1)}, {1: ("var", "k"), 0: Fraction(3)})
    {2: ('var', 'k'), 1: Fraction(3, 1)}
    """
    out: SPoly = {}
    for da, ca in a.items():
        for db, cb in b.items():
            out[da + db] = sadd(out.get(da + db, Fraction(0)), smul(ca, cb))
    return out


def contains_var(expr: Any, name: str) -> bool:
    """
    Checks if variable *name* appears in *expr*.

    Parameters
    ----------
    expr : AST or list
        Physika expression.
    name : str
        Variable name.

    Examples
    --------
    >>> from physika.solver.sym_utils import contains_var
    >>> expr = ("call", "cos", [("var", "θ")])
    >>> contains_var(expr, "θ"), contains_var(expr, "x")
    (True, False)
    """
    if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "var":
        return expr[1] == name
    if isinstance(expr, (tuple, list)):
        return any(contains_var(s, name) for s in expr)
    return False


def lit_or_expr(expr: Any) -> Any:
    """
    Returns a ``("num", v)`` literal as a number, any other expression
    unchanged.

    Parameters
    ----------
    expr : AST
        Physika expression.

    Examples
    --------
    >>> from physika.solver.sym_utils import lit_or_expr
    >>> lit_or_expr(("num", 3)), lit_or_expr(("var", "a"))
    (Fraction(3, 1), ('var', 'a'))
    """
    if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "num":
        v = expr[1]
        return v if isinstance(v, float) else Fraction(int(v))
    return expr


def deriv_args(expr: Any) -> Optional[Tuple[Any, str]]:
    """
    Returns ``(f, x)`` of a ``deriv(f, x)`` call whose *x* is a variable,
    ``None`` for any other expression.

    Parameters
    ----------
    expr : AST
        Physika expression.

    Examples
    --------
    >>> from physika.solver.sym_utils import deriv_args
    >>> f = ("call", "sin", [("var", "t")])
    >>> deriv_args(("call", "deriv", [f, ("var", "t")]))
    (('call', 'sin', [('var', 't')]), 't')
    """
    if (isinstance(expr, tuple) and len(expr) == 3 and expr[0] == "call"
            and expr[1] in DERIV_NAMES and isinstance(expr[2], list)
            and len(expr[2]) == 2 and isinstance(expr[2][1], tuple)
            and expr[2][1][:1] == ("var", )):
        return expr[2][0], expr[2][1][1]
    return None


def rewrite_once(expr: Any, rule: Callable[[Any, Any], Any],
                 data: Any) -> Tuple[Any, bool]:
    """
    One top-down pass over *expr*: a tuple node that ``rule(node, data)``
    rewrites (by returning ``(new, changed)``) is replaced and not visited
    further, every other node is walked into. Returns the new expression
    and whether anything changed.

    Parameters
    ----------
    expr : AST or list
        Physika expression.
    rule : callable
        ``rule(node, data)`` returning ``(new, changed)`` or ``None``.
    data : Any
        Extra argument passed to *rule*.

    Examples
    --------
    >>> from physika.solver.sym_utils import rewrite_once, defs_rule
    >>> rewrite_once(("neg", ("var", "a")), defs_rule, {"a": ("num", 2)})
    (('neg', ('num', 2)), True)
    """
    if isinstance(expr, tuple):
        hit = rule(expr, data)
        if hit is not None:
            return hit
    if not isinstance(expr, (tuple, list)):
        return expr, False
    parts, changed = [], False
    for s in expr:
        s2, c = rewrite_once(s, rule, data)
        parts.append(s2)
        changed = changed or c
    return (tuple(parts) if isinstance(expr, tuple) else parts), changed


def rewrite(expr: Any, rule: Callable[[Any, Any], Any], data: Any,
            depth: int) -> Any:
    """
    Repeats ``rewrite_once`` until nothing changes, at most *depth* times.

    Parameters
    ----------
    expr : AST or list
        Physika expression.
    rule : callable
        ``rule(node, data)`` returning ``(new, changed)`` or ``None``.
    data : Any
        Extra argument passed to *rule*.
    depth : int
        Maximum number of passes.

    Examples
    --------
    >>> from physika.solver.sym_utils import rewrite, defs_rule
    >>> defs = {"a": ("var", "b"), "b": ("num", 2)}
    >>> rewrite(("var", "a"), defs_rule, defs, 8)
    ('num', 2)
    """
    for _ in range(depth):
        expr, changed = rewrite_once(expr, rule, data)
        if not changed:
            break
    return expr


def defs_rule(node: tuple, defs: Dict[Any, Any]) -> Optional[Tuple[Any, bool]]:
    """
    ``rewrite`` rule replacing a variable that has a definition in *defs*.

    Parameters
    ----------
    node : tuple
        AST node.
    defs : dict
        Definitions, by variable name.

    Examples
    --------
    >>> from physika.solver.sym_utils import defs_rule
    >>> defs_rule(("var", "a"), {"a": ("num", 2)})
    (('num', 2), True)
    """
    if len(node) == 2 and node[0] == "var" and node[1] in defs:
        return defs[node[1]], True
    return None


def chain_rule(node: tuple, chain: Dict[Any,
                                        Any]) -> Optional[Tuple[Any, bool]]:
    """
    ``rewrite`` rule for ``substitute_chain``. A node that is a key of
    *chain* or a variable with a definition is replaced. In
    ``deriv(f, x)``, *x* names the variable to differentiate in, so it is
    kept and not substituted inside *f*.

    Parameters
    ----------
    node : tuple
        AST node.
    chain : dict
        Definitions, by variable name or by AST subtree.

    Examples
    --------
    >>> from physika.solver.sym_utils import chain_rule
    >>> square = ("pow", ("var", "q"), ("num", 2))
    >>> chain_rule(square, {square: ("var", "s")})
    (('var', 's'), True)
    """
    d = deriv_args(node)
    if d is not None:
        f, wrt = d
        new_f = substitute_chain(f, {
            k: v
            for k, v in chain.items() if k != wrt
        })
        return ("call", node[1], [new_f, node[2][1]]), new_f != f
    try:
        if node in chain:
            return chain[node], True
    except TypeError:
        pass  # a node holding a list is not hashable
    return defs_rule(node, chain)


def expand_rule(node: tuple, _: Any) -> Optional[Tuple[Any, bool]]:
    """
    ``rewrite`` rule replacing ``deriv(f, x)`` with its derivative from
    ``sym_deriv``.

    Parameters
    ----------
    node : tuple
        AST node.
    _ : Any
        Unused.

    Examples
    --------
    >>> from physika.solver.sym_utils import expand_rule
    >>> square = ("pow", ("var", "t"), ("num", 2))
    >>> expand_rule(("call", "deriv", [square, ("var", "t")]), None)
    (('mul', ('num', 2), ('var', 't')), True)
    """
    d = deriv_args(node)
    if d is None:
        return None
    val = sym_deriv(*d)
    return None if val is None else (to_expr(val), True)


def substitute_defs(expr: Any, defs: Dict[str, Any], depth: int = 8) -> Any:
    """
    Replaces every variable that has a definition in *defs*, repeated up
    to *depth* times so definitions that use other definitions unroll.

    Parameters
    ----------
    expr : AST
        Physika expression.
    defs : dict
        Definitions, by variable name.
    depth : int, optional
        Maximum number of passes.

    Examples
    --------
    >>> from physika.solver.sym_utils import substitute_defs
    >>> defs = {"a": ("mul", ("num", 2), ("var", "b")), "b": ("var", "c")}
    >>> substitute_defs(("var", "a"), defs)
    ('mul', ('num', 2), ('var', 'c'))
    """
    return rewrite(expr, defs_rule, defs, depth)


def substitute_chain(expr: Any, chain: Dict[Any, Any]) -> Any:
    """
    Like ``substitute_defs``, but *chain* may also have AST subtrees as
    keys, e.g. ``q**2`` from a chain line ``q**2 == ...``. The *x* of
    ``deriv(f, x)`` is never substituted.

    Parameters
    ----------
    expr : AST
        Physika expression.
    chain : dict
        Definitions, by variable name or by AST subtree.

    Examples
    --------
    >>> from physika.solver.sym_utils import substitute_chain
    >>> variable = lambda n: ("var", n)
    >>> chain = {"t": ("num", 0), "v": ("mul", variable("a"), variable("t"))}
    >>> substitute_chain(("call", "deriv", [variable("v"), variable("t")]),
    ...                  chain)
    ('call', 'deriv', [('mul', ('var', 'a'), ('var', 't')), ('var', 't')])
    """
    return rewrite(expr, chain_rule, chain, 8)


def sym_expand_calls(expr: Any) -> Any:
    """
    Replaces every ``deriv(f, x)`` in *expr* with its derivative.
    ``sym_poly_of`` keeps a subexpression without the unknown as an opaque
    coefficient, so a ``deriv`` call in it must be expanded beforehand.

    Parameters
    ----------
    expr : AST
        Physika expression.

    Examples
    --------
    >>> from physika.solver.sym_utils import sym_expand_calls
    >>> variable = lambda n: ("var", n)
    >>> f = ("mul", variable("a"), variable("t"))
    >>> sym_expand_calls(("add", ("call", "deriv", [f, variable("t")]),
    ...                   variable("x")))
    ('add', ('var', 'a'), ('var', 'x'))
    """
    return rewrite(expr, expand_rule, None, 6)


def sym_inline_calls(expr: Any, functions: dict, depth: int = 4) -> Any:
    """
    Replaces every call to a function in *functions* with its body, the
    parameters and locals substituted, so a report line shows the
    arithmetic instead of ``f(...)``. At most *depth* nested calls are
    inlined, calls to other names are kept.

    Parameters
    ----------
    expr : AST
        Physika expression.
    functions : dict
        Function definitions, by name.
    depth : int, optional
        Maximum number of nested inlinings.

    Examples
    --------
    >>> from physika.solver.sym_utils import sym_inline_calls
    >>> variable = lambda n: ("var", n)
    >>> funcs = {"sq": {"params": [("y", "ℝ")], "statements": [],
    ...                 "body": ("mul", variable("y"), variable("y"))}}
    >>> sym_inline_calls(("call", "sq", [variable("a")]), funcs)
    ('mul', ('var', 'a'), ('var', 'a'))
    """
    if depth <= 0:
        return expr
    if (isinstance(expr, tuple) and expr and expr[0] == "call"
            and expr[1] in functions):
        fdef = functions[expr[1]]
        params = [p[0] for p in fdef.get("params", [])]
        args = [sym_inline_calls(a, functions, depth) for a in expr[2]]
        if len(args) == len(params):
            local: Dict[str, Any] = dict(zip(params, args))
            for stmt in fdef.get("statements", []):
                if (isinstance(stmt, tuple)
                        and stmt[0] in ("body_assign", "body_decl")):
                    local[stmt[1]] = substitute_defs(stmt[-1], local, 1)
            body = substitute_defs(fdef.get("body"), local, 1)
            return sym_inline_calls(body, functions, depth - 1)
    if isinstance(expr, (tuple, list)):
        parts = [sym_inline_calls(s, functions, depth) for s in expr]
        return tuple(parts) if isinstance(expr, tuple) else parts
    return expr


def sym_poly_of(expr: Any, unknown: str) -> Optional[SPoly]:
    """
    Returns `expr` as a polynomial in `unknown` whose coefficients may be
    symbolic, or ``None`` if it is not a polynomial in *unknown*. A
    subexpression without *unknown* (``cos(θ)``, a call, ...) is kept as
    an opaque coefficient.

    Parameters
    ----------
    expr : AST
        Physika expression.
    unknown : str
        Name of the unknown.

    Examples
    --------
    >>> from physika.solver.sym_utils import sym_poly_of
    >>> variable = lambda n: ("var", n)
    >>> cos = ("call", "cos", [variable("θ")])
    >>> sym_poly_of(("add", ("mul", cos, variable("x")), ("num", 1)), "x")
    {1: ('call', 'cos', [('var', 'θ')]), 0: Fraction(1, 1)}
    >>> sym_poly_of(("div", ("num", 1), variable("x")), "x") is None
    True
    """
    if not contains_var(expr, unknown):
        return {0: lit_or_expr(expr)}
    if not isinstance(expr, tuple):
        return None
    op = expr[0]
    if op == "var":
        return {1: Fraction(1)}
    if op == "neg":
        p = sym_poly_of(expr[1], unknown)
        return None if p is None else sp_mul(p, {0: Fraction(-1)})
    if op in ("add", "sub", "mul"):
        a, b = sym_poly_of(expr[1], unknown), sym_poly_of(expr[2], unknown)
        if a is None or b is None:
            return None
        return sp_mul(a, b) if op == "mul" else sp_add(a, b, op == "sub")
    if op == "div":
        # only division by an expression without the unknown
        a = sym_poly_of(expr[1], unknown)
        if a is None or contains_var(expr[2], unknown):
            return None
        return {d: sdiv(c, lit_or_expr(expr[2])) for d, c in a.items()}
    if op == "pow":
        base = sym_poly_of(expr[1], unknown)
        if base is None or not (isinstance(expr[2], tuple)
                                and expr[2][0] == "num"):
            return None
        out: SPoly = {0: Fraction(1)}
        for _ in range(int(expr[2][1])):
            out = sp_mul(out, base)
        return out
    # deriv(f, x) is differentiated with sym_deriv, other calls are not
    # polynomials in the unknown
    d = deriv_args(expr)
    val = None if d is None else sym_deriv(*d)
    return None if val is None else sym_poly_of(val, unknown)


def int_exp(node: Any) -> Optional[int]:
    """
    Returns the integer exponent of a ``pow``, a literal or a negated
    literal (``x**-3`` parses as ``("neg", ("num", 3))``), else ``None``.

    Parameters
    ----------
    node : AST
        Exponent of a ``pow`` node.

    Examples
    --------
    >>> from physika.solver.sym_utils import int_exp
    >>> int_exp(("neg", ("num", 3))), int_exp(("var", "k"))
    (-3, None)
    """
    if isinstance(node, tuple) and node[0] == "num":
        return int(node[1])
    if (isinstance(node, tuple) and node[0] == "neg"
            and isinstance(node[1], tuple) and node[1][0] == "num"):
        return -int(node[1][1])
    return None


# Derivative of a one-argument function at its argument u.
SYM_DERIV_TABLE: Dict[str, Any] = {
    "sin": lambda u: ("call", "cos", [u]),
    "cos": lambda u: ("neg", ("call", "sin", [u])),
    "sqrt": lambda u: sdiv(Fraction(1), smul(Fraction(2),
                                             ("call", "sqrt", [u]))),
    "exp": lambda u: ("call", "exp", [u]),
    "log": lambda u: sdiv(Fraction(1), lit_or_expr(u)),
}


def sym_deriv(expr: Any, wrt: str) -> Optional[Any]:
    """
    Returns the derivative of *expr* in *wrt*, with the sum, product,
    quotient and power rules, and the chain rule through ``sin``, ``cos``,
    ``sqrt``, ``exp`` and ``log``. Other variables are constants. Returns
    ``None`` for any other call or a non-integer exponent.

    Parameters
    ----------
    expr : AST
        Physika expression.
    wrt : str
        Name of the variable to differentiate in.

    Examples
    --------
    >>> from physika.solver.sym_utils import sym_deriv
    >>> variable = lambda n: ("var", n)
    >>> sym_deriv(("call", "sin", [("mul", ("num", 2), variable("t"))]), "t")
    ('mul', ('call', 'cos', [('mul', ('num', 2), ('var', 't'))]), ('num', 2))
    """
    if not isinstance(expr, tuple):
        return Fraction(0)
    op = expr[0]
    if op in ("num", "imaginary", "complex"):
        return Fraction(0)
    if op == "var":
        return Fraction(1) if expr[1] == wrt else Fraction(0)
    if op == "neg":
        d = sym_deriv(expr[1], wrt)
        return None if d is None else sneg(d)
    if op in ("add", "sub", "mul", "div"):
        du, dv = sym_deriv(expr[1], wrt), sym_deriv(expr[2], wrt)
        if du is None or dv is None:
            return None
        if op == "add":
            return sadd(du, dv)
        if op == "sub":
            return ssub(du, dv)
        u, v = lit_or_expr(expr[1]), lit_or_expr(expr[2])
        if op == "mul":
            return sadd(smul(du, v), smul(u, dv))
        return sdiv(ssub(smul(du, v), smul(u, dv)), smul(v, v))
    if op == "pow":
        k = int_exp(expr[2])
        dbase = None if k is None else sym_deriv(expr[1], wrt)
        if k is None or dbase is None:
            return None
        if k in (0, 1):
            return dbase if k == 1 else Fraction(0)
        base = lit_or_expr(expr[1])
        lower = base if k == 2 else ("pow", to_expr(base), ("num", k - 1))
        return smul(smul(Fraction(k), lower), dbase)
    if op == "call" and len(expr[2]) == 1 and expr[1] in SYM_DERIV_TABLE:
        dinner = sym_deriv(expr[2][0], wrt)
        if dinner is None:
            return None
        return smul(SYM_DERIV_TABLE[expr[1]](expr[2][0]), dinner)
    return None


def eval_numeric(expr: Any,
                 subs: Dict[str, Any],
                 functions: Optional[dict] = None) -> Any:
    """
    Evaluates *expr* to a ``float``, or a ``complex`` when it uses the
    imaginary unit. Raises ``ValueError`` when it cannot be evaluated.

    Parameters
    ----------
    expr : AST
        Physika expression.
    subs : dict
        Value of every free variable.
    functions : dict, optional
        Function definitions, by name.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.sym_utils import eval_numeric
    >>> variable = lambda n: ("var", n)
    >>> eval_numeric(("mul", variable("a"), ("call", "sqrt", [("num", 4)])),
    ...              {"a": Fraction(3, 2)})
    3.0
    """
    if not isinstance(expr, tuple):
        return float(expr)
    functions = functions or {}
    op = expr[0]
    if op == "num":
        return float(expr[1])
    if op == "imaginary":
        return complex(0, 1)
    if op == "complex":
        return complex(expr[1])
    if op == "var":
        # not cast to float, a substituted root may be complex
        return subs[expr[1]]
    if op == "neg":
        return -eval_numeric(expr[1], subs, functions)
    if op in ARITH:
        return ARITH[op](eval_numeric(expr[1], subs, functions),
                         eval_numeric(expr[2], subs, functions))
    if op != "call":
        raise ValueError(f"cannot numerically evaluate {expr!r}")
    d = deriv_args(expr)
    if d is not None:
        val = sym_deriv(*d)
        if val is None:
            raise ValueError(f"cannot evaluate deriv of {expr!r}")
        return eval_numeric(val, subs, functions)
    fname = expr[1]
    vals = [eval_numeric(a, subs, functions) for a in expr[2]]
    if fname in MATH_UNARY and len(vals) == 1:
        if isinstance(vals[0], complex) and fname in CMATH_UNARY:
            return CMATH_UNARY[fname](vals[0])
        return float(MATH_UNARY[fname](vals[0]))
    if fname == "atan2" and len(vals) == 2:
        return float(math.atan2(vals[0], vals[1]))
    if fname in functions:
        fdef = functions[fname]
        params = [p[0] for p in fdef.get("params", [])]
        if len(vals) == len(params):
            fsubs: Dict[str, Any] = dict(zip(params, vals))
            for stmt in fdef.get("statements", []):
                if (isinstance(stmt, tuple)
                        and stmt[0] in ("body_assign", "body_decl")):
                    fsubs[stmt[1]] = eval_numeric(stmt[-1], fsubs, functions)
            return eval_numeric(fdef.get("body"), fsubs, functions)
    raise ValueError(f"cannot numerically evaluate {expr!r}")


class NotInRingFragment(Exception):
    """
    Raised while translating an expression to a ``Sym`` term when it uses
    something outside the commutative ring fragment: a division, a
    non-integer literal, a call with a compound argument, or a symbolic or
    negative exponent. ``verify_ring_identity`` turns it into ``None``.
    """


def check_ring_int(v: Any) -> None:
    """
    Raises ``NotInRingFragment`` when *v* is a non-integer number.

    Parameters
    ----------
    v : Any
        Literal value.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.sym_utils import check_ring_int
    >>> check_ring_int(Fraction(4))
    >>> check_ring_int(2.5)
    Traceback (most recent call last):
    ...
    physika.solver.sym_utils.NotInRingFragment: non-integer literal 2.5
    """
    if ((isinstance(v, Fraction) and v.denominator != 1)
            or (isinstance(v, float) and not v.is_integer())):
        raise NotInRingFragment(f"non-integer literal {v}")


def call_key(expr: tuple) -> Tuple[str, tuple]:
    """
    Returns the ``Sym`` atom key of an opaque call such as ``cos(θ)``. The
    arguments must be variables or integer literals, so that ``cos(a + b)``
    and ``cos(b + a)`` never become two different atoms.

    Parameters
    ----------
    expr : tuple
        ``("call", fname, args)`` node.

    Examples
    --------
    >>> from physika.solver.sym_utils import call_key
    >>> call_key(("call", "cos", [("var", "θ")]))
    ('cos', (('var', 'θ'),))
    """
    keys = []
    for arg in expr[2]:
        if isinstance(arg, tuple) and arg[0] == "var":
            keys.append(("var", arg[1]))
            continue
        if isinstance(arg, tuple) and arg[0] != "num":
            raise NotInRingFragment(f"call argument {arg!r} is not a bare "
                                    f"variable/literal")
        v = arg[1] if isinstance(arg, tuple) else arg
        check_ring_int(v)
        keys.append(("num", int(v)))
    return expr[1], tuple(keys)


def ring_degree(expr: Any, atoms: Dict[Any, int]) -> int:
    """
    Returns the total degree of *expr* as a polynomial in its variables
    and opaque calls, and adds each of them to *atoms* (key -> index).
    The degree sizes the ``Sym`` basis, too small a basis would truncate
    terms. Raises ``NotInRingFragment`` outside the ring fragment.

    An opaque call may satisfy relations ``Sym`` does not know, such as
    ``cos(θ)² + sin(θ)² = 1``, see ``verify_ring_identity``.

    Parameters
    ----------
    expr : AST
        Physika expression.
    atoms : dict
        Atom indices, by variable name or ``call_key``, filled in place.

    Examples
    --------
    >>> from physika.solver.sym_utils import ring_degree
    >>> variable = lambda n: ("var", n)
    >>> atoms = {}
    >>> ring_degree(("mul", ("pow", variable("x"), ("num", 2)),
    ...              ("call", "cos", [variable("θ")])), atoms)
    3
    >>> atoms
    {'x': 0, ('cos', (('var', 'θ'),)): 1}
    """
    if not isinstance(expr, tuple) or expr[0] == "num":
        check_ring_int(expr[1] if isinstance(expr, tuple) else expr)
        return 0
    op = expr[0]
    if op in ("var", "call"):
        key = expr[1] if op == "var" else call_key(expr)
        atoms.setdefault(key, len(atoms))
        return 1
    if op == "neg":
        return ring_degree(expr[1], atoms)
    if op in ("add", "sub"):
        return max(ring_degree(expr[1], atoms), ring_degree(expr[2], atoms))
    if op == "mul":
        return ring_degree(expr[1], atoms) + ring_degree(expr[2], atoms)
    if op == "pow":
        base_deg = ring_degree(expr[1], atoms)
        k = int_exp(expr[2])
        if k is None or k < 0:
            raise NotInRingFragment("non-constant or negative exponent")
        return base_deg * k
    raise NotInRingFragment(f"operator {op!r} not in ring fragment")


def sym_app(name: str, *args: Expr) -> Expr:
    """
    Returns ``Sym.<name>`` applied to *args*.

    Parameters
    ----------
    name : str
        ``Sym`` operation, e.g. ``"add"``.
    *args : Expr
        Arguments.

    Examples
    --------
    >>> from physika.core.expr import Lit
    >>> from physika.solver.sym_utils import sym_app
    >>> sym_app("atom", Lit(0))
    App(func=Const(name='Sym.atom', levels=()), arg=Lit(val=0))
    """
    term: Expr = Const(f"Sym.{name}", ())
    for a in args:
        term = App(term, a)
    return term


def ast_to_sym(expr: Any, atoms: Dict[Any, int]) -> Expr:
    """
    Returns *expr* as a ``Sym`` CIC term. *expr* must be in the ring
    fragment and its atoms already in *atoms* (from ``ring_degree``).

    Parameters
    ----------
    expr : AST
        Physika expression.
    atoms : dict
        Atom indices, by variable name or ``call_key``.

    Examples
    --------
    >>> from physika.solver.sym_utils import ast_to_sym
    >>> term = ast_to_sym(("neg", ("var", "x")), {"x": 0})
    >>> term.func
    Const(name='Sym.neg', levels=())
    """
    if not isinstance(expr, tuple) or expr[0] == "num":
        return sym_app("lit", int_lit(int(lit_or_expr(expr))))
    op = expr[0]
    if op in ("var", "call"):
        key = expr[1] if op == "var" else call_key(expr)
        return sym_app("atom", Lit(atoms[key]))
    if op == "neg":
        return sym_app("neg", ast_to_sym(expr[1], atoms))
    if op == "pow":
        k = int_exp(expr[2])
        if not k:
            return sym_app("lit", int_lit(1))
        base = ast_to_sym(expr[1], atoms)
        term = base
        for _ in range(k - 1):
            term = sym_app("mul", term, base)
        return term
    if op not in ("add", "sub", "mul"):
        raise NotInRingFragment(f"operator {op!r} not in ring fragment")
    a, b = ast_to_sym(expr[1], atoms), ast_to_sym(expr[2], atoms)
    if op == "sub":
        return sym_app("add", a, sym_app("neg", b))
    return sym_app(op, a, b)


def verify_ring_identity(lhs: Any, rhs: Any) -> Optional[bool]:
    """
    CIC kernel checks ``lhs == rhs`` with ``Sym`` ring.
    ``Sym.eqb (Sym.normalize lhs) (Sym.normalize rhs)`` reduces to
    ``Bool.true`` or ``Bool.false``.
    Variables and opaque calls with simple arguments (``cos(θ)``)
    are atoms. Returns ``None`` outside the ring fragment,
    so the caller uses ``verify_identity`` instead.

    ``Bool.false`` is not trusted when there is an opaque call, since the
    identity may hold because of a relation ``Sym`` is not supported yets,
    returns None.

    Parameters
    ----------
    lhs : AST
        Left hand side.
    rhs : AST
        Right hand side.

    Examples
    --------
    >>> from physika.solver.sym_utils import verify_ring_identity
    >>> variable = lambda n: ("var", n)
    >>> a, b = variable("a"), variable("b")
    >>> verify_ring_identity(("add", a, b), ("add", b, a))
    True
    >>> verify_ring_identity(("add", a, b), a)
    False
    >>> verify_ring_identity(("div", a, b), a) is None
    True
    """
    try:
        atoms: Dict[Any, int] = {}
        max_degree = max(ring_degree(lhs, atoms), ring_degree(rhs, atoms))
        num_atoms = max(len(atoms), 1)
        if len(monomial_basis(num_atoms, max_degree)) > MAX_SYM_BASIS:
            return None
        lhs_term, rhs_term = ast_to_sym(lhs, atoms), ast_to_sym(rhs, atoms)
        env = mk_builtin_env()
        reg_sym_ops(env, num_atoms, max_degree)
        lctx, mctx = LocalContext(), MetaVarContext()
        nlhs = whnf(sym_app("normalize", lhs_term), env, lctx, mctx)
        nrhs = whnf(sym_app("normalize", rhs_term), env, lctx, mctx)
        result = whnf(sym_app("eqb", nlhs, nrhs), env, lctx, mctx)
    except (NotInRingFragment, KeyError, ValueError, OverflowError,
            RecursionError):
        return None
    if not isinstance(result, Const):
        return None
    if result.name == "Bool.true":
        return True
    opaque = any(not isinstance(k, str) for k in atoms)
    return False if result.name == "Bool.false" and not opaque else None


def verify_identity(lhs: Any,
                    rhs: Any,
                    free: List[str],
                    functions: Optional[dict] = None,
                    unknown: Optional[str] = None,
                    root: Any = None,
                    seed: int = 7654321) -> Optional[bool]:
    """
    Checks ``lhs == rhs`` numerically at 3 random rational values of the
    names in *free*. Given *unknown* and *root*, the root is evaluated and
    substituted for the unknown first. Not a kernel proof. Returns
    ``None`` when every attempt hits a domain error (``sqrt`` of a
    negative number, ...).

    Parameters
    ----------
    lhs : AST
        Left hand side.
    rhs : AST
        Right hand side.
    free : list[str]
        Names that get random values.
    functions : dict, optional
        Function definitions, by name.
    unknown : str, optional
        Name of the unknown that *root* is substituted for.
    root : AST, optional
        Claimed value of *unknown*.
    seed : int, optional
        Seed of the random values.

    Examples
    --------
    >>> from physika.solver.sym_utils import verify_identity
    >>> variable = lambda n: ("var", n)
    >>> lhs = ("pow", ("call", "sin", [variable("t")]), ("num", 2))
    >>> rhs = ("sub", ("num", 1),
    ...        ("pow", ("call", "cos", [variable("t")]), ("num", 2)))
    >>> verify_identity(lhs, rhs, ["t"])
    True
    """
    rng = random.Random(seed)
    checked = 0
    for _ in range(12):
        subs: Dict[str, Any] = {
            name: Fraction(rng.randint(2, 97), rng.randint(1, 7))
            for name in free
        }
        try:
            if unknown is not None:
                subs[unknown] = eval_numeric(root, subs, functions)
            lv = eval_numeric(lhs, subs, functions)
            rv = eval_numeric(rhs, subs, functions)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if abs(lv - rv) > 1e-6 * max(1.0, abs(lv), abs(rv)):
            return False
        checked += 1
        if checked == 3:
            return True
    return True if checked else None
