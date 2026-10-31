from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from physika.solver.utils import expr_str, fraction_isqrt, free_vars, to_expr
from physika.solver.sym_utils import (SPoly, contains_var, is_num, sadd, sdiv,
                                      smul, sneg, sp_add, ssub,
                                      substitute_chain, sym_expand_calls,
                                      sym_inline_calls, sym_poly_of,
                                      verify_identity, verify_ring_identity)

NOT_CONSEQUENCE = ("✗ NOT a pure consequence of the previous steps — "
                   "treated as new input")


def is_var(node: Any) -> bool:
    """
    Checks if *node* is a variable ``("var", name)``.

    Parameters
    ----------
    node : AST
        Physika expression node.

    Examples
    --------
    >>> from physika.solver.solver_chain import is_var
    >>> is_var(("var", "q")), is_var(("num", 1))
    (True, False)
    """
    return isinstance(node, tuple) and len(node) == 2 and node[0] == "var"


def ssqrt(a: Any) -> Any:
    """
    Returns the square root of a coefficient, exact for a rational square
    and a ``sqrt`` call otherwise.

    Parameters
    ----------
    a : Fraction, int, float or AST
        Coefficient.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_chain import ssqrt
    >>> ssqrt(Fraction(9, 4)), ssqrt(("var", "d"))
    (Fraction(3, 2), ('call', 'sqrt', [('var', 'd')]))
    """
    if is_num(a) and Fraction(a) >= 0:
        exact = fraction_isqrt(Fraction(a))
        if exact is not None:
            return exact
    return ("call", "sqrt", [to_expr(a)])


def fresh_pattern(side: Any, chain: Dict[Any, Any], var: str) -> Any:
    """
    Returns the key that a chain line ``side == value`` defines, when
    *side* is a new variable ``q`` (key ``"q"``) or its power ``q**k``
    (key would be ``pow`` node), else ``None``. The unknown *var* and
    names already in *chain* are never defined.

    Parameters
    ----------
    side : AST
        One side of the chain line.
    chain : dict
        Definitions so far.
    var : str
        Name of the unknown.

    Examples
    --------
    >>> from physika.solver.solver_chain import fresh_pattern
    >>> variable = lambda n: ("var", n)
    >>> fresh_pattern(variable("q"), {}, "x")
    'q'
    >>> fresh_pattern(("pow", variable("q"), ("num", 2)), {}, "x")
    ('pow', ('var', 'q'), ('num', 2))
    >>> fresh_pattern(variable("x"), {}, "x") is None
    True
    """
    if is_var(side):
        return side[1] if side[1] != var and side[1] not in chain else None
    if (isinstance(side, tuple) and len(side) == 3 and side[0] == "pow"
            and is_var(side[1]) and side[1][1] != var
            and side[1][1] not in chain and side not in chain):
        return side
    return None


def fresh_definition(lhs_raw: Any, rhs_raw: Any, lhs: Any, rhs: Any,
                     chain: Dict[Any,
                                 Any], var: str) -> Optional[Tuple[Any, Any]]:
    """
    Returns ``(key, value)`` when one side of a chain line is a new
    variable or power (``fresh_pattern``) and the other side does not use
    it, else ``None``. The left side is tried first.

    Parameters
    ----------
    lhs_raw : AST
        Left side as written.
    rhs_raw : AST
        Right side as written.
    lhs : AST
        Left side with the chain substituted.
    rhs : AST
        Right side with the chain substituted.
    chain : dict
        Definitions so far.
    var : str
        Name of the unknown.

    Examples
    --------
    >>> from physika.solver.solver_chain import fresh_definition
    >>> variable = lambda n: ("var", n)
    >>> value = ("add", variable("a"), variable("b"))
    >>> fresh_definition(variable("q"), value, variable("q"), value, {}, "x")
    ('q', ('add', ('var', 'a'), ('var', 'b')))
    >>> loop = ("add", variable("q"), ("num", 1))  # q == q + 1
    >>> fresh_definition(variable("q"), loop, variable("q"), loop, {},
    ...                  "x") is None
    True
    """
    for side_raw, value in ((lhs_raw, rhs), (rhs_raw, lhs)):
        key = fresh_pattern(side_raw, chain, var)
        if key is None:
            continue
        name = key if isinstance(key, str) else key[1][1]
        # occurs check: q == f(q) is a claim, not a definition
        if not contains_var(value, name):
            return key, value
    return None


def fresh_linear_pattern(lhs_raw: Any, rhs_raw: Any, lhs: Any, rhs: Any,
                         chain: Dict[Any, Any], var: str,
                         functions: dict) -> Optional[Tuple[str, Any]]:
    """
    Returns ``(name, value / coeff)`` for a chain line
    ``coeff * name == value`` (any order) that defines a new variable
    *name*. For exmample, ``(2*N + 1) * alpha == N*pi - pi/2`` defines
    ``alpha``.

    ``None`` otherwise. Only a top-level product is matched, and *coeff*
    must not be a new bare variable itself (in ``ħ * ω`` neither factor is
    the coefficient), must not use *name* or the unknown *var*, and
    *value* must not use *name*.

    Parameters
    ----------
    lhs_raw : AST
        Left side as written.
    rhs_raw : AST
        Right side as written.
    lhs : AST
        Left side with the chain substituted.
    rhs : AST
        Right side with the chain substituted.
    chain : dict
        Definitions so far.
    var : str
        Name of the unknown.
    functions : dict
        Function definitions, by name.

    Examples
    --------
    >>> from physika.solver.solver_chain import fresh_linear_pattern
    >>> variable = lambda n: ("var", n)
    >>> side = ("mul", ("num", 2), variable("alpha"))
    >>> fresh_linear_pattern(side, variable("pi"), side, variable("pi"), {},
    ...                      "x", {})
    ('alpha', ('div', ('var', 'pi'), ('num', 2)))
    """
    for side_raw, value in ((lhs_raw, rhs), (rhs_raw, lhs)):
        if not (isinstance(side_raw, tuple) and len(side_raw) == 3
                and side_raw[0] == "mul"):
            continue
        for factor, coeff_raw in ((side_raw[1], side_raw[2]), (side_raw[2],
                                                               side_raw[1])):
            if not is_var(factor):
                continue
            name = factor[1]
            if name == var or name in chain:
                continue
            coeff_free = free_vars(coeff_raw, set())
            if name in coeff_free or var in coeff_free:
                continue
            if is_var(coeff_raw) and coeff_raw[1] not in chain:
                continue
            if name in free_vars(value, set()):
                continue
            coeff = sym_inline_calls(substitute_chain(coeff_raw, chain),
                                     functions)
            return name, sdiv(value, coeff)
    return None


def sym_solve_poly(poly: SPoly) -> Optional[List[Any]]:
    """
    Returns the roots of ``poly == 0`` as Physika expressions, with the
    quadratic formula for degree 2. ``None`` when the degree, after
    dropping zero leading coefficients, is 0 or above 2.

    Parameters
    ----------
    poly : SPoly
        Polynomial in the unknown with symbolic coefficients.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_chain import sym_solve_poly
    >>> sym_solve_poly({1: ("var", "k"), 0: ("var", "F")})  # k x + F == 0
    [('div', ('neg', ('var', 'F')), ('var', 'k'))]
    >>> sym_solve_poly({2: Fraction(1), 0: Fraction(-4)})  # x² - 4 == 0
    [Fraction(2, 1), Fraction(-2, 1)]
    """
    poly = dict(poly)
    deg = max(poly, default=0)
    while deg > 0 and is_num(poly[deg]) and poly[deg] == 0:
        del poly[deg]
        deg = max(poly, default=0)
    if deg == 0 or deg > 2:
        return None
    a0 = poly.get(0, Fraction(0))
    if deg == 1:
        return [sdiv(sneg(a0), poly[1])]
    a2, a1 = poly[2], poly.get(1, Fraction(0))
    root = ssqrt(ssub(smul(a1, a1), smul(Fraction(4), smul(a2, a0))))
    two_a2 = smul(Fraction(2), a2)
    return [
        sdiv(sadd(sneg(a1), root), two_a2),
        sdiv(ssub(sneg(a1), root), two_a2)
    ]


def chain_side(side: Any, chain: Dict[Any, Any], functions: dict) -> Any:
    """
    Returns one side of a chain line with the definitions substituted,
    the function calls inlined and ``deriv`` calls expanded. The second
    substitution replaces the ``x`` of an expanded ``deriv(f, x)``, which
    is kept free while differentiating.

    Parameters
    ----------
    side : AST
        Side as written.
    chain : dict
        Definitions so far.
    functions : dict
        Function definitions, by name.

    Examples
    --------
    >>> from physika.solver.solver_chain import chain_side
    >>> variable = lambda n: ("var", n)
    >>> f = ("mul", variable("v"), variable("t"))
    >>> chain_side(("call", "deriv", [f, variable("t")]),
    ...            {"v": ("num", 3), "t": ("num", 0)}, {})
    ('num', 3)
    """
    inlined = sym_inline_calls(substitute_chain(side, chain), functions)
    return substitute_chain(sym_expand_calls(inlined), chain)


def solve_chain(
        conds: List[tuple],
        var: str,
        defs: Dict[str, Any],
        funcs: dict,
        tag: str,
        tname: str,
        bindings: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
    """
    Reads a ``solve`` block for one ℝ unknown as a derivation
    chain. A line before the last one either defines a new name
    (``fresh_definition``, ``fresh_linear_pattern``) or is a claim checked
    against the definitions so far, with the ``Sym`` ring when possible,
    else at random instances. A failed claim is reported and kept. The
    last line is solved for the unknown (degree ≤ 2) and each root is
    checked. Returns the report lines, or ``None`` when the last line is
    not such a polynomial. A single verified root is added to *bindings*.

    Parameters
    ----------
    conds : list[AST]
        ``("cond_eq", lhs, rhs)`` chain lines.
    var : str
        Name of the unknown.
    defs : dict
        Right hand side of every declaration, by name (``raw_defs``).
    funcs : dict
        Function definitions, by name.
    tag : str
        Prefix for every report line.
    tname : str
        Type symbol shown in the report line.
    bindings : dict, optional
        Solved unknowns, filled in place.

    Examples
    --------
    >>> from physika.solver.solver_chain import solve_chain
    >>> variable = lambda n: ("var", n)
    >>> a, b, q = variable("a"), variable("b"), variable("q")
    >>> conds = [("cond_eq", q, ("add", a, b)),
    ...          ("cond_eq", ("mul", variable("y"), q), a)]
    >>> for line in solve_chain(conds, "y", {}, {}, "Line 1: solve y", "ℝ"):
    ...     print(line)
    Line 1: solve y: q := (a + b)   (definition)
    Line 1: solve y: symbolic in a, b
    Line 1: solve y : ℝ  →  y = (a / (a + b))   (✓ verified at random instances, not a kernel proof)
    """  # noqa: E501
    chain: Dict[Any, Any] = dict(defs)
    out: List[str] = []
    for i, (_, lhs_raw, rhs_raw) in enumerate(conds[:-1]):
        lhs = chain_side(lhs_raw, chain, funcs)
        rhs = chain_side(rhs_raw, chain, funcs)
        found = fresh_definition(lhs_raw, rhs_raw, lhs, rhs, chain, var)
        if found is not None:
            key, value = found
            chain[key] = value
            label = expr_str(key) if isinstance(key, tuple) else key
            out.append(f"{tag}: {label} := {expr_str(value)}   (definition)")
            continue
        lin_def = fresh_linear_pattern(lhs_raw, rhs_raw, lhs, rhs, chain, var,
                                       funcs)
        if lin_def is not None:
            name, value = lin_def
            chain[name] = value
            out.append(f"{tag}: {name} := {expr_str(value)}   (definition, "
                       f"solved from {expr_str(lhs)} == {expr_str(rhs)})")
            continue
        ok = verify_ring_identity(lhs, rhs)
        if ok is not None:
            verdict = ("✓ verified (algebraic consequence)" if ok else
                       NOT_CONSEQUENCE) + " [kernel-verified: Sym ring]"
        else:
            free = sorted(free_vars(lhs, set()) | free_vars(rhs, set()))
            ok = verify_identity(lhs, rhs, free, funcs)
            verdict = ("✓ verified (algebraic consequence)"
                       if ok else "unverified (no valid random instance)"
                       if ok is None else NOT_CONSEQUENCE)
        out.append(f"{tag}: step {i + 1}: {expr_str(lhs)} == "
                   f"{expr_str(rhs)}   ({verdict})")

    if not conds:
        return None
    lhs = chain_side(conds[-1][1], chain, funcs)
    rhs = chain_side(conds[-1][2], chain, funcs)
    spl, spr = sym_poly_of(lhs, var), sym_poly_of(rhs, var)
    roots = None if spl is None or spr is None else sym_solve_poly(
        sp_add(spl, spr, sub=True))
    if roots is None:
        return None
    free = sorted(free_vars(lhs, {var}) | free_vars(rhs, {var}))
    out.append(f"{tag}: symbolic in {', '.join(free) or '(none)'}")
    verified = []
    for root in roots:
        # kernel proof when the equation with the root substituted stays
        # in the ring fragment
        # if not checked at random instances
        if verify_ring_identity(substitute_chain(lhs, {var: root}),
                                substitute_chain(rhs, {var: root})):
            ok = True
            mark = "✓ kernel-verified: Sym ring"
        else:
            ok = verify_identity(lhs,
                                 rhs,
                                 free,
                                 funcs,
                                 unknown=var,
                                 root=root,
                                 seed=1234567)
            mark = ("✓ verified at random instances"
                    if ok else "✗ FAILS at a random instance" if ok is False
                    else "unverified (no valid random instance)")
            mark += ", not a kernel proof"
        out.append(f"{tag} : {tname}  →  {var} = {expr_str(root)}   "
                   f"({mark})")
        if ok:
            verified.append(root)
    if bindings is not None and len(verified) == 1:
        bindings[var] = verified[0]
    return out
