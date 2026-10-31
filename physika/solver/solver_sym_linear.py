from fractions import Fraction
from typing import Any, Dict, List, Optional

from physika.units import unit_to_str
from physika.solver.utils import expr_str, free_vars, to_expr
from physika.solver.solver_chain import chain_side
from physika.solver.sym_utils import (contains_var, is_num, lit_or_expr, sdiv,
                                      smul, sneg, ssub, substitute_chain,
                                      sym_expand_calls, sym_inline_calls,
                                      sym_poly_of, verify_identity,
                                      verify_ring_identity)


def is_zero(c: Any) -> bool:
    """
    Checks if a coefficient is the number 0.

    Parameters
    ----------
    c : Fraction, int, float or AST
        Coefficient.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_sym_linear import is_zero
    >>> is_zero(Fraction(0)), is_zero(("var", "k"))
    (True, False)
    """
    return is_num(c) and c == 0


def sym_linear_rows(conds: List[tuple], unknowns: List[str], defs: Dict[str,
                                                                        Any],
                    functions: dict) -> Optional[List[List[Any]]]:
    """
    Returns the augmented matrix ``[c_1, ..., c_n, -c_0]`` of each
    equation ``lhs - rhs == c_1 u_1 + ... + c_n u_n + c_0``, where the
    coefficients may be symbolic. ``None`` when an equation is not linear
    in the unknowns or has a product of two unknowns.

    Parameters
    ----------
    conds : list[AST]
        ``("cond_eq", lhs, rhs)`` equations.
    unknowns : list[str]
        Names of the unknowns.
    defs : dict
        Right hand side of every declaration, by name.
    functions : dict
        Function definitions, by name.

    Examples
    --------
    >>> from physika.solver.solver_sym_linear import sym_linear_rows
    >>> variable = lambda n: ("var", n)
    >>> eq = ("cond_eq", ("mul", variable("k"), variable("x")), variable("F"))
    >>> sym_linear_rows([eq], ["x"], {}, {})
    [[('var', 'k'), ('var', 'F')]]
    """
    zero_all = {u: ("num", 0) for u in unknowns}
    rows = []
    for _, lhs_raw, rhs_raw in conds:
        expr = ("sub", chain_side(lhs_raw, defs, functions),
                chain_side(rhs_raw, defs, functions))
        row: List[Any] = []
        for u in unknowns:
            poly = sym_poly_of(expr, u)
            if poly is None or max(poly, default=0) > 1:
                return None
            c = poly.get(1, Fraction(0))
            if any(contains_var(to_expr(c), w) for w in unknowns):
                return None
            row.append(c)
        # constant term: the degree 0 part in each unknown in turn
        const: Any = expr
        for u in unknowns:
            poly = sym_poly_of(to_expr(const), u)
            if poly is None:
                const = substitute_chain(expr, zero_all)
                break
            const = poly.get(0, Fraction(0))
        rows.append(row + [sneg(lit_or_expr(const))])
    return rows


def sym_gauss_jordan(matrix: List[List[Any]]) -> Optional[List[Any]]:
    """
    Gauss–Jordan elimination on an ``n x (n + 1)`` augmented matrix with
    symbolic entries. Returns the solution, or ``None`` when no pivot is
    a non-zero entry. *matrix* is changed in place.

    Parameters
    ----------
    matrix : list[list]
        Rows ``[c_1, ..., c_n, b]`` of ``c_1 u_1 + ... + c_n u_n == b``.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from physika.solver.solver_sym_linear import sym_gauss_jordan
    >>> # 2 x == s, y == t
    >>> sym_gauss_jordan([[Fraction(2), Fraction(0), ("var", "s")],
    ...                   [Fraction(0), Fraction(1), ("var", "t")]])
    [('div', ('var', 's'), ('num', 2)), ('var', 't')]
    """
    n = len(matrix)
    for col in range(n):
        piv = next((r for r in range(col, n) if not is_zero(matrix[r][col])),
                   None)
        if piv is None:
            return None
        matrix[col], matrix[piv] = matrix[piv], matrix[col]
        p = matrix[col][col]
        matrix[col] = [sdiv(x, p) for x in matrix[col]]
        for r in range(n):
            if r != col and not is_zero(matrix[r][col]):
                f = matrix[r][col]
                matrix[r] = [
                    ssub(x, smul(f, y))
                    for x, y in zip(matrix[r], matrix[col])
                ]
    return [row[n] for row in matrix]


def solve_sym_linear(
        conds: List[tuple],
        variables: List[str],
        defs: Dict[str, Any],
        funcs: dict,
        tag: str,
        tname: str,
        unit: dict,
        bindings: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
    """
    Solves ``n`` equations linear in ``n`` ℝ unknowns whose
    coefficients are symbolic, irrational or complex.

    ``v0 == a*i*sqrt(k/m) - b*i*sqrt(k/m)``, which linear solver path verify.

    The solution is substituted into every equation and
    checked with the ``Sym`` ring when possible, else at random instances.
    Returns the report lines, or ``None`` when the block is not such a
    system.

    A verified solution is added to `bindings`.

    Parameters
    ----------
    conds : list[AST]
        ``("cond_eq", lhs, rhs)`` equations.
    variables : list[str]
        Names of the unknowns.
    defs : dict
        Right hand side of every declaration, by name (``raw_defs``).
    funcs : dict
        Function definitions, by name.
    tag : str
        Prefix for every report line.
    tname : str
        Type symbol shown in the report line.
    unit : dict
        Declared unit of the unknowns.
    bindings : dict, optional
        Solved unknowns, filled in place.

    Examples
    --------
    >>> from physika.solver.solver_sym_linear import solve_sym_linear
    >>> variable = lambda n: ("var", n)
    >>> x, y, s = variable("x"), variable("y"), variable("s")
    >>> conds = [("cond_eq", ("mul", ("num", 2), x), s),
    ...          ("cond_eq", y, ("mul", variable("k"), x))]
    >>> for line in solve_sym_linear(conds, ["x", "y"], {}, {},
    ...                              "Line 1: solve x, y", "ℝ", {}):
    ...     print(line)
    Line 1: solve x, y: linear system in x, y, symbolic coefficients (Gauss–Jordan)
    Line 1: solve x, y: symbolic in k, s
    Line 1: solve x, y : ℝ  →  x = (s / 2), y = (k * (s / 2))   (✓ verified at random instances, not a kernel proof)
    """  # noqa: E501
    if len(conds) != len(variables):
        return None
    rows = sym_linear_rows(conds, list(variables), defs, funcs)
    sol = None if rows is None else sym_gauss_jordan(rows)
    if sol is None:
        return None
    sol_ast = {u: to_expr(v) for u, v in zip(variables, sol)}

    # substitute the solution into every equation and check it
    verdicts: List[Optional[bool]] = []
    proven = True
    for _, lhs_raw, rhs_raw in conds:
        lhs, rhs = (sym_expand_calls(
            sym_inline_calls(
                substitute_chain(substitute_chain(raw, defs), sol_ast), funcs))
                    for raw in (lhs_raw, rhs_raw))
        if verify_ring_identity(lhs, rhs) is True:
            verdicts.append(True)
            continue
        proven = False
        free = sorted(free_vars(lhs, set()) | free_vars(rhs, set()))
        verdicts.append(verify_identity(lhs, rhs, free, funcs))
    if all(v is True for v in verdicts):
        mark = ("✓ kernel-verified: Sym ring" if proven else
                "✓ verified at random instances, not a kernel proof")
        if bindings is not None:
            bindings.update(sol_ast)
    elif any(v is False for v in verdicts):
        mark = "✗ FAILS at a random instance"
    else:
        mark = "unverified (no valid random instance)"

    free_syms = sorted(set().union(*(free_vars(v, set())
                                     for v in sol_ast.values())))
    body = ", ".join(f"{u} = {expr_str(sol_ast[u])}" for u in variables)
    unit_s = f"  [{unit_to_str(unit)}]" if unit else ""
    return [
        f"{tag}: linear system in {', '.join(variables)}, symbolic "
        f"coefficients (Gauss–Jordan)",
        f"{tag}: symbolic in {', '.join(free_syms) or '(none)'}",
        f"{tag} : {tname}  →  {body}{unit_s}   ({mark})"
    ]
