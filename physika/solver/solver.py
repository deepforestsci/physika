from typing import Any, Dict, List, Optional

from physika.core.environment import Environment
from physika.solver.solver_candidate import (solve_candidate,
                                             unsupported_reason)
from physika.solver.solver_dims import check_dimensions
from physika.solver.utils import (KIND_SYMBOL, free_vars, kind_and_unit)
from physika.units import sig_unit_spec


def solve_block(node: tuple,
                env: Environment,
                unified_ast: dict,
                bindings: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Solver function used for every case of ``solve_decl`` AST tag by
    called from ``solve_equations`` dispatcher and again by
    ``solve_bindings``. First, checks the unkown type and after `where`
    keyword local constans and equations are unpacked. Dimensional analysis
    is performed and exits if there is a mismatch. Then, first path is checked
    for verify a candidate in an equation.


    Parameters
    ----------
    node : AST
        ``("solve_decl", variables, type_spec, candidates, conds, lineno)``.
    env : Environment
        Environment used for every kernel check.
    unified_ast : dict
        Dictionary oftop-level constants, functions and classes.
    bindings : dict, optional
        Dictionary that contains the names of unknowns and solved values passed
        to codegen for runtime binding.

    Examples
    --------
    >>> from physika.solver.solver import solve_block
    >>> from physika.core.inductive import mk_builtin_env  # noqa: E501
    >>> variable = lambda n: ("var", n)
    >>> eq = ("cond_eq", ("mul", variable("x"), variable("x")), ("add", variable("x"), ("num", 12)))
    >>> node = ("solve_decl", ["x"], "ℕ", [("num", 4)], [eq], 1)
    >>> bindings = {}
    >>> solve_block(node, mk_builtin_env(), {"program": []}, bindings)
    ['Line 1: solve x : ℕ = 4  ✓ verified']
    >>> bindings # empty is expeced for path 1 (candidtae verifier)
    {}
    """
    _, variables, type_spec, candidates, conds, lineno = node
    tag = f"Line {lineno}: solve {', '.join(variables)}"
    kind, unit = kind_and_unit(type_spec)
    is_int = kind == "int"

    block_locals: List[Any] = []
    balance_unit: Optional[dict] = None
    if (isinstance(conds, tuple) and conds and conds[0] == "solve_block"):
        block_locals = list(conds[1])
        if len(conds) > 3 and conds[3] is not None:
            balance_unit = sig_unit_spec(conds[3])
        conds = list(conds[2]) if isinstance(conds[2], list) else [conds[2]]

    if kind is None:
        return [
            f"{tag}: only ℕ / ℤ / ℝ unknowns are supported "
            f"(got {type_spec!r})"
        ]
    tname = KIND_SYMBOL[kind]
    if not conds or any(not (isinstance(c, tuple) and c[0] == "cond_eq")
                        for c in conds):
        return [f"{tag}: every constraint must be an equation `lhs == rhs`"]

    lines, dims_ok = check_dimensions(variables, unit, conds, block_locals,
                                      balance_unit, unified_ast, env, tag)
    if not dims_ok:
        return lines

    # path 1 - candidate verification
    if candidates is not None:
        return lines + solve_candidate(variables, kind, candidates, conds, env,
                                       tag, tname)

    reason = unsupported_reason(conds, variables[0], is_int)
    if reason is not None:
        return lines + [f"{tag}: {reason}"]
    return lines + [
        f"{tag} : {tname}  ✗ the equation is neither linear nor "
        f"a polynomial in {', '.join(variables)}"
    ]


def solve_equations(unified_ast: dict, cic_env: Environment) -> List[str]:
    """
    Run ``solve`` statements in *unified_ast* through the CIC kernel aned
    return the results. This functions is called after CIC dimensional analysis
    checker.

    Parameters
    ----------
    unified_ast : dict
        Unified AST from ``build_unified_ast``.
    cic_env : Environment
        CIC environment with ``Nat.eqb`` / ``Int.eqb``
        bodies from ``mk_builtin_env``.

    Examples
    --------
    >>> from physika.solver import solve_equations
    >>> from physika.core.inductive import mk_builtin_env
    >>> variable = lambda n: ("var", n)
    >>> ast = {"program": [
    ...     ("solve_decl", ["x"], "ℕ", [("num", 4)],
    ...      [("cond_eq", ("mul", variable("x"), variable("x")),
    ...        ("add", variable("x"), ("num", 12)))], 1),
    ... ]}
    >>> for line in solve_equations(ast, mk_builtin_env()):
    ...     print(line)
    Line 1: solve x : ℕ = 4  ✓ verified
    """
    out: List[str] = []
    for s in unified_ast.get("program", []):
        if isinstance(s, tuple) and s and s[0] == "solve_decl":
            out += solve_block(s, cic_env, unified_ast)
    return out


def solve_bindings(unified_ast: dict, cic_env: Environment) -> Dict[str, Any]:
    """
    Runs after `solve_equations` from main execution path. Retuens a dictionary
    of solved unknowns and is used in codegen for getting runtime bindings.

    Parameters
    ----------
    unified_ast : dict
        Unified AST from ``build_unified_ast``.
    cic_env : Environment
        Environment used for ``solve_equations``.

    Examples
    --------
    >>> from physika.solver import solve_bindings  # noqa: E501
    >>> from physika.core.inductive import mk_builtin_env
    >>> variable = lambda n: ("var", n)
    >>> ast = {"program": [
    ...     ("solve_decl", ["x"], "ℕ", [("num", 4)],
    ...      [("cond_eq", ("mul", variable("x"), variable("x")),
    ...        ("add", variable("x"), ("num", 12)))], 1),
    ... ]}
    >>> solve_bindings(ast, mk_builtin_env()) # expected empty for cn=andidate verifier
    {}
    """
    bindings: Dict[str, Any] = {}
    for s in unified_ast.get("program", []):
        if isinstance(s, tuple) and s and s[0] == "solve_decl":
            solve_block(s, cic_env, unified_ast, bindings=bindings)

    top_level = {
        s[1]
        for s in unified_ast.get("program", [])
        if isinstance(s, tuple) and s and s[0] in ("decl", "unit_decl")
    }
    changed = True
    while changed:
        changed = False
        resolvable = set(bindings) | top_level
        for name, root in list(bindings.items()):
            if not free_vars(root, set()) <= resolvable:
                del bindings[name]
                changed = True
    return bindings
