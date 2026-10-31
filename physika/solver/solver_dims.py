from typing import Any, Dict, List, Optional, Tuple

from physika.core.environment import Environment
from physika.core.expr import App, Const, Expr
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.reduction import is_def_eq
from physika.units import (UnitError, check_func_dims, expr_unit_to_dimvec_cic,
                           local_annotation, sig_unit_spec, unit_to_dimvec_cic,
                           unit_to_str, update_unit_cic_env)
from physika.utils.cic_utils.inductive_utils import (SI_BASE_UNITS,
                                                     read_dim_vec_literal)
from physika.solver.utils import expr_str


def is_number(expr: Any) -> bool:
    """
    Checks if an expression is a number with no variables.

    Parameters
    ----------
    expr : ASTNode
        Physika expression for solve block containing units.

    Examples
    --------
    >>> from physika.solver.solver_dims import is_number
    >>> is_number(("mul", ("num", 2), ("num", 3)))
    True
    >>> is_number(("mul", ("num", 2), ("var", "x")))
    False
    """
    if not isinstance(expr, tuple):
        return False
    if expr[0] == "num":
        return True
    if expr[0] == "neg":
        return is_number(expr[1])
    if expr[0] in ("add", "sub", "mul", "div", "pow"):
        return is_number(expr[1]) and is_number(expr[2])
    return False


def add_terms(expr: Any) -> List[Any]:
    """
    Splits a sum or difference into its terms. Used for checking each
    term against the expected return unit.

    Parameters
    ----------
    expr : ASTNode
        Physika expression for solve block containing units.

    Examples
    --------
    >>> from physika.solver.solver_dims import add_terms
    >>> variable = lambda n: ("var", n)
    >>> add_terms(("sub", ("add", variable("a"), variable("b")),
    ...             variable("c")))
    [('var', 'a'), ('var', 'b'), ('var', 'c')]
    """
    if isinstance(expr, tuple) and expr and expr[0] in ("add", "sub"):
        return add_terms(expr[1]) + add_terms(expr[2])
    if isinstance(expr, tuple) and expr and expr[0] == "neg":
        return add_terms(expr[1])
    return [expr]


def same_dim(d1: Expr, d2: Expr, env: Environment) -> bool:
    """
    CIC kernel checks if two ``DimVec`` terms are the same
    dimension.

    Parameters
    ----------
    d1 : Expr
        ``DimVec`` term.
    d2 : Expr
        ``DimVec`` term to compare against.
    env : Environment
        Environment used during kernel check.

    Examples
    --------
    >>> from physika.solver.solver_dims import same_dim
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.units import unit_to_dimvec_cic
    >>> env = mk_builtin_env()
    >>> newton = unit_to_dimvec_cic({"kg": 1, "m": 1, "s": -2}, env)
    >>> same_dim(newton, unit_to_dimvec_cic({"kg": 1}, env), env)
    False
    """
    ok, _ = is_def_eq(App(Const("Quantity", ()), d1),
                      App(Const("Quantity", ()), d2),
                      env,
                      LocalContext(),
                      MetaVarContext(),
                      allow_assign=False)
    return ok


def dim_str(d: Expr, env: Environment) -> str:
    """
    Returns a ``DimVec`` term as a string for error messages and ``?`` for a
    non-SI unit.

    Parameters
    ----------
    d : Expr
        ``DimVec`` term.
    env : Environment
        Environment used for reducing *d*.

    Examples
    --------
    >>> from physika.solver.solver_dims import dim_str
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.units import unit_to_dimvec_cic
    >>> env = mk_builtin_env()
    >>> dim_str(unit_to_dimvec_cic({"kg": 1, "m": 1, "s": -2}, env), env)
    'kg·m·s⁻²'
    """
    exps = read_dim_vec_literal(d, env, LocalContext(), MetaVarContext())
    if exps is None:
        return "?"
    return unit_to_str({b: e for b, e in zip(SI_BASE_UNITS, exps) if e})


def dim_env(variables: List[str], unit: dict, block_locals: List[Any],
            unified_ast: dict,
            env: Environment) -> Tuple[Dict[str, Expr], Dict[str, Expr]]:
    """
    Returns two dictionaries of ``DimVec`` terms. One for variable names
    (top-level constants, block locals and unknowns) and one for function
    return dimensions.

    Parameters
    ----------
    variables : list[str]
        Name of unknowns.
    unit : dict
        Declared unit of unknowns.
    block_locals : list
        Local constants declared in the ``where:`` block.
    unified_ast : dict
        Dictionary of top-level constants, functions and classes.
    env : Environment
        Environment used for building ``DimVec`` terms.

    Examples
    --------
    >>> from physika.solver.solver_dims import dim_env, dim_str
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()
    >>> ast = {"program": [
    ...     ("unit_decl", "m", "ℝ", [("kg", 1)], ("num", 2.0), 1)]}
    >>> dims, func_dims = dim_env(["t"], {"s": 1}, [], ast, env)
    >>> dim_str(dims["m"], env), dim_str(dims["t"], env)
    ('kg', 's')
    """
    func_dims: Dict[str, Expr] = {}
    for fname, fdef in unified_ast.get("functions", {}).items():
        check_func_dims(fname, fdef, env, lambda _: None, func_dims)
    denv: Dict[str, Expr] = {}
    for stmt in unified_ast.get("program", []):
        update_unit_cic_env(stmt, denv, env)
    for stmt in block_locals:
        ts = stmt[2]
        if isinstance(ts, tuple) and ts and ts[0] == "unit_typed":
            denv[stmt[1]] = unit_to_dimvec_cic(sig_unit_spec(ts[2]), env)
    for v in variables:
        denv[v] = unit_to_dimvec_cic(unit, env)
    return denv, func_dims


def has_units(unit: dict, block_locals: List[Any], unified_ast: dict) -> bool:
    """
    Checks if unknowns at top-level constant inside a block local or a
    function parameter has a ``[unit]`` annotation.

    Parameters
    ----------
    unit : dict
        Declared unit of unknowns.
    block_locals : list
        Local constants declared in the ``where:`` block.
    unified_ast : dict
        Dictionary of top-level constants, functions and classes.

    Examples
    --------
    >>> from physika.solver.solver_dims import has_units
    >>> ast = {"program": [
    ...     ("unit_decl", "m", "ℝ", [("kg", 1)], ("num", 2.0), 1)]}
    >>> has_units({}, [], ast), has_units({}, [], {"program": []})
    (True, False)
    """
    # unknowns with a declared unit
    if unit:
        return True
    # top-level constants with a unit
    for stmt in unified_ast.get("program", []):
        if isinstance(stmt, tuple) and stmt and stmt[0] == "unit_decl":
            return True
    # block local constants with a unit
    for stmt in block_locals:
        if local_annotation(stmt[2]) is not None:
            return True
    # function parameters with unit
    for fdef in unified_ast.get("functions", {}).values():
        for _, ts in fdef.get("params", []):
            if local_annotation(ts) is not None:
                return True
    return False


def check_dimensions(variables: List[str], unit: dict, conds: List[tuple],
                     block_locals: List[Any], balance_unit: Optional[dict],
                     unified_ast: dict, env: Environment,
                     tag: str) -> Tuple[List[str], bool]:
    """
    Dimensional analysis for ``solve`` statements in Physika's CIC
    units checker, called from ``solve_block``. Each side
    of an equation gets a ``DimVec`` term. Then CIC kernel checks if they are
    equal.

    Parameters
    ----------
    variables : list[str]
        Names of unknowns.
    unit : dict
        Declared unit of unknowns.
    conds : list[AST]
        ``("cond_eq", lhs, rhs)`` equations.
    block_locals : list
        Local constants declared in the ``where:`` block.
    balance_unit : dict, optional
        Unit every term must have, from ``where [unit]:``.
    unified_ast : dict
        Dictionary of top-level constants, functions and classes.
    env : Environment
        Environment used during kernel check.
    tag : str
        Prefix for every report line.

    Examples
    --------
    >>> from physika.solver.solver_dims import check_dimensions
    >>> from physika.core.inductive import mk_builtin_env
    >>> variable = lambda n: ("var", n)
    >>> ast = {"program": [
    ...     ("unit_decl", "m", "ℝ", [("kg", 1)], ("num", 2.0), 1),
    ...     ("unit_decl", "a", "ℝ", [("m", 1), ("s", -2)], ("num", 3.0), 2)]}
    >>> eq = ("cond_eq", variable("F"),
    ...       ("mul", variable("m"), variable("a")))
    >>> lines, ok = check_dimensions(["F"], {"kg": 1}, [eq], [], None, ast,
    ...                              mk_builtin_env(), "Line 3: solve F")
    >>> ok  # F declared [kg], but m * a is [kg·m·s⁻²]
    False
    >>> lines
    ['Line 3: solve F: the equation is dimensionally inconsistent: [kg] vs [kg·m·s⁻²]']
    """  # noqa: E501
    if not (has_units(unit, block_locals, unified_ast)
            or balance_unit is not None):
        return [], True
    denv, fdims = dim_env(variables, unit, block_locals, unified_ast, env)

    errors: List[str] = []
    if balance_unit is not None:
        target = unit_to_dimvec_cic(balance_unit, env)
        shown = unit_to_str(balance_unit)
        for side in conds[0][1:]:
            if is_number(side):
                continue
            for term in add_terms(side):
                label = expr_str(term)
                try:
                    d = expr_unit_to_dimvec_cic(term, denv, env, fdims)
                except UnitError:
                    errors.append(f"the term  {label}  is not dimensionally "
                                  f"valid")
                    continue
                if d is None:
                    errors.append(f"could not determine the dimension of  "
                                  f"{label}")
                elif not same_dim(d, target, env):
                    errors.append(f"the term  {label}  has dimension "
                                  f"[{dim_str(d, env)}], not the declared "
                                  f"[{shown}]")
        if errors:
            return [f"{tag}: {e}" for e in errors], False
        return [f"{tag}: every term is [{shown}]"], True

    for i, (_, lhs, rhs) in enumerate(conds, 1):
        which = "the equation" if len(conds) == 1 else f"equation {i}"
        try:
            # only one side is checked
            dl = dr = None
            if not is_number(lhs):
                dl = expr_unit_to_dimvec_cic(lhs, denv, env, fdims)
            if not is_number(rhs):
                dr = expr_unit_to_dimvec_cic(rhs, denv, env, fdims)
        except UnitError as exc:
            errors.append(f"{which} is not dimensionally well-formed: {exc}")
            continue
        if dl is not None and dr is not None and not same_dim(dl, dr, env):
            errors.append(f"{which} is dimensionally inconsistent: "
                          f"[{dim_str(dl, env)}] vs [{dim_str(dr, env)}]")
    if errors:
        return [f"{tag}: {e}" for e in errors], False
    shown_u = unit_to_str(unit) if unit else "dimensionless"
    return [f"{tag}: dimensions consistent  ([{shown_u}])"], True
