from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    # for avoiding circular imports
    from physika.utils.cic_utils.elab_utils import ElabT
from physika.utils.cic_utils.elab_utils import (
    safe_infer_type,
    HandlerResult,
    build_ite_term,
    lit_to_real,
    elaborate_condition_bool,
    bind_local,
    find_accum_loop_index,
    try_elaborate_sum_accumulator,
    invalidate_local,
    invalidate_loop_locals,
    bind_declared_local,
    try_elaborate_dependent_fold_loop,
    try_elaborate_fold_loop,
    try_elaborate_index_write_loop,
    merge_branch_values,
    vec_shape_of,
    elaborate_grad_call,
    elaborate_call,
    safe_whnf,
    resolve_name,
    expected_elem_type_of,
    coerce_to_fin,
    is_def_eq,
    head_const_of,
    elaborate_binop,
    vec_len,
)
from physika.core.expr import FVarId, Proj
from physika.core.metavar import MetaVarKind
from physika.core.elab.dim_typespec import struct_field_names
from physika.utils.ast_utils import ASTNode

from physika.core.expr import (
    App,
    Const,
    Expr,
    FVar,
    Lit,
    MVar,
)
from physika.utils.cic_utils.expr_utils import get_app_fn_args
from physika.core.reduction import whnf

from physika.core.elab.dim_typespec import (_REAL_CONST, _TYPE_0_LEVEL,
                                            _NAT_CONST, _NAT_SUB,
                                            typespec_to_cic_resolved)


def expr_to_str(e: Expr) -> str:
    """
    Expression to string helper fucntion used for writing when ``t``/``dim``
    isn't one of ``error_type_str``'s own named cases (Const, Vec
    App, FVar, MVar, Lit).

    Parameters
    ----------
    e : Expr
        CIC term to render.

    Example
    -------
    >>> from physika.core.elab.body_elab import expr_to_str
    >>> from physika.core.expr import App, Const
    >>> expr_to_str(Const("Real", ()))
    'Real'
    >>> node = App(App(Const("Nat.add", ()), Const("n", ())), Const("one", ()))
    >>> expr_to_str(node)
    'Nat.add n one'
    """
    if isinstance(e, Const):
        return e.name
    if isinstance(e, App):
        head, args = get_app_fn_args(e)
        if not isinstance(head, Const):
            return "?"
        args_s = " ".join(expr_to_str(a) for a in args)
        return f"{head.name} {args_s}" if args_s else head.name
    return "?"


def error_type_str(t: Expr, elab: ElabT) -> str:
    """
    Convert a CIC type to Physika-style notation for error messages.

    Parameters
    ----------
    t : Expr
        A CIC type, not necessarily whnf-reduced (this instantiates
        solved MVars and reduces it itself).
    elab : Elab
        Elaborator whose env/lctx/mctx resolve ``t``.

    Example
    -------
    >>> from physika.core.elab.body_elab import error_type_str
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> error_type_str(_REAL_CONST, elab)
    'ℝ'
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> error_type_str(vec_ty, elab)
    'ℝ[3]'
    """
    t = elab.state.mctx.instantiate_mvars(t)
    t = whnf(t, elab.state.env, elab.state.lctx, elab.state.mctx)
    if isinstance(t, Const):
        return {"Real": "ℝ", "Nat": "ℕ"}.get(t.name, t.name)
    if isinstance(t, App):
        head, args = get_app_fn_args(t)
        if isinstance(head, Const) and head.name == "Vec" and len(args) == 2:
            dim = elab.state.mctx.instantiate_mvars(args[1])
            if isinstance(dim, FVar):
                decl = elab.state.lctx.find(dim.id)
                dim_s = decl.user_name if decl else "?"
            elif isinstance(dim, MVar):
                dim_s = "?"
            elif isinstance(dim, Lit):
                dim_s = str(dim.val)
            else:
                dim_s = expr_to_str(dim)
            inner = error_type_str(args[0], elab)
            if inner.startswith("ℝ["):
                return f"ℝ[{dim_s}, {inner[2:-1]}]"
            if inner == "ℝ":
                return f"ℝ[{dim_s}]"
            return f"{inner}[{dim_s}]"
    if isinstance(t, FVar):
        decl = elab.state.lctx.find(t.id)
        return decl.user_name if decl else "?"
    if isinstance(t, MVar):
        return "?"
    if isinstance(t, Lit):
        return str(t.val)
    return expr_to_str(t)


def elaborate_expr(node: ASTNode,
                   fvar_env: Dict[str, Expr],
                   elab: ElabT,
                   errors: List[str],
                   expected_type: Optional[Expr] = None) -> Expr:
    """
    Translate a Physika AST expression node to a CIC ``Expr``.

    Parameters
    ----------
    node : ASTNode
        A Physika AST expression node.
    fvar_env : Dict[str, Expr]
        Maps a varn name in scope to its CIC value.
    elab : Elab
        Elaborator providing environment, local context, metavar contxt, MVar
        and FVar information.
    errors : List[str]
        List to append a type-mismatch.
    expected_type : Optional[Expr]
        CIC type associated with this expression either when declared or
        inferred.

    Example
    -------
    >>> from physika.core.elab.body_elab import elaborate_expr
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> elaborate_expr(("num", 1.0), {}, elab, [])
    Lit(val=1.0)
    >>> elaborate_expr(("add", ("num", 1.0), ("num", 2.0)), {}, elab, [])  # noqa: E501
    App(func=App(func=Const(name='Real.add', levels=()), arg=Lit(val=1.0)), arg=Lit(val=2.0))
    """
    if node is None or not isinstance(node, tuple):
        return elab.new_mvar("_", _TYPE_0_LEVEL)
    handler = EXPR_TAG_HANDLERS.get(node[0])
    if handler is None:
        return elab.new_mvar("_" + str(node[0]), _TYPE_0_LEVEL)
    return handler(node, fvar_env, elab, errors, expected_type)


def elaborate_stmts_with_return(
    statements: Optional[list], body_node: ASTNode, fvar_env: Dict[str, Expr],
    elab: ElabT, errors: List[str], context_label: str
) -> Tuple[Expr, ElabT, Dict[str, Expr], List[Tuple[str, Expr]], dict]:
    """
    Elaborate body statements and return expressions from fucntions and classes
    with its local context and metavariables into CIC terms.

    Parameters
    ----------
    statements : Optional[list]
        List of statement declarations and assignments (without return)
    body_node : ASTNode
        Return expression as AST node (or ``None``).
    fvar_env : Dict[str, Expr]
        Mapping of function arguments from var name toelaborated CIC term.
    elab : Elab
        Elaborator to extend with FVar for each local declaration.
    errors : List[str]
        List to append error messages.
    context_label : str
        String referrencing when error occurs.

    Example
    -------
    >>> from physika.core.elab.body_elab import elaborate_stmts_with_return
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import FVar
    >>> elab = Elab(Environment())
    >>> stmts = [("body_decl", "y", "ℝ", ("num", 1.0))]
    >>> body_cic, new_elab, cur_env, local_decls, fvar_names = (
    ...     elaborate_stmts_with_return(
    ...         stmts, ("var", "y"), {}, elab, [], "In function 'f'"))
    >>> isinstance(body_cic, FVar)
    True
    >>> local_decls
    [('y', Lit(val=1.0))]
    >>> list(cur_env)
    ['y']
    """
    cur_elab = elab
    cur_env = dict(fvar_env)
    fvar_names: dict = {
        v.id: nm
        for nm, v in fvar_env.items() if isinstance(v, FVar)
    }
    terminal_node = body_node
    stmts = statements or []

    local_decls: List[Tuple[str, Expr]] = []
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        if not (isinstance(stmt, tuple) and stmt):
            i += 1
            continue
        handler = TAG_HANDLERS.get(stmt[0])
        if handler is None:
            i += 1
            continue
        cur_env, cur_elab, local_decls, fvar_names, control = handler(
            stmt=stmt,
            i=i,
            stmts=stmts,
            body_node=body_node,
            terminal_node=terminal_node,
            cur_env=cur_env,
            cur_elab=cur_elab,
            local_decls=local_decls,
            fvar_names=fvar_names,
            errors=errors,
            context_label=context_label,
        )
        if control is None:
            i += 1
        elif control[0] == "skip_to":
            i = control[1]
        elif control[0] == "break":
            if control[1] is not None:
                terminal_node = control[1]
            break
        elif control[0] == "return":
            return control[1]
    if terminal_node is None:
        return (elab.new_mvar("_body", _TYPE_0_LEVEL), cur_elab, cur_env,
                local_decls, fvar_names)
    body_cic = elaborate_expr(terminal_node, cur_env, cur_elab, errors)
    return body_cic, cur_elab, cur_env, local_decls, fvar_names


def elab_body_decl(stmt: tuple, i: int, stmts: List[tuple],
                   body_node: Optional[ASTNode],
                   terminal_node: Optional[ASTNode], cur_env: Dict[str, Expr],
                   cur_elab: ElabT, local_decls: List[Tuple[str, Expr]],
                   fvar_names: Dict[FVarId, str], errors: List[str],
                   context_label: str) -> HandlerResult:
    """
    Handle a ``body_decl`` statement:
    ``x: T = expr``


    Parameters
    ----------
    stmt : tuple
        ``body_decl`` statement.
    i : int
        Statement's index within ``stmts``.
    stmts : list
        Function/branch body's statement list.
    body_node, terminal_node : object
        Unused by this handler — part of uniform handler signature
        every ``TAG_HANDLERS`` entry shares (see that dict's docstring).
    cur_env, cur_elab, local_decls, fvar_names : object
        Threaded elaboration state, as in ``elaborate_stmts_with_return``.
    errors : list
        Diagnostics list to append to.
    context_label : str
        Location prefix for any appended diagnostic.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_body_decl
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> stmt = ("body_decl", "x", "ℝ", ("num", 5.0))
    >>> env, elab, decls, names, ctrl = elab_body_decl(
    ...     stmt, 0, [stmt], None, None, {}, elab, [], {}, [], "f")
    >>> ctrl is None
    True
    """
    _, var_name, type_spec, expr_node = stmt

    if (type_spec == "ℝ" and isinstance(expr_node, tuple)
            and len(expr_node) == 2 and expr_node[0] == "num"
            and float(expr_node[1]) == 0.0):
        j = find_accum_loop_index(stmts, i + 1, var_name)
        next_stmt = stmts[j] if j is not None else None
        sum_cic = try_elaborate_sum_accumulator(var_name, type_spec, next_stmt,
                                                cur_env, cur_elab, errors)
        if sum_cic is not None:
            # try_elaborate_sum_accumulator only succeeds when next_stmt is
            # a real ("body_for", ...) tuple, which itself only happens
            # when j is not None (next_stmt := stmts[j]).
            assert next_stmt is not None and j is not None
            cur_env = {**cur_env, var_name: sum_cic}
            cur_env = invalidate_loop_locals(next_stmt[2],
                                             cur_env,
                                             cur_elab,
                                             exclude={var_name})
            return (cur_env, cur_elab, local_decls, fvar_names, ("skip_to",
                                                                 j + 1))

    # A dim var in type_spec is resolved in local env
    decl_type = typespec_to_cic_resolved(type_spec, cur_env.get)
    rhs_cic = elaborate_expr(expr_node,
                             cur_env,
                             cur_elab,
                             errors,
                             expected_type=decl_type)
    cur_env, cur_elab = bind_declared_local(var_name, decl_type, rhs_cic,
                                            cur_env, cur_elab, local_decls,
                                            fvar_names, errors, context_label)
    return cur_env, cur_elab, local_decls, fvar_names, None


def elab_body_assign(stmt: tuple, i: int, stmts: List[tuple],
                     body_node: Optional[ASTNode],
                     terminal_node: Optional[ASTNode], cur_env: Dict[str,
                                                                     Expr],
                     cur_elab: ElabT, local_decls: List[Tuple[str, Expr]],
                     fvar_names: Dict[FVarId, str], errors: List[str],
                     context_label: str) -> HandlerResult:
    """
    Elaboration for ``body_assign`` ASTNode (``y = expr``).

    Parameters
    ----------
    stmt : tuple
        ``body_assign`` statement.
    i : int
        Statement's index within ``stmts``.
    stmts : list
        String list of statements.
    cur_env, cur_elab, local_decls, fvar_names : object
        Threaded elaboration state, as in ``elaborate_stmts_with_return``.
    errors : list
       Error list for appending messages.
    context_label : str
        Location prefix for any appended errors.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_body_assign
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> elab, x_fv = elab.with_local("x", _REAL_CONST)
    >>> stmt = ("body_assign", "x", ("num", 2.0))
    >>> env, elab, decls, names, ctrl = elab_body_assign(
    ...     stmt, 0, [stmt], None, None, {"x": x_fv}, elab, [], {}, [], "f")
    >>> env["x"] is x_fv
    False
    """
    _, var_name, expr_node = stmt
    prev_fv = cur_env.get(var_name)
    prev_decl = (cur_elab.state.lctx.find(prev_fv.id) if isinstance(
        prev_fv, FVar) else None)
    rhs_cic = elaborate_expr(
        expr_node,
        cur_env,
        cur_elab,
        errors,
        expected_type=prev_decl.type if prev_decl is not None else None)
    if isinstance(rhs_cic, MVar):
        rhs_type = None
    else:
        rhs_type = safe_infer_type(cur_elab, rhs_cic)

    if rhs_type is not None:
        if prev_decl is not None and not cur_elab.unify(
                rhs_type, prev_decl.type):
            exp_s = error_type_str(prev_decl.type, cur_elab)
            got_s = error_type_str(rhs_type, cur_elab)
            errors.append(f"{context_label}: variable '{var_name}' "
                          f"reassigned value of type {got_s}, "
                          f"previously {exp_s}")
        cur_env, cur_elab = bind_local(var_name, rhs_cic, rhs_type, cur_env,
                                       cur_elab, local_decls, fvar_names)
    elif isinstance(prev_fv, FVar):
        # RHS failed elaboration
        # add var_name as an mvar
        cur_env = invalidate_local(var_name, cur_env, cur_elab)
    return cur_env, cur_elab, local_decls, fvar_names, None


def elab_body_for(stmt: tuple, i: int, stmts: List[tuple],
                  body_node: Optional[ASTNode],
                  terminal_node: Optional[ASTNode], cur_env: Dict[str, Expr],
                  cur_elab: ElabT, local_decls: List[Tuple[str, Expr]],
                  fvar_names: Dict[FVarId, str], errors: List[str],
                  context_label: str) -> HandlerResult:
    """
    Handler for elaborating ``body_for``/``body_for_range`` loop ASTNode.

    Elaborate a physka program of the form:
    ``for i: ...`` loop (implicit bound)
    ``for i: ℕ(a, b):`` loop (explicit range)

    Parameters
    ----------
    stmt : tuple
        ``body_for``/``body_for_range`` statement.
    i : int
        Statement's index within ``stmts``.
    stmts : list
        Function's statement list.
    cur_env: Dict[str, Expr]
        Current environment with solved MVars
    cur_elab: Elab
        Threaded elaboration state.
    local_decls: List[Tuple[str, Expr]]
        List of tuples that contains solved and unsolved variables names with
        its CIC term.
    fvar_names: Dict[FVarId, str].
        Free variables names present in current scope.
    errors : list
        Diagnostics list to append to.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_body_for
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import App, Const, Lit
    >>> elab = Elab(Environment())
    >>> elab, x_fv = elab.with_local("x", _REAL_CONST)
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> elab, arr_fv = elab.with_local("arr", vec_ty)
    >>> stmt = ("body_for", "k", [("loop_assign", "x", ("num", 1.0))], ["arr"])
    >>> env, elab, decls, names, ctrl = elab_body_for(
    ...     stmt, 0, [stmt], None, None,
    ...     {"x": x_fv, "arr": arr_fv}, elab, [], {}, [], "f")
    >>> env["x"] is x_fv
    False
    """
    if stmt[0] == "body_for":
        loop_body = stmt[2]
    else:
        loop_body = stmt[4]
    # fold means an accumulator present
    fold_cic = None
    if stmt[0] == "body_for":
        loop_var, indexed_arrays = stmt[1], stmt[3]
        fold_cic = try_elaborate_fold_loop(loop_var, loop_body, indexed_arrays,
                                           cur_env, cur_elab, errors)
    # No accumulator, but the whole loop body may still be a pure
    # per-index write to one already-bound array (`for i: dst[i] =
    # expr`, expr not referencing dst) — that's a plain Vec.tabulate,
    # not a fold, so try it before giving up.
    index_write = None
    if fold_cic is None and stmt[0] == "body_for":
        index_write = try_elaborate_index_write_loop(loop_var, loop_body,
                                                     cur_env, cur_elab, errors)
    multi_updates = None
    if (fold_cic is None and index_write is None
            and stmt[0] == "body_for_range"):
        # exlicit for loops: `for i:ℕ(a, b):`
        range_var, start_node, end_node = stmt[1], stmt[2], stmt[3]
        start_cic = elaborate_expr(start_node, cur_env, cur_elab, errors)
        end_cic = elaborate_expr(end_node, cur_env, cur_elab, errors)
        if not isinstance(start_cic, MVar) and not isinstance(end_cic, MVar):
            n_cic = App(App(_NAT_SUB, end_cic), start_cic)
            multi_updates = try_elaborate_dependent_fold_loop(
                range_var, loop_body, n_cic, cur_env, cur_elab, errors)
    if fold_cic is not None:
        cur_env = {**cur_env, loop_body[0][1]: fold_cic}
    elif index_write is not None:
        array_name, tabulate_cic = index_write
        cur_env = {**cur_env, array_name: tabulate_cic}
    elif multi_updates is not None:
        cur_env = {**cur_env, **multi_updates}
    else:
        # No general loop translation
        cur_env = invalidate_loop_locals(loop_body, cur_env, cur_elab)
    return cur_env, cur_elab, local_decls, fvar_names, None


def elab_body_if(stmt: tuple, i: int, stmts: List[tuple],
                 body_node: Optional[ASTNode],
                 terminal_node: Optional[ASTNode], cur_env: Dict[str, Expr],
                 cur_elab: ElabT, local_decls: List[Tuple[str, Expr]],
                 fvar_names: Dict[FVarId, str], errors: List[str],
                 context_label: str) -> HandlerResult:
    """
    Handler for elaborating ``body_if``/``body_if_else`` ASTNode where branches
    reassign rather than return.

    Parameters
    ----------
    stmt : tuple
        ``body_if``/``body_if_else`` statement.
    i : int
        Statement's index for ``stmts`` list.
    stmts : list
        Function body's statement list.
    cur_env: Dict[str, Expr]
        Current environment with solved MVars
    cur_elab: Elab
        Threaded elaboration state.
    local_decls: List[Tuple[str, Expr]]
        List of tuples that contains solved and unsolved variables names with
        its CIC term.
    fvar_names: Dict[FVarId, str].
        Free variables names present in current scope.
    errors : list
        List to append errors.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_body_if
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> elab, x_fv = elab.with_local("x", _REAL_CONST)
    >>> cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
    >>> stmt = ("body_if", cond, [("body_assign", "x", ("num", 9.0))])
    >>> env, elab, decls, names, ctrl = elab_body_if(
    ...     stmt, 0, [stmt], None, None, {"x": x_fv}, elab, [], {}, [], "f")
    >>> env["x"] is x_fv
    False
    """
    cond_node = stmt[1]
    then_stmts = stmt[2]
    if stmt[0] == "body_if_else":
        else_stmts = stmt[3]
    else:
        else_stmts = None

    merged = merge_branch_values(cond_node, then_stmts, else_stmts, cur_env,
                                 cur_elab, errors)
    for var_name, value_cic in merged.items():
        value_type = safe_infer_type(cur_elab, value_cic)
        if value_type is None:
            prev_fv = cur_env.get(var_name)
            prev_decl = (cur_elab.state.lctx.find(prev_fv.id) if isinstance(
                prev_fv, FVar) else None)
            prev_type = prev_decl.type if prev_decl is not None else _REAL_CONST  # noqa: E501
            cur_env = {
                **cur_env, var_name:
                cur_elab.new_mvar(f"_unelaborated_{var_name}",
                                  prev_type,
                                  kind=MetaVarKind.SYNTHETIC)
            }
            continue
        cur_env, cur_elab = bind_local(var_name, value_cic, value_type,
                                       cur_env, cur_elab, local_decls,
                                       fvar_names)
    return cur_env, cur_elab, local_decls, fvar_names, None


def elab_body_if_return(stmt: tuple, i: int, stmts: List[tuple],
                        body_node: Optional[ASTNode],
                        terminal_node: Optional[ASTNode], cur_env: Dict[str,
                                                                        Expr],
                        cur_elab: ElabT, local_decls: List[Tuple[str, Expr]],
                        fvar_names: Dict[FVarId, str], errors: List[str],
                        context_label: str) -> HandlerResult:
    """
    Handle a ``body_if_return`` (``if cond: return then_expr``, no
    else) — execution falls through to whatever follows when cond is
    false.

    Real.ite/Nat.ite(cond, then_expr, value). "value" recursively elaborated
    as else branch.

    Parameters
    ----------
    stmt : tuple
        ``body_if_return`` statement.
    i : int
        Statement's index in ``stmts``.
    stmts : list
        Function body's statement list.
    body_node : object
        Function's return-expression node as else branch
    cur_env: Dict[str, Expr]
        Current environment with solved MVars
    cur_elab: Elab
        Threaded elaboration state.
    local_decls: List[Tuple[str, Expr]]
        List of tuples that contains solved and unsolved variables names with
        its CIC term.
    fvar_names: Dict[FVarId, str].
        Free variables names present in current scope.
    errors : list
        List to append to append errors.
    context_label : str
        Location in a physika file for appending an error.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_body_if_return
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
    >>> stmt = ("body_if_return", cond, ("num", 1.0))
    >>> env, elab, decls, names, ctrl = elab_body_if_return(
    ...     stmt, 0, [stmt], ("num", 2.0), None, {}, elab, [], {}, [], "f")
    >>> ctrl[0]
    'return'

    """
    cond_node, then_node = stmt[1], stmt[2]
    cond_cic = elaborate_condition_bool(cond_node, cur_env, cur_elab, errors)
    then_cic = elaborate_expr(then_node, cur_env, cur_elab, errors)
    else_cic, else_elab, _else_env, else_local_decls, else_fvar_names = (
        elaborate_stmts_with_return(stmts[i + 1:], body_node, cur_env,
                                    cur_elab, errors, context_label))
    all_local_decls = local_decls + else_local_decls

    merged_fvar_names = {**fvar_names, **else_fvar_names}
    if (isinstance(cond_cic, MVar) or isinstance(then_cic, MVar)
            or isinstance(else_cic, MVar)):
        mvar_result: Expr = else_elab.new_mvar("_if_return", _TYPE_0_LEVEL)
        result = (mvar_result, else_elab, cur_env, all_local_decls,
                  merged_fvar_names)
        return cur_env, cur_elab, local_decls, fvar_names, ("return", result)
    then_type = safe_infer_type(else_elab, then_cic, whnf_reduce=True)
    else_type = safe_infer_type(else_elab, else_cic, whnf_reduce=True)
    if then_type == _REAL_CONST and else_type != _REAL_CONST:
        else_cic = lit_to_real(else_cic, then_type)
    elif else_type == _REAL_CONST and then_type != _REAL_CONST:
        then_cic = lit_to_real(then_cic, else_type)
        try:
            then_type = whnf(else_elab.infer_type(then_cic),
                             else_elab.state.env, else_elab.state.lctx,
                             else_elab.state.mctx)
        except Exception:
            pass
    merged_cic = build_ite_term(cond_cic, then_cic, else_cic, then_type)
    result = (merged_cic, else_elab, cur_env, all_local_decls,
              merged_fvar_names)
    return cur_env, cur_elab, local_decls, fvar_names, ("return", result)


def elab_body_if_else_return(stmt: tuple, i: int, stmts: List[tuple],
                             body_node: Optional[ASTNode],
                             terminal_node: Optional[ASTNode],
                             cur_env: Dict[str, Expr], cur_elab: ElabT,
                             local_decls: List[Tuple[str, Expr]],
                             fvar_names: Dict[FVarId, str], errors: List[str],
                             context_label: str) -> HandlerResult:
    """
    Handle for ``body_if_else_return``.

    Parameters
    ----------
    stmt : tuple
        ``body_if_else_return`` statement
    terminal_node : object
        Function's terminal expression.
    cur_env, cur_elab, local_decls, fvar_names : object
        Threaded elaboration state, as in ``elaborate_stmts_with_return``.
    cur_env: Dict[str, Expr]
        Current environment with solved MVars
    cur_elab: Elab
        Threaded elaboration state.
    local_decls: List[Tuple[str, Expr]]
        List of tuples that contains solved and unsolved variables names with
        its CIC term.
    fvar_names: Dict[FVarId, str].
        Free variables names present in current scope.
    errors : list
        List to append to append errors.
    context_label : str
        Location in a physika file for appending an error.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_body_if_else_return
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
    >>> stmt = ("body_if_else_return", cond, ("num", 1.0), ("num", 2.0))
    >>> elab_body_if_else_return(
    ...     stmt, 0, [stmt], None, None, {}, elab, [], {}, [], "f")[4][0]
    'break'

    """
    if terminal_node is not None:
        return cur_env, cur_elab, local_decls, fvar_names, None
    return cur_env, cur_elab, local_decls, fvar_names, ("break", stmt)


def elab_expr_var(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                  errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle for ``var`` ASTnode. This is elaborating a variable name reference.


    Parameters
    ----------
    node : tuple
        ``("var", name)`` node.
    fvar_env : Dict[str, Expr]
        Maps a varnname in scope (parameter, local, loop var) to
        its CIC term.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar context.
    expected_type : Optional[Expr]
        CIC type as declared in Physika program or inferred.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_var
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> elab, x_fv = elab.with_local("x", _REAL_CONST)
    >>> elab_expr_var(("var", "x"), {"x": x_fv}, elab, [], None) is x_fv
    True
    """
    return resolve_name(node[1], fvar_env, elab)


def elab_expr_num(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                  errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``num`` ASTNode elaboration for a numeric literal.

    An number literal is elaborated by using OfNat instance matching the type
    of surrounding context expects ("ℕ" or "ℝ"), following how
    Lean 4 resolves numeric literals.

    Parameters
    ----------
    node : tuple
        ``("num", value)`` node.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    expected_type : Optional[Expr]
        When "ℝ" is declared or inferred, we use ``instOfNatReal`` instance
        instead of defaulting to ``instOfNatNat``.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_num
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> elab_expr_num(("num", 3.0), {}, elab, [], None)
    Lit(val=3.0)
    """
    raw = node[1]
    if isinstance(raw, float):
        return Lit(raw)  # type: ignore[arg-type]
    target = safe_whnf(elab,
                       expected_type) if expected_type is not None else None
    # defaults to Nat
    if target == _REAL_CONST:
        inst_name = "instOfNatReal"
    else:
        inst_name = "instOfNatNat"
    return Proj("OfNat", 0, App(Const(inst_name, ()), Lit(raw)))


def elab_expr_imaginary(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                        errors: List[str],
                        expected_type: Optional[Expr]) -> Expr:
    """
    Handle an ``imaginary`` ASTNode.

    Physika's lexer emits IMAGINARY for this token everywhere except the
    ``for i : ...``.

    Parameters
    ----------
    node : tuple
        ``("imaginary",)`` node.
    fvar_env : Dict[str, Expr]
        Variable Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_imaginary
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> elab, i_fv = elab.with_local("i", _REAL_CONST)  # noqa: E501
    >>> elab_expr_imaginary(("imaginary",), {"i": i_fv}, elab, [], None) is i_fv
    True
    """
    if "i" in fvar_env:
        return fvar_env["i"]
    return elab.new_mvar("_imaginary", _TYPE_0_LEVEL)


def elab_expr_for(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                  errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``for_expr`` node at top level.
    ``for i : ℕ(n) -> body``.

    Elaborates to ``Vec.tabulate n (fun i : Fin n => body)``, which supports
    dependent functions from indices to values.

    Parameters
    ----------
    node : tuple
        ``("for_expr", loop_var, bound_node, body_node)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope, extended with ``loop_var`` for the body.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains elaboration error meesagges for failing
        ``elaborate_expr`` calls.
    expected_type : Optional[Expr]
        When it's a ``Vec`` type, commonly a tensor, each elemt type is
        verified and registerd (ℝ[3] is checked to be three elemets of type ℝ)

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_for
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import MVar
    >>> elab = Elab(Environment())
    >>> node = ("for_expr", "j", ("num", 2), ("num", 1.0))
    >>> isinstance(elab_expr_for(node, {}, elab, [], None), MVar)
    False
    """
    loop_var, bound_node, body_node = node[1], node[2], node[3]
    n_cic = elaborate_expr(bound_node, fvar_env, elab, errors)
    if isinstance(n_cic, MVar):
        return elab.new_mvar("_for_expr", _TYPE_0_LEVEL)
    fin_n = App(Const("Fin", ()), n_cic)
    child_elab, i_fv = elab.with_local(loop_var, fin_n)
    child_env = {**fvar_env, loop_var: i_fv}

    expected_elem_type = expected_elem_type_of(elab, expected_type)
    body_cic = elaborate_expr(body_node,
                              child_env,
                              child_elab,
                              errors,
                              expected_type=expected_elem_type)
    if isinstance(body_cic, MVar):
        return elab.new_mvar("_for_expr", _TYPE_0_LEVEL)
    # Follows Lean 4 Vec.tabulate which is polymorphic over element type
    # infer it from the body (usually ℝ)
    try:
        elem_type = child_elab.infer_type(body_cic)
    except Exception:
        elem_type = _REAL_CONST
    fn = child_elab.state.lctx.mk_lambda([i_fv], body_cic)
    return App(App(App(Const("Vec.tabulate", ()), elem_type), n_cic), fn)


def elab_expr_array(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                    errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle an ``array`` ASTNode.

    Elaborates to ``Vec.cons`` with concrete length, not an opaque MVar useful
    at functions calls.
    For example, a top level array:
    ``u3 : ℝ[3] = [1.0, 2.0, 3.0]``
    CIC equivalent would be:
    ``Vec Real 3``

    Parameters
    ----------
    node : tuple
        ``("array", elements)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope, passed to each element's elaboration.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.
    expected_type : Optional[Expr]
        Declared or inferred type of array.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_array
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import MVar
    >>> elab = Elab(Environment())
    >>> node = ("array", [("num", 1.0), ("num", 2.0)])
    >>> isinstance(elab_expr_array(node, {}, elab, [], None), MVar)
    False
    """
    elements = node[1] if len(node) > 1 else []
    if not elements:
        return elab.new_mvar("_array", _TYPE_0_LEVEL)
    expected_elem_type = expected_elem_type_of(elab, expected_type)

    elem_cics = [
        elaborate_expr(e,
                       fvar_env,
                       elab,
                       errors,
                       expected_type=expected_elem_type) for e in elements
    ]
    if any(isinstance(c, MVar) for c in elem_cics):
        return elab.new_mvar("_array", _TYPE_0_LEVEL)
    try:
        elem_type = whnf(elab.infer_type(elem_cics[0]), elab.state.env,
                         elab.state.lctx, elab.state.mctx)
    except Exception:
        return elab.new_mvar("_array", _TYPE_0_LEVEL)
    if expected_elem_type == _REAL_CONST and elem_type != _REAL_CONST:
        elem_type = expected_elem_type
    coerced: List[Expr] = []
    for c in elem_cics:
        c = lit_to_real(c, elem_type)
        try:
            c_type = whnf(elab.infer_type(c), elab.state.env, elab.state.lctx,
                          elab.state.mctx)
        except Exception:
            return elab.new_mvar("_array", _TYPE_0_LEVEL)
        ok, _ = is_def_eq(c_type, elem_type, elab.state.env, elab.state.lctx,
                          elab.state.mctx)
        if not ok:
            return elab.new_mvar("_array", _TYPE_0_LEVEL)
        coerced.append(c)
    vec: Expr = App(Const("Vec.nil", ()), elem_type)
    length = 0
    for elem in reversed(coerced):
        vec = App(
            App(
                App(App(Const("Vec.cons", ()), elem_type),
                    Lit(length)),  # type: ignore[arg-type]
                elem),
            vec)
        length += 1
    return vec


def elab_expr_index(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                    errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle an ``index`` ASTNode.

    ``index`` is elaborated for this handler when we have an index in a Physika
    program:
    ``u[i]``

    ``u`` is a var name (not a nested var node) and ``i`` is usually
    for's loop variable.

    Parameters
    ----------
    node : tuple
        ``("index", obj_name, idx_node)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_index
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import App, Const, Lit, MVar
    >>> elab = Elab(Environment())
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> elab, arr_fv = elab.with_local("arr", vec_ty)
    >>> elab, i_fv = elab.with_local("i", App(Const("Fin", ()), Lit(3)))
    >>> env = {"arr": arr_fv, "i": i_fv}
    >>> node = ("index", "arr", ("var", "i"))
    >>> isinstance(elab_expr_index(node, env, elab, [], None), MVar)
    False
    """
    obj_name, idx_node = node[1], node[2]
    obj_cic = resolve_name(obj_name, fvar_env, elab)
    idx_cic = elaborate_expr(idx_node, fvar_env, elab, errors)
    if not isinstance(obj_cic, MVar) and not isinstance(idx_cic, MVar):
        obj_type = safe_infer_type(elab, obj_cic, whnf_reduce=True)
        if obj_type is not None:
            n_expr, elem_type = vec_shape_of(obj_type)
            if n_expr is not None and elem_type is not None:
                idx_cic = coerce_to_fin(idx_cic, n_expr, elab, errors)
                return App(
                    App(App(App(Const("Vec.get", ()), elem_type), n_expr),
                        obj_cic), idx_cic)
    return elab.new_mvar("_index", _TYPE_0_LEVEL)


def elab_expr_indexn(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                     errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle elaboration of ``indexN`` ASTNode. These elaboration rules
    are very simliar to ``index`` case but for more indexed dimensions.

    ``ℝ[a,b]`` is represented in CIC as nested ``Vec (Vec Real b) a``, so
    ``x[i,j]`` is ``x[i][j]``. We use ``Vec`` inductive type recursor rule
    ``Vec.get`` per index, re-inferring shape at each step.

    Parameters
    ----------
    node : tuple
        ``("indexN", obj_name, index_items)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_indexn
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import App, Const, Lit, MVar
    >>> elab = Elab(Environment())
    >>> row_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> mat_ty = App(App(Const("Vec", ()), row_ty), Lit(2))
    >>> elab, m_fv = elab.with_local("m", mat_ty)
    >>> elab, i_fv = elab.with_local("i", App(Const("Fin", ()), Lit(2)))
    >>> elab, j_fv = elab.with_local("j", App(Const("Fin", ()), Lit(3)))
    >>> env = {"m": m_fv, "i": i_fv, "j": j_fv}
    >>> idx = [("index_item", ("var", "i")), ("index_item", ("var", "j"))]  # noqa: E501
    >>> isinstance(elab_expr_indexn(("indexN", "m", idx), env, elab, [], None), MVar)
    True
    """
    obj_name, raw_idx_nodes = node[1], node[2]

    idx_nodes = [
        item[1] if (isinstance(item, tuple) and len(item) == 2
                    and item[0] == "index_item") else item
        for item in raw_idx_nodes
    ]
    obj_cic = resolve_name(obj_name, fvar_env, elab)
    if isinstance(obj_cic, MVar):
        return elab.new_mvar("_indexN", _TYPE_0_LEVEL)
    for idx_node in idx_nodes:
        idx_cic = elaborate_expr(idx_node, fvar_env, elab, errors)
        if isinstance(idx_cic, MVar):
            return elab.new_mvar("_indexN", _TYPE_0_LEVEL)
        obj_type = safe_infer_type(elab, obj_cic, whnf_reduce=True)
        if obj_type is None:
            return elab.new_mvar("_indexN", _TYPE_0_LEVEL)
        n_expr, elem_type = vec_shape_of(obj_type)
        if n_expr is None or elem_type is None:
            return elab.new_mvar("_indexN", _TYPE_0_LEVEL)
        idx_cic = coerce_to_fin(idx_cic, n_expr, elab, errors)
        obj_cic = App(
            App(App(App(Const("Vec.get", ()), elem_type), n_expr), obj_cic),
            idx_cic)
    return obj_cic


DEFAULT_OPEN_ALIASES = {
    "sum": "Vec.sum",
    "cons": "Vec.cons",
}


def elab_expr_call(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                   errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle a function ``call`` ASTNode

    Parameters
    ----------
    node : tuple
        ``("call", func_name, args)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_call
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import App, Const, Lit
    >>> elab = Elab(Environment())
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> elab, arr_fv = elab.with_local("arr", vec_ty)
    >>> node = ("call", "len", [("var", "arr")])
    >>> elab_expr_call(node, {"arr": arr_fv}, elab, [], None)
    Lit(val=3)
    """
    func_name = node[1]
    raw_args = node[2] if len(node) > 2 else []
    expr_args = [
        a for a in raw_args
        if not (isinstance(a, tuple) and a[0] in ("string", "equation_string"))
    ]
    func_name = DEFAULT_OPEN_ALIASES.get(func_name, func_name)
    arg_cics = [elaborate_expr(a, fvar_env, elab, errors) for a in expr_args]
    if func_name == "grad":
        return elaborate_grad_call(arg_cics, elab, errors)
    if func_name == "len" and len(arg_cics) == 1:
        if not isinstance(arg_cics[0], MVar):
            arg_type = safe_infer_type(elab, arg_cics[0], whnf_reduce=True)
            n_expr = vec_len(arg_type) if arg_type is not None else None
            if n_expr is not None:
                return n_expr
        return elab.new_mvar("_len", _NAT_CONST)
    return elaborate_call(func_name, arg_cics, elab, errors)


def elab_expr_field_access(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                           errors: List[str],
                           expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``field_access`` ASTNode elaboration to CIC term.

    Parameters
    ----------
    node : tuple
        ``("field_access", obj_node, field_name)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_field_access
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import MVar
    >>> elab = Elab(Environment())
    >>> node = ("field_access", ("var", "obj"), "x")
    >>> isinstance(elab_expr_field_access(node, {}, elab, [], None), MVar)
    True
    """
    obj_node, field_name = node[1], node[2]
    obj_cic = elaborate_expr(obj_node, fvar_env, elab, errors)
    if not isinstance(obj_cic, MVar):
        head = head_const_of(elab, obj_cic)
        if head is not None:
            fnames = struct_field_names(head.name, elab.state.env)
            if fnames is not None and field_name in fnames:
                return Proj(head.name, fnames.index(field_name), obj_cic)
    return elab.new_mvar("_" + field_name, _TYPE_0_LEVEL)


def elab_expr_method_call(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                          errors: List[str],
                          expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``method_call`` ASTNode elaboration.

    For a class method call:
    ``a.dot(b)``

    Physika elborates ``a``'s actual class type to find
    ``ClassName.method`` and register it. Then reuses
    ``elaborate_call`` with ``a`` prepended as method's
    leading ``this`` argument.
    ``ClassName.method``'s registered
    type is ``∀{dims}, (this:ClassName) -> params -> Return``

    Parameters
    ----------
    node : tuple
        ``("method_call", obj_node, method_name, raw_args)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_method_call
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import MVar
    >>> elab = Elab(Environment())
    >>> node = ("method_call", ("var", "obj"), "m", [])
    >>> isinstance(elab_expr_method_call(node, {}, elab, [], None), MVar)
    True
    """
    obj_node, method_name, raw_args = node[1], node[2], node[3]

    if (isinstance(obj_node, tuple) and obj_node[0] == "var"
            and obj_node[1] not in fvar_env):
        qualified = f"{obj_node[1]}.{method_name}"
        if elab.state.env.constants.get(qualified) is not None:
            arg_cics = [
                elaborate_expr(a, fvar_env, elab, errors) for a in raw_args
            ]
            return elaborate_call(qualified, arg_cics, elab, errors)
    obj_cic = elaborate_expr(obj_node, fvar_env, elab, errors)
    if not isinstance(obj_cic, MVar):
        head = head_const_of(elab, obj_cic)
        if head is not None:
            qualified = f"{head.name}.{method_name}"
            if elab.state.env.constants.get(qualified) is not None:
                arg_cics = [
                    elaborate_expr(a, fvar_env, elab, errors) for a in raw_args
                ]
                return elaborate_call(qualified, [obj_cic, *arg_cics], elab,
                                      errors)
    return elab.new_mvar("_method_call", _TYPE_0_LEVEL)


def elab_expr_binop(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                    errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle an ``add``/``sub``/``mul``/``div`` ASTNode elaboration.

    Shared binary arithmetic operator.

    Parameters
    ----------
    node : tuple
        ``(tag, lhs_node, rhs_node)`` node, ``tag`` one of
        ``"add"``/``"sub"``/``"mul"``/``"div"``.
    fvar_env : Dict[str, Expr]
        Variable names in scope
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_binop  # noqa: E501
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> elab_expr_binop(("add", ("num", 1.0), ("num", 2.0)), {}, elab, [], None)
    App(func=App(func=Const(name='Real.add', levels=()), arg=Lit(val=1.0)), arg=Lit(val=2.0))
    """
    a_cic = elaborate_expr(node[1], fvar_env, elab, errors)
    b_cic = elaborate_expr(node[2], fvar_env, elab, errors)
    return elaborate_binop(node[0], a_cic, b_cic, elab)


def elab_expr_matmul(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                     errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``matmul`` node

    Elaborates ``A @ B`` amtrix mutiplication symbol into valid CIC term.
    Physika represents both matrices and column vectors as 2D arrays (a
    vector is ``ℝ[n, 1]``). This elaboration rules covers matrix-vector
    and matrix-matrix with one axiom (``Mat.matmul``).

    Parameters
    ----------
    node : tuple
        ``("matmul", a_node, b_node)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_matmul
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import App, Const, Lit, MVar
    >>> elab = Elab(Environment())
    >>> row_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> mat_ty = App(App(Const("Vec", ()), row_ty), Lit(2))
    >>> elab, m_fv = elab.with_local("m", mat_ty)
    >>> env = {"m": m_fv}
    >>> node = ("matmul", ("var", "m"), ("var", "m"))
    >>> isinstance(elab_expr_matmul(node, env, elab, [], None), MVar)
    False
    """
    a_cic = elaborate_expr(node[1], fvar_env, elab, errors)
    b_cic = elaborate_expr(node[2], fvar_env, elab, errors)
    if not isinstance(a_cic, MVar) and not isinstance(b_cic, MVar):
        a_type = safe_infer_type(elab, a_cic, whnf_reduce=True)
        b_type = safe_infer_type(elab, b_cic, whnf_reduce=True)
        if a_type is not None and b_type is not None:

            m_expr, a_elem = vec_shape_of(a_type)
            k_expr, _ = vec_shape_of(a_elem) if a_elem is not None else (None,
                                                                         None)
            k2_expr, b_elem = vec_shape_of(b_type)
            n_expr, _ = vec_shape_of(b_elem) if b_elem is not None else (None,
                                                                         None)
            if (m_expr is not None and k_expr is not None
                    and k2_expr is not None and n_expr is not None):
                return App(
                    App(
                        App(App(App(Const("Mat.matmul", ()), m_expr), k_expr),
                            n_expr), a_cic), b_cic)
    return elab.new_mvar("_matmul", _TYPE_0_LEVEL)


def elab_expr_pow(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                  errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``pow`` ASTNode

    ``Real.pow``'s base and exponent are both Real typed. However, a lit
    numeral (like ``x ** 2``) defaults to Nat OfNat instance.

    Parameters
    ----------
    node : tuple
        ``("pow", base_node, exp_node)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_pow  # noqa: E501
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> elab_expr_pow(("pow", ("num", 2.0), ("num", 3)), {}, elab, [], None)
    App(func=App(func=Const(name='Real.pow', levels=()), arg=Lit(val=2.0)), arg=Proj(type_name='OfNat', idx=0, expr=App(func=Const(name='instOfNatReal', levels=()), arg=Lit(val=3))))
    """
    a_cic = elaborate_expr(node[1],
                           fvar_env,
                           elab,
                           errors,
                           expected_type=_REAL_CONST)
    b_cic = elaborate_expr(node[2],
                           fvar_env,
                           elab,
                           errors,
                           expected_type=_REAL_CONST)
    if isinstance(a_cic, MVar) or isinstance(b_cic, MVar):
        return elab.new_mvar("_pow", _TYPE_0_LEVEL)
    return App(App(Const("Real.pow", ()), a_cic), b_cic)


def elab_expr_neg(node: tuple, fvar_env: Dict[str, Expr], elab: ElabT,
                  errors: List[str], expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``neg`` elabroation OF ASTNode.

    Unary minus (``-J``) defaults to ``Real.neg``. ℕ is unsigned (Nat
    has no negation axiom).

    Parameters
    ----------
    node : tuple
        ``("neg", operand_node)`` node.
    fvar_env : Dict[str, Expr]
        Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_neg
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> elab_expr_neg(("neg", ("num", 1.0)), {}, elab, [], None)
    App(func=Const(name='Real.neg', levels=()), arg=Lit(val=1.0))
    """
    a_cic = elaborate_expr(node[1], fvar_env, elab, errors)
    if isinstance(a_cic, MVar):
        return elab.new_mvar("_neg", _TYPE_0_LEVEL)
    return App(Const("Real.neg", ()), a_cic)


def elab_expr_if_else_return(node: tuple, fvar_env: Dict[str, Expr],
                             elab: ElabT, errors: List[str],
                             expected_type: Optional[Expr]) -> Expr:
    """
    Handle a ``body_if_else_return`` node.

    ``
    if <cond>:
        return t
    else:
        return f
    ``

    Both branches return, so this elaborates to ``<Nat|Real>.ite cond t
    f`` (kernel-checked conditional value).

    Parameters
    ----------
    node : tuple
        ``("body_if_else_return", cond_node, then_node, else_node)``
        node.
    fvar_env : Dict[str, Expr]
        Variable names in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar/FVar information.
    errors : List[str]
        List that contains error messages.

    Example
    -------
    >>> from physika.core.elab.body_elab import elab_expr_if_else_return
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import MVar
    >>> elab = Elab(Environment())
    >>> cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
    >>> node = ("body_if_else_return", cond, ("num", 1.0), ("num", 2.0))
    >>> isinstance(elab_expr_if_else_return(node, {}, elab, [], None), MVar)
    False
    """
    cond_node, then_node, else_node = node[1], node[2], node[3]
    cond_cic = elaborate_condition_bool(cond_node, fvar_env, elab, errors)
    then_cic = elaborate_expr(then_node, fvar_env, elab, errors)
    else_cic = elaborate_expr(else_node, fvar_env, elab, errors)
    if (isinstance(cond_cic, MVar) or isinstance(then_cic, MVar)
            or isinstance(else_cic, MVar)):
        return elab.new_mvar("_if_else", _TYPE_0_LEVEL)
    then_type = safe_infer_type(elab, then_cic, whnf_reduce=True)
    return build_ite_term(cond_cic, then_cic, else_cic, then_type)


EXPR_TAG_HANDLERS = {
    "var": elab_expr_var,
    "num": elab_expr_num,
    "imaginary": elab_expr_imaginary,
    "for_expr": elab_expr_for,
    "array": elab_expr_array,
    "index": elab_expr_index,
    "indexN": elab_expr_indexn,
    "call": elab_expr_call,
    "field_access": elab_expr_field_access,
    "method_call": elab_expr_method_call,
    "add": elab_expr_binop,
    "sub": elab_expr_binop,
    "mul": elab_expr_binop,
    "div": elab_expr_binop,
    "matmul": elab_expr_matmul,
    "pow": elab_expr_pow,
    "neg": elab_expr_neg,
    "body_if_else_return": elab_expr_if_else_return,
}

TAG_HANDLERS = {
    "body_decl": elab_body_decl,
    "body_assign": elab_body_assign,
    "body_for": elab_body_for,
    "body_for_range": elab_body_for,
    "body_if": elab_body_if,
    "body_if_else": elab_body_if,
    "body_if_return": elab_body_if_return,
    "body_if_else_return": elab_body_if_else_return,
}
