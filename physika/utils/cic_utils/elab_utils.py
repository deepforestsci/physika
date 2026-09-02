from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from physika.core.expr import (
    App,
    BVar,
    Const,
    Expr,
    FVar,
    FVarId,
    ForallE,
    Lit,
    MVar,
    Proj,
    Sort,
    BinderInfo,
)
from physika.core.level import LSucc, LZero
from physika.core.metavar import MetaVarKind
from physika.core.reduction import is_def_eq, lit_nat_int, whnf
from physika.utils.cic_utils.expr_utils import (
    get_app_fn_args,
    instantiate1,
)
from physika.utils.ast_utils import ASTNode

if TYPE_CHECKING:
    # t avoud circular imports
    from physika.core.elab.elab import Elab as ElabT

HandlerResult = Tuple[Dict[str, Expr], "ElabT", List[Tuple[str, Expr]],
                      Dict[FVarId, str], Optional[tuple]]

_NAT_CONST = Const("Nat", ())
_REAL_CONST = Const("Real", ())
_VEC_CONST = Const("Vec", ())
_NAT_ADD = Const("Nat.add", ())
_TYPE_0_LEVEL = Sort(LSucc(LZero()))


def return_type_contains_more_elements(tp: Expr) -> bool:
    """
    Structurally recognize ``∀{m:Nat}, Vec Real m → Real → Vec Real
    (m+"n")``. Returns wether a Pi type has a body that outputs an array
    with more elements than original.

    Parameters
    ----------
    tp : Expr
        A function's registered Pi-type.

    Example
    -------
    >>> from physika.core.expr import ForallE, BinderInfo, App, BVar, Lit  # noqa: E501
    >>> from physika.utils.cic_utils.elab_utils import return_type_contains_more_elements
    >>> shape = ForallE("m", _NAT_CONST,
    ...     ForallE("x", App(App(_VEC_CONST, _REAL_CONST), BVar(0)),
    ...         ForallE("v", _REAL_CONST,
    ...             App(App(_VEC_CONST, _REAL_CONST),
    ...                 App(App(_NAT_ADD, BVar(2)), Lit(1))),
    ...             BinderInfo.DEFAULT),
    ...         BinderInfo.DEFAULT),
    ...     BinderInfo.IMPLICIT)
    >>> return_type_contains_more_elements(shape)
    True
    >>> return_type_contains_more_elements(_REAL_CONST)
    False

    """
    if not (isinstance(tp, ForallE) and tp.binder_info == BinderInfo.IMPLICIT
            and tp.binder_type == _NAT_CONST):
        return False
    lvl1 = tp.body
    if not (isinstance(lvl1, ForallE)
            and lvl1.binder_info == BinderInfo.DEFAULT and lvl1.binder_type
            == App(App(_VEC_CONST, _REAL_CONST), BVar(0))):
        return False
    lvl2 = lvl1.body
    if not (isinstance(lvl2, ForallE) and lvl2.binder_info
            == BinderInfo.DEFAULT and lvl2.binder_type == _REAL_CONST):
        return False
    ret = lvl2.body
    expected = App(App(_VEC_CONST, _REAL_CONST),
                   App(App(_NAT_ADD, BVar(2)),
                       Lit(1)))  # type: ignore[arg-type]
    return ret == expected


def flatten_loop_assigns(
        loop_body: list) -> Optional[List[List[Tuple[str, ASTNode]]]]:
    """
    Helper function to flatten loop body of ``loop_assign``/
    ``loop_tuple_unpack`` statements into ordered groups of
    ``(name, rhs_node)`` pairs.

    Parameters
    ----------
    loop_body : list
        A loop's statement list.

    Example
    -------
    >>> flatten_loop_assigns([("loop_assign", "x", ("num", 1.0))])
    [[('x', ('num', 1.0))]]
    >>> flatten_loop_assigns([("loop_if", None, [])]) is None
    True
    """
    groups: List[List[Tuple[str, ASTNode]]] = []
    for stmt in loop_body:
        if not (isinstance(stmt, tuple) and stmt):
            return None
        if stmt[0] == "loop_assign" and len(stmt) == 3:
            groups.append([(stmt[1], stmt[2])])
        elif stmt[0] == "loop_tuple_unpack" and len(stmt) == 3:
            names, expr_list_node = stmt[1], stmt[2]
            if not (isinstance(expr_list_node, tuple) and len(expr_list_node)
                    == 2 and expr_list_node[0] == "expr_list"
                    and len(expr_list_node[1]) == len(names)):
                return None
            groups.append(list(zip(names, expr_list_node[1])))
        else:
            return None
    return groups


def loop_body_assigned_names(loop_body: list) -> set:
    """
    Return a variable name that is being reassingned inside a loop body.


    Parameters
    ----------
    loop_body : list
        A loop's statement list.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import loop_body_assigned_names
    >>> loop_body_assigned_names([("loop_assign", "total", ("num", 1.0))])
    {'total'}
    """
    names = set()
    for stmt in loop_body or []:
        if not (isinstance(stmt, tuple) and stmt):
            continue
        tag = stmt[0]
        if tag in ("loop_assign", "loop_pluseq", "loop_index_assign_nd",
                   "loop_index_pluseq"):
            names.add(stmt[1])
        elif tag == "loop_if":
            names |= loop_body_assigned_names(stmt[2])
        elif tag == "loop_if_else":
            names |= loop_body_assigned_names(stmt[2])
            names |= loop_body_assigned_names(stmt[3])
        elif tag == "loop_for_range":
            names |= loop_body_assigned_names(stmt[4])
    return names


def body_stmts_assigned_names(stmts: list) -> set:
    """
    Return name of variables of function's body statement list that are
    reassinged.

    Parameters
    ----------
    stmts : list
        A function/branch body's statement list.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import body_stmts_assigned_names  # noqa: E501
    >>> body_stmts_assigned_names([("body_assign", "x", ("num", 1.0))])
    {'x'}
    """
    names = set()
    for stmt in stmts or []:
        if not (isinstance(stmt, tuple) and stmt):
            continue
        tag = stmt[0]
        if tag in ("body_decl", "body_assign", "body_zeros_decl",
                   "body_index_assign", "body_index_assign_nd"):
            names.add(stmt[1])
        elif tag == "body_if":
            names |= body_stmts_assigned_names(stmt[2])
        elif tag == "body_if_else":
            names |= body_stmts_assigned_names(stmt[2])
            names |= body_stmts_assigned_names(stmt[3])
        elif tag in ("body_for", "body_for_accum", "body_for_map"):
            names |= loop_body_assigned_names(stmt[2])
        elif tag == "body_for_range":
            names |= loop_body_assigned_names(stmt[4])
    return names


def collect_calls_to(target_names: set, body_node,
                     statements: list) -> List[Tuple[str, list]]:
    """
    Return ``("call", name, args)`` nodes from ``body_node`` and
    ``statements``. Used to find self and mutual recursive calls for
    termination checking.

    Parameters
    ----------
    target_names : set
        Recursive callee name to look for.
    body_node :
        Function's recursive expression as AST Node.
    statements : list
        A function's statement list.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import collect_calls_to
    >>> collect_calls_to({"f"}, ("call", "f", [("var", "x")]), [])
    [('f', [('var', 'x')])]
    """
    found = []

    def walk(node: object) -> None:
        """
        Recurse into node and append ``call`` to target_names.

        Parameters
        ----------
        node : ASTNode
            AST node (tuple, list, or leaf) to search for recursive calls.
        """
        if isinstance(node, tuple) and node:
            if node[0] == "call" and node[1] in target_names:
                found.append((node[1], node[2] if len(node) > 2 else []))
            for child in node[1:]:
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body_node)
    for stmt in statements or []:
        walk(stmt)
    return found


def arg_decreases(arg: Tuple, param_name: str, param_type: str) -> bool:
    """
    Checks if ``arg`` is syntactic shape ``param_name - positive_constant``
    . A int: ``ℕ` parameter requires constant to be 1. A ``ℝ`` typed
    parameter allows any positive constant.

    Parameters
    ----------
    arg : Tuple
        Recursive call's argument.
    param_name : str
        Recursive parameter's name.
    param_type : str
        Recursive parameter's declared type (``"ℕ"``, ``"ℝ"``, ...).

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import arg_decreases
    >>> arg_decreases(("sub", ("var", "n"), ("num", 1.0)), "n", "ℕ")
    True
    >>> arg_decreases(("sub", ("var", "n"), ("num", 2.0)), "n", "ℕ")
    False
    """
    if not (isinstance(arg, tuple) and len(arg) == 3 and arg[0] == "sub"
            and isinstance(arg[1], tuple) and arg[1][0] == "var" and arg[1][1]
            == param_name and isinstance(arg[2], tuple) and arg[2][0] == "num"
            and isinstance(arg[2][1], (int, float)) and arg[2][1] > 0):
        return False
    if param_type == "ℕ":
        return arg[2][1] == 1
    return True


def safe_infer_type(elab: ElabT,
                    cic: Expr,
                    whnf_reduce: bool = False) -> Optional[Expr]:
    """
    Infer type of CIC term during elaboration. Repeated across
    ``elab_body_*`` handlers and helpers.

    Parameters
    ----------
    elab : Elab
        Elaborator object where ``infer_type`` resolves ``cic``.
    cic : Expr
        Elaborated CIC term to infer the type of.
    whnf_reduce : bool
        Whether to additionally whnf-reduce the inferred type.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import safe_infer_type
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> safe_infer_type(elab, Lit(1.0))
    Const(name='Real', levels=())
    """
    try:
        t = elab.infer_type(cic)
        return whnf(t, elab.state.env, elab.state.lctx,
                    elab.state.mctx) if whnf_reduce else t
    except Exception:
        return None


def bind_local(name: str, value_cic: Expr, value_type: Expr,
               cur_env: Dict[str, Expr], cur_elab: ElabT,
               local_decls: List[Tuple[str, Expr]],
               fvar_names: Dict[FVarId, str]) -> Tuple[Dict[str, Expr], ElabT]:
    """
    Bind ``name`` to a fresh FVar of ``value_type``.
    ``local_decls``/``fvar_names`` are mutated in place. ``cur_env`` and
    ``cur_elab`` are reassigned and returned.

    Parameters
    ----------
    name : str
        Name being bound.
    value_cic : Expr
        Its already-elaborated CIC value.
    value_type : Expr
        Its type — the fresh FVar is minted at this type.
    cur_env : Dict[str, Expr]
        Name → CIC value map to rebind ``name`` in.
    cur_elab : Elab
        Elaborator to mint the fresh FVar in.
    local_decls : List[Tuple[str, Expr]]
        Mutated in place with ``(name, value_cic)``.
    fvar_names : Dict[FVarId, str]
        Mutated in place with the new FVar's id → ``name``.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import bind_local
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> decls, names = [], {}
    >>> env, elab = bind_local("x", Lit(1.0), _REAL_CONST, {}, elab, decls, names)  # noqa: E501
    >>> decls
    [('x', Lit(val=1.0))]
    >>> list(env)
    ['x']
    """
    local_decls.append((name, value_cic))
    cur_elab, fv = cur_elab.with_local(name, value_type)
    cur_env = {**cur_env, name: fv}
    fvar_names[fv.id] = name
    return cur_env, cur_elab


def bind_declared_local(name: str, decl_type: Expr, rhs_cic: Expr,
                        cur_env: Dict[str, Expr], cur_elab: ElabT,
                        local_decls: List[Tuple[str, Expr]],
                        fvar_names: Dict[FVarId, str], errors: List[str],
                        context_label: str) -> Tuple[Dict[str, Expr], ElabT]:
    """
    Bind ``name`` at its declared ``decl_type``, unconditionally. A unify
    mismatch between ``rhs_cic``'s inferred type and
    ``decl_type`` is reported via ``errors``.

    Parameters
    ----------
    name : str
        Name being bound.
    decl_type : Expr
        Its declared type — the fresh FVar is minted at this type.
    rhs_cic : Expr
        Its already-elaborated CIC value, checked (not required) against
        ``decl_type``.
    cur_env : Dict[str, Expr]
        Name → CIC value map to rebind ``name`` in.
    cur_elab : Elab
        Elaborator used to infer/unify ``rhs_cic``'s type and mint the
        fresh FVar.
    local_decls : List[Tuple[str, Expr]]
        Mutated in place with ``(name, rhs_cic)``.
    fvar_names : Dict[FVarId, str]
        Mutated in place with the new FVar's id → ``name``.
    errors : List[str]
        Diagnostics list a type mismatch is appended to.
    context_label : str
        Location prefix for any appended diagnostic.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import bind_declared_local
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> decls, names, errors = [], {}, []
    >>> env, elab = bind_declared_local(
    ...     "x", _REAL_CONST, Lit(1.0), {}, elab, decls, names, errors, "f")
    >>> errors
    []
    >>> list(env)
    ['x']
    """
    if not isinstance(rhs_cic, MVar):
        try:
            rhs_type = cur_elab.infer_type(rhs_cic)
            if not cur_elab.unify(rhs_type, decl_type):
                from physika.core.elab.body_elab import error_type_str
                exp_s = error_type_str(decl_type, cur_elab)
                got_s = error_type_str(rhs_type, cur_elab)
                errors.append(f"{context_label}: variable '{name}' declared "
                              f"{exp_s} but assigned value of type {got_s}")
        except Exception:
            pass
    return bind_local(name, rhs_cic, decl_type, cur_env, cur_elab, local_decls,
                      fvar_names)


def invalidate_local(name: str, cur_env: Dict[str, Expr],
                     cur_elab: ElabT) -> Dict[str, Expr]:
    """
    If ``name`` is bound to an FVar, rebind it to a new synthetic MVar of
    that FVar's declared type. ``cur_env`` is unchanged if ``name`` isn't
    bound to an FVar.

    Parameters
    ----------
    name: str
    cur_env: Dict[str, Expr]
    cur_elab: Elab

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import invalidate_local
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import MVar
    >>> elab = Elab(Environment())
    >>> elab, x_fv = elab.with_local("x", _REAL_CONST)
    >>> isinstance(invalidate_local("x", {"x": x_fv}, elab)["x"], MVar)
    True
    >>> invalidate_local("y", {}, elab)
    {}
    """
    prev = cur_env.get(name)
    if not isinstance(prev, FVar):
        return cur_env
    prev_decl = cur_elab.state.lctx.find(prev.id)
    prev_type = prev_decl.type if prev_decl is not None else _REAL_CONST
    return {
        **cur_env, name:
        cur_elab.new_mvar(f"_unelaborated_{name}",
                          prev_type,
                          kind=MetaVarKind.SYNTHETIC)
    }


def lit_to_real(cic: Expr, expected_type: Expr) -> Expr:
    """
    Converts a integer literal to ``Real`` when ``expected_type`` is Real
    following Lean 4 implementation.

    Parameters
    ----------
    cic : Expr
        Elaborated CIC term for value being coerced.
    expected_type : Expr
        target declared/inferred type.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import lit_to_real
    >>> from physika.core.expr import Lit, App, Const, Proj
    >>> from physika.core.elab.dim_typespec import _REAL_CONST, _NAT_CONST
    >>> lit_to_real(Lit(0), _REAL_CONST)
    Lit(val=0.0)
    >>> lit_to_real(Lit(0), _NAT_CONST)
    Lit(val=0)
    >>> ofnat_zero = Proj("OfNat", 0, App(Const("instOfNatNat", ()), Lit(0)))
    >>> lit_to_real(ofnat_zero, _REAL_CONST)  # noqa: E501
    Proj(type_name='OfNat', idx=0, expr=App(func=Const(name='instOfNatReal', levels=()), arg=Lit(val=0)))
    >>> lit_to_real(ofnat_zero, _NAT_CONST) is ofnat_zero
    True
    """
    if expected_type != _REAL_CONST:
        return cic
    if (isinstance(cic, Lit) and isinstance(cic.val, int)
            and not isinstance(cic.val, bool)):
        return Lit(float(cic.val))  # type: ignore[arg-type]
    if (isinstance(cic, Proj) and cic.type_name == "OfNat" and cic.idx == 0
            and isinstance(cic.expr, App) and isinstance(cic.expr.func, Const)
            and cic.expr.func.name == "instOfNatNat"
            and isinstance(cic.expr.arg, Lit)
            and isinstance(cic.expr.arg.val, int)):
        return Proj("OfNat", 0, App(Const("instOfNatReal", ()), cic.expr.arg))
    return cic


def vec_len(t: Expr) -> Optional[Expr]:
    """
    Return length expression ``n`` if ``t`` is ``Vec α n``, else ``None``.


    Parameters
    ----------
    t : Expr
        whnf-reduced CIC type.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import vec_len
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> vec_len(vec_ty)
    Lit(val=3)
    >>> vec_len(_REAL_CONST) is None
    True
    """
    if not isinstance(t, App):
        return None
    head, args = get_app_fn_args(t)
    if isinstance(head, Const) and head.name == "Vec" and len(args) == 2:
        return args[1]
    return None


def vec_elem_type(t: Expr) -> Optional[Expr]:
    """
    Return element type ``α`` if ``t`` is ``Vec α n``, else ``None``.

    Needed since ``Vec.get`` and ``Vec.tabulate`` inductive types
    are polymorphic over element type. A nested ``Vec``, like a matrix
    ``Vec (Vec Real n) m``, needs element type itself to be
    Vec-typed, not just Real. ``t`` must already be reduced to whnf by
    caller.

    Parameters
    ----------
    t : Expr
        A whnf-reduced CIC type.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import vec_elem_type
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> vec_elem_type(vec_ty)
    Const(name='Real', levels=())
    >>> vec_elem_type(_REAL_CONST) is None
    True
    """
    if not isinstance(t, App):
        return None
    head, args = get_app_fn_args(t)
    if isinstance(head, Const) and head.name == "Vec" and len(args) == 2:
        return args[0]
    return None


def build_ite_term(cond_cic: Expr, then_cic: Expr, else_cic: Expr,
                   then_type: Optional[Expr]) -> Expr:
    """
    Build a ``Nat.ite``/``Real.ite``/``Vec.ite`` application from an
    elaborated condition and branch values.

    Use ``ite`` variant matching branches' own type
    (``then_type``, in whnf form, ``None`` if it couldn't be
    inferred, in which case ``Real.ite`` is default).
    ``Vec.ite`` (``∀ (n:Nat), Bool → Vec Real n → Vec Real n → Vec Real n``)
    needs shared length ``n`` as an explicit leading argument.
    ``Nat.ite``/``Real.ite`` don't since these are monomorphic.

    Parameters
    ----------
    cond_cic : Expr
        Elaborated ``Bool`` condition term.
    then_cic : Expr
        Elaborated "then" branch value.
    else_cic : Expr
       Elaborated "else" branch value.
    then_type : Optional[Expr]
        then-branch's whnf-reduced type, or ``None`` if unknown.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import build_ite_term
    >>> from physika.core.expr import Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> cond = Const("true", ())
    >>> build_ite_term(cond, Lit(1.0), Lit(0.0), _REAL_CONST)  # noqa: E501
    App(func=App(func=App(func=Const(name='Real.ite', levels=()), arg=Const(name='true', levels=())), arg=Lit(val=1.0)), arg=Lit(val=0.0))
    """
    n_expr = vec_len(then_type) if then_type is not None else None
    if n_expr is not None:
        return App(
            App(App(App(Const("Vec.ite", ()), n_expr), cond_cic), then_cic),
            else_cic)
    ite_name = "Nat.ite" if then_type == _NAT_CONST else "Real.ite"
    return App(App(App(Const(ite_name, ()), cond_cic), then_cic), else_cic)


def mk_prod_chain(values: List[Expr], elab: ElabT) -> Optional[Expr]:
    """
    Build a right associative ``Prod.mk`` nesting over ``values``.

    Same shape ``elaborate_expr``'s ``tuple_return`` case builds
    (``Prod.mk t1 (Prod t2 t3...) v1 (Prod.mk v2 ...)``), factored out so
    ``try_elaborate_dependent_fold_loop`` can build same shape for a
    fold accumulator.

    Returns ``None`` if any value's type can't be
    inferred.

    Parameters
    ----------
    values : List[Expr]
        Elaborated CIC terms to nest, outermost-first.
    elab : Elab
        Elaborator used to infer each value's type.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import mk_prod_chain
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> mk_prod_chain([Lit(1.0), Lit(2.0)], elab)  # noqa: E501
    App(func=App(func=App(func=App(func=Const(name='Prod.mk', levels=()), arg=Const(name='Real', levels=())), arg=Const(name='Real', levels=())), arg=Lit(val=1.0)), arg=Lit(val=2.0))
    >>> mk_prod_chain([], elab) is None
    True
    """
    if not values:
        return None
    result = values[-1]
    for v in reversed(values[:-1]):
        try:
            t_v = elab.infer_type(v)
            t_result = elab.infer_type(result)
        except Exception:
            return None
        result = App(App(App(App(Const("Prod.mk", ()), t_v), t_result), v),
                     result)
    return result


def try_elaborate_dependent_fold_loop(loop_var: str, loop_body: list,
                                      n_cic: Expr, cur_env: dict, elab: ElabT,
                                      errors: List[str]) -> Optional[dict]:
    """
    A loop that reassigns bound locals together at each iteration, and elaborate
    via ``Nat.rec``.

    Some of reassigned locals may grow in shape (e.g. ``x = append(x,
    v)``). A same-type ``Vec.foldl`` can't represent an accumulator
    whose shape changes at every step, so this uses a Nat-indexed motive
    instead.

    Parameters
    ----------
    loop_var : str
        Loop's bound variable name.
    loop_body : list
        Loop's statement list.
    n_cic : Expr
        Elaborated CIC term for loop's iteration count.
    cur_env : dict
        Name to CIC value map in scope before loop.
    elab : Elab
        Elaborator lctx/mctx loop
    errors : List[str]
        List to append any elaboration errors raised while building the
        step function's body.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import try_elaborate_dependent_fold_loop  # noqa: E501
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> try_elaborate_dependent_fold_loop(
    ...     "k", [], Lit(3), {}, elab, []) is None
    True
    """
    from physika.core.elab.body_elab import elaborate_expr

    groups = flatten_loop_assigns(loop_body)
    if not groups:
        return None

    assigned_order: List[str] = []
    seen = set()
    for group in groups:
        for name, _ in group:
            if name not in seen:
                assigned_order.append(name)
                seen.add(name)
    carried = [n for n in assigned_order if isinstance(cur_env.get(n), FVar)]
    if not carried:
        return None

    growth_dim0: dict = {}
    growth_elem: dict = {}
    fixed_type: dict = {}
    for name in carried:
        prev_fv = cur_env[name]
        prev_decl = elab.state.lctx.find(prev_fv.id)
        if prev_decl is None:
            return None
        prev_type = prev_decl.type
        rhs_node = next(r for group in groups for n, r in group if n == name)
        is_growth = False
        if (isinstance(rhs_node, tuple) and len(rhs_node) == 3
                and rhs_node[0] == "call"):
            callee, call_args = rhs_node[1], rhs_node[2]
            if (call_args and isinstance(call_args[0], tuple)
                    and call_args[0] == ("var", name)):
                callee_ci = elab.state.env.constants.get(callee)
                if callee_ci is not None and return_type_contains_more_elements(  # noqa: E501
                        callee_ci.type):
                    is_growth = True
        if is_growth:
            try:
                prev_type_w = whnf(prev_type, elab.state.env, elab.state.lctx,
                                   elab.state.mctx)
            except Exception:
                return None
            l0 = vec_len(prev_type_w)
            elem = vec_elem_type(prev_type_w)
            if l0 is None or elem is None:
                return None
            growth_dim0[name] = l0
            growth_elem[name] = elem
        else:
            fixed_type[name] = prev_type

    def type_at(name: str, k_expr: Expr) -> Expr:
        """
        Nat.add/Nat.rec/Nat.sub all recurse on their *second*
        argument, so only `Nat.add(var, const)` reduces when variable
        side is free.

        Parameters
        ----------
        """
        if name in growth_dim0:

            dim = App(App(_NAT_ADD, k_expr), growth_dim0[name])
            return App(App(_VEC_CONST, growth_elem[name]), dim)
        return fixed_type[name]

    def build_prod_type(k_expr: Expr) -> Expr:
        """
        Elaborate CIC term for Prod type. Type-level counterpart of
        ``mk_prod_chain``'s value nesting — the tupled accumulator's
        type after ``k_expr`` iterations, one ``type_at`` per carried
        local.

        Parameters
        ----------
        k_expr : Expr
            Nat.rec step count so far.
        """
        types = [type_at(n, k_expr) for n in carried]
        result = types[-1]
        for t in reversed(types[:-1]):
            result = App(App(Const("Prod", ()), t), result)
        return result

    base = mk_prod_chain([cur_env[n] for n in carried], elab)
    if base is None:
        return None

    motive_elab, mk_fv = elab.with_local("k", _NAT_CONST)
    motive = motive_elab.state.lctx.mk_lambda([mk_fv], build_prod_type(mk_fv))

    step_elab, k_fv = elab.with_local(loop_var, _NAT_CONST)
    step_elab, acc_fv = step_elab.with_local("_acc", build_prod_type(k_fv))
    all_fvars = [k_fv, acc_fv]

    comp_env = {**cur_env, loop_var: k_fv}
    remaining: Expr = acc_fv
    for idx, name in enumerate(carried):
        if idx < len(carried) - 1:
            proj_val: Expr = Proj("Prod", 0, remaining)
            next_remaining: Optional[Expr] = Proj("Prod", 1, remaining)
        else:
            proj_val = remaining
            next_remaining = None
        try:
            proj_type = step_elab.infer_type(proj_val)
        except Exception:
            return None
        proj_type = step_elab.state.mctx.instantiate_mvars(proj_type)
        step_elab, proj_fv = step_elab.with_let(name, proj_type, proj_val)
        all_fvars.append(proj_fv)
        comp_env[name] = proj_fv
        if next_remaining is not None:
            remaining = next_remaining

    for group in groups:
        computed = []
        for var_name, rhs_node in group:
            rhs_cic = elaborate_expr(rhs_node, comp_env, step_elab, errors)
            if isinstance(rhs_cic, MVar):
                return None
            try:
                rhs_type = step_elab.infer_type(rhs_cic)
            except Exception:
                return None

            rhs_cic = step_elab.state.mctx.instantiate_mvars(rhs_cic)
            rhs_type = step_elab.state.mctx.instantiate_mvars(rhs_type)
            computed.append((var_name, rhs_type, rhs_cic))
        for var_name, rhs_type, rhs_cic in computed:
            step_elab, new_fv = step_elab.with_let(var_name, rhs_type, rhs_cic)
            all_fvars.append(new_fv)
            comp_env[var_name] = new_fv

    new_acc = mk_prod_chain([comp_env[n] for n in carried], step_elab)
    if new_acc is None:
        return None
    step_fn = step_elab.state.lctx.mk_lambda(all_fvars, new_acc)

    rec_ref = Const("Nat.rec", (LSucc(LZero()), ))
    fold_result = App(App(App(App(rec_ref, motive), base), step_fn), n_cic)

    updates: dict = {}
    remaining = fold_result
    for idx, name in enumerate(carried):
        if idx < len(carried) - 1:
            updates[name] = Proj("Prod", 0, remaining)
            remaining = Proj("Prod", 1, remaining)
        else:
            updates[name] = remaining
    return updates


def fin_lit_chain(k: int, n: int) -> Expr:
    """
    Build ``Fin n`` value for the literal index ``k``.

    Emits the ``Fin.succ^k (Fin.zero)`` chain, with each constructor's
    implicit ``{m}`` supplied as ``Lit(m)``.

    Parameters
    ----------
    k : int
        Literal index (0-based), ``0 <= k < n``.
    n : int
        Vector length or ``Fin`` bound.

    Examples
    --------
    >>> from physika.utils.cic_utils.elab_utils import fin_lit_chain  # noqa: E501
    >>> fin_lit_chain(0, 3)  # index 0 in Fin 3
    App(func=Const(name='Fin.zero', levels=()), arg=Lit(val=2))
    >>> fin_lit_chain(1, 3)  # index 1 in Fin 3
    App(func=App(func=Const(name='Fin.succ', levels=()), arg=Lit(val=2)), arg=App(func=Const(name='Fin.zero', levels=()), arg=Lit(val=1)))
    """
    if k == 0:
        return App(Const("Fin.zero", ()), Lit(n - 1))  # type: ignore[arg-type]
    return App(
        App(Const("Fin.succ", ()), Lit(n - 1)),  # type: ignore[arg-type]
        fin_lit_chain(k - 1, n - 1))


def coerce_to_fin(idx_cic: Expr,
                  n_expr: Expr,
                  elab: ElabT,
                  errors: Optional[List[str]] = None) -> Expr:
    """
    A for loop's bound variable is Fin-typed (bound via ``with_local`` directly
    at type ``Fin n``), but a literal index (``results[0, 0]``) elaborates to
    ``Nat`` ``Lit``. ``Vec.get`` requires ``Fin n`` argument. No operation when
    ``idx_cic`` is ``Fin``-typed or its type can't be determined.

    A literal index with a known bound is emitted as ``Fin.succ``/``Fin.zero``
    chain that be checked through the kernel.

    Parameters
    ----------
    idx_cic : Expr
        Elaborated index term.
    n_expr : Expr
        Vec length, i.e. ``n`` in target ``Fin n``.
    elab : Elab
        Elaborator used to infer ``idx_cic``'s type.
    errors : Optional[List[str]]
        List of error messages when getting an index out of range case.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import coerce_to_fin
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> coerce_to_fin(Lit(0), Lit(3), elab)  # noqa: E501
    App(func=Const(name='Fin.zero', levels=()), arg=Lit(val=2))
    """
    try:
        idx_type = whnf(elab.infer_type(idx_cic), elab.state.env,
                        elab.state.lctx, elab.state.mctx)
    except Exception:
        return idx_cic
    head, _ = get_app_fn_args(idx_type)
    if isinstance(head, Const) and head.name == "Fin":
        return idx_cic

    # Literal index with bound should return Fin.succ/Fin.zero chain
    try:
        idx_w = whnf(idx_cic, elab.state.env, elab.state.lctx, elab.state.mctx)
        n_w = whnf(n_expr, elab.state.env, elab.state.lctx, elab.state.mctx)
    except Exception:
        idx_w, n_w = idx_cic, n_expr
    k = lit_nat_int(idx_w)
    n = lit_nat_int(n_w)
    if k is not None and n is not None:
        if 0 <= k < n:
            return fin_lit_chain(k, n)
        if errors is not None:
            errors.append(
                f"index {k} is out of range for a vector of length {n}")
        return elab.new_mvar("_fin_oob", _TYPE_0_LEVEL)

    return App(App(Const("Fin.ofNat", ()), n_expr), idx_cic)


def elaborate_binop(tag: str, a_cic: Expr, b_cic: Expr, elab: ElabT) -> Expr:
    """
    Elaborate a binary arithmetic operations. We need to choose between
    Real/Nat/Vec/Mat inductive types to elaborate into CIC term by inspecting
    each operand's inferred type.

    Implemented to support current Physika operations with same length Vec
    add/mul (``Vec.vadd``/``Vec.vmul``), mismatched-length Vec as
    ``Vec.concat``, Vec-scalar ``mul`` as ``Vec.scale``, matrix add/broadcast
    (``Mat.madd``/``Mat.add_scalar``/ ``Mat.concat_rows``), and basic Nat/Real
    arithmetic.

    A MVar is added when either operand is itself degraded, or operand types
    don't match any registered axiom.

    Parameters
    ----------
    tag : str
        One of ``"add"``, ``"sub"``, ``"mul"``, ``"div"``.
    a_cic : Expr
        Elaborated left operand.
    b_cic : Expr
        Elaborated right operand.
    elab : Elab
        Elaborator used to infer each operand's type.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import elaborate_binop
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> elaborate_binop("add", Lit(1.0), Lit(2.0), elab)  # noqa: E501
    App(func=App(func=Const(name='Real.add', levels=()), arg=Lit(val=1.0)), arg=Lit(val=2.0))
    """
    if isinstance(a_cic, MVar) or isinstance(b_cic, MVar):
        return elab.new_mvar("_" + tag, _TYPE_0_LEVEL)
    try:
        a_type = whnf(elab.infer_type(a_cic), elab.state.env, elab.state.lctx,
                      elab.state.mctx)
        b_type = whnf(elab.infer_type(b_cic), elab.state.env, elab.state.lctx,
                      elab.state.mctx)
    except Exception:
        return elab.new_mvar("_" + tag, _TYPE_0_LEVEL)

    a_len, b_len = vec_len(a_type), vec_len(b_type)

    # A matrix (ℝ[m,k], CIC-represented as Vec (Vec Real k) m) is a Vec
    # where element type is itself a Vec
    def _is_mat(vec_type: Expr) -> bool:
        elem = vec_elem_type(vec_type)
        if elem is None:
            return False
        try:
            elem_w = whnf(elem, elab.state.env, elab.state.lctx,
                          elab.state.mctx)
        except Exception:
            return False
        return vec_len(elem_w) is not None

    a_is_mat = a_len is not None and _is_mat(a_type)
    b_is_mat = b_len is not None and _is_mat(b_type)

    if tag == "add" and (a_is_mat or b_is_mat):
        if a_is_mat and b_is_mat:
            # a_is_mat/b_is_mat both required a_len/b_len is not None.
            assert a_len is not None and b_len is not None
            a_elem = vec_elem_type(a_type)
            assert a_elem is not None  # a_is_mat's own _is_mat check found it
            k_expr = vec_len(a_elem)
            assert k_expr is not None
            try:
                same_len, _ = is_def_eq(a_len, b_len, elab.state.env,
                                        elab.state.lctx, elab.state.mctx)
            except Exception:
                same_len = False
            if same_len:
                return App(
                    App(App(App(Const("Mat.madd", ()), a_len), k_expr), a_cic),
                    b_cic)
            # Non unifiable outer lengths, same inner k.
            return App(
                App(
                    App(App(App(Const("Mat.concat_rows", ()), a_len), b_len),
                        k_expr), a_cic), b_cic)
        if a_is_mat and b_len is None:
            assert a_len is not None
            # Matrix + scalar (e.g. `w1 @ z + b1`) — broadcast.
            if b_type == _NAT_CONST:
                b_cic = App(Const("Nat.toReal", ()), b_cic)
            a_elem = vec_elem_type(a_type)
            assert a_elem is not None
            k_expr = vec_len(a_elem)
            assert k_expr is not None
            return App(
                App(App(App(Const("Mat.add_scalar", ()), a_len), k_expr),
                    a_cic), b_cic)
        if b_is_mat and a_len is None:
            assert b_len is not None
            if a_type == _NAT_CONST:
                a_cic = App(Const("Nat.toReal", ()), a_cic)
            b_elem = vec_elem_type(b_type)
            assert b_elem is not None
            k_expr = vec_len(b_elem)
            assert k_expr is not None
            return App(
                App(App(App(Const("Mat.add_scalar", ()), b_len), k_expr),
                    b_cic), a_cic)

        # No registered axiom
        return elab.new_mvar("_" + tag, _TYPE_0_LEVEL)

    if a_len is None and b_len is None:
        if a_type == _NAT_CONST and b_type == _NAT_CONST:
            # Nat.div has no registered axiom
            # (matches mk_builtin_env only add/mul/sub are defined)
            op = {
                "add": "Nat.add",
                "sub": "Nat.sub",
                "mul": "Nat.mul"
            }.get(tag)
        else:
            # Mixed Nat/Real
            if a_type == _NAT_CONST:
                a_cic = App(Const("Nat.toReal", ()), a_cic)
            if b_type == _NAT_CONST:
                b_cic = App(Const("Nat.toReal", ()), b_cic)
            op = {
                "add": "Real.add",
                "sub": "Real.sub",
                "mul": "Real.mul",
                "div": "Real.div"
            }.get(tag)
        if op is None:
            return elab.new_mvar("_" + tag, _TYPE_0_LEVEL)
        return App(App(Const(op, ()), a_cic), b_cic)

    if a_len is not None and b_len is not None and tag == "add":
        try:
            same_len, _ = is_def_eq(a_len, b_len, elab.state.env,
                                    elab.state.lctx, elab.state.mctx)
        except Exception:
            same_len = False
        if not same_len:
            # Different length
            return App(
                App(App(App(Const("Vec.concat", ()), a_len), b_len), a_cic),
                b_cic)
        return App(App(App(Const("Vec.vadd", ()), a_len), a_cic), b_cic)
    if a_len is not None and b_len is not None and tag == "mul":
        return App(App(App(Const("Vec.vmul", ()), a_len), a_cic), b_cic)

    if tag == "mul" and a_len is not None and b_len is None:
        return App(App(App(Const("Vec.scale", ()), a_len), b_cic), a_cic)
    if tag == "mul" and b_len is not None and a_len is None:
        return App(App(App(Const("Vec.scale", ()), b_len), a_cic), b_cic)

    # No registered axiom for this shape combination
    return elab.new_mvar("_" + tag, _TYPE_0_LEVEL)


COND_OP_SUFFIX = {
    "cond_eq": "eqb",
    "cond_neq": "neb",
    "cond_lt": "ltb",
    "cond_gt": "gtb",
    "cond_leq": "leb",
    "cond_geq": "geb",
}


def elaborate_condition_bool(cond_node: Optional[tuple], fvar_env: Dict[str,
                                                                        Expr],
                             elab: ElabT, errors: List[str]) -> Expr:
    """
    Translate a boolean ``cond`` AST node (``if`` condition) to a
    ``Bool`` CIC term.

    Uses ``Nat.*b``/``Real.*b`` comparison axioms, chosen
    by left operand's actual inferred type. Degrades to a
    fresh ``Bool`` MVar if condition shape isn't one of six
    comparison operators, or operand type can't be resolved.

    Parameters
    ----------
    cond_node : Optional[tuple]
        A ``cond_eq``/``cond_neq``/``cond_lt``/``cond_gt``/``cond_leq``/
        ``cond_geq`` AST node, ``(tag, lhs, rhs)``.
    fvar_env : Dict[str, Expr]
        Names currently in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar information.
    errors : List[str]
        List forwarded to ``elaborate_expr`` for operands with current error messages.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import elaborate_condition_bool
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
    >>> elaborate_condition_bool(cond, {}, elab, [])  # noqa: E501
    App(func=App(func=Const(name='Real.ltb', levels=()), arg=Lit(val=1.0)), arg=Lit(val=2.0))
    """
    from physika.core.elab.body_elab import elaborate_expr

    if isinstance(cond_node,
                  tuple) and cond_node and cond_node[0] in COND_OP_SUFFIX:
        l_cic = elaborate_expr(cond_node[1], fvar_env, elab, errors)
        r_cic = elaborate_expr(cond_node[2], fvar_env, elab, errors)
        if not isinstance(l_cic, MVar) and not isinstance(r_cic, MVar):
            try:
                l_type = whnf(elab.infer_type(l_cic), elab.state.env,
                              elab.state.lctx, elab.state.mctx)
            except Exception:
                l_type = None
            prefix = "Nat" if l_type == _NAT_CONST else "Real"
            op = f"{prefix}.{COND_OP_SUFFIX[cond_node[0]]}"
            return App(App(Const(op, ()), l_cic), r_cic)
    return elab.new_mvar("_cond", Const("Bool", ()))


def elaborate_branch_values(stmts: list, base_env: Dict[str, Expr],
                            elab: ElabT, errors: List[str]) -> Dict[str, Expr]:
    """
    Elaborate nested ``if``/``if-else`` branch's statement list, returning a
    dict mapping every name in ``base_env`` to its final CIC value
    expression at end of branch.

    Parameters
    ----------
    stmts : list
        Branch's statement list.
    base_env : Dict[str, Expr]
        Name to CIC value map in scope before branch.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar information.
    errors : List[str]
        List passed to ``elaborate_expr``/``merge_branch_values``.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import elaborate_branch_values
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> stmts = [("body_assign", "y", ("num", 5.0))]
    >>> elaborate_branch_values(stmts, {"y": Lit(1.0)}, elab, [])
    {'y': Lit(val=5.0)}
    """
    from physika.core.elab.body_elab import elaborate_expr

    values = dict(base_env)
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        if not (isinstance(stmt, tuple) and stmt):
            i += 1
            continue
        tag = stmt[0]
        if tag in ("body_decl", "body_assign"):
            name = stmt[1]
            expr_node = stmt[3] if tag == "body_decl" else stmt[2]
            values[name] = elaborate_expr(expr_node, values, elab, errors)
        elif tag == "body_if_else":
            merged = merge_branch_values(stmt[1], stmt[2], stmt[3], values,
                                         elab, errors)
            values.update(merged)
        elif tag == "body_if":
            merged = merge_branch_values(stmt[1], stmt[2], None, values, elab,
                                         errors)
            values.update(merged)
        else:
            for name in body_stmts_assigned_names(stmts[i:]):
                if name in values:
                    values[name] = elab.new_mvar(f"_unelaborated_{name}",
                                                 _TYPE_0_LEVEL,
                                                 kind=MetaVarKind.SYNTHETIC)
            break
        i += 1
    return values


def merge_branch_values(cond_node: Optional[tuple], then_stmts: list,
                        else_stmts: Optional[list], base_env: Dict[str, Expr],
                        elab: ElabT, errors: List[str]) -> Dict[str, Expr]:
    """
    Elaborate an ``if``/``if-else`` as a Real.ite/Nat.ite
    conditional value.

    Parameters
    ----------
    cond_node : Optional[tuple]
        ``if``'s condition AST node, forwarded to ``elaborate_condition_bool``
    then_stmts : list
        "then" branch's statement list.
    else_stmts : Optional[list]
        "else" branch's statement list, or ``None`` for a ``if`` statement
        with no else.
    base_env : Dict[str, Expr]
        Name to CIC value map in scope before ``if``.
    elab : Elab
        Elaborator providing env/lctx/mctx and MVar information.
    errors : List[str]
        List forwarded to ``elaborate_branch_values``/
        ``elaborate_condition_bool``.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import merge_branch_values
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
    >>> then_stmts = [("body_assign", "y", ("num", 1.0))]
    >>> else_stmts = [("body_assign", "y", ("num", 2.0))]
    >>> merged = merge_branch_values(
    ...     cond, then_stmts, else_stmts, {"y": Lit(0.0)}, elab, [])
    >>> merged  # noqa: E501
    {'y': App(func=App(func=App(func=Const(name='Real.ite', levels=()), arg=App(func=App(func=Const(name='Real.ltb', levels=()), arg=Lit(val=1.0)), arg=Lit(val=2.0))), arg=Lit(val=1.0)), arg=Lit(val=2.0))}
    """
    cond_cic = elaborate_condition_bool(cond_node, base_env, elab, errors)
    then_values = elaborate_branch_values(then_stmts, base_env, elab, errors)
    else_values = (elaborate_branch_values(else_stmts, base_env, elab, errors)
                   if else_stmts is not None else base_env)
    touched = {
        name
        for name in base_env if then_values.get(name) is not base_env[name]
        or else_values.get(name) is not base_env[name]
    }
    if isinstance(cond_cic, MVar):
        return {
            name:
            elab.new_mvar(f"_unelaborated_{name}",
                          _TYPE_0_LEVEL,
                          kind=MetaVarKind.SYNTHETIC)
            for name in touched
        }
    merged: Dict[str, Expr] = {}
    for name in touched:
        t_val, e_val = then_values[name], else_values[name]
        if isinstance(t_val, MVar) or isinstance(e_val, MVar):
            merged[name] = elab.new_mvar(f"_unelaborated_{name}",
                                         _TYPE_0_LEVEL,
                                         kind=MetaVarKind.SYNTHETIC)
            continue
        try:
            t_type = whnf(elab.infer_type(t_val), elab.state.env,
                          elab.state.lctx, elab.state.mctx)
        except Exception:
            t_type = None
        try:
            e_type = whnf(elab.infer_type(e_val), elab.state.env,
                          elab.state.lctx, elab.state.mctx)
        except Exception:
            e_type = None

        if t_type == _REAL_CONST and e_type != _REAL_CONST:
            e_val = lit_to_real(e_val, t_type)
        elif e_type == _REAL_CONST and t_type != _REAL_CONST:
            t_val = lit_to_real(t_val, e_type)
            try:
                t_type = whnf(elab.infer_type(t_val), elab.state.env,
                              elab.state.lctx, elab.state.mctx)
            except Exception:
                pass
        merged[name] = build_ite_term(cond_cic, t_val, e_val, t_type)
    return merged


def elaborate_grad_call(arg_cics: List[Expr], elab: ElabT,
                        errors: List[str]) -> Expr:
    """
    A single ``grad`` Const can't have two different types.
    ''elaborate_grad_call'' picks one of two registered axioms matches its
    argument shapes at elaboration time.

    For example:
    ``grad : ℝl → ℝ → ℝ``

    Represents a scalar case type:
    ``grad(f(x0), x0)`` for ``f : ℝ → ℝ``

    Since all types where ℝ, then whe use ``Real`` axioms

    However, for a Jacobian example:
    ``Vec.grad : {n}{m} → Vec Real n → Vec Real m → Vec (Vec Real m) n``

    ``grad(f(x), x)`` for ``f : Vec Real m → Vec Real n``

    Both arguments are ``Vec Real _`` type, with ``n``/``m`` supplied
    explicitly from the shapes just inferred.

    For arity mismatch, finding a MVar argument, mismatched or unrecognized
    shapes, falls through to ``elaborate_call("grad", ...)`` path.

    Parameters
    ----------
    arg_cics : List[Expr]
        Elaborated ``grad(...)`` call arguments.
    elab : Elab
        Elaborator used to infer each argument's type.
    errors : List[str]
        List forwarded to ``elaborate_call`` on fallback.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import elaborate_grad_call
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import App, Const, Lit, FVar
    >>> from physika.utils.cic_utils.expr_utils import get_app_fn_args
    >>> elab = Elab(Environment())
    >>> vec_n = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> vec_m = App(App(Const("Vec", ()), _REAL_CONST), Lit(2))
    >>> elab, x_fv = elab.with_local("x", vec_n)
    >>> elab, y_fv = elab.with_local("y", vec_m)
    >>> # Both arguments are Vec Real _, so this routes to the Jacobian
    >>> # `Vec.grad` axiom (built directly, no registered `grad` needed).
    >>> head, args = get_app_fn_args(elaborate_grad_call([x_fv, y_fv], elab, []))  # noqa: E501
    >>> head
    Const(name='Vec.grad', levels=())
    >>> args[:2]
    [Lit(val=3), Lit(val=2)]
    >>> isinstance(args[2], FVar) and isinstance(args[3], FVar)
    True
    """
    if len(arg_cics) != 2 or any(isinstance(a, MVar) for a in arg_cics):
        return elaborate_call("grad", arg_cics, elab, errors)

    if (isinstance(arg_cics[0], Const)
            and elab.state.env.constants.get(arg_cics[0].name) is not None
            and arg_cics[0].name not in ("grad", "Vec.grad")):
        probe_errors: list = []
        applied = elaborate_call(arg_cics[0].name, [arg_cics[1]], elab,
                                 probe_errors)
        if not probe_errors and not isinstance(applied, MVar):
            arg_cics = [applied, arg_cics[1]]
    try:
        t0 = whnf(elab.infer_type(arg_cics[0]), elab.state.env,
                  elab.state.lctx, elab.state.mctx)
        t1 = whnf(elab.infer_type(arg_cics[1]), elab.state.env,
                  elab.state.lctx, elab.state.mctx)
    except Exception:
        return elaborate_call("grad", arg_cics, elab, errors)
    if t0 == _REAL_CONST and t1 == _REAL_CONST:
        return elaborate_call("grad", arg_cics, elab, errors)
    n_expr, n_elem = vec_len(t0), vec_elem_type(t0)
    m_expr, m_elem = vec_len(t1), vec_elem_type(t1)
    if (n_expr is not None and n_elem == _REAL_CONST and m_expr is not None
            and m_elem == _REAL_CONST):
        return App(
            App(App(App(Const("Vec.grad", ()), n_expr), m_expr), arg_cics[0]),
            arg_cics[1])
    return elaborate_call("grad", arg_cics, elab, errors)


def elaborate_call(func_name: str, explicit_args: List[Expr], elab: ElabT,
                   errors: List[str]) -> Expr:
    """
    Elaborate a function call with implicit arguments (dependent types) and
    type checking.

    IMPLICIT binders get a fresh MVar, which lets CIC infer dim variables by
    unification (like ``n`` in ``{n} → Vec Real n → …`` dependent type).
    Explicit binders are consumed one by one. Argument's CIC type
    is unified against expected binder type. A mismatch is
    recorded in ``errors``.

    Parameters
    ----------
    func_name : str
       Constant (or class) name being called.
    explicit_args : List[Expr]
        Elaborated explicit call arguments, in source order.
    elab : Elab
        Elaborator providing env/lctx/mctx, unification, and MVar information.
    errors : List[str]
        List to append error messages.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import elaborate_call
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment, ConstantInfo
    >>> from physika.core.expr import ForallE, Const, Lit
    >>> env = Environment()
    >>> real = Const("Real", ())
    >>> env.add_constant(ConstantInfo("double", (), ForallE("x", real, real), None))  # noqa: E501
    >>> elab = Elab(env)
    >>> elaborate_call("double", [Lit(1.0)], elab, [])
    App(func=Const(name='double', levels=()), arg=Lit(val=1.0))
    """
    from physika.core.elab.body_elab import error_type_str

    target_name = func_name
    ii = elab.state.env.inductives.get(func_name)
    if ii is not None and len(ii.decl.constructors) == 1:
        target_name = ii.decl.constructors[0].name
    func_ci = elab.state.env.constants.get(target_name)
    if func_ci is None:
        return elab.new_mvar(func_name + "_result", _TYPE_0_LEVEL)
    result: Expr = Const(target_name, ())
    current_type = func_ci.type
    remaining = list(explicit_args)
    for _ in range(32):
        cw = whnf(current_type, elab.state.env, elab.state.lctx,
                  elab.state.mctx)
        if not isinstance(cw, ForallE):
            break
        if cw.binder_info == BinderInfo.IMPLICIT:
            mv = elab.new_mvar(cw.binder_name, cw.binder_type)
            result = App(result, mv)
            current_type = instantiate1(cw.body, mv)
        elif remaining:
            arg = remaining.pop(0)
            arg = lit_to_real(arg, cw.binder_type)
            if not isinstance(arg, MVar):
                try:
                    arg_type = elab.infer_type(arg)
                    ok = elab.unify(arg_type, cw.binder_type)
                    if not ok:
                        exp_s = error_type_str(cw.binder_type, elab)
                        got_s = error_type_str(arg_type, elab)
                        errors.append(f"In call to '{func_name}': "
                                      f"expected {exp_s} but got {got_s}")
                except Exception:
                    pass
            result = App(result, arg)
            current_type = instantiate1(cw.body, arg)
        else:
            break
    return result


def find_accum_loop_index(stmts: list, start: int,
                          var_name: str) -> Optional[int]:
    """
    Looks for acummulators in a declare-then-accumulate local's declaration
    (index ``start``) used in for loops (``for i: var_name += expr`).

    Parameters
    ----------
    stmts : list
        Statement list.
    start : int
        Index to begin scanning from.
    var_name : str
        Accumulator local's name.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import find_accum_loop_index
    >>> stmts = [
    ...     ("body_decl", "total", "ℝ", ("num", 0.0)),
    ...     ("body_decl", "temp", "ℝ", ("num", 0.0)),
    ...     ("body_for", "i", [("loop_pluseq", "total", ("var", "i"))],
    ...      ["arr"]),
    ... ]
    >>> find_accum_loop_index(stmts, 1, "total")
    2
    >>> find_accum_loop_index([("body_assign", "x", ("num", 1.0))], 0, "total") is None  # noqa: E501
    True
    """
    j = start
    while j < len(stmts):
        s = stmts[j]
        if not (isinstance(s, tuple) and s):
            return None
        if s[0] == "body_for":
            return j
        if s[0] in ("body_decl", "body_zeros_decl") and s[1] != var_name:
            j += 1
            continue
        return None
    return None


def infer_reduction_bound(node: Optional[tuple], red_var: str,
                          cur_env: Dict[str,
                                        Expr], elab: ElabT) -> Optional[Expr]:
    """
    Search ``node`` for an indexed-array read using ``red_var`` as one of its
    index positions, and return that array's own declared length at that
    position.

    For example:
    ``k`` in ``C[i,j] += A[i,k] * B[k,j]``

    A reduction loop variable (``k``) does not gets an explicit range unlike
    an output dimension. So its intended bound is recoverable from how it's
    used to read one of arrays on right-hand side.
    Returns ``None`` if no such use is found (or array's shape can't
    be resolved), leaving caller to decline whole construction.

    Parameters
    ----------
    node : Optional[tuple]
        A AST expression node to search.
    red_var : str
        Reduction loop variable to look for as an index.
    cur_env : Dict[str, Expr]
        Name to CIC value map used to resolve an indexed array's type.
    elab : Elab
        Elaborator used to infer each candidate array's type.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import infer_reduction_bound
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> elab, a_fv = elab.with_local("A", vec_ty)
    >>> node = ("index", "A", ("var", "k"))
    >>> infer_reduction_bound(node, "k", {"A": a_fv}, elab)
    Lit(val=3)
    """
    if not isinstance(node, tuple) or not node:
        return None
    tag = node[0]
    if tag in ("index", "indexN") and len(node) >= 3:
        arr_name = node[1]
        if tag == "index":
            idx_nodes = [node[2]]
        else:
            idx_nodes = [item[1] for item in node[2]]
        for pos, idx_node in enumerate(idx_nodes):
            name = None
            if isinstance(idx_node,
                          tuple) and idx_node and idx_node[0] == "var":
                name = idx_node[1]
            elif isinstance(idx_node,
                            tuple) and idx_node and idx_node[0] == "imaginary":
                name = "i"
            if name != red_var:
                continue
            arr_ref = cur_env.get(arr_name)
            if arr_ref is None:
                return None
            try:
                arr_type = whnf(elab.infer_type(arr_ref), elab.state.env,
                                elab.state.lctx, elab.state.mctx)
            except Exception:
                return None
            for _ in range(pos):
                elem_type = vec_elem_type(arr_type)
                if elem_type is None:
                    return None
                arr_type = whnf(elem_type, elab.state.env, elab.state.lctx,
                                elab.state.mctx)
            bound = vec_len(arr_type)
            if bound is not None:
                return bound
    for child in node[1:]:
        if isinstance(child, tuple):
            found = infer_reduction_bound(child, red_var, cur_env, elab)
            if found is not None:
                return found
        elif isinstance(child, list):
            for c in child:
                if isinstance(c, tuple):
                    found = infer_reduction_bound(c, red_var, cur_env, elab)
                    if found is not None:
                        return found
    return None


def try_elaborate_fold_loop(loop_var: str, loop_body: list, indexed_arrays,
                            cur_env: Dict[str, Expr], elab: ElabT,
                            errors: List[str]) -> Optional[Expr]:
    """
    Elaborate a single statement reassignment loop ``for k: x = expr``
    where ``expr`` may reference both ``k`` and pre-loop value of
    ``x``. This expression is elaborated, following Lean 4 approach, as:
    ``Vec.foldl α n (fun k acc => expr[x := acc]) x_initial``.

    This elaboration represents a sequential fold. Each iteration result
    depends on previous iterations.

    Parameters
    ----------
    loop_var : str
        Loop's own bound variable name.
    loop_body : list
        Loop's statement list.
    indexed_arrays : list
        Array names read in loop.
    cur_env : Dict[str, Expr]
        Name to CIC value map in scope before loop.
    elab : Elab
        Elaborator providing env/lctx/mctx and FVar information.
    errors : List[str]
        List forwarded to ``elaborate_expr``.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import try_elaborate_fold_loop
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.utils.cic_utils.expr_utils import get_app_fn_args
    >>> elab = Elab(Environment())
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(2))
    >>> elab, x_fv = elab.with_local("x", _REAL_CONST)
    >>> elab, arr_fv = elab.with_local("arr", vec_ty)
    >>> loop_body = [("loop_assign", "x", ("num", 1.0))]
    >>> cur_env = {"x": x_fv, "arr": arr_fv}
    >>> result = try_elaborate_fold_loop(
    ...     "k", loop_body, ["arr"], cur_env, elab, [])
    >>> head, args = get_app_fn_args(result)
    >>> head
    Const(name='Vec.foldl', levels=())
    >>> args[-1] is x_fv
    True
    """
    from physika.core.elab.body_elab import elaborate_expr

    if not (isinstance(loop_body, list) and len(loop_body) == 1):
        return None
    inner = loop_body[0]
    if not (isinstance(inner, tuple) and len(inner) == 3
            and inner[0] == "loop_assign"):
        return None
    var_name, rhs_node = inner[1], inner[2]
    prev_fv = cur_env.get(var_name)
    if not isinstance(prev_fv, FVar):
        return None
    prev_decl = elab.state.lctx.find(prev_fv.id)
    if prev_decl is None:
        return None
    alpha = prev_decl.type
    n_cic = None
    for arr_name in indexed_arrays:
        if arr_name in cur_env:
            try:
                arr_type = whnf(elab.infer_type(cur_env[arr_name]),
                                elab.state.env, elab.state.lctx,
                                elab.state.mctx)
                n_cic = vec_len(arr_type)
                if n_cic is not None:
                    break
            except Exception:
                continue
    if n_cic is None:
        return None
    fin_n = App(Const("Fin", ()), n_cic)
    child_elab, k_fv = elab.with_local(loop_var, fin_n)
    grandchild_elab, acc_fv = child_elab.with_local(var_name, alpha)
    child_env = {**cur_env, loop_var: k_fv, var_name: acc_fv}
    body_cic = elaborate_expr(rhs_node, child_env, grandchild_elab, errors)
    if isinstance(body_cic, MVar):
        return None

    fn = grandchild_elab.state.lctx.mk_lambda([k_fv, acc_fv], body_cic)
    return App(App(App(App(Const("Vec.foldl", ()), alpha), n_cic), fn),
               prev_fv)


def try_elaborate_index_write_loop(
        loop_var: str, loop_body: list, cur_env: Dict[str, Expr], elab: ElabT,
        errors: List[str]) -> Optional[Tuple[str, Expr]]:
    """
    Similar to ``try_elaborate_fold_loop``, elaborates ``for i: arr[i] = expr``
     where ``expr`` does not reference ``arr`` itself as:
    ``arr := Vec.tabulate n (fun i => expr)``.

    Parameters
    ----------
    loop_var : str
        Loop's own bound variable name.
    loop_body : list
        Loop's statement list.
    cur_env : Dict[str, Expr]
        Name to CIC map before loop.
    elab : Elab
        Elaborator providing env/lctx/mctx and FVar information.
    errors : List[str]
        List passed to ``elaborate_expr`` for adding error messages.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import try_elaborate_index_write_loop  # noqa: E501
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.utils.cic_utils.expr_utils import get_app_fn_args
    >>> elab = Elab(Environment())
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> elab, arr_fv = elab.with_local("arr", vec_ty)
    >>> elab, dst_fv = elab.with_local("dst", vec_ty)
    >>> # RHS is a plain literal (no arithmetic/indexing), so this needs
    >>> # no registered constants beyond the Vec/Fin/Real shape above.
    >>> loop_body = [("loop_index_assign_nd", "dst",
    ...               [("index_item", ("imaginary",))],
    ...               ("num", 2.0))]
    >>> name, result = try_elaborate_index_write_loop(
    ...     "i", loop_body, {"arr": arr_fv, "dst": dst_fv}, elab, [])
    >>> name
    'dst'
    >>> head, _ = get_app_fn_args(result)
    >>> head
    Const(name='Vec.tabulate', levels=())
    """
    from physika.core.elab.body_elab import elaborate_expr

    if not (isinstance(loop_body, list) and len(loop_body) == 1):
        return None
    inner = loop_body[0]
    if not (isinstance(inner, tuple) and len(inner) == 4
            and inner[0] == "loop_index_assign_nd"):
        return None
    array_name, index_items, rhs_node = inner[1], inner[2], inner[3]
    if not (isinstance(index_items, list) and len(index_items) == 1):
        return None
    item = index_items[0]
    if not (isinstance(item, tuple) and len(item) == 2
            and item[0] == "index_item"):
        return None
    idx = item[1]
    if not (isinstance(idx, tuple) and idx and idx[0] == "imaginary"):
        return None

    def references(node: object, name: str) -> bool:
        """
        Whether raw AST ``node`` reads ``name`` anywhere (as a ``var``/
        ``index``/``indexN`` target), walking tuples and lists.

        Parameters
        ----------
        node : object
            Raw (pre-elaboration) AST node or sub-node to search.
        name : str
            Variable name to look for.
        """
        if isinstance(node, tuple) and node:
            if (node[0] in ("var", "index", "indexN") and len(node) >= 2
                    and node[1] == name):
                return True
            return any(references(child, name) for child in node[1:])
        if isinstance(node, list):
            return any(references(child, name) for child in node)
        return False

    if references(rhs_node, array_name):
        return None

    prev = cur_env.get(array_name)
    if not isinstance(prev, FVar):
        return None
    try:
        arr_type = whnf(elab.infer_type(prev), elab.state.env, elab.state.lctx,
                        elab.state.mctx)
    except Exception:
        return None
    n_cic = vec_len(arr_type)
    elem_type = vec_elem_type(arr_type)
    if n_cic is None or elem_type is None:
        return None
    fin_n = App(Const("Fin", ()), n_cic)
    child_elab, i_fv = elab.with_local(loop_var, fin_n)
    child_env = {**cur_env, loop_var: i_fv}
    body_cic = elaborate_expr(rhs_node,
                              child_env,
                              child_elab,
                              errors,
                              expected_type=elem_type)
    if isinstance(body_cic, MVar):
        return None
    fn = child_elab.state.lctx.mk_lambda([i_fv], body_cic)
    result = App(App(App(Const("Vec.tabulate", ()), elem_type), n_cic), fn)
    return array_name, result


def invalidate_loop_locals(loop_body: list,
                           cur_env: Dict[str, Expr],
                           cur_elab: ElabT,
                           exclude: Optional[set] = None) -> Dict[str, Expr]:
    """
    Invalidate local that ``loop_body`` reassigns, replacing each with a new
    unresolved MVar.

    Parameters
    ----------
    loop_body : list
        Loop's statement list.
    cur_env : Dict[str, Expr]
        Name to CIC value map in scope before loop.
    cur_elab : Elab
        Elaborator to look up each mutated local's pre-loop type.
    exclude : Optional[set]
        Names to leave untouched even if ``loop_body`` reassigns them.
        Defaults to an empty set — not a mutable default argument since
        a fresh one is only ever built inside the function.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import invalidate_loop_locals
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import MVar
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> elab, total_fv = elab.with_local("total", _REAL_CONST)
    >>> loop_body = [("loop_assign", "total", ("num", 1.0))]
    >>> new_env = invalidate_loop_locals(
    ...     loop_body, {"total": total_fv}, elab)
    >>> isinstance(new_env["total"], MVar)
    True
    """
    exclude = exclude if exclude is not None else set()
    for mutated_name in loop_body_assigned_names(loop_body) - exclude:
        prev = cur_env.get(mutated_name)
        if not isinstance(prev, FVar):
            continue
        prev_decl = cur_elab.state.lctx.find(prev.id)
        prev_type = prev_decl.type if prev_decl is not None else _REAL_CONST
        cur_env = {
            **cur_env, mutated_name:
            cur_elab.new_mvar(f"_unelaborated_{mutated_name}",
                              prev_type,
                              kind=MetaVarKind.SYNTHETIC)
        }
    return cur_env


def try_elaborate_sum_accumulator(var_name: str, type_spec,
                                  next_stmt: Optional[tuple],
                                  fvar_env: Dict[str, Expr], elab: ElabT,
                                  errors: List[str]) -> Optional[Expr]:
    """
    Looks for ``result: ℝ`` followed by ``for i: result += expr`` and elaborate
    it as:
    ``Vec.sum n (Vec.tabulate n (fun i => expr))``

    loop body may contain other statements than ``result += expr`` which are
    ignored by using ``invalidate_loop_locals``.
    Returns ``None`` if shape or bound can't be determined.

    Parameters
    ----------
    var_name : str
        Accumulator local's name (``"result"``).
    type_spec : Union[str, tuple]
        ``var_name``'s declared type spec
    next_stmt : Optional[tuple]
        statement immediately following ``var_name``'s declaration
    fvar_env : Dict[str, Expr]
        Names currently in scope.
    elab : Elab
        Elaborator providing env/lctx/mctx and FVar information.
    errors : List[str]
        List forwarded to ``try_elaborate_accumulator_body``.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import try_elaborate_sum_accumulator  # noqa: E501
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(2))
    >>> elab, arr_fv = elab.with_local("arr", vec_ty)
    >>> next_stmt = ("body_for", "i",
    ...              [("loop_pluseq", "total", ("num", 1.0))], ["arr"])
    >>> result = try_elaborate_sum_accumulator(
    ...     "total", "ℝ", next_stmt, {"arr": arr_fv}, elab, [])
    >>> result
    App(func=App(func=Const(name='Vec.sum', levels=()), arg=Lit(val=2)), arg=App(func=App(func=App(func=Const(name='Vec.tabulate', levels=()), arg=Const(name='Real', levels=())), arg=Lit(val=2)), arg=Lam(binder_name='i', binder_type=App(func=Const(name='Fin', levels=()), arg=Lit(val=2)), body=Lit(val=1.0), binder_info=<BinderInfo.DEFAULT: 1>)))
    """
    if type_spec != "ℝ":
        return None
    if not (isinstance(next_stmt, tuple) and next_stmt
            and next_stmt[0] == "body_for"):
        return None
    _, loop_var, loop_body, indexed_arrays = next_stmt
    if not isinstance(loop_body, list):
        return None
    n_cic = None
    for arr_name in indexed_arrays:
        if arr_name in fvar_env:
            try:
                arr_type = whnf(elab.infer_type(fvar_env[arr_name]),
                                elab.state.env, elab.state.lctx,
                                elab.state.mctx)
                n_cic = vec_len(arr_type)
                if n_cic is not None:
                    break
            except Exception:
                continue
    if n_cic is None:
        return None
    fin_n = App(Const("Fin", ()), n_cic)
    child_elab, i_fv = elab.with_local(loop_var, fin_n)
    child_env = {**fvar_env, loop_var: i_fv}
    body_cic = try_elaborate_accumulator_body(loop_body, var_name, child_env,
                                              child_elab, errors)
    if body_cic is None:
        return None
    fn = child_elab.state.lctx.mk_lambda([i_fv], body_cic)
    vec_cic = App(App(App(Const("Vec.tabulate", ()), _REAL_CONST), n_cic), fn)
    return App(App(Const("Vec.sum", ()), n_cic), vec_cic)


def try_elaborate_accumulator_body(loop_body: list, var_name: str,
                                   child_env: Dict[str,
                                                   Expr], child_elab: ElabT,
                                   errors: List[str]) -> Optional[Expr]:
    """
    Elaborate a ``result += expr`` accumulator loop value.

    Recognizes three shapes for ``loop_body``. First, ``result += expr``
    summand is ``expr`` itself. Second, ``if cond: result += expr``
    (``loop_if``) the summand is ``Real.ite cond expr 0.0``. Finally,
    ``if cond: result += a else: result += b`` (``loop_if_else``) summand is
    ``Real.ite cond a b``.

    Parameters
    ----------
    loop_body : list
        Loop's statement list.
    var_name : str
        Accumulator local's name.
    child_env : Dict[str, Expr]
        Variable names in scope inside loop (including loop variable).
    child_elab : Elab
        Elaborator scoped to inside loop.
    errors : List[str]
        List passed to ``elaborate_expr``/``elaborate_condition_bool``.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import try_elaborate_accumulator_body  # noqa: E501
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> elab = Elab(Environment())
    >>> loop_body = [("loop_pluseq", "total", ("num", 1.0))]
    >>> try_elaborate_accumulator_body(loop_body, "total", {}, elab, [])
    Lit(val=1.0)
    """
    from physika.core.elab.body_elab import elaborate_expr

    def _single_pluseq_rhs(stmts: list) -> Optional[ASTNode]:
        matches = [
            s for s in stmts if isinstance(s, tuple) and len(s) == 3
            and s[0] == "loop_pluseq" and s[1] == var_name
        ]
        if len(matches) != 1:
            return None
        return matches[0][2]

    plain_rhs = _single_pluseq_rhs(loop_body)
    if plain_rhs is not None:
        rhs_cic = elaborate_expr(plain_rhs,
                                 child_env,
                                 child_elab,
                                 errors,
                                 expected_type=_REAL_CONST)
        if isinstance(rhs_cic, MVar):
            return None
        return rhs_cic

    if_stmts = [
        s for s in loop_body
        if isinstance(s, tuple) and s and s[0] in ("loop_if", "loop_if_else")
    ]
    if len(if_stmts) != 1 or len(loop_body) != 1:
        return None
    if_stmt = if_stmts[0]
    cond_node = if_stmt[1]
    cond_cic = elaborate_condition_bool(cond_node, child_env, child_elab,
                                        errors)
    if isinstance(cond_cic, MVar):
        return None
    then_rhs = _single_pluseq_rhs(if_stmt[2])
    if then_rhs is None:
        return None
    then_cic = elaborate_expr(then_rhs,
                              child_env,
                              child_elab,
                              errors,
                              expected_type=_REAL_CONST)
    if isinstance(then_cic, MVar):
        return None
    if if_stmt[0] == "loop_if":
        else_cic: Expr = Lit(0.0)  # type: ignore[arg-type]
    else:
        else_rhs = _single_pluseq_rhs(if_stmt[3])
        if else_rhs is None:
            return None
        else_cic = elaborate_expr(else_rhs,
                                  child_env,
                                  child_elab,
                                  errors,
                                  expected_type=_REAL_CONST)
        if isinstance(else_cic, MVar):
            return None
    return App(App(App(Const("Real.ite", ()), cond_cic), then_cic), else_cic)


def safe_whnf(elab: ElabT, t: Expr) -> Optional[Expr]:
    """
    whnf-reduce, returns ``None`` on failure.

    Reducing known type expression so it can be compared structurally with
    types at type infer.

    Parameters
    ----------
    elab : Elab
        Elaborator whose env/lctx/mctx resolve ``t``.
    t : Expr
        Type expression to reduce.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import safe_whnf
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> safe_whnf(elab, _REAL_CONST) == _REAL_CONST
    True
    """
    try:
        return whnf(t, elab.state.env, elab.state.lctx, elab.state.mctx)
    except Exception:
        return None


def resolve_name(name: str, fvar_env: Dict[str, Expr], elab: ElabT) -> Expr:
    """
    Resolve a var name to a CIC value: a local in ``fvar_env`` first,
    else a registered global constant, else a fresh opaque MVar.

    Parameters
    ----------
    name : str
        Name to resolve.
    fvar_env : Dict[str, Expr]
        Names currently in scope, checked first.
    elab : Elab
        Elaborator whose env supplies registered constants and mints
        the MVar fallback.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import resolve_name
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> elab = Elab(Environment())
    >>> elab, x_fv = elab.with_local("x", _REAL_CONST)
    >>> resolve_name("x", {"x": x_fv}, elab) is x_fv
    True
    """
    if name in fvar_env:
        return fvar_env[name]
    if elab.state.env.constants.get(name) is not None:
        return Const(name, ())
    return elab.new_mvar(name, _TYPE_0_LEVEL)


def expected_elem_type_of(elab: ElabT,
                          expected_type: Optional[Expr]) -> Optional[Expr]:
    """
    Looks for inner element types of Vec types.

    Parameters
    ----------
    elab : Elab
        Elaborator whose env/lctx/mctx resolve ``expected_type``.
    expected_type : Optional[Expr]
        Caller's known type, if any — peeled one Vec layer.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import expected_elem_type_of
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> from physika.core.expr import App, Const, Lit
    >>> elab = Elab(Environment())
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> expected_elem_type_of(elab, vec_ty) == _REAL_CONST
    True
    >>> expected_elem_type_of(elab, None) is None
    True
    """
    if expected_type is None:
        return None
    t = safe_whnf(elab, expected_type)
    return vec_elem_type(t) if t is not None else None


def head_const_of(elab: ElabT, cic: Expr) -> Optional[Const]:
    """
    Infer ``cic``'s type and return its applied head ``Const`` for a Physika
    class, if any else ``None``.

    Parameters
    ----------
    elab : Elab
        Elaborator used to infer ``cic``'s type.
    cic : Expr
        Already-elaborated CIC value to inspect.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import head_const_of
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> head_const_of(elab, Lit(1.0))
    Const(name='Real', levels=())
    """
    t = safe_infer_type(elab, cic, whnf_reduce=True)
    if t is None:
        return None
    head, _ = get_app_fn_args(t)
    return head if isinstance(head, Const) else None


def vec_shape_of(t: Expr) -> Tuple[Optional[Expr], Optional[Expr]]:
    """
    Return ``(n_expr, elem_type)`` for ``Vec elem_type
    n_expr`` type in whnf.

    Parameters
    ----------
    t : Expr
        Already whnf-reduced type to read the shape off.

    Example
    -------
    >>> from physika.utils.cic_utils.elab_utils import vec_shape_of
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.elab.dim_typespec import _REAL_CONST
    >>> vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
    >>> vec_shape_of(vec_ty)
    (Lit(val=3), Const(name='Real', levels=()))
    """
    return vec_len(t), vec_elem_type(t)
