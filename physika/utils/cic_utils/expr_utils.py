from physika.core.expr import (
    Expr,
    BVar,
    FVar,
    MVar,
    Sort,
    Const,
    App,
    Lam,
    ForallE,
    LetE,
    MData,
    Proj,
    MVarId,
)
from typing import Tuple, Optional, List
from physika.core.level import Level
from physika.utils.cic_utils.level_utils import instantiate_level_params
from physika.core.expr import BinderInfo


def get_app_fn(e: Expr) -> Expr:
    """
    Return the function at the top of an application (``App``) chain.

    Parameters
    ----------
    e : Expr
        An application chain. If ``e`` is not  an ``App`` instance,
        returns ``e`` as is.

    Returns
    -------
    Expr
        Head of the chain.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import get_app_fn
    >>> from physika.core.expr import App, Const, Lit
    >>> add = Const("Nat.add", ())
    >>> call = App(App(add, Lit(2)), Lit(3))  # equivalent to Nat.add 2 3
    >>> get_app_fn(call) == add
    True
    """
    while isinstance(e, App):
        e = e.func
    return e


def get_app_args(e: Expr) -> List[Expr]:
    """
    Return the arguments of an application chain from left to right.

    Parameters
    ----------
    e : Expr
        An application chain. If ``e`` is not ``App``, an empty list is
        returned.

    Returns
    -------
    List[Expr]
        Function's arguments.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import get_app_args
    >>> from physika.core.expr import App, Const, Lit
    >>> add = Const("Nat.add", ())
    >>> call = App(App(add, Lit(2)), Lit(3))
    >>> get_app_args(call)
    [Lit(val=2), Lit(val=3)]
    >>> get_app_args(add)
    []
    """
    args = []
    while isinstance(e, App):
        args.append(e.arg)
        e = e.func
    args.reverse()
    return args


def get_app_fn_args(e: Expr) -> Tuple[Expr, List[Expr]]:
    """
    Return both the head and the arguments of an application chain in
    one call. Is the combination of `get_app_fn` and `get_app_args`.

    Parameters
    ----------
    e : Expr
        An application chain.

    Returns
    -------
    Tuple[Expr, List[Expr]]
        Head of function and its arguments.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import get_app_fn_args
    >>> from physika.core.expr import App, Const, Lit
    >>> add = Const("Nat.add", ())
    >>> call = App(App(add, Lit(2)), Lit(3))
    >>> head, args = get_app_fn_args(call)
    >>> head == add
    True
    >>> args
    [Lit(val=2), Lit(val=3)]
    """
    return get_app_fn(e), get_app_args(e)


def abstract_fvars(expr: Expr, fvars: List[FVar]) -> Expr:
    """
    Replace ``fvars[i]`` with ``BVar(i)``, inspired by Lean 4 locally
    nameless approach.

    Using de Bruijn indices (``BVar``) in local contexts (e.g. inside
    a binder) can lead to index shifts each time a binder is accesed.
    In Physika, inspired by Lean 4, when opening a binder, its
    ``BVar(0)`` is transformed into a globally unique ``FVar``, the
    body is elaborated with that name (does not read de Bruijn
    indices). Once the binder is fully elaborated, we go back from
    ``FVar`` to the original ``BVar`` and  ``Lam``/``ForallE`` node.

    ``abstract_fvars` produces the ``FVar`` to ``BVar`` transformation.

    Parameters
    ----------
    expr : Expr
        Expression node referencing ``fvars``.
    fvars : List[FVar]
        Free variables to abstract.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import abstract_fvars  # noqa: E501
    >>> from physika.core.expr import FVar, FVarId, Const, App
    >>> x, y = FVar(FVarId("x.0")), FVar(FVarId("y.1"))
    >>> body = App(App(Const("Real.add", ()), x), y)
    >>> abstract_fvars(body, [x, y])
    App(func=App(func=Const(name='Real.add', levels=()), arg=BVar(idx=0)), arg=BVar(idx=1))
    """
    if not fvars:
        return expr
    ids = [fv.id.id for fv in fvars]
    return abstract(expr, ids, 0)


def abstract(e: Expr, fvar_ids: List[str], depth: int) -> Expr:
    """
    Convert a set of free variables (`FVar`) to bound variables (`BVar`.

    `abstract` recurse over `e` Expression tree and when an `FVar` is found (and
    its id is in `fvar_ids`), replace with `BVar(depth + i)`. `i` refers to the
    variable's position in the list where `FVar` was found.  `depth` is a counter
    of how many binder scopes (Lam/ForallE/LetE bodies) we are currently at.
    Other node type recurses into its children, threading depth through
    unchanged.

    Parameters
    ----------
    e: Expr
        Expression node to check for FVar occurrencies.
    fvar_ids: List[str]
        List of `FVar` id's as strings present in a given context
    depth: int
        Integer number that represents the current depth `abstract` is.

    Example
    -------
    >>> from physika.utils.cic_utils.expr_utils import abstract  # noqa: E501
    >>> from physika.core.expr import FVar, FVarId, Const, App, Lam, BinderInfo
    >>> x = FVar(FVarId("x.0"))
    >>> term = App(Const("Real.mul", ()), x)
    >>> abstract(term, ["x.0"], 0)
    App(func=Const(name='Real.mul', levels=()), arg=BVar(idx=0))
    >>> nested = Lam("y", Const("Real", ()), App(Const("Real.mul", ()), x), BinderInfo.DEFAULT)
    >>> abstract(nested, ["x.0"], 0)
    Lam(binder_name='y', binder_type=Const(name='Real', levels=()), body=App(func=Const(name='Real.mul', levels=()), arg=BVar(idx=1)), binder_info=<BinderInfo.DEFAULT: 1>)
    """
    if isinstance(e, FVar):
        try:
            i = fvar_ids.index(e.id.id)
            return BVar(depth + i)
        except ValueError:
            return e  # not in the list, keep as FVar
    elif isinstance(e, App):
        nf = abstract(e.func, fvar_ids, depth)
        na = abstract(e.arg, fvar_ids, depth)
        return App(nf, na) if (nf is not e.func or na is not e.arg) else e
    elif isinstance(e, Lam):
        nt = abstract(e.binder_type, fvar_ids, depth)
        nb = abstract(e.body, fvar_ids, depth + 1)
        return Lam(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, ForallE):
        nt = abstract(e.binder_type, fvar_ids, depth)
        nb = abstract(e.body, fvar_ids, depth + 1)
        return ForallE(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, LetE):
        nt = abstract(e.type, fvar_ids, depth)
        nv = abstract(e.value, fvar_ids, depth)
        nb = abstract(e.body, fvar_ids, depth + 1)
        return LetE(e.binder_name, nt, nv, nb, e.non_dep) if (
            nt is not e.type or nv is not e.value or nb is not e.body) else e
    elif isinstance(e, MData):
        ne = abstract(e.expr, fvar_ids, depth)
        return MData(e.kvs, ne) if ne is not e.expr else e
    elif isinstance(e, Proj):
        ne = abstract(e.expr, fvar_ids, depth)
        return Proj(e.type_name, e.idx, ne) if ne is not e.expr else e
    else:  # BVar, MVar, Sort, Const, Lit
        return e


def loose_bvar_range(e: Expr) -> int:
    """
    Determines the number of loose BVars in an expression.

    Loose variable refers to a variable that is not bound to any binder in the  # noqa: E501
    expression and need outer binders to be resolved.

    ``BVar(0)`` would have a loose variable range of 1, while ``BVar(1)``
    would have a loose variable range of 2. A closed expression have a
    loose variable range of 0, which normally refers to ``FVar, MVar,
    Sort, Const, Lit`` expressions.

    Parameters
    ----------
    e : Expr
        Expression to measure.

    Examples
    --------
    >>> from physika.core.expr import BVar, Lam, Const, BinderInfo
    >>> from physika.utils.cic_utils.expr_utils import loose_bvar_range
    >>> loose_bvar_range(BVar(0))
    1
    >>> id_fn = Lam("x", Const("Nat", ()), BVar(0), BinderInfo.DEFAULT)
    >>> loose_bvar_range(id_fn)
    0
    >>> open_under_one = Lam("x", Const("Nat", ()), BVar(1), BinderInfo.DEFAULT)
    >>> loose_bvar_range(open_under_one)
    1
    """
    if isinstance(e, BVar):
        return e.idx + 1
    elif isinstance(e, App):
        return max(loose_bvar_range(e.func), loose_bvar_range(e.arg))
    elif isinstance(e, (Lam, ForallE)):
        t_range = loose_bvar_range(e.binder_type)
        b_range = loose_bvar_range(e.body)
        return max(t_range, max(0, b_range - 1))
    elif isinstance(e, LetE):
        t_range = loose_bvar_range(e.type)
        v_range = loose_bvar_range(e.value)
        b_range = loose_bvar_range(e.body)
        return max(t_range, v_range, max(0, b_range - 1))
    elif isinstance(e, MData):
        return loose_bvar_range(e.expr)
    elif isinstance(e, Proj):
        return loose_bvar_range(e.expr)
    else:  # FVar, MVar, Sort, Const, Lit
        return 0


def has_mvar(e: Expr, mvar_id: Optional[MVarId] = None) -> bool:
    """
    True if e contains any MVar, or a specific one if mvar_id is given.

    Parameters
    ----------
    e : Expr
        Expression to search.
    mvar_id : Optional[MVarId], default None
        If given, only match this specific MVarId; otherwise match any
        MVar.

    Examples
    --------
    >>> from physika.core.expr import MVar, MVarId, Const, App
    >>> from physika.utils.cic_utils.expr_utils import has_mvar
    >>> mv = MVar(MVarId("m.0"))
    >>> has_mvar(App(Const("f", ()), mv))
    True
    >>> has_mvar(mv, MVarId("m.0"))
    True
    >>> has_mvar(mv, MVarId("n.0"))
    False
    """
    if isinstance(e, MVar):
        return mvar_id is None or e.id == mvar_id
    elif isinstance(e, App):
        return has_mvar(e.func, mvar_id) or has_mvar(e.arg, mvar_id)
    elif isinstance(e, (Lam, ForallE)):
        return has_mvar(e.binder_type, mvar_id) or has_mvar(e.body, mvar_id)
    elif isinstance(e, LetE):
        return (has_mvar(e.type, mvar_id) or has_mvar(e.value, mvar_id)
                or has_mvar(e.body, mvar_id))
    elif isinstance(e, MData):
        return has_mvar(e.expr, mvar_id)
    elif isinstance(e, Proj):
        return has_mvar(e.expr, mvar_id)
    return False


def lift(e: Expr, depth: int, n: int) -> Expr:
    """
    Increments ``BVar(k)`` to ``BVar(k+n)`` for every ``k >= depth`` in
    ``e``, recursing through binders with ``depth`` incremented by one
    each time is accesed.

    Used in ``substitute`` to push a substituted term (written in a
    outer scope) under the binders it's being placed inside,
    so its own loose ``BVar``s keep pointing at the same outer bindings.

    Parameters
    ----------
    e : Expr
        Expression to lift.
    depth : int
        ``BVar`` indices at or above this value are lifted.
    n : int
        Amount to shift ``BVar`` indices.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import lift  # noqa: E501
    >>> from physika.core.expr import BVar, Lam, Sort, BinderInfo
    >>> from physika.core.level import LZero
    >>> lift(BVar(0), 0, 2)
    BVar(idx=2)
    >>> lift(BVar(0), 1, 2)
    BVar(idx=0)
    >>> lam = Lam("x", Sort(LZero()), BVar(1), BinderInfo.DEFAULT)
    >>> lift(lam, 0, 1)
    Lam(binder_name='x', binder_type=Sort(level=LZero()), body=BVar(idx=2), binder_info=<BinderInfo.DEFAULT: 1>)
    """
    if n == 0:
        return e
    if isinstance(e, BVar):
        return BVar(e.idx + n) if e.idx >= depth else e
    elif isinstance(e, App):
        nf = lift(e.func, depth, n)
        na = lift(e.arg, depth, n)
        return App(nf, na) if (nf is not e.func or na is not e.arg) else e
    elif isinstance(e, Lam):
        nt = lift(e.binder_type, depth, n)
        nb = lift(e.body, depth + 1, n)
        return Lam(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, ForallE):
        nt = lift(e.binder_type, depth, n)
        nb = lift(e.body, depth + 1, n)
        return ForallE(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, LetE):
        nt = lift(e.type, depth, n)
        nv = lift(e.value, depth, n)
        nb = lift(e.body, depth + 1, n)
        return LetE(e.binder_name, nt, nv, nb, e.non_dep) if (
            nt is not e.type or nv is not e.value or nb is not e.body) else e
    elif isinstance(e, MData):
        ne = lift(e.expr, depth, n)
        return MData(e.kvs, ne) if ne is not e.expr else e
    elif isinstance(e, Proj):
        ne = lift(e.expr, depth, n)
        return Proj(e.type_name, e.idx, ne) if ne is not e.expr else e
    else:  # FVar, MVar, Sort, Const, Lit
        return e


def substitute(e: Expr, subst: List[Expr], depth: int) -> Expr:
    """
    Core substitution: replaces ``BVar(depth+i)`` with
    ``lift(subst[i], 0, depth)`` throughout ``e``.

    ``depth`` tracks how many binders have been entered during the
    recursion. ``depth`` increases by 1 each time a binder is crossed,
    so that ``BVar`` indices that were "loose" at the original call site stay
    correctly identified relative to the binders now surrounding them.
    Any ``BVar`` beyond the substitution range has its index shifted
    down by ``len(subst)`` to account for the binders being removed.

    Parameters
    ----------
    e : Expr
        Expression to substitute into.
    subst : List[Expr]
        Replacement terms. ``subst[i]`` replaces ``BVar(depth+i)``.
    depth : int
        Number of binders already crossed in the recursion so far

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import substitute
    >>> from physika.core.expr import BVar, Lit
    >>> substitute(BVar(0), [Lit(5)], 0)
    Lit(val=5)
    >>> substitute(BVar(1), [Lit(5)], 0)
    BVar(idx=0)
    """
    if isinstance(e, BVar):
        k = e.idx
        if k < depth:
            return e  # bound by an inner binder
        elif k - depth < len(subst):
            # Replace with the substituted term, lifted into the current depth.
            return lift(subst[k - depth], 0, depth)
        else:
            # BVar refers beyond the substitution range.
            # After removing len(subst) binders, indices shift down.
            return BVar(k - len(subst))
    elif isinstance(e, App):
        nf = substitute(e.func, subst, depth)
        na = substitute(e.arg, subst, depth)
        return App(nf, na) if (nf is not e.func or na is not e.arg) else e
    elif isinstance(e, Lam):
        nt = substitute(e.binder_type, subst, depth)
        nb = substitute(e.body, subst, depth + 1)
        return Lam(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, ForallE):
        nt = substitute(e.binder_type, subst, depth)
        nb = substitute(e.body, subst, depth + 1)
        return ForallE(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, LetE):
        nt = substitute(e.type, subst, depth)
        nv = substitute(e.value, subst, depth)
        nb = substitute(e.body, subst, depth + 1)
        return LetE(e.binder_name, nt, nv, nb, e.non_dep) if (
            nt is not e.type or nv is not e.value or nb is not e.body) else e
    elif isinstance(e, MData):
        ne = substitute(e.expr, subst, depth)
        return MData(e.kvs, ne) if ne is not e.expr else e
    elif isinstance(e, Proj):
        ne = substitute(e.expr, subst, depth)
        return Proj(e.type_name, e.idx, ne) if ne is not e.expr else e
    else:  # FVar, MVar, Sort, Const, Lit
        return e


def instantiate(body: Expr, subst: List[Expr]) -> Expr:
    """
    Replaces ``BVar(i)`` with ``subst[i]`` throughout ``body``.

    ``subst[0]`` replaces ``BVar(0)`` and ``BVar(k)``
    for ``k >= len(subst)`` is replaced by ``BVar(k - len(subst))``.
    Its index shifts down once the substituted binders are removed and is
    used when applying a function to its arguments or opening a let-binding.

    Parameters
    ----------
    body : Expr
        Expression to substitute into.
    subst : List[Expr]
        Replacement terms, innermost binder first.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import instantiate
    >>> from physika.core.expr import BVar, Lit
    >>> instantiate(BVar(0), [Lit(1), Lit(2)])
    Lit(val=1)
    >>> instantiate(BVar(2), [Lit(1), Lit(2)])
    BVar(idx=0)
    """
    if not subst:
        return body
    return substitute(body, subst, 0)


def instantiate1(body: Expr, s: Expr) -> Expr:
    """
    Replaces ``BVar(0)`` with ``s`` in ``body``. Standard
    beta-reduction step. ``(Lam x A body)`` applied to ``arg`` reduces
    to ``instantiate1(body, arg)``.

    Parameters
    ----------
    body : Expr
        Lambda/Forall/LetE body to substitute into.
    s : Expr
        Term to substitute for ``BVar(0)``.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import instantiate1
    >>> from physika.core.expr import BVar, Lit
    >>> instantiate1(BVar(0), Lit(7))
    Lit(val=7)
    """
    return substitute(body, [s], 0)


def instantiate_level_params_in_expr(e: Expr, params: List[str],
                                     levels: List[Level]) -> Expr:
    """
    Recursively replaces universe ``LParam`` nodes throughout ``e`` via
    ``instantiate_level_params`` on every ``Sort``/``Const`` level found.

    ``instantiate_level_params_in_expr`` is employ when instantiating a
    universe-polymorphic constant with concrete levels.

    Parameters
    ----------
    e : Expr
        Expression tree to walk and rewrite.
    params : List[str]
        Universe parameter names to replace.
    levels : List[Level]
        Levels where universe constants where instantiated.

    Examples
    --------
    >>> from physika.utils.cic_utils.expr_utils import instantiate_level_params_in_expr  # noqa: E501
    >>> from physika.core.expr import Sort
    >>> from physika.core.level import LParam, LSucc, LZero
    >>> instantiate_level_params_in_expr(Sort(LParam("u")), ["u"], [LSucc(LZero())])
    Sort(level=LSucc(pred=LZero()))
    """
    if not params:
        return e
    if isinstance(e, Sort):
        return Sort(instantiate_level_params(e.level, params, levels))
    elif isinstance(e, Const):
        return Const(e.name,
                     tuple(
                         instantiate_level_params(lvl, params, levels)
                         for lvl in e.levels))  # noqa: E501
    elif isinstance(e, App):
        nf = instantiate_level_params_in_expr(e.func, params, levels)
        na = instantiate_level_params_in_expr(e.arg, params, levels)
        return App(nf, na) if (nf is not e.func or na is not e.arg) else e
    elif isinstance(e, Lam):
        nt = instantiate_level_params_in_expr(e.binder_type, params, levels)
        nb = instantiate_level_params_in_expr(e.body, params, levels)
        return Lam(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, ForallE):
        nt = instantiate_level_params_in_expr(e.binder_type, params, levels)
        nb = instantiate_level_params_in_expr(e.body, params, levels)
        return ForallE(e.binder_name, nt, nb, e.binder_info) if (
            nt is not e.binder_type or nb is not e.body) else e
    elif isinstance(e, LetE):
        nt = instantiate_level_params_in_expr(e.type, params, levels)
        nv = instantiate_level_params_in_expr(e.value, params, levels)
        nb = instantiate_level_params_in_expr(e.body, params, levels)
        return LetE(e.binder_name, nt, nv, nb, e.non_dep) if (
            nt is not e.type or nv is not e.value or nb is not e.body) else e
    elif isinstance(e, MData):
        ne = instantiate_level_params_in_expr(e.expr, params, levels)
        return MData(e.kvs, ne) if ne is not e.expr else e
    elif isinstance(e, Proj):
        ne = instantiate_level_params_in_expr(e.expr, params, levels)
        return Proj(e.type_name, e.idx, ne) if ne is not e.expr else e
    else:  # BVar, FVar, MVar, Lit
        return e


def mk_arrow(domain: Expr, codomain: Expr) -> ForallE:
    """
    Build non-dependent function type ``domain -> codomain``.

    A ``ForallE`` with binder name (``"_"``) whose body does
    not mention the bound variable.

    ``codomain`` has no loose ``BVar`` (it is a closed term
    or contains only ``FVar``s). Since elaborator operates with FVar,
    mk_arrow holds during elaboration.

    Parameters
    ----------
    domain : Expr
        Argument type.
    codomain : Expr
        Result type, must not contain loose ``BVar``s.

    Examples
    --------
    >>> from physika.core.expr import Const
    >>> from physika.utils.cic_utils.expr_utils import mk_arrow
    >>> nat = Const("Nat", ())
    >>> mk_arrow(nat, nat)  # Nat -> Nat  (e.g. Nat.succ)  # noqa: E501
    ForallE(binder_name='_', binder_type=Const(name='Nat', levels=()), body=Const(name='Nat', levels=()), binder_info=<BinderInfo.DEFAULT: 1>)
    >>> # nests right: Nat -> Nat -> Nat
    >>> mk_arrow(nat, mk_arrow(nat, nat)).body.binder_name
    '_'
    """
    return ForallE("_", domain, codomain, BinderInfo.DEFAULT)
