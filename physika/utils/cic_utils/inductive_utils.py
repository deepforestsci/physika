from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from physika.core.expr import (Const, Expr, Lam, App, ForallE, MData, LetE,
                               Proj, BVar, FVar, BinderInfo, TYPE_0, Sort,
                               PROP)
from physika.core.level import LParam, LSucc, LZero
from physika.core.environment import ConstantInfo, Environment, InductiveInfo
from physika.core.inductive import (InductiveDecl, Recursor, RecursorRule,
                                    Constructor)
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.reduction import is_def_eq, whnf
from physika.core.kernel import check as kernel_check, KernelException
from physika.utils.cic_utils.expr_utils import (abstract_fvars,
                                                get_app_fn_args, instantiate,
                                                instantiate1, mk_arrow)


def name_appears(name: str, expr: Expr) -> bool:
    """
    Recursively checks if an inductive type ``Const(name)`` appears inside
    an `Expr` when declaring an inductive type.

    Parameters
    ----------
    name : str
        Name of inductive type to search.
    expr : Expr
        Expression node to check for an inductive type.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import name_appears
    >>> from physika.core.expr import Const, ForallE
    >>> nat = Const("Nat", ())
    >>> succ_type = ForallE("n", nat, nat)  # Nat -> Nat
    >>> name_appears("Nat", succ_type)
    True
    >>> name_appears("Bool", succ_type)
    False
    """
    if isinstance(expr, Const):
        return expr.name == name
    if isinstance(expr, App):
        # Check appearance of inductive type inside a function application
        return name_appears(name, expr.func) or name_appears(name, expr.arg)
    if isinstance(expr, (Lam, ForallE)):
        # Check the type of a binder and body
        # Lam/ForallE expressions are (arg : binder_type) => body.
        return (name_appears(name, expr.binder_type)
                or name_appears(name, expr.body))
    if isinstance(expr, LetE):
        # Checks declared type, bounded value, and body
        return (name_appears(name, expr.type)
                or name_appears(name, expr.value)
                or name_appears(name, expr.body))
    if isinstance(expr, (MData, Proj)):
        # checks if inductve type is inside MData or Proj sub epxressions
        return name_appears(name, expr.expr)
    return False  # Case expr is BVar, FVar, MVar, Sort, or Lit


def strict_positive_check(type_name: str, expr: Expr,
                          is_inductive_former: Callable[[str], bool]) -> bool:
    """
    A constructor is strictly positive for the inductive type T if T
    appears in positive (covariant) positions in each field type.

    Strict positivity checks that an inductive type is constructed correctly
    before it is elaborated or the kernel sees it. First, inductive type ``T``
    must not appear in expr. Second, an inductive type must not be inside the
    domain of an arrow type or function application. This imples anywhere
    inside the domain, no matter how deeply nested arrows are within it.
    In other words, the domain of a constructor's field cannot contain a self
    referenced inductive type, because this can lead to logical
    inconsistencies. Finally, since a constructor is a chain of fields (one
    ``ForallE`` arrow per field), this domain check is applied separately
    to each field along that chain.

    Negative occurrence example since ``Bad`` appears in the domain of the
    field:
        Bad : (Bad → Nat) → Bad #

    Parameters
    ----------
    type_name : str
        Name of the inductive type being checked.
    expr : Expr
        Expression node to check for a negative occurrence of
        ``type_name``.
    is_inductive_former : Callable[[str], bool]
        True an inductive type name being declared or already
        registered.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import strict_positive_check  # noqa: E501
    >>> from physika.core.expr import Const, ForallE, App
    >>> vec, alpha, n = Const("Vec", ()), Const("Real", ()), Const("n", ())
    >>> field = App(App(vec, alpha), n)  # Vec.cons's tl : Vec alpha n
    >>> strict_positive_check("Vec", field, lambda name: name == "Vec")
    True
    >>> bad, nat = Const("Bad", ()), Const("Nat", ())
    >>> domain = ForallE("_", bad, nat)  # Bad -> Nat, a constructor's field
    >>> strict_positive_check("Bad", domain, lambda name: name == "Bad")
    False
    """

    # 1) Inductive type (`type_name`) does not appear in `expr`.
    if not name_appears(type_name, expr):
        return True
    # 2) Arrow type: A → B
    # type_name must not appear in A (at any depth), then check B
    if isinstance(expr, ForallE):
        if name_appears(type_name, expr.binder_type):
            return False
        return strict_positive_check(type_name, expr.body, is_inductive_former)
    # 3) Each argument of a function application (App) is checked recursively
    head, args = get_app_fn_args(expr)
    if not (isinstance(head, Const) and is_inductive_former(head.name)):
        return False
    return all(
        strict_positive_check(type_name, a, is_inductive_former) for a in args)


def check_positivity_for_inductive(decl: "InductiveDecl",
                                   is_inductive_former=None) -> Optional[str]:
    """
    Strict positivity check for an inductive type declaration (InductiveDecl).
    This step is done before elaboration, when adding inductive types that will
    be used in `Environment`.


    Parameters
    ----------
    decl : InductiveDecl
        The inductive type being checked.
    is_inductive_former : Callable[[str], bool]
        True an inductive type name being declared or already
        registered.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import check_positivity_for_inductive # noqa: E501
    >>> from physika.core.inductive import InductiveDecl, Constructor
    >>> from physika.core.expr import Const, ForallE, TYPE_0
    >>> nat = Const("Nat", ())
    >>> nat_decl = InductiveDecl(
    ...     name="Nat", level_params=(), num_params=0, type=TYPE_0,
    ...     constructors=(Constructor("Nat.zero", nat),
    ...                   Constructor("Nat.succ", ForallE("n", nat, nat))),
    ...     is_recursive=True,
    ... )
    >>> check_positivity_for_inductive(nat_decl) is None
    True
    >>> bad = Const("Bad", ())
    >>> bad_ctor_type = ForallE("x", ForallE("_", bad, nat), bad)  # (Bad -> Nat) -> Bad
    >>> bad_decl = InductiveDecl(
    ...     name="Bad", level_params=(), num_params=0, type=TYPE_0,
    ...     constructors=(Constructor("Bad.bad", bad_ctor_type),),
    ...     is_recursive=True,
    ... )
    >>> check_positivity_for_inductive(bad_decl)
    "constructor 'Bad.bad' violates strict positivity: 'Bad' appears in a negative position in a field type"
    """
    if is_inductive_former is None:
        is_inductive_former = lambda name: name == decl.name  # noqa: E731
    for ctor in decl.constructors:
        tp = ctor.type
        for _ in range(decl.num_params):
            if isinstance(tp, ForallE):
                tp = tp.body
        while isinstance(tp, ForallE):
            field_type = tp.binder_type
            if not strict_positive_check(decl.name, field_type,
                                         is_inductive_former):
                return (
                    f"constructor '{ctor.name}' violates strict positivity: "
                    f"'{decl.name}' appears in a negative position in a field type"  # noqa: E501
                )
            tp = tp.body
    return None


NAT = Const("Nat", ())
BOOL = Const("Bool", ())
ZERO = Const("Nat.zero", ())
SUCC = Const("Nat.succ", ())
INT = Const("Int", ())


def self_reference_indices(type_name: str, param_fvars: List["FVar"],
                           field_type: Expr) -> Optional[List[Expr]]:
    """
    Classify a constructor field as a recursive occurrence of the
    inductive type being declared. If so, return the index expressions it
    carries.

    A field type is self referenced when the inductive ``T`` applied to
    parameter FVars, followed by ``K`` index
    expressions:

    ``T pf_0 .. pf_{P-1} j_0 .. j_{K-1}``

    A recursor needs these index to build the field's induction hypothesis and its
    ι-rule recursive call.

    Parameters
    ----------
    type_name : str
        Name of the inductive being declared (``decl.name``).
    param_fvars : List[FVar]
        Uniform parameter free variables.
    field_type : Expr
        Constructor field's type.


    Examples
    --------
    >>> from physika.core.expr import App, Const, FVar, FVarId
    >>> from physika.utils.cic_utils.inductive_utils import self_reference_indices  # noqa: E501
    >>> a = FVar(FVarId("a.0"))
    >>> n = FVar(FVarId("n.1"))
    >>> # Vec.cons's `tl : Vec a n`
    >>> self_reference_indices("Vec", [a], App(App(Const("Vec", ()), a), n))
    [FVar(id=FVarId(id='n.1'))]
    >>> # List.cons's `tl : List a` is recursive and dont have indices
    >>> self_reference_indices("List", [a], App(Const("List", ()), a))
    []
    """
    head, args = get_app_fn_args(field_type)
    if not (isinstance(head, Const) and head.name == type_name):
        return None
    p = len(param_fvars)
    if len(args) < p or list(args[:p]) != list(param_fvars):
        return None
    return list(args[p:])


def decl_is_prop_sorted(decl: "InductiveDecl") -> bool:
    """
    Indicates ifwhethe an inductive is of type ``Prop``.

    Clrears the parametes and index telescope from ``decl.type`` and checks
    if resulting sort is ``Sort 0`` (i.e. ``Prop``).

    Parameters
    ----------
    decl : InductiveDecl
        Inductive declaration to check on.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     decl_is_prop_sorted, mk_nat_decl)
    >>> decl_is_prop_sorted(mk_nat_decl())   # Nat : Type₀
    False
    """
    tp = decl.type
    while isinstance(tp, ForallE):
        tp = tp.body
    return isinstance(tp, Sort) and isinstance(tp.level, LZero)


def open_index_tele(
        cur_lctx: LocalContext,
        index_tele: Expr,
        K: int,
        bi: BinderInfo = BinderInfo.DEFAULT
) -> Tuple[LocalContext, List["FVar"]]:
    """
    Open the first ``K`` binders of an index telescope with fresh FVars.

    ``index_tele`` is of the form:
        ``∀ (i_0:I_0) .. (i_{K-1}:I_{K-1}), Sort _``

    ``open_index_tele`` is to get ``K`` binders that are properly typed for
    inductive's indices and local context that contains their types.

    Parameters
    ----------
    cur_lctx : LocalContext
        Context to push the new locals.
    index_tele : Expr
        A ``ForallE`` chain of length ``>= K``.
    K : int
        Number of leading binders to open.
    bi : BinderInfo, optional
        Binder info to record on each opened local (``DEFAULT`` for the
        motive, ``IMPLICIT`` for recursor telescope). Defaults to
        ``BinderInfo.DEFAULT``.

    Examples
    --------
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.expr import ForallE, Const, Sort, BinderInfo
    >>> from physika.core.level import LSucc, LZero
    >>> from physika.utils.cic_utils.inductive_utils import open_index_tele
    >>> # Vec's index telescope after params are stripped
    >>> tele = ForallE("n", Const("Nat", ()), Sort(LSucc(LZero())),
    ...                BinderInfo.DEFAULT)
    >>> lctx, idx = open_index_tele(LocalContext(), tele, 1)
    >>> len(idx) # on FVar for "n"
    1
    >>> lctx.fvar_type(idx[0])
    Const(name='Nat', levels=())
    >>> open_index_tele(LocalContext(), tele, 0)[1]   # K=0 opens nothing
    []
    """
    fvs: List["FVar"] = []
    t = index_tele
    for _ in range(K):
        assert isinstance(t, ForallE), "index_tele shorter than K"
        cur_lctx, f = cur_lctx.push_local(t.binder_name, t.binder_type, bi)
        fvs.append(f)
        t = instantiate1(t.body, f)
    return cur_lctx, fvs


def app_all(fn: Expr, args: Iterable[Expr]) -> Expr:
    """Apply ``fn`` to every expression in ``args`` from left to right.

    ``app_all(f, [a, b, c])`` would return:
    ``App(App(App(f, a), b), c)`` — t

    This curried application is used in ``derive_recursor`` when
    aadding constructor, recursor, and motive applications.

    Parameters
    ----------
    fn : Expr
        Head expression.
    args : Iterable[Expr]
        Arguments to apply ``fn`` from left to right.


    Examples
    --------
    >>> from physika.core.expr import Const
    >>> from physika.utils.cic_utils.inductive_utils import app_all
    >>> app_all(Const("f", ()), [Const("a", ()), Const("b", ())])  # noqa: E501
    App(func=App(func=Const(name='f', levels=()), arg=Const(name='a', levels=())), arg=Const(name='b', levels=()))
    """
    for a in args:
        fn = App(fn, a)
    return fn


def derive_recursor(decl: "InductiveDecl") -> "Recursor":
    """
    Automatically derive a universe polymorphic recursor when adding and
    inductive ``decl`` for an inductive type.

    The constructed recursor is built with FVars using
    ``LocalContext.push_local`` and ``mk_forall`` rather than modifying de
    Bruijn depths. General form of recursor is as followns:
    ``params -> motive -> minors -> indices -> (major) -> motive indices
    major``.

    Parameters
    ----------
    decl : InductiveDecl
        Inductive declaration to build a recursor for.


    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     derive_recursor, mk_vec_decl)
    >>> rec = derive_recursor(mk_vec_decl())
    >>> rec.name
    'Vec.rec'
    >>> (rec.num_params, rec.num_indices, rec.num_minors, rec.arity())
    (1, 1, 2, 6)
    >>> [r.ctor_name for r in rec.rules]
    ['Vec.nil', 'Vec.cons']
    """
    name = decl.name
    P = decl.num_params  # number of parameters
    U = LParam("u")  # Universe level variable
    T = Const(name, ())  # inductive type for which the recursor is derived
    num_ctors = len(decl.constructors)

    if decl_is_prop_sorted(decl) and num_ctors > 1:
        raise ValueError(
            "cannot derive a recursor for a proposition with more than one "
            f"constructor: '{name}' is Prop-sorted with {num_ctors} "
            "constructors")

    lctx = LocalContext()

    # open decl.type
    param_fvars = []
    tp = decl.type
    for _ in range(P):
        if not isinstance(tp, ForallE):
            raise ValueError(f"'{name}' declares num_params={P} but its "
                             "type has fewer leading binders")
        lctx, pf = lctx.push_local(tp.binder_name, tp.binder_type,
                                   BinderInfo.IMPLICIT)
        param_fvars.append(pf)
        tp = instantiate1(tp.body, pf)

    # Walks decl.type (ForallE chain) and counts indices (K)
    index_tele: Expr = tp
    K = 0
    foralle_chain = index_tele
    while isinstance(foralle_chain, ForallE):
        K += 1
        foralle_chain = foralle_chain.body

    # build motive as CIC term
    m_lctx, m_idx = open_index_tele(lctx, index_tele, K)
    m_lctx, t_fv = m_lctx.push_local(
        "_t", app_all(T,
                      list(param_fvars) + list(m_idx)))
    motive_type = m_lctx.mk_forall(list(m_idx) + [t_fv], Sort(U))
    lctx, motive_fv = lctx.push_local("motive", motive_type,
                                      BinderInfo.DEFAULT)

    # Drerive minor premise for each constructor
    ctor_infos: List[Dict[str, Any]] = []
    for ctor in decl.constructors:
        c_lctx = lctx
        ctp: Expr = ctor.type
        for j in range(P):
            if not isinstance(ctp, ForallE):
                raise ValueError(
                    f"'{ctor.name}' have less than num_params={P} binders")
            ctp = instantiate1(ctp.body, param_fvars[j])

        field_fvars: List["FVar"] = []
        field_recs: List[Optional[List[Expr]]] = []
        while isinstance(ctp, ForallE):
            ftype = ctp.binder_type
            sref = self_reference_indices(name, param_fvars, ftype)
            if sref is None and name_appears(name, ftype):
                raise ValueError(
                    f"'{ctor.name}' field '{ctp.binder_name}' is a nested"
                    "occurrence")
            c_lctx, ff = c_lctx.push_local(ctp.binder_name, ftype)
            field_fvars.append(ff)
            field_recs.append(sref)
            ctp = instantiate1(ctp.body, ff)

        _, ret_args = get_app_fn_args(ctp)
        j_exprs = list(ret_args[P:])
        if len(j_exprs) != K:
            raise ValueError(
                f"'{ctor.name}' returns {len(j_exprs)} index argument(s) but"
                f" '{name}' declares num_indices={K}")

        ctor_app = app_all(Const(ctor.name, ()),
                           list(param_fvars) + list(field_fvars))

        # derive constructor conclusion
        concl = app_all(motive_fv, j_exprs + [ctor_app])

        binders = []
        for ff, sref in zip(field_fvars, field_recs):
            binders.append(ff)
            if sref is not None:
                ih_type = app_all(motive_fv, list(sref) + [ff])
                c_lctx, ih_fv = c_lctx.push_local(f"_ih_{ff.id.id}", ih_type)
                binders.append(ih_fv)

        # minor premises as CIC temrs
        minor_type = c_lctx.mk_forall(binders, concl)
        ctor_infos.append({
            "field_fvars": field_fvars,
            "field_recs": field_recs,
            "minor_type": minor_type,
        })

    minor_fvars: List["FVar"] = []
    for i, ci in enumerate(ctor_infos):
        lctx, mf = lctx.push_local(f"_minor_{i}", ci["minor_type"],
                                   BinderInfo.DEFAULT)
        minor_fvars.append(mf)

    # recurso's tail
    lctx, r_idx = open_index_tele(lctx, index_tele, K, BinderInfo.IMPLICIT)
    lctx, major_fv = lctx.push_local(
        "_major", app_all(T,
                          list(param_fvars) + list(r_idx)), BinderInfo.DEFAULT)
    concl_body = app_all(motive_fv, list(r_idx) + [major_fv])

    rec_type = lctx.mk_forall(
        list(param_fvars) + [motive_fv] + list(minor_fvars) + list(r_idx) +
        [major_fv], concl_body)

    # add iota(ι) elimination rules
    rules: List[RecursorRule] = []
    for i, ctor in enumerate(decl.constructors):
        ci = ctor_infos[i]
        field_fvars = ci["field_fvars"]
        field_recs = ci["field_recs"]
        nf = len(field_fvars)

        rhs: Expr = minor_fvars[i]
        for ff, sref in zip(field_fvars, field_recs):
            rhs = App(rhs, ff)
            if sref is not None:
                rec_call: Expr = Const(f"{name}.rec", (U, ))
                rec_call = app_all(
                    rec_call,
                    list(param_fvars) + [motive_fv] + list(minor_fvars) +
                    list(sref) + [ff])
                rhs = App(rhs, rec_call)

        rhs = abstract_fvars(
            rhs,
            list(field_fvars) + list(param_fvars) + [motive_fv] +
            list(minor_fvars))
        rules.append(RecursorRule(ctor_name=ctor.name, nfields=nf, rhs=rhs))

    return Recursor(
        name=f"{name}.rec",
        type=rec_type,
        num_params=P,
        num_indices=K,
        num_motives=1,
        num_minors=num_ctors,
        rules=tuple(rules),
        level_params=("u", ),
    )


def verify_recursor_rules(
        decl: "InductiveDecl",
        ctors: Dict[str, "ConstantInfo"],  # noqa: E501
        recursor_ci: "ConstantInfo",
        rec_info: "Recursor",
        env: "Environment") -> Optional[str]:
    """
    Kernel checj that each ``RecursorRule`` in ``rec_info`` reduces to a term
    of the type its motive promises for the matching constructor. Returns
    ``None`` if ``rec_info`` is valid for ``decl``, or an error if is found.

    Parameters
    ----------
    decl : InductiveDecl
        Inductive declaration ``rec_info``contains.
    ctors : Dict[str, ConstantInfo]
        Declaration's constructors, keyed by name.
    recursor_ci : ConstantInfo
        Recursor's constant entry.
    rec_info : Recursor
        Recursor metadata and ι-rules to verify.
    env : Environment
        Environment to resolve constants while kernel checking each
        rule.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     mk_nat_decl, verify_recursor_rules,
    ... )
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()
    >>> decl = mk_nat_decl()
    >>> ii = env.inductives["Nat"]
    >>> verify_recursor_rules(decl, ii.ctors, ii.recursor, ii.rec_info, env) is None  # noqa: E501
    True
    """
    name = decl.name

    if rec_info.num_motives != 1:
        return (f"verify_recursor_rules: '{rec_info.name}' has "
                f"num_motives={rec_info.num_motives}; only single-motive "
                "recursors are supported")

    if rec_info.num_minors != len(decl.constructors):
        return (f"verify_recursor_rules: '{rec_info.name}' declares "
                f"num_minors={rec_info.num_minors} but '{name}' has "
                f"{len(decl.constructors)} constructors")

    ctor_names = [c.name for c in decl.constructors]
    rule_names = [r.ctor_name for r in rec_info.rules]
    if sorted(rule_names) != sorted(ctor_names) or len(rule_names) != len(
            set(rule_names)):
        return (f"'{rec_info.name}' rules {sorted(rule_names)} do not match"
                f" constructors {sorted(ctor_names)}.")

    mctx = MetaVarContext()
    lctx0 = LocalContext()
    rec_type: Expr = rec_info.type
    rc_type: Expr = recursor_ci.type
    ok, mctx = is_def_eq(rec_type, rc_type, env, lctx0, mctx)
    if not ok:
        return (f"{rec_info.name}': ConstantInfo.type and Recursor.type"
                " disagree")

    # Open the recursor's own declared telescope with fresh FVars
    arity = rec_info.arity()
    lctx = lctx0
    tp: Expr = rec_type
    all_fvars = []
    for _ in range(arity):
        tp_w = whnf(tp, env, lctx, mctx)
        if not isinstance(tp_w, ForallE):
            return (f"verify_recursor_rules: '{rec_info.name}': declared "
                    f"type has fewer than arity()={arity} leading binders")
        lctx, fv = lctx.push_local(tp_w.binder_name, tp_w.binder_type)
        all_fvars.append(fv)
        tp = instantiate1(tp_w.body, fv)

    num_params = rec_info.num_params
    param_fvars = all_fvars[:num_params]
    motive_fvar = all_fvars[num_params]
    minor_fvars = all_fvars[num_params + 1:num_params + 1 +
                            rec_info.num_minors]  # noqa: E501
    index_fvars = all_fvars[num_params + 1 + rec_info.num_minors:num_params +
                            1 + rec_info.num_minors + rec_info.num_indices]
    major_fvar = all_fvars[-1]
    conclusion_template = abstract_fvars(tp,
                                         list(index_fvars) +
                                         [major_fvar])  # noqa: E501

    non_major_args = list(param_fvars) + [motive_fvar] + list(minor_fvars)
    rules_by_ctor = {r.ctor_name: r for r in rec_info.rules}

    for ctor in decl.constructors:
        rule = rules_by_ctor[ctor.name]
        ctor_type: Expr = ctors[ctor.name].type

        # Open constructor's telescope
        ctp: Expr = ctor_type
        c_lctx = lctx
        for i in range(num_params):
            ctp_w = whnf(ctp, env, c_lctx, mctx)
            if not isinstance(ctp_w, ForallE):
                return (f"Constructor '{ctor.name}' has fewer than "
                        "num_params={num_params} binders")
            ctp = instantiate1(ctp_w.body, param_fvars[i])

        field_fvars = []
        while True:
            ctp_w = whnf(ctp, env, c_lctx, mctx)
            if not isinstance(ctp_w, ForallE):
                break
            c_lctx, ffv = c_lctx.push_local(ctp_w.binder_name,
                                            ctp_w.binder_type)
            field_fvars.append(ffv)
            ctp = instantiate1(ctp_w.body, ffv)

        if len(field_fvars) != rule.nfields:
            return (f"verify_recursor_rules: rule for '{ctor.name}' "
                    f"declares nfields={rule.nfields} but the constructor "
                    f"has {len(field_fvars)} field(s)")

        # ctp is now the constructor's return type:

        _, ctor_ret_args = get_app_fn_args(ctp)
        ctor_index_exprs = ctor_ret_args[num_params:]
        if len(ctor_index_exprs) != rec_info.num_indices:
            return (f"verify_recursor_rules: constructor '{ctor.name}' "
                    f"asserts {len(ctor_index_exprs)} index argument(s) "
                    f"but '{rec_info.name}' declares num_indices="
                    f"{rec_info.num_indices}")

        ctor_app: Expr = Const(ctor.name, ())
        for pf in param_fvars:
            ctor_app = App(ctor_app, pf)
        for ff in field_fvars:
            ctor_app = App(ctor_app, ff)

        expected = instantiate(conclusion_template,
                               list(ctor_index_exprs) + [ctor_app])

        subst: List[Expr] = [*field_fvars, *non_major_args, *index_fvars]
        actual = instantiate(rule.rhs, subst)

        try:
            kernel_check(actual, expected, env, c_lctx, mctx)
        except KernelException as e:
            return (f"Recursor rule for '{ctor.name}' does not pass type-check"
                    f" against its motive: {e}")

    return None


def mk_nat_decl() -> InductiveDecl:
    """
    Build the declaration of Nat inductive (natural numbers).

    ``Nat : Type0`` with constructors ``Nat.zero : Nat`` and
    ``Nat.succ : Nat → Nat``.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import mk_nat_decl
    >>> decl = mk_nat_decl()
    >>> decl.name, decl.num_params, [c.name for c in decl.constructors]
    ('Nat', 0, ['Nat.zero', 'Nat.succ'])
    """
    return InductiveDecl(
        name="Nat",
        level_params=(),
        num_params=0,
        type=TYPE_0,
        constructors=(
            Constructor("Nat.zero", NAT),
            Constructor("Nat.succ", mk_arrow(NAT, NAT)),
        ),
        is_recursive=True,
    )


def mk_nat_add_value() -> Expr:
    """
    Build reducible body of ``Nat.add``.

    ``Nat.add n m := Nat.rec (fun _ => Nat) n (fun k ih => Nat.succ ih) m``

    Support recursion on the second argument.


    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.reduction import whnf
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> env = mk_builtin_env()
    >>> two_plus_three = App(App(Const("Nat.add", ()), Lit(2)), Lit(3))
    >>> whnf(two_plus_three, env, LocalContext(), MetaVarContext())
    Lit(val=5)
    """
    motive = Lam("_", NAT, NAT, BinderInfo.DEFAULT)
    # innermost first:
    # m=BVar(0), n=BVar(1).
    base = BVar(1)
    step = Lam("k", NAT, Lam("ih", NAT, App(SUCC, BVar(0)),
                             BinderInfo.DEFAULT), BinderInfo.DEFAULT)
    body = App(
        App(App(App(Const("Nat.rec", (LSucc(LZero()), )), motive), base),
            step), BVar(0))  # major premise = m
    return Lam("n", NAT, Lam("m", NAT, body, BinderInfo.DEFAULT),
               BinderInfo.DEFAULT)


def mk_nat_mul_value() -> Expr:
    """
    Build reducible body of ``Nat.mul``.

    ``Nat.mul n m := Nat.rec (fun _ => Nat) Nat.zero (fun k ih => Nat.add ih n)
    m``

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.reduction import whnf
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> env = mk_builtin_env()
    >>> expr = App(App(Const("Nat.mul", ()), Lit(3)), Lit(4))
    >>> whnf(expr, env, LocalContext(), MetaVarContext())
    Lit(val=12)
    """
    motive = Lam("_", NAT, NAT, BinderInfo.DEFAULT)
    # innermost first:
    # m=BVar(0), n=BVar(1).
    base = ZERO

    step = Lam(
        "k", NAT,
        Lam("ih", NAT, App(App(Const("Nat.add", ()), BVar(0)), BVar(3)),
            BinderInfo.DEFAULT), BinderInfo.DEFAULT)
    body = App(
        App(App(App(Const("Nat.rec", (LSucc(LZero()), )), motive), base),
            step), BVar(0))  # major premise = m
    return Lam("n", NAT, Lam("m", NAT, body, BinderInfo.DEFAULT),
               BinderInfo.DEFAULT)


def mk_nat_pred_value() -> Expr:
    """
    Build reducible body of ``Nat.pred`` (predecessor).

    ``Nat.pred n := Nat.rec (fun _ => Nat) Nat.zero (fun k ih => k) n``

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.reduction import whnf
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> env = mk_builtin_env()
    >>> whnf(App(Const("Nat.pred", ()), Lit(5)),
    ...      env, LocalContext(), MetaVarContext())
    Lit(val=NatLit(val=4))
    """
    motive = Lam("_", NAT, NAT, BinderInfo.DEFAULT)
    base = ZERO
    # Body needs k (BVar(1))
    step = Lam("k", NAT, Lam("ih", NAT, BVar(1), BinderInfo.DEFAULT),
               BinderInfo.DEFAULT)
    body = App(
        App(App(App(Const("Nat.rec", (LSucc(LZero()), )), motive), base),
            step), BVar(0))  # major premise = n
    return Lam("n", NAT, body, BinderInfo.DEFAULT)


def mk_nat_sub_value() -> Expr:
    """
    Build reducible body of ``Nat.sub``, which represents trucates substraction
    follwoing Lean 4 implementation.

    ``Nat.sub n m := Nat.rec (fun _ => Nat) n (fun k ih => Nat.pred ih) m``

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.reduction import whnf
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> env = mk_builtin_env()
    >>> whnf(App(App(Const("Nat.sub", ()), Lit(5)), Lit(3)),
    ...      env, LocalContext(), MetaVarContext())
    Lit(val=2)
    """
    motive = Lam("_", NAT, NAT, BinderInfo.DEFAULT)
    base = BVar(1)
    step = Lam(
        "k", NAT,
        Lam("ih", NAT, App(Const("Nat.pred", ()), BVar(0)),
            BinderInfo.DEFAULT), BinderInfo.DEFAULT)
    body = App(
        App(App(App(Const("Nat.rec", (LSucc(LZero()), )), motive), base),
            step), BVar(0))  # major premise = m
    return Lam("n", NAT, Lam("m", NAT, body, BinderInfo.DEFAULT),
               BinderInfo.DEFAULT)


def mk_int_decl() -> InductiveDecl:
    """
    Declaration of the integer inductive type.

    ``Int : Type0`` and ``Int.ofNat : Nat → Int`` (denotes ``n``) and
    ``Int.negSucc : Nat → Int`` (denotes ``-(n+1)``).

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import mk_int_decl
    >>> decl = mk_int_decl()
    >>> decl.name, decl.is_recursive, [c.name for c in decl.constructors]
    ('Int', False, ['Int.ofNat', 'Int.negSucc'])
    """
    return InductiveDecl(
        name="Int",
        level_params=(),
        num_params=0,
        type=TYPE_0,
        constructors=(
            Constructor("Int.ofNat", mk_arrow(NAT, INT)),
            Constructor("Int.negSucc", mk_arrow(NAT, INT)),
        ),
        is_recursive=False,
    )


def mk_bool_decl() -> InductiveDecl:
    """
    Declaration of the boolean inductive type.

    ``Bool : Type₀`` with the two null constructors ``Bool.false`` and
    ``Bool.true``.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import mk_bool_decl
    >>> [c.name for c in mk_bool_decl().constructors]
    ['Bool.false', 'Bool.true']
    """
    return InductiveDecl(
        name="Bool",
        level_params=(),
        num_params=0,
        type=TYPE_0,
        constructors=(
            Constructor("Bool.false", BOOL),
            Constructor("Bool.true", BOOL),
        ),
        is_recursive=False,
    )


def mk_fin_decl() -> InductiveDecl:
    """
    Declaration of ``Fin`` inductive type for indexing.

    ``Fin : Nat → Type₀``:

    - ``Fin.zero : ∀ {n : Nat}, Fin (Nat.succ n)``
    - ``Fin.succ : ∀ {n : Nat} (k : Fin n), Fin (Nat.succ n)``

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import mk_fin_decl
    >>> decl = mk_fin_decl()
    >>> decl.name, [c.name for c in decl.constructors]
    ('Fin', ['Fin.zero', 'Fin.succ'])
    """
    FIN = Const("Fin", ())

    fin_zero_type = ForallE("n", NAT, App(FIN, App(SUCC, BVar(0))),
                            BinderInfo.IMPLICIT)

    fin_succ_type = ForallE(
        "n", NAT,
        ForallE("k", App(FIN, BVar(0)), App(FIN, App(SUCC, BVar(1))),
                BinderInfo.DEFAULT), BinderInfo.IMPLICIT)

    return InductiveDecl(
        name="Fin",
        level_params=(),
        num_params=0,
        type=mk_arrow(NAT, TYPE_0),  # Fin : Nat → Type₀
        constructors=(
            Constructor("Fin.zero", fin_zero_type),
            Constructor("Fin.succ", fin_succ_type),
        ),
        is_recursive=False,
    )


def mk_vec_decl() -> InductiveDecl:
    """
    Inductive type for ``Vec`` which represnets length indexed vectors.

    ``Vec : Type₀ → Nat → Type₀`` with one uniform parameter ``α`` and one
    index ``n`` (``num_params=1``, ``num_indices=1``).
    CIC representation of a Physika tensor ``ℝ[n]``:

    - ``Vec.nil  : ∀ {α}, Vec α Nat.zero``
    - ``Vec.cons : ∀ {α} {n} (hd : α) (tl : Vec α n), Vec α (Nat.succ n)``


    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import mk_vec_decl
    >>> decl = mk_vec_decl()
    >>> decl.name, decl.num_params, [c.name for c in decl.constructors]
    ('Vec', 1, ['Vec.nil', 'Vec.cons'])
    """
    VEC = Const("Vec", ())

    # Vec type constructor: Type₀ → Nat → Type₀
    vec_kind = ForallE("α", TYPE_0, mk_arrow(NAT, TYPE_0), BinderInfo.DEFAULT)

    # Vec.nil : ∀ {α : Type₀}, Vec α Nat.zero
    # ForallE("α", TYPE_0, App(App(Vec, BVar(0)), Nat.zero))
    nil_type = ForallE("α", TYPE_0, App(App(VEC, BVar(0)), ZERO),
                       BinderInfo.IMPLICIT)

    # Return:
    # Vec α (Nat.succ n) = App(App(Vec, BVar(3)), App(Nat.succ, BVar(2)))
    cons_type = ForallE(
        "α",
        TYPE_0,
        ForallE(
            "n",
            NAT,
            ForallE(
                "hd",
                BVar(1),  # hd : α
                ForallE(
                    "tl",
                    App(App(VEC, BVar(2)), BVar(1)),  # tl : Vec α n
                    App(App(VEC, BVar(3)), App(SUCC,
                                               BVar(2))),  # Vec α (succ n)
                    BinderInfo.DEFAULT),
                BinderInfo.DEFAULT),
            BinderInfo.IMPLICIT),
        BinderInfo.IMPLICIT)

    return InductiveDecl(
        name="Vec",
        level_params=(),
        num_params=1,  # α is uniform
        type=vec_kind,
        constructors=(
            Constructor("Vec.nil", nil_type),
            Constructor("Vec.cons", cons_type),
        ),
        is_recursive=True,
    )


def mk_prod_decl() -> InductiveDecl:
    """
    Declaration of the non-dependent pair type.

    ``Prod : Type₀ → Type₀ → Type₀`` with a single constructor:

    ``Prod.mk : ∀ {α β}, α → β → Prod α β``.

    Examples
    --------
    >>> from physika.utils.cic_utils.inductive_utils import mk_prod_decl
    >>> decl = mk_prod_decl()
    >>> decl.name, decl.num_params, [c.name for c in decl.constructors]
    ('Prod', 2, ['Prod.mk'])
    """
    PRODC = Const("Prod", ())
    prod_kind = ForallE("α", TYPE_0,
                        ForallE("β", TYPE_0, TYPE_0, BinderInfo.DEFAULT),
                        BinderInfo.DEFAULT)
    mk_type = ForallE(
        "α",
        TYPE_0,
        ForallE(
            "β",
            TYPE_0,
            ForallE(
                "fst",
                BVar(1),  # fst : α
                ForallE(
                    "snd",
                    BVar(1),  # snd : β
                    App(App(PRODC, BVar(3)), BVar(2)),  # Prod α β
                    BinderInfo.DEFAULT),
                BinderInfo.DEFAULT),
            BinderInfo.IMPLICIT),
        BinderInfo.IMPLICIT)
    return InductiveDecl(
        name="Prod",
        level_params=(),
        num_params=2,
        type=prod_kind,
        constructors=(Constructor("Prod.mk", mk_type), ),
        is_recursive=False,
    )


def reg_nat_ops(env: Environment) -> None:
    """
    Register ``Nat`` operators.

    Boolean comparisons as axioms:
    ``Nat.eqb``/``neb``/``ltb``/``gtb``/``leb``/``geb`` and ``Nat.ite`` used
    for elaborating an if condtion with natural numbers.

    Definitions as ``Nat.rec``-based bodies that the kernel computes with:
    ``Nat.add``, ``Nat.mul``, ``Nat.pred``, ``Nat.sub``
    This is what allows a dependent shape like ``ℝ[n+1]`` to be checked against
    a body that infers ``succ n``.

    Parameters
    ----------
    env : Environment
        Environment to add the constants to.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()   # calls reg_nat_ops internally
    >>> env.constants["Nat.add"].value is not None
    True
    >>> env.constants["Nat.ltb"].value is None
    True
    """
    nat_binop = mk_arrow(NAT, mk_arrow(NAT, NAT))
    nat_cmp_bool = mk_arrow(NAT, mk_arrow(NAT, BOOL))
    for name, tp in [
        ("Nat.eqb", nat_cmp_bool),
        ("Nat.neb", nat_cmp_bool),
        ("Nat.ltb", nat_cmp_bool),
        ("Nat.gtb", nat_cmp_bool),
        ("Nat.leb", nat_cmp_bool),
        ("Nat.geb", nat_cmp_bool),
            # Nat.ite b t f : Bool → Nat → Nat → Nat
        ("Nat.ite", mk_arrow(BOOL, mk_arrow(NAT, mk_arrow(NAT, NAT)))),
    ]:
        env.add_constant(
            ConstantInfo(name=name, level_params=(), type=tp, value=None))

    env.add_constant(
        ConstantInfo(name="Nat.add",
                     level_params=(),
                     type=nat_binop,
                     value=mk_nat_add_value()))
    env.add_constant(
        ConstantInfo(name="Nat.mul",
                     level_params=(),
                     type=nat_binop,
                     value=mk_nat_mul_value()))
    env.add_constant(
        ConstantInfo(name="Nat.pred",
                     level_params=(),
                     type=mk_arrow(NAT, NAT),
                     value=mk_nat_pred_value()))
    env.add_constant(
        ConstantInfo(name="Nat.sub",
                     level_params=(),
                     type=nat_binop,
                     value=mk_nat_sub_value()))


def reg_real_ops(env: Environment) -> None:
    """
    Register ``Real`` type and Real-typed operator as opaque
    axioms. However, ``Real`` implemention can be improed as Mathlib does.
    Kernel type checks ``Real`` but does not computes with it. This is done
    by pytorch generated code if succesfully passes CIC checks.

    Registers ``Real`` type, arithmetic operations (``Real.add``/``mul``/
    ``sub``/``div``/``neg``/``pow``), ``Prop`` orderings (``Real.lt``/
    ``le``), basic functions (``log``/``exp``/``cos``/``sin``/``sqrt``/
    ``abs``), and ``Nat.toReal`` coercion ``elaborate_binop`` inserts
    around a Nat operand of a mixed ``Nat``/``Real`` arithmetic op.

    Parameters
    ----------
    env : Environment
        Environment to add the constants to.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()
    >>> "Real" in env.constants and env.constants["Real.add"].value is None
    True
    """
    real = Const("Real", ())
    real_binop = mk_arrow(real, mk_arrow(real, real))
    real_cmp_bool = mk_arrow(real, mk_arrow(real, BOOL))
    real_ops: List[Tuple[str, Expr]] = [
        ("Real", TYPE_0),
        ("Real.add", real_binop),
        ("Real.mul", real_binop),
        ("Real.sub", real_binop),
        ("Real.div", real_binop),
        ("Real.neg", mk_arrow(real, real)),
        ("Real.pow", real_binop),
        ("Real.lt", mk_arrow(real, mk_arrow(real, PROP))),
        ("Real.le", mk_arrow(real, mk_arrow(real, PROP))),
        ("Real.eqb", real_cmp_bool),
        ("Real.neb", real_cmp_bool),
        ("Real.ltb", real_cmp_bool),
        ("Real.gtb", real_cmp_bool),
        ("Real.leb", real_cmp_bool),
        ("Real.geb", real_cmp_bool),
        ("Real.ite", mk_arrow(BOOL, mk_arrow(real, mk_arrow(real, real)))),
        # basic functions: Real → Real
        ("log", mk_arrow(real, real)),
        ("exp", mk_arrow(real, real)),
        ("cos", mk_arrow(real, real)),
        ("sin", mk_arrow(real, real)),
        ("sqrt", mk_arrow(real, real)),
        ("abs", mk_arrow(real, real)),
        # Nat.toReal : Nat → Real
        ("Nat.toReal", mk_arrow(NAT, real)),
    ]
    for name, tp in real_ops:
        env.add_constant(
            ConstantInfo(name=name, level_params=(), type=tp, value=None))


def reg_autodiff(env: Environment) -> None:
    """
    Register the grad axioms.

    ``grad(f(x), x) : Real`` for a scalar call, and for Jacobian``Vec.grad :
    {n}{m} → Vec Real n → Vec Real m → Vec (Vec Real m) n`` for a vector grad
    call.


    Parameters
    ----------
    env : Environment
        Environment to add the constants to.


    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()
    >>> "grad" in env.constants and "Vec.grad" in env.constants
    True
    """
    real = Const("Real", ())
    vec = Const("Vec", ())
    env.add_constant(
        ConstantInfo(name="grad",
                     level_params=(),
                     type=mk_arrow(real, mk_arrow(real, real)),
                     value=None))

    # {n}{m} implicit
    vec_grad_type = ForallE(
        "n",
        NAT,
        ForallE(
            "m",
            NAT,
            ForallE(
                "_",
                App(App(vec, real), BVar(1)),  # Vec Real n
                ForallE(
                    "_",
                    App(App(vec, real), BVar(1)),  # Vec Real m
                    App(
                        App(vec, App(App(vec, real),
                                     BVar(2))),  # Vec (Vec Real m) n
                        BVar(3)),
                    BinderInfo.DEFAULT,
                ),
                BinderInfo.DEFAULT,
            ),
            BinderInfo.IMPLICIT,
        ),
        BinderInfo.IMPLICIT,
    )
    env.add_constant(
        ConstantInfo(name="Vec.grad",
                     level_params=(),
                     type=vec_grad_type,
                     value=None))


def reg_ofnat(env: Environment) -> None:
    """Register ``OfNat`` and its instances.

    Physika elaborates integer numeral as:
        ``Proj("OfNat", 0, App(Const(<instance>), Lit(n)))``


    ``OfNat`` is a single field struct (``InductiveDecl``, one
    constructor and ``Proj`` field access, no recursor). ``instOfNatNat`` has
    a real body so a concrete literal collapses straight back to ``Lit(k)`` via
    β/ι-reduction.


    Parameters
    ----------
    env : Environment
        Environment to add ``OfNat`` inductive and instances to.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.expr import App, Const, Proj, Lit
    >>> from physika.core.reduction import whnf
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> env = mk_builtin_env()
    >>> term = Proj("OfNat", 0, App(Const("instOfNatNat", ()), Lit(3)))
    >>> whnf(term, env, LocalContext(), MetaVarContext())
    Lit(val=3)
    """
    real = Const("Real", ())
    ofnat = Const("OfNat", ())
    ofnat_kind = ForallE(
        "α",
        TYPE_0,
        ForallE("n", NAT, TYPE_0, BinderInfo.DEFAULT),
        BinderInfo.DEFAULT,
    )
    # OfNat.mk :
    # {α : Type₀} → {n : Nat} → (ofNat : α) → OfNat α n
    ofnat_mk_type = ForallE(
        "α",
        TYPE_0,
        ForallE(
            "n",
            NAT,
            ForallE("ofNat", BVar(1), App(App(ofnat, BVar(2)), BVar(1)),
                    BinderInfo.DEFAULT),
            BinderInfo.IMPLICIT,
        ),
        BinderInfo.IMPLICIT,
    )
    ofnat_decl = InductiveDecl(
        name="OfNat",
        level_params=(),
        num_params=2,
        type=ofnat_kind,
        constructors=(Constructor("OfNat.mk", ofnat_mk_type), ),
        is_recursive=False,
    )
    env.add_inductive(
        InductiveInfo(
            decl=ofnat_decl,
            ctors={
                "OfNat.mk":
                ConstantInfo(name="OfNat.mk",
                             level_params=(),
                             type=ofnat_mk_type,
                             value=None)
            },
            recursor=ConstantInfo(name="OfNat.rec",
                                  level_params=(),
                                  type=TYPE_0,
                                  value=None),
            rec_info=None,
        ))

    # instOfNatNat :
    # (n : Nat) → OfNat Nat n := fun n => OfNat.mk Nat n n
    inst_lctx = LocalContext()
    inst_lctx, ofnat_n_fv = inst_lctx.push_local("n", NAT)
    inst_nat_value = inst_lctx.mk_lambda(
        [ofnat_n_fv],
        App(App(App(Const("OfNat.mk", ()), NAT), ofnat_n_fv), ofnat_n_fv),
    )
    env.add_constant(
        ConstantInfo(
            name="instOfNatNat",
            level_params=(),
            type=ForallE("n", NAT, App(App(ofnat, NAT), BVar(0)),
                         BinderInfo.DEFAULT),
            value=inst_nat_value,
        ))
    env.add_constant(
        ConstantInfo(
            name="instOfNatReal",
            level_params=(),
            type=ForallE("n", NAT, App(App(ofnat, real), BVar(0)),
                         BinderInfo.DEFAULT),
            value=None,
        ))


def reg_vec_ops(env: Environment) -> None:
    """
    Register vector operators as axioms.

    Arithmetics (``Vec.vadd``/``vmul``/``dot``/``scale``/
    ``norm_sq``/``sum``), ``Vec.ite`` branch, the ``Vec.zeros``
    zero-decl target, ``Vec.concat`` (different-length ``+``),
    element-type-polymorphic ``Vec.get`` / ``Vec.tabulate`` / ``Vec.foldl``
    ``Vec`` for a matrix ``ℝ[m,n]``, and ``Fin.ofNat`` coercion non-literal
    index.


    Parameters
    ----------
    env : Environment
        Environment to add the constants to.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()
    >>> all(n in env.constants for n in
    ...     ("Vec.dot", "Vec.get", "Vec.tabulate", "Vec.foldl", "Fin.ofNat"))
    True
    """
    real = Const("Real", ())
    vec = Const("Vec", ())
    fin = Const("Fin", ())

    def vr(n_bvar: Expr) -> Expr:
        """
        ``Vec Real n_bvar``.

        Flat vector type of length ``n_bvar``

        Parameters
        ----------
        n_bvar: Expr
            A ``BVar`` at the current binder depth.
        """
        return App(App(vec, real), n_bvar)

    vec_ops = [
        # Vec.vadd/vmul :
        # ∀ (n:Nat), Vec Real n → Vec Real n → Vec Real n
        ("Vec.vadd",
         ForallE(
             "n", NAT,
             ForallE(
                 "u", vr(BVar(0)),
                 ForallE("v", vr(BVar(1)), vr(BVar(2)), BinderInfo.DEFAULT),
                 BinderInfo.DEFAULT), BinderInfo.DEFAULT)),
        ("Vec.vmul",
         ForallE(
             "n", NAT,
             ForallE(
                 "u", vr(BVar(0)),
                 ForallE("v", vr(BVar(1)), vr(BVar(2)), BinderInfo.DEFAULT),
                 BinderInfo.DEFAULT), BinderInfo.DEFAULT)),
        # Vec.dot :
        # ∀ (n:Nat), Vec Real n → Vec Real n → Real
        ("Vec.dot",
         ForallE(
             "n", NAT,
             ForallE("u", vr(BVar(0)),
                     ForallE("v", vr(BVar(1)), real, BinderInfo.DEFAULT),
                     BinderInfo.DEFAULT), BinderInfo.DEFAULT)),
        # Vec.norm_sq :
        # ∀ (n:Nat), Vec Real n → Real
        ("Vec.norm_sq",
         ForallE("n", NAT, ForallE("v", vr(BVar(0)), real, BinderInfo.DEFAULT),
                 BinderInfo.DEFAULT)),
        # Vec.scale :
        # ∀ (n:Nat), Real → Vec Real n → Vec Real n
        ("Vec.scale",
         ForallE(
             "n", NAT,
             ForallE(
                 "c", real,
                 ForallE("v", vr(BVar(1)), vr(BVar(2)), BinderInfo.DEFAULT),
                 BinderInfo.DEFAULT), BinderInfo.DEFAULT)),
        # Vec.sum :
        # ∀ {n:Nat}, Vec Real n → Real
        ("Vec.sum",
         ForallE("n", NAT, ForallE("v", vr(BVar(0)), real, BinderInfo.DEFAULT),
                 BinderInfo.IMPLICIT)),
        # Vec.ite :
        # ∀ (n:Nat), Bool → Vec Real n → Vec Real n → Vec Real n
        ("Vec.ite",
         ForallE(
             "n", NAT,
             ForallE(
                 "cond", BOOL,
                 ForallE(
                     "t", vr(BVar(1)),
                     ForallE("e", vr(BVar(2)), vr(BVar(3)),
                             BinderInfo.DEFAULT), BinderInfo.DEFAULT),
                 BinderInfo.DEFAULT), BinderInfo.DEFAULT)),
        # Vec.zeros :
        # ∀ (n:Nat), Vec Real n
        ("Vec.zeros", ForallE("n", NAT, vr(BVar(0)), BinderInfo.DEFAULT)),
        # Vec.concat : ∀ (m n:Nat), Vec Real m → Vec Real n →
        #               Vec Real (Nat.add m n)
        ("Vec.concat",
         ForallE(
             "m", NAT,
             ForallE(
                 "n", NAT,
                 ForallE(
                     "u", vr(BVar(1)),
                     ForallE(
                         "v", vr(BVar(1)),
                         vr(App(App(Const("Nat.add", ()), BVar(3)), BVar(2))),
                         BinderInfo.DEFAULT), BinderInfo.DEFAULT),
                 BinderInfo.DEFAULT), BinderInfo.DEFAULT)),
    ]
    for name, tp in vec_ops:
        env.add_constant(
            ConstantInfo(name=name, level_params=(), type=tp, value=None))

    # Vec.get :
    # ∀ {α:Type₀} (n:Nat), Vec α n → Fin n → α
    vec_get_type = ForallE(
        "α",
        TYPE_0,
        ForallE(
            "n",
            NAT,
            ForallE(
                "v",
                App(App(vec, BVar(1)), BVar(0)),  # Vec α n
                ForallE("i", App(fin, BVar(1)), BVar(3), BinderInfo.DEFAULT),
                BinderInfo.DEFAULT),
            BinderInfo.DEFAULT),
        BinderInfo.IMPLICIT)
    # Vec.tabulate :
    # ∀ {α:Type₀} (n:Nat), (Fin n → α) → Vec α n
    vec_tab_f_type = ForallE("i", App(fin, BVar(0)), BVar(2),
                             BinderInfo.DEFAULT)
    vec_tabulate_type = ForallE(
        "α", TYPE_0,
        ForallE(
            "n", NAT,
            ForallE("f", vec_tab_f_type, App(App(vec, BVar(2)), BVar(1)),
                    BinderInfo.DEFAULT), BinderInfo.DEFAULT),
        BinderInfo.IMPLICIT)
    for name, tp in [("Vec.get", vec_get_type),
                     ("Vec.tabulate", vec_tabulate_type)]:
        env.add_constant(
            ConstantInfo(name=name, level_params=(), type=tp, value=None))

    # Fin.ofNat :
    # ∀ {n : Nat}, Nat → Fin n
    env.add_constant(
        ConstantInfo(name="Fin.ofNat",
                     level_params=(),
                     type=ForallE(
                         "n", NAT,
                         ForallE("i", NAT, App(fin, BVar(1)),
                                 BinderInfo.DEFAULT), BinderInfo.IMPLICIT),
                     value=None))

    # Vec.foldl :
    # ∀ {α:Type₀} (n:Nat), (Fin n → α → α) → α → α
    foldl_f_type = ForallE(
        "k", App(fin, BVar(0)),
        ForallE("acc", BVar(2), BVar(3), BinderInfo.DEFAULT),
        BinderInfo.DEFAULT)
    vec_foldl_type = ForallE(
        "α", TYPE_0,
        ForallE(
            "n", NAT,
            ForallE("f", foldl_f_type,
                    ForallE("init", BVar(2), BVar(3), BinderInfo.DEFAULT),
                    BinderInfo.DEFAULT), BinderInfo.DEFAULT),
        BinderInfo.IMPLICIT)
    env.add_constant(
        ConstantInfo(name="Vec.foldl",
                     level_params=(),
                     type=vec_foldl_type,
                     value=None))


def reg_mat_ops(env: Environment) -> None:
    """
    Register matrix operators as axioms.

    Physika represents a matrix ``ℝ[m,n]`` as a nested ``Vec (Vec Real n) m``.

    ``Mat.matmul`` (``A @ B``), ``Mat.madd`` (elementwise, same shape),
    ``Mat.add_scalar`` (broadcast a scalar), ``Mat.concat_rows``
    (row-append).

    Parameters
    ----------
    env : Environment
        Environment to add the constants to.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> env = mk_builtin_env()
    >>> all(n in env.constants for n in
    ...     ("Mat.matmul", "Mat.madd", "Mat.add_scalar", "Mat.concat_rows"))
    True
    """
    real = Const("Real", ())
    vec = Const("Vec", ())

    def vr(n_bvar: Expr) -> Expr:
        """
        ``Vec Real n_bvar``
        Mtrix row of width ``n_bvar``.

        Parameters
        ----------
        n_bvar: Expr
            A ``BVar`` at the current binder depth.
        """
        return App(App(vec, real), n_bvar)

    # Mat.matmul :
    # ∀ (m k n : Nat), Vec (Vec Real k) m → Vec (Vec Real n) k
    #                  → Vec (Vec Real n) m
    mat_matmul_type = ForallE(
        "m", NAT,
        ForallE(
            "k", NAT,
            ForallE(
                "n", NAT,
                ForallE(
                    "A", App(App(vec, vr(BVar(1))), BVar(2)),
                    ForallE("B", App(App(vec, vr(BVar(1))), BVar(2)),
                            App(App(vec, vr(BVar(2))), BVar(4)),
                            BinderInfo.DEFAULT), BinderInfo.DEFAULT),
                BinderInfo.DEFAULT), BinderInfo.DEFAULT), BinderInfo.DEFAULT)
    env.add_constant(
        ConstantInfo(name="Mat.matmul",
                     level_params=(),
                     type=mat_matmul_type,
                     value=None))

    # Mat.madd :
    #  ∀ (m k : Nat), Vec (Vec Real k) m → Vec (Vec Real k) m
    #            → Vec (Vec Real k) m
    mat_madd_type = ForallE(
        "m", NAT,
        ForallE(
            "k", NAT,
            ForallE(
                "A", App(App(vec, vr(BVar(0))), BVar(1)),
                ForallE("B", App(App(vec, vr(BVar(1))), BVar(2)),
                        App(App(vec, vr(BVar(2))), BVar(3)),
                        BinderInfo.DEFAULT), BinderInfo.DEFAULT),
            BinderInfo.DEFAULT), BinderInfo.DEFAULT)
    env.add_constant(
        ConstantInfo(name="Mat.madd",
                     level_params=(),
                     type=mat_madd_type,
                     value=None))

    # Mat.add_scalar :
    # ∀ (m k : Nat), Vec (Vec Real k) m → Real
    #                  → Vec (Vec Real k) m
    mat_add_scalar_type = ForallE(
        "m", NAT,
        ForallE(
            "k", NAT,
            ForallE(
                "A", App(App(vec, vr(BVar(0))), BVar(1)),
                ForallE("c", real, App(App(vec, vr(BVar(2))), BVar(3)),
                        BinderInfo.DEFAULT), BinderInfo.DEFAULT),
            BinderInfo.DEFAULT), BinderInfo.DEFAULT)
    env.add_constant(
        ConstantInfo(name="Mat.add_scalar",
                     level_params=(),
                     type=mat_add_scalar_type,
                     value=None))

    # Mat.concat_rows :
    # ∀ (m n k:Nat), Vec (Vec Real k) m → Vec (Vec Real k) n
    #                   → Vec (Vec Real k) (Nat.add m n)
    mat_concat_rows_type = ForallE(
        "m", NAT,
        ForallE(
            "n", NAT,
            ForallE(
                "k", NAT,
                ForallE(
                    "A", App(App(vec, vr(BVar(0))), BVar(2)),
                    ForallE(
                        "B", App(App(vec, vr(BVar(1))), BVar(2)),
                        App(App(vec, vr(BVar(2))),
                            App(App(Const("Nat.add", ()), BVar(4)), BVar(3))),
                        BinderInfo.DEFAULT), BinderInfo.DEFAULT),
                BinderInfo.DEFAULT), BinderInfo.DEFAULT), BinderInfo.DEFAULT)
    env.add_constant(
        ConstantInfo(name="Mat.concat_rows",
                     level_params=(),
                     type=mat_concat_rows_type,
                     value=None))
