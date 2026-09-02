from typing import List, Optional, Tuple

from physika.core.level import Level, LMVar
from physika.core.expr import (
    App,
    BVar,
    BinderInfo,
    Const,
    Expr,
    FVar,
    FloatLit,
    ForallE,
    Lam,
    LetE,
    Lit,
    MData,
    MVar,
    NatLit,
    Proj,
    Sort,
)
from physika.utils.cic_utils.expr_utils import (
    abstract_fvars,
    get_app_fn_args,
    instantiate,
    instantiate1,
    instantiate_level_params_in_expr,
)
from physika.utils.cic_utils.level_utils import levels_equal
from physika.core.local_context import LocalContext, LocalDeclDef
from physika.core.metavar import MetaVarContext, LMVarId
from physika.core.environment import Environment


def is_level_def_eq(l1: Level, l2: Level,
                    mctx: MetaVarContext) -> Tuple[bool, MetaVarContext]:
    """
    Checks if two ``Level`` variables (``l1`` and ``l2``) are equal.
    ``is_level_def_eq`` solves level metavariable as needed.

    First, instantiate solved ``LMVars`` and look for structural equality.
    Then, if either side is an unassigned LMVar, solve it against the other
    side. If ``l1`` and ``l2`` are not ``LMVar`` instances, check two levels
    are equal by using a general ``level_equal`` function from utils.

    Parameters
    ----------
    l1: Level
        First ``Level`` to be checked for equality.
    l2: Level
        Second ``Level`` to be checked for equality.
    mctx: MetaVarContext
        Metavariable context where LMVar are instantiated.

    Example
    -------
    >>> from physika.core.reduction import is_level_def_eq
    >>> from physika.core.level import LMVar
    >>> from physika.core.metavar import MetaVarContext
    >>> l1 = LMVar("u1")
    >>> l2 = LMVar("u2")
    >>> mctx = MetaVarContext()
    >>> is_level_eq, metavar_ctx = is_level_def_eq(l1, l2, mctx)
    >>> is_level_eq
    True
    """
    l1 = mctx.instantiate_level_mvars(l1)
    l2 = mctx.instantiate_level_mvars(l2)
    if l1 == l2:
        return True, mctx
    if isinstance(l1, LMVar):
        lid = LMVarId(l1.id)
        ok, _ = mctx.is_valid_level_assignment(lid, l2)
        if ok:
            mctx.level_assignments[lid] = l2  # ?lid := l2
            return True, mctx
    if isinstance(l2, LMVar):
        lid = LMVarId(l2.id)
        ok, _ = mctx.is_valid_level_assignment(lid, l1)
        if ok:
            mctx.level_assignments[lid] = l1  # ?lid := l1
            return True, mctx
    return levels_equal(l1, l2), mctx


def lit_nat_int(e: Expr) -> Optional[int]:
    """
    Checks if ``e`` is not a negative integer ``Lit``/``NatLit`` and returns
    its value (``None`` if negative). Used at ``is_def_eq`` and ``try_iota``
    to convert a level Nat literal against an actual ``Nat.zero``/``Nat.succ``
    chain value.

    Parameters
    ----------
    e: Expr
        Expression to inspect for a non negative integer literal.

    Example
    -------
    >>> from physika.core.reduction import lit_nat_int
    >>> from physika.core.expr import Lit
    >>> lit_nat_int(Lit(3))
    3
    >>> lit_nat_int(Lit(-1)) is None
    True
    """
    if not isinstance(e, Lit):
        return None
    v = e.val
    if isinstance(v, (NatLit, FloatLit)):
        raw = v.val
    else:
        raw = v
    if isinstance(raw, int) and raw >= 0:
        return raw
    return None


def succ_chain_nat_int(e: Expr, env: Environment, lctx: LocalContext,
                       mctx: MetaVarContext) -> Optional[int]:
    """
    Checks if ``e`` is a ``Nat.succ`` and if true, returns a Nat it
    integer. ``whnf`` reduces at the start and after going through each ``Nat.succ``
    layer, then checks whether the chain closes with ``Nat.zero`` or a
    ``Lit`` int.

    ``whnf`` reduction at each step means a tail like
    ``Nat.succ (Nat.add (Lit 1) (Lit 2))``can be reduced once
    ``Nat.add (Lit 1) (Lit 2)`` reduces to ``Lit 3``.

    Parameters
    ----------
    e: Expr
        Expression to inspect for ``Nat.succ``
    env: Environment
        CIC environment, to reduce ``e`` using ``whnf``.
    lctx: LocalContext
        Local context, to reduce ``e`` using ``whnf``.
    mctx: MetaVarContext
        Metavariable context.

    Example
    -------
    >>> from physika.core.reduction import succ_chain_nat_int
    >>> from physika.core.expr import Const, App
    >>> from physika.core.environment import Environment
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> two = App(Const("Nat.succ", ()), App(Const("Nat.succ", ()), Const("Nat.zero", ())))
    >>> succ_chain_nat_int(two, Environment(), LocalContext(), MetaVarContext())  # noqa: E501
    2
    """
    n = 0
    cur = whnf(e, env, lctx, mctx)
    while True:
        if isinstance(cur, Const) and cur.name == "Nat.zero":
            return n
        if (isinstance(cur, App) and isinstance(cur.func, Const)
                and cur.func.name == "Nat.succ"):
            n += 1
            cur = whnf(cur.arg, env, lctx, mctx)
            continue
        base = lit_nat_int(cur)
        if base is not None:
            return n + base
        return None


def whnf(expr: Expr, env: Environment, lctx: LocalContext,
         mctx: MetaVarContext) -> Expr:
    """
    Reduces ``expr`` to weak-head normal form (WHNF). ``whnf`` applies
    β, δ, ζ, ι, and Proj reduction rules for checking types at elaboration
    time.

    Apply reduction rules to an `expr:Expr` until the head is irreducible
    (``Sort``, ``Lam``, ``ForallE``, ``Const``, ``FVar``, ``MVar``, or ``BVar``
    ). First,by  unfolding a let-binding or a let-bound local (ζ(zeta)-
    reduction). Second, reducing a defined constant (δ(delta)-reduction). Third
    , reducing a lambda application (β(beta)-reduction) and firing a recursor's
    ι-rule against a constructor (iota reduction).

    Parameters
    ----------
    expr: Expr
        Expression to reduce to weak-head normal form.
    env: Environment
        CIC environment contains constant definitions for δ-reduction.
    lctx: LocalContext
        Local context provides let-bound local definitions for
        ζ-reduction.
    mctx: MetaVarContext
        Metavariable context used to follow an assigned MVar's
        solution.

    Example
    -------
    >>> from physika.core.reduction import whnf
    >>> from physika.core.expr import Const, App, Lit
    >>> from physika.core.environment import Environment
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> expr = App(App(Const("Nat.add", ()), Lit(1)), Lit(2))
    >>> whnf(expr, Environment(), LocalContext(), MetaVarContext())
    Lit(val=3)
    """
    while True:
        # Apply assignment chain for MVar
        if isinstance(expr, MVar):
            a = mctx.expr_assignments.get(expr.id.id)
            if a is not None:
                expr = a
                continue
            return expr

        # zeta reduction (binding elimination)
        if isinstance(expr, LetE):
            expr = instantiate1(expr.body, expr.value)
            continue
        # zeta-reduction for let-bound FVars
        if isinstance(expr, FVar):
            decl = lctx.find(expr.id)
            if isinstance(decl, LocalDeclDef):

                expr = decl.value
                continue
            return expr

        # delta-reduction (constant unfolding)
        if isinstance(expr, Const):
            ci = env.constants.get(expr.name)
            if ci is not None and ci.value is not None:
                val = ci.value
                # Instantiate universe polymorphism parameters, if any.
                if ci.level_params and expr.levels:
                    val = instantiate_level_params_in_expr(
                        val, list(ci.level_params), list(expr.levels))
                expr = val
                continue
            return expr

        if isinstance(expr, App):
            # try literal fast path, then β, then iota reductions
            fast = try_nat_literal_binop(expr, env, lctx, mctx)
            if fast is not None:
                expr = fast
                continue
            fn = whnf(expr.func, env, lctx, mctx)
            if isinstance(fn, Lam):
                # apply beta reduction
                expr = instantiate1(fn.body, expr.arg)
                continue
            # ι(iota)-reduction recursor applied to a constructor
            iota = try_iota(fn, expr.arg, env, lctx, mctx)
            if iota is not None:
                expr = iota
                continue
            # fn was already reduced which means the whole App whnf, so return
            # as is
            if fn is expr.func:
                return expr
            return App(fn, expr.arg)

        # unwrap MData expr
        if isinstance(expr, MData):
            expr = expr.expr
            continue

        # Proj reduce field projection on a constructor
        if isinstance(expr, Proj):
            inner = whnf(expr.expr, env, lctx, mctx)
            head, args = get_app_fn_args(inner)
            if isinstance(head, Const):
                ii = env.inductives.get(expr.type_name)
                if ii is not None and len(ii.decl.constructors) == 1:
                    ctor_name = ii.decl.constructors[0].name
                    if head.name == ctor_name:
                        field_idx = ii.decl.num_params + expr.idx
                        if field_idx < len(args):
                            expr = args[field_idx]
                            continue
            if inner is not expr.expr:
                return Proj(expr.type_name, expr.idx, inner)
            return expr

        return expr


def pattern_mvar_app(
        t: Expr, mctx: MetaVarContext) -> Optional[Tuple[MVar, List[FVar]]]:
    """
    Checks if ``t`` is a metavariable applied to to different free variables.

    A function application expression (``App``), have its head and args.
    Miller unification works for head as unassigned ``MVar`` and at least one
    argument. Also, Miller patter shape checks that t (unassigned metavariable)
    is being applied to distince free variables.


    Parameters
    ----------
    t: Expr
        Expression to check for the Miller pattern shape.
    mctx: MetaVarContext
        Metavariable context, used to check if a metavarible is unassigned

    Example
    -------
    >>> from physika.core.reduction import pattern_mvar_app
    >>> from physika.core.expr import Sort, App
    >>> from physika.core.level import LZero
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> lctx = LocalContext()
    >>> lctx, x = lctx.push_local("x", Sort(LZero()))
    >>> mctx = MetaVarContext()
    >>> mv = mctx.mk_mvar("m", lctx, Sort(LZero()))
    >>> pattern_mvar_app(App(mv, x), mctx) == (mv, [x])
    True
    """
    head, args = get_app_fn_args(t)
    if not isinstance(head, MVar) or not args:
        return None
    if mctx.expr_assignments.get(head.id.id) is not None:
        return None
    seen = set()
    fvars: List[FVar] = []
    for a in args:
        if not isinstance(a, FVar) or a.id in seen:
            return None
        seen.add(a.id)
        fvars.append(a)
    return head, fvars


def solve_pattern(mvar: MVar, pat_fvars: List[FVar], rhs: Expr,
                  lctx: "LocalContext", mctx: "MetaVarContext") -> bool:
    """
    Attempts to build a lambda function that takes pattern variables
    ``pat_fvars`` as parameters, and returns ``rhs`` as its body.
    Returns ``True``, mutating ``mctx``, on success, or ``False``
    (``mctx`` left untouched) if the assignment isn't valid.

    Builds ``λpat_fvars. rhs`` by abstracting ``rhs``, and each
    outer binder's own type, over pattern variables. Other variable's
    type may dependently reference an earlier ones. For example,
    ``pat_fvars = [n, v]`` with ``v : Vec Real n``. Then, assigns it
    to ``?m`` if ``mctx.is_valid_assignment`` accepts it. If ``solve_pattern``
    fails, probably meaning an invalid assingment, returns False. This means,
    ``is_def_eq`` tries a more general unification rather that Miller-patterN
    shortcut.

    Parameters
    ----------
    mvar: MVar
        Unassigned metavariable to solve.
    pat_fvars: List[FVar]
        The pattern variables ``mvar`` is applied to.
    rhs: Expr
        Right hand side to abstract over ``pat_fvars`` and assign.
    lctx: LocalContext
        Local context, used to look for pattern variable's type.
    mctx: MetaVarContext
        Metavariable context mutated in place on success.

    Example
    -------
    >>> from physika.core.reduction import solve_pattern  # noqa: E501
    >>> from physika.core.expr import Sort, Lit
    >>> from physika.core.level import LZero
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> lctx = LocalContext()
    >>> lctx, x = lctx.push_local("x", Sort(LZero()))
    >>> mctx = MetaVarContext()
    >>> mv = mctx.mk_mvar("m", lctx, Sort(LZero()))
    >>> mv.id.id
    'm.1'
    >>> # patter variable is " x : Sort 0:"
    >>> # solving: ?m x =?= 3
    >>> solve_pattern(mv, [x], Lit(3), lctx, mctx)
    True
    >>> # solved to λx. 3
    >>> mctx.expr_assignments[mv.id.id]
    Lam(binder_name='x', binder_type=Sort(level=LZero()), body=Lit(val=3), binder_info=<BinderInfo.DEFAULT: 1>)
    """
    result = abstract_fvars(rhs, list(reversed(pat_fvars)))
    for i in range(len(pat_fvars) - 1, -1, -1):
        fv = pat_fvars[i]
        decl = lctx.find(fv.id)
        if decl is None:
            return False
        btype = abstract_fvars(decl.type, list(reversed(pat_fvars[:i])))
        result = Lam(decl.user_name, btype, result, BinderInfo.DEFAULT)
    ok, _ = mctx.is_valid_assignment(mvar.id, result)
    if not ok:
        return False

    mctx.expr_assignments[mvar.id.id] = result
    return True


def is_def_eq(t1: Expr,
              t2: Expr,
              env: Environment,
              lctx: LocalContext,
              mctx: MetaVarContext,
              allow_assign: bool = True) -> Tuple[bool, MetaVarContext]:
    """
    Checks if ``t1`` and ``t2`` are definitionally equal, solving
    expression metavariables as needed. ``is_def_eq`` returns
    ``(True, mctx)`` if equal, ``(False, mctx)`` otherwise.

    Instantiates solved MVars, and compare ``t1`` and ``t2`` node by node
    checking if they are literally equal, and reduces both sides to WHNF.
    If either side is a metavariable applied only to distinct local variables
    (Miller pattern), solve it directly. Otherwise falls through MVar
    assignment, checking if a Nat literal resolves to a ``Nat.succ`` chain,
    and structural comparison (``App``, binder-opening on ``Lam``/``ForallE``,
    one side is a ``Lam`` and the other isn't).

    Returns ``mctx: MetaVarContext``, mutated in place if any MVar got assigned
    during the call.


    Parameters
    ----------
    t1: Expr
        First expression to be checked for definitional equality.
    t2: Expr
        Second expression to be checked for definitional equality.
    env: Environment
        CIC environment, providing constant definitions for reduction.
    lctx: LocalContext
        Local context, extended with a fresh local when opening a
        binder.
    mctx: MetaVarContext
        Metavariable context mutated in place as MVars get assigned.
    allow_assign: bool
        Whether an unassigned metavariable may be solved during the
        comparison. ``False`` makes this a trusted kernel decision.

    Example
    -------
    >>> from physika.core.reduction import is_def_eq
    >>> from physika.core.expr import Const, App, Lit
    >>> from physika.core.environment import Environment
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> lhs = App(App(Const("Nat.add", ()), Lit(1)), Lit(2))
    >>> is_eq, _ = is_def_eq(lhs, Lit(3), Environment(), LocalContext(), MetaVarContext())  # noqa: E501
    >>> is_eq
    True
    """
    # instantiate solved MVars
    t1 = mctx.instantiate_mvars(t1)
    t2 = mctx.instantiate_mvars(t2)
    # structural equality
    if t1 == t2:
        return True, mctx

    # reduce t1 and t2 to whnf
    t1r = whnf(t1, env, lctx, mctx)
    t2r = whnf(t2, env, lctx, mctx)

    # pattern (Miller) unification
    if allow_assign:
        pat1 = pattern_mvar_app(t1r, mctx)
        if pat1 is not None and not isinstance(t2r, MVar):
            mvar, pat_fvars = pat1
            if solve_pattern(mvar, pat_fvars, t2r, lctx, mctx):
                return True, mctx
        pat2 = pattern_mvar_app(t2r, mctx)
        if pat2 is not None and not isinstance(t1r, MVar):
            mvar, pat_fvars = pat2
            if solve_pattern(mvar, pat_fvars, t1r, lctx, mctx):
                return True, mctx

    # MVar instances
    if isinstance(t1r, MVar) and isinstance(t2r, MVar):
        if t1r.id == t2r.id:
            return True, mctx
        if not allow_assign:
            return False, mctx
        # Try assigning t1r := t2r
        ok, _ = mctx.is_valid_assignment(t1r.id, t2r)
        if ok:

            mctx.expr_assignments[t1r.id.id] = t2r
            return True, mctx
        # Try assigning t2r := t1r
        ok, _ = mctx.is_valid_assignment(t2r.id, t1r)
        if ok:
            mctx.expr_assignments[t2r.id.id] = t1r
            return True, mctx
        return False, mctx

    if isinstance(t1r, MVar):
        if not allow_assign:
            return False, mctx
        ok, _ = mctx.is_valid_assignment(t1r.id, t2r)
        if ok:
            mctx.expr_assignments[t1r.id.id] = t2r
            return True, mctx
        return False, mctx

    if isinstance(t2r, MVar):
        if not allow_assign:
            return False, mctx
        ok, _ = mctx.is_valid_assignment(t2r.id, t1r)
        if ok:
            mctx.expr_assignments[t2r.id.id] = t1r
            return True, mctx
        return False, mctx

    # Nat literal resolves to succ chain
    lit1 = lit_nat_int(t1r)
    lit2 = lit_nat_int(t2r)
    if lit1 is not None and lit2 is None:
        chain2 = succ_chain_nat_int(t2r, env, lctx, mctx)
        if chain2 is not None:
            return lit1 == chain2, mctx
    if lit2 is not None and lit1 is None:
        chain1 = succ_chain_nat_int(t1r, env, lctx, mctx)
        if chain1 is not None:
            return chain1 == lit2, mctx

    # structural comparison
    # Sort
    if isinstance(t1r, Sort) and isinstance(t2r, Sort):
        return is_level_def_eq(t1r.level, t2r.level, mctx)

    # Const equal if same name and same level arguments
    if isinstance(t1r, Const) and isinstance(t2r, Const):
        if t1r.name != t2r.name:
            return False, mctx
        if len(t1r.levels) != len(t2r.levels):
            return False, mctx
        for l1, l2 in zip(t1r.levels, t2r.levels):
            ok, _ = is_level_def_eq(l1, l2, mctx)
            if not ok:
                return False, mctx
        return True, mctx

    # FVar same id
    if isinstance(t1r, FVar) and isinstance(t2r, FVar):
        return t1r.id == t2r.id, mctx

    # BVar same de Bruijn index
    if isinstance(t1r, BVar) and isinstance(t2r, BVar):
        return t1r.idx == t2r.idx, mctx

    # Lit syntactic equality
    if isinstance(t1r, Lit) and isinstance(t2r, Lit):

        def _lit_raw(lit: Lit) -> object:
            v = lit.val
            return v.val if isinstance(v, (NatLit, FloatLit)) else v

        return _lit_raw(t1r) == _lit_raw(t2r), mctx

    # App check function and argument
    if isinstance(t1r, App) and isinstance(t2r, App):
        ok, mctx = is_def_eq(t1r.func, t2r.func, env, lctx, mctx, allow_assign)
        if not ok:
            return False, mctx
        return is_def_eq(t1r.arg, t2r.arg, env, lctx, mctx, allow_assign)

    # Lam binder types equal
    if isinstance(t1r, Lam) and isinstance(t2r, Lam):
        ok, mctx = is_def_eq(t1r.binder_type, t2r.binder_type, env, lctx, mctx,
                             allow_assign)
        if not ok:
            return False, mctx
        # bodies equal under a fresh FVar
        lctx2, fv = lctx.push_local(t1r.binder_name, t1r.binder_type)
        b1 = instantiate1(t1r.body, fv)
        b2 = instantiate1(t2r.body, fv)
        return is_def_eq(b1, b2, env, lctx2, mctx, allow_assign)

    # ForallE equal domain
    if isinstance(t1r, ForallE) and isinstance(t2r, ForallE):
        ok, mctx = is_def_eq(t1r.binder_type, t2r.binder_type, env, lctx, mctx,
                             allow_assign)
        if not ok:
            return False, mctx
        # codomain equal under a fresh FVar
        lctx2, fv = lctx.push_local(t1r.binder_name, t1r.binder_type)
        cod1 = instantiate1(t1r.body, fv)
        cod2 = instantiate1(t2r.body, fv)
        return is_def_eq(cod1, cod2, env, lctx2, mctx, allow_assign)

    # Proj compare inner expressions (head unification)
    if isinstance(t1r, Proj) and isinstance(t2r, Proj):
        if t1r.type_name != t2r.type_name or t1r.idx != t2r.idx:
            return False, mctx
        return is_def_eq(t1r.expr, t2r.expr, env, lctx, mctx, allow_assign)

    # After whnf, if one side is a Lam and the other is not, η-expand
    # the non Lam by applying it to a fresh FVar and compare bodies.
    if isinstance(t1r, Lam) and not isinstance(t2r, Lam):
        lctx2, fv = lctx.push_local(t1r.binder_name, t1r.binder_type)
        b1 = instantiate1(t1r.body, fv)
        b2 = App(t2r, fv)
        return is_def_eq(b1, b2, env, lctx2, mctx, allow_assign)

    if isinstance(t2r, Lam) and not isinstance(t1r, Lam):
        lctx2, fv = lctx.push_local(t2r.binder_name, t2r.binder_type)
        b1 = App(t1r, fv)
        b2 = instantiate1(t2r.body, fv)
        return is_def_eq(b1, b2, env, lctx2, mctx, allow_assign)

    # Different head forms
    return False, mctx


# Must be a binop from Nat.rec for shortcut
NAT_LITERAL_BINOPS = {
    "Nat.add": lambda a, b: a + b,
    "Nat.mul": lambda a, b: a * b,
    "Nat.sub": lambda a, b: max(a - b, 0),
}


def try_nat_literal_binop(expr: Expr, env: Environment, lctx: LocalContext,
                          mctx: MetaVarContext) -> Optional[Expr]:
    """
    Checks if ``expr`` is ``Nat.add``/``Nat.mul``/``Nat.sub`` applied to
    two literal ``Nat``, and returns the computed ``Lit``.

    Parameters
    ----------
    expr: Expr
        Expression to check for a literal ``Nat`` binop application.
    env: Environment
        CIC environment to compute ``whnf`` on each operand.
    lctx: LocalContext
        Local context  to compute ``whnf`` on each operand.
    mctx: MetaVarContext
        Metavariable context to compute `whnf`` on each operand.

    Example
    -------
    >>> from physika.core.reduction import try_nat_literal_binop
    >>> from physika.core.expr import Const, App, Lit
    >>> from physika.core.environment import Environment
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> expr = App(App(Const("Nat.mul", ()), Lit(3)), Lit(4))
    >>> try_nat_literal_binop(expr, Environment(), LocalContext(), MetaVarContext())  # noqa: E501
    Lit(val=12)
    """
    head, args = get_app_fn_args(expr)
    if not (isinstance(head, Const) and head.name in NAT_LITERAL_BINOPS
            and len(args) == 2):
        return None
    a = lit_nat_int(whnf(args[0], env, lctx, mctx))
    if a is None:
        return None
    b = lit_nat_int(whnf(args[1], env, lctx, mctx))
    if b is None:
        return None
    return Lit(NAT_LITERAL_BINOPS[head.name](a, b))


def try_iota(fn: Expr, last_arg: Expr, env: Environment, lctx: LocalContext,
             mctx: MetaVarContext) -> Optional[Expr]:
    """
    Iota-reduction on ``App(fn, last_arg)``, where ``fn`` is already
    in ``whnf``. ``try_iota`` returns the reduced expression on success.

    Parameters
    ----------
    fn: Expr
        whnf function being applied.
    last_arg: Expr
        Final argument of the application being reduced.
    env: Environment
        CIC environment, used to look up the recursor's inductive info.
    lctx: LocalContext
        Local context to use``whnf`` on the major premise.
    mctx: MetaVarContext
        Metavariable context to use ``whnf`` on the major premise.

    Example
    -------
    >>> from physika.core.reduction import try_iota
    >>> from physika.core.expr import Const, App, Lit, BVar, TYPE_0
    >>> from physika.core.inductive import InductiveDecl, Constructor, Recursor, RecursorRule
    >>> from physika.core.environment import Environment, InductiveInfo, ConstantInfo
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> nat = Const("Nat", ())
    >>> nat_decl = InductiveDecl(
    ...     name="Nat", level_params=(), num_params=0, type=TYPE_0,
    ...     constructors=(Constructor("Nat.zero", nat), Constructor("Nat.succ", nat)),
    ...     is_recursive=True,
    ... )
    >>> zero_rule = RecursorRule("Nat.zero", nfields=0, rhs=BVar(1))
    >>> nat_rec = Recursor(
    ...     name="Nat.rec", type=TYPE_0, num_params=0, num_indices=0,
    ...     num_motives=1, num_minors=2, rules=(zero_rule,),
    ... )
    >>> env = Environment()
    >>> env.add_inductive(InductiveInfo(
    ...     decl=nat_decl,
    ...     ctors={"Nat.zero": ConstantInfo("Nat.zero", (), nat, None),
    ...            "Nat.succ": ConstantInfo("Nat.succ", (), nat, None)},
    ...     recursor=ConstantInfo("Nat.rec", (), TYPE_0, None),
    ...     rec_info=nat_rec,
    ... ))
    >>> fn = App(App(App(Const("Nat.rec", ()), nat), Lit(10)), nat)
    >>> try_iota(fn, Const("Nat.zero", ()), env, LocalContext(), MetaVarContext())  # noqa: E501
    Lit(val=10)
    """
    head, args = get_app_fn_args(App(fn, last_arg))
    if not isinstance(head, Const):
        return None

    ii = next((cand for cand in env.inductives.values()
               if cand.recursor.name == head.name), None)
    if ii is None or ii.rec_info is None:
        return None

    rec = ii.rec_info
    arity = rec.arity()
    # verify the right number of args for a full application.

    if len(args) < arity:
        return None

    # Split any application args
    core_args = list(args[:arity])
    extra_args = list(args[arity:])

    major_premise = core_args[-1]
    non_major_args = core_args[:-1]  # [motive, minor_0, minor_1, ...]

    # Reduce the major premise to expose the constructor
    major_whnf = whnf(major_premise, env, lctx, mctx)

    # Nat literals are an alternative of a Nat.zero/Nat.succ chain
    ctor_head: Expr
    ctor_args: List[Expr]
    if ii.decl.name == "Nat":
        n = lit_nat_int(major_whnf)
        if n is not None:
            if n == 0:
                ctor_head, ctor_args = Const("Nat.zero", ()), []
            else:
                ctor_head, ctor_args = Const("Nat.succ",
                                             ()), [Lit(NatLit(n - 1))]
        else:
            ctor_head, ctor_args = get_app_fn_args(major_whnf)
    else:
        ctor_head, ctor_args = get_app_fn_args(major_whnf)

    if not isinstance(ctor_head, Const):
        return None  # major premise not in constructor form

    # Find matching RecursorRule
    rule = None
    for r in rec.rules:
        if r.ctor_name == ctor_head.name:
            rule = r
            break
    if rule is None:
        return None

    # Build substitution list of constructor fields, motive, minor, major
    # premises, etc
    num_params = ii.decl.num_params
    ctor_fields = list(ctor_args[num_params:num_params + rule.nfields])
    subst = ctor_fields + non_major_args

    rule_rhs = rule.rhs

    if rec.level_params and head.levels:
        rule_rhs = instantiate_level_params_in_expr(rule_rhs,
                                                    list(rec.level_params),
                                                    list(head.levels))

    result = instantiate(rule_rhs, subst)

    # Apply any application arguments left
    for extra in extra_args:
        result = App(result, extra)

    return result
