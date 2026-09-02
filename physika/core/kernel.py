from physika.core.expr import (
    App,
    BVar,
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
    instantiate1,
    instantiate_level_params_in_expr,
)
from physika.core.level import (
    LZero,
    LSucc,
)
from physika.utils.cic_utils.level_utils import mk_level_imax
from physika.core.reduction import whnf, is_def_eq
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.environment import Environment


class KernelException(Exception):
    """Raised when the kernel rejects a term as bad typed term."""


def proj_field_type(type_name: str, idx: int, struct_expr: Expr,
                    env: Environment, lctx: LocalContext,
                    mctx: MetaVarContext) -> Expr:
    """
    Get the type of a Proj expression. This means, that for a give physika
    class (``struct_expr``) retrurn the CIC type of its ``idx``'th field.

    Look for ``type_name`` single-constructor inductive and infer the type of
    ``struct_expr`` type arguments (the parameters applied to the inductive
    type head). Then, walk the constructor's ForallE chain and skipping ``idx`` field
    binders to return the next binder type (class field's type).

    Parameters
    ----------
    type_name : str
        Inductive type name of Proj expression.
    idx: int
        Field index number to retrieve its type
    struct_expr: Expr
        Proj expression
    env: Environment
        Global environment containg inductive types definitions.
    lctx: LocalContext
        Local scope for free variables ``struct_expr`` references.
    mctx: MetaVarContext
        Metavariable assignment table.

    Example
    -------
    >>> from physika.core.environment import Environment, ConstantInfo, InductiveInfo  # noqa: E501
    >>> from physika.core.inductive import InductiveDecl, Constructor
    >>> from physika.core.expr import ForallE, Const, App, Lit, NatLit, TYPE_0
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.core.kernel import proj_field_type
    >>> nat = Const("Nat", ())
    >>> ctor_type = ForallE("n", nat, ForallE("m", nat, Const("Box", ())))
    >>> decl = InductiveDecl(
    ...     name="Box", level_params=(), num_params=0, type=TYPE_0,
    ...     constructors=(Constructor("Box.mk", ctor_type),),
    ...     is_recursive=False,
    ... )
    >>> env = Environment()
    >>> env.add_inductive(InductiveInfo(
    ...     decl=decl,
    ...     ctors={"Box.mk": ConstantInfo("Box.mk", (), ctor_type, None)},
    ...     recursor=ConstantInfo("Box.rec", (), TYPE_0, None),
    ... ))
    >>> b = App(App(Const("Box.mk", ()), Lit(NatLit(3))), Lit(NatLit(5)))
    >>> proj_field_type("Box", 0, b, env, LocalContext(), MetaVarContext())
    Const(name='Nat', levels=())
    """
    ii = env.inductives.get(type_name)
    if ii is None:
        raise KernelException(
            f"kernel Proj: inductive '{type_name}' not in environment")
    if len(ii.decl.constructors) != 1:
        raise KernelException(
            f"kernel Proj: '{type_name}' is not a structure "
            f"(expected 1 constructor, got {len(ii.decl.constructors)})")

    ctor_name = ii.decl.constructors[0].name
    ctor_ci = env.constants.get(ctor_name)

    if ctor_ci is None:
        raise KernelException(
            f"kernel Proj: constructor '{ctor_name}' not found")

    # Infer class type and extract type arguments
    struct_type = infer_type(struct_expr, env, lctx, mctx)
    struct_whnf = whnf(struct_type, env, lctx, mctx)
    _, type_args = get_app_fn_args(struct_whnf)

    # Constructor type for universe polymorphic constructors
    ctor_type: Expr = ctor_ci.type
    if ctor_ci.level_params and ii.decl.level_params:
        # Use level zero params
        zero_levels = tuple(LZero() for _ in ctor_ci.level_params)
        ctor_type = instantiate_level_params_in_expr(
            ctor_type, list(ctor_ci.level_params), list(zero_levels))

    # Skip num_params implicit Pi-binders by instantiating with type_args
    tp = ctor_type
    num_params = ii.decl.num_params
    for i in range(num_params):
        tp_w = whnf(tp, env, lctx, mctx)
        if not isinstance(tp_w, ForallE):
            raise KernelException(
                f"kernel Proj: constructor type too short at param {i}")
        if i >= len(type_args):
            raise KernelException(
                f"kernel Proj: struct_expr's inferred type supplies only "
                f"{len(type_args)} type argument(s), expected {num_params}")
        tp = instantiate1(tp_w.body, type_args[i])

    # Skip idx-1 field binders by instantiating with
    # Proj(type_name, j, struct_expr)
    for j in range(idx):
        tp_w = whnf(tp, env, lctx, mctx)
        if not isinstance(tp_w, ForallE):
            raise KernelException(
                f"kernel Proj: constructor type too short at field {j}")
        field_j = Proj(type_name, j, struct_expr)
        tp = instantiate1(tp_w.body, field_j)

    # target field
    tp_w = whnf(tp, env, lctx, mctx)
    if not isinstance(tp_w, ForallE):
        raise KernelException(
            f"kernel Proj: could not find field type at index {idx} "
            f"in constructor '{ctor_name}'")
    return tp_w.binder_type


def infer_type(expr: Expr, env: Environment, lctx: LocalContext,
               mctx: MetaVarContext) -> Expr:
    """
    Infer the type of ``exp: Expr``.

    Kernel's trusted type-inference function handles 12 Expr constructors
    If a term is not fully elaborated, a KernelException is raised.

    Parameters
    ----------
    expr : Expr
        A fully elaborated CIC expression without unresolved MVar.
    env : Environment
        The global constant environment.
    lctx : LocalContext
        Local context with FVar declarations currently in scope.
    mctx : MetaVarContext
        Metavariable context that should no contain unassigned MVars.

    Example
    -------
    >>> from physika.core.kernel import infer_type
    >>> from physika.core.expr import Sort, Lit, NatLit
    >>> from physika.core.level import LZero
    >>> from physika.core.environment import Environment
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
    >>> infer_type(Lit(NatLit(5)), env, lctx, mctx)
    Const(name='Nat', levels=())
    >>> infer_type(Sort(LZero()), env, lctx, mctx)  # Prop : Type 0
    Sort(level=LSucc(pred=LZero()))
    """

    if isinstance(expr, MData):
        return infer_type(expr.expr, env, lctx, mctx)

    if isinstance(expr, Sort):
        return Sort(LSucc(expr.level))

    if isinstance(expr, BVar):
        raise KernelException(f"loose BVar({expr.idx})")

    if isinstance(expr, MVar):
        decl = mctx.find_decl(expr.id)
        if decl is None:
            raise KernelException(f"MVar '{expr.id.id}' not declared in mctx")
        assigned = mctx.expr_assignments.get(expr.id.id)
        if assigned is not None:
            return infer_type(assigned, env, lctx, mctx)
        raise KernelException(
            f"Unresolved MVar '{expr.id.id}'"
            "term must be fully elaborated before kernel checking")

    if isinstance(expr, FVar):
        fvar_decl = lctx.find(expr.id)
        if fvar_decl is None:
            raise KernelException(f"FVar '{expr.id.id}' not in local context")
        return fvar_decl.type

    if isinstance(expr, Const):
        ci = env.constants.get(expr.name)
        if ci is None:
            raise KernelException(f"Unknown constant '{expr.name}'")
        if ci.level_params and len(expr.levels) != len(ci.level_params):

            raise KernelException(
                f"constant '{expr.name}' expects "
                f"{len(ci.level_params)} universe level argument(s), got "
                f"{len(expr.levels)}")
        tp = ci.type
        if ci.level_params and expr.levels:
            tp = instantiate_level_params_in_expr(tp, list(ci.level_params),
                                                  list(expr.levels))
        return tp

    if isinstance(expr, App):
        fn_type = infer_type(expr.func, env, lctx, mctx)
        fn_whnf = whnf(fn_type, env, lctx, mctx)
        if not isinstance(fn_whnf, ForallE):
            raise KernelException(f"Function has non-Pi type "
                                  f"(got {type(fn_whnf).__name__})")

        arg_type = infer_type(expr.arg, env, lctx, mctx)
        ok, _ = is_def_eq(arg_type,
                          fn_whnf.binder_type,
                          env,
                          lctx,
                          mctx,
                          allow_assign=False)
        if not ok:
            raise KernelException(
                "Argument type does not match expected "
                f"parameter type (binder {fn_whnf.binder_name!r})")
        return instantiate1(fn_whnf.body, expr.arg)

    if isinstance(expr, Lam):
        dom = expr.binder_type
        new_lctx, fv = lctx.push_local(expr.binder_name, dom)
        body_open = instantiate1(expr.body, fv)
        cod_open = infer_type(body_open, env, new_lctx, mctx)
        cod_closed = abstract_fvars(cod_open, [fv])
        return ForallE(expr.binder_name, dom, cod_closed, expr.binder_info)

    if isinstance(expr, ForallE):
        # domain
        tp_inferred = infer_type(expr.binder_type, env, lctx, mctx)
        tp_whnf = whnf(tp_inferred, env, lctx, mctx)
        if isinstance(tp_whnf, Sort):
            dom_level = tp_whnf.level
        else:
            raise KernelException(
                f"kernel: expression is not a type (expected Sort, got {type(tp_whnf).__name__})"  # noqa: E501
            )
        new_lctx, fv = lctx.push_local(expr.binder_name, expr.binder_type)
        # codomain
        cod_open = instantiate1(expr.body, fv)
        tp_inferred_cod = infer_type(cod_open, env, new_lctx, mctx)
        tp_whnf_cod = whnf(tp_inferred_cod, env, new_lctx, mctx)
        if isinstance(tp_whnf_cod, Sort):
            cod_level = tp_whnf_cod.level
        else:
            raise KernelException(
                f"kernel: expression is not a type (expected Sort, got {type(tp_whnf_cod).__name__})"  # noqa: E501
            )
        return Sort(mk_level_imax(dom_level, cod_level))

    if isinstance(expr, LetE):
        reduced = instantiate1(expr.body, expr.value)
        return infer_type(reduced, env, lctx, mctx)

    if isinstance(expr, Lit):
        v = expr.val
        if isinstance(v, (int, NatLit)):
            return Const("Nat", ())
        if isinstance(v, (float, FloatLit)):
            return Const("Real", ())
        raise KernelException(f"Unknown literal type {type(v).__name__}")

    if isinstance(expr, Proj):
        return proj_field_type(expr.type_name, expr.idx, expr.expr, env, lctx,
                               mctx)

    raise KernelException(f"Unhandled Expr node {type(expr).__name__}")


def check(expr: Expr, expected: Expr, env: Environment, lctx: LocalContext,
          mctx: MetaVarContext) -> None:
    """
    Infer type of ``expr`` and verifies is definitionally equal to
    ``expected``.

    Parameters
    ----------
    expr : Expr
        Term to be verified
    expected : Expr
        Declared type to compared against
    env: Environment
        Global constant environment
    lctx: LocalContext
        Local context that contains FVars
    mctx:
        Metavriable context that should not contain unresolved metavariables

    Example
    -------
    >>> from physika.core.kernel import check
    >>> from physika.core.expr import Lit, NatLit, Const
    >>> from physika.core.environment import Environment
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
    >>> check(Lit(NatLit(5)), Const("Nat", ()), env, lctx, mctx)
    """
    try:
        inferred = infer_type(expr, env, lctx, mctx)
    except KernelException:
        raise
    except Exception as e:
        raise KernelException(f"infer_type failed: {e}") from e

    ok, _ = is_def_eq(inferred, expected, env, lctx, mctx, allow_assign=False)
    if not ok:
        raise KernelException("inferred type does not match declared type")
