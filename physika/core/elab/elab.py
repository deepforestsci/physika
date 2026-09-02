from typing import Dict, List, Optional, Tuple
from physika.core.environment import ConstantInfo, InductiveInfo
from physika.core.inductive import InductiveDecl, Constructor
from physika.core.level import (
    Level,
    LSucc,
)
from physika.utils.cic_utils.level_utils import mk_level_imax
from physika.core.expr import (
    App,
    BVar,
    Const,
    Expr,
    FVar,
    FVarId,
    ForallE,
    Lam,
    LetE,
    Lit,
    MData,
    MVar,
    Proj,
    Sort,
    BinderInfo,
)
from physika.utils.cic_utils.expr_utils import (
    instantiate1,
    instantiate_level_params_in_expr,
    has_mvar,
)
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext, MetaVarContextState, MetaVarKind  # noqa: E501
from physika.core.environment import Environment
from physika.core.reduction import is_def_eq, whnf
from physika.core import kernel
from physika.core.kernel import KernelException

from physika.core.elab.dim_typespec import (
    _TYPE_0_LEVEL,
    collect_dim_vars_ordered,
    elaborate_func_type,
    elaborate_method_type,
    elaborate_struct_kind_and_ctor,
    struct_field_names,
    typespec_to_cic_resolved,
)
from physika.core.elab.termination import (
    check_mutual_recursion_termination,
    check_recursive_termination,
)
# from .tactics import (
#     _tactic_run,
# )
from physika.core.elab.body_elab import (
    # elaborate_dist_call,
    elaborate_expr,
    elaborate_stmts_with_return,
    error_type_str,
)


def open_telescope(cic_type: Expr, elab: "Elab") -> Tuple:
    """
    Open a ForallE chain, introducing every binder as a new FVar.

    After adding each binder, ``open_telescope`` returns a new Elab with
    all binders in its local context. Also, maps each ``binder_name`` to FVar
    (dim vars AND params). When two binders share a display name (e.g.
    ``def f(v: ℝ[n], n: ℝ)``), an implicit dim var and an explicit param
    both named ``n``, ``fvar_env`` keeps only the last one under that its
    declared name. ``open_telescope`` also tracks parameters in order and
    return type.

    Parameters
    ----------
    cic_type : Expr
        CIC type, opened while it's a ``ForallE`` chain.
    elab : Elab
        Elaborator to extend with one fresh FVar per binder.


    Example
    -------
    >>> from physika.core.elab.elab import open_telescope, Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import ForallE, Const, BinderInfo
    >>> elab = Elab(Environment())
    >>> real = Const("Real", ())
    >>> pi = ForallE("x", real, real, BinderInfo.DEFAULT)
    >>> inner, fvar_env, binder_order, current = open_telescope(pi, elab)
    >>> list(fvar_env)
    ['x']
    >>> current == real
    True
    """
    inner = elab
    fvar_env: dict = {}
    binder_order: List[Tuple[str, FVar]] = []
    current = cic_type
    while isinstance(current, ForallE):
        inner, fv = inner.with_local(current.binder_name, current.binder_type)
        fvar_env[current.binder_name] = fv
        binder_order.append((current.binder_name, fv))
        current = instantiate1(current.body, fv)
    return inner, fvar_env, binder_order, current


def resolve_binder_names(
        binder_order: List[Tuple[str, FVar]],
        is_dim_var: List[bool]) -> Tuple[Dict[FVarId, str], List[str]]:
    """
    Assign each binder in ``binder_order`` a identifier,
    when dim var display name conflicts with a non-dim-var binder's own name
    (e.g. ``def f(v: ℝ[n], n: ℝ)``), or a
    method's ``class_dim_vars``/``method_dim_vars`` with its
    own explicit param names.

    Non dim var binder always keeps its name. The conflicting dim-var
    occurrence is renamed, to ``__dim_<name>``.

    Parameters
    ----------
    binder_order : list
        Binders in Pi-scope order, duplicates included.
    is_dim_var : list
        True at each position that's an implicit dim-var binder.

    Example
    -------
    >>> from physika.core.elab.elab import resolve_binder_names
    >>> from physika.core.expr import FVar, FVarId
    >>> n_dim_fv = FVar(FVarId("n.0"))
    >>> n_param_fv = FVar(FVarId("n.1"))
    >>> binder_order = [("n", n_dim_fv), ("n", n_param_fv)]
    >>> is_dim_var = [True, False]
    >>> compiled_names, param_order = resolve_binder_names(binder_order, is_dim_var)  # noqa: E501
    >>> compiled_names == {n_dim_fv.id: "__dim_n", n_param_fv.id: "n"}
    True
    >>> param_order
    ['__dim_n', 'n']
    """
    other_names = {
        name
        for (name, _), dv in zip(binder_order, is_dim_var) if not dv
    }
    compiled_names: Dict[FVarId, str] = {}
    param_order: List[str] = []
    for (name, fv), dv in zip(binder_order, is_dim_var):
        compiled = f"__dim_{name}" if dv and name in other_names else name
        compiled_names[fv.id] = compiled
        param_order.append(compiled)
    return compiled_names, param_order


class ElabError(Exception):
    """Raised when elaboration or type inference fails."""


def find_synthetic_hole_names(mctx: "MetaVarContext", *exprs:
                              Expr) -> List[str]:
    """
    Names of unassigned SYNTHETIC mvars referenced in ``exprs``.

    Parameters
    ----------
    mctx : MetaVarContext
        Metavariable context whose declarations are looked up.
    *exprs : Expr
        One or more CIC terms to scan for unassigned mvars.

    Example
    -------
    >>> from physika.core.elab.elab import Elab, find_synthetic_hole_names
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Const
    >>> from physika.core.metavar import MetaVarKind
    >>> elab = Elab(Environment())
    >>> mv = elab.new_mvar("_hole", Const("Real", ()), kind=MetaVarKind.SYNTHETIC)  # noqa: E501
    >>> find_synthetic_hole_names(elab.state.mctx, mv)
    ['_hole']
    """
    names: List[str] = []
    seen: set = set()
    for e in exprs:
        for mvar_id in mctx.unassigned_mvars(e):
            if mvar_id.id in seen:
                continue
            seen.add(mvar_id.id)
            decl = mctx.find_decl(mvar_id)
            if decl is not None and decl.kind is MetaVarKind.SYNTHETIC:
                names.append(decl.user_name)
    return names


class ElabState:
    """
    Mutable context carried through elaboration.

    Parameters
    ----------
    env : Environment
        Global constants plus inductive types.
    lctx : LocalContext
        Local variables currently in scope.
    mctx : MetaVarContext
        Metavariables and solved CIC terms.
    errors : Optional[List[str]]
        Error messages accumulated during elaboration.

    Example
    -------
    >>> from physika.core.elab.elab import ElabState
    >>> from physika.core.environment import Environment
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> state = ElabState(Environment(), LocalContext(), MetaVarContext())
    >>> state.errors
    []
    """

    def __init__(self,
                 env: Environment,
                 lctx: LocalContext,
                 mctx: MetaVarContext,
                 errors: Optional[List[str]] = None) -> None:
        self.env = env
        self.lctx = lctx
        self.mctx = mctx
        self.errors = errors if errors is not None else []


class Elab:
    """
    A ``Elab`` instance serves for whole elaboration step of a Physika program.
    ``lctx`` is replaced ``with_local`` and mctx is mutated by ``new_mvar`` and
    ``unify`` all in place — one instance lives for the duration of
    elaborating a whole program (or a speculative sub-attempt of one,
    via ``with_local``/``with_let``'s child instances).

    Parameters
    ----------
    env : Environment
        Global constants and inductive types this elaborator resolves
        names against.

    Example
    -------
    >>> from physika.core.elab.elab import Elab
    >>> from physika.core.environment import Environment
    >>> from physika.core.expr import Lit
    >>> elab = Elab(Environment())
    >>> elab.infer_type(Lit(1.0))
    Const(name='Real', levels=())
    """

    def __init__(self, env: Environment) -> None:
        self.state = ElabState(
            env=env,
            lctx=LocalContext(),
            mctx=MetaVarContext(),
        )

    def with_local(self,
                   name: str,
                   type_: Expr,
                   bi: BinderInfo = BinderInfo.DEFAULT) -> Tuple["Elab", FVar]:
        """Extend lctx with a a  name : type_.

        Binder information ``bi`` is recorded on the pushed ``LocalDeclVar``
        so that closing back over this FVar (via lctx.mk_lambda/mk_forall)
        reproduces the same binder kind.

        Returns a new ``Elab`` sharing the same env and mctx (mutations to
        mctx in the inner scope are visible in the outer scope), and the
        fresh FVar representing the local variable.

        Parameters
        ----------
        name : str
            Display name of the new local declaration.
        type_ : Expr
            Its type.
        bi : BinderInfo
            Binder kind to record (default EXPLICIT).


        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Const
        >>> elab = Elab(Environment())
        >>> child, x_fv = elab.with_local("x", Const("Real", ()))
        >>> child is elab
        False
        >>> child.state.lctx.find(x_fv.id).type == Const("Real", ())
        True
        """
        new_lctx, fv = self.state.lctx.push_local(name, type_, bi)
        child = Elab.__new__(Elab)
        child.state = ElabState(
            env=self.state.env,
            lctx=new_lctx,
            mctx=self.state.mctx,
            errors=self.state.errors,
        )
        return child, fv

    def with_let(self, name: str, type_: Expr,
                 value: Expr) -> Tuple["Elab", FVar]:
        """
        Extend lctx with a let-binding:
        ``name : type_ := value``

        Same as ``with_local``, but the pushed declaration also carries
        ``value``. Closing back over the returned FVar (``lctx.mk_lambda``
        /``mk_forall``) produces a real ``LetE`` node instead of an
        opaque binder.

        Parameters
        ----------
        name : str
            Display name of the new binding.
        type_ : Expr
            Its type.
        value : Expr
            Its bound value.

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Const, Lit
        >>> elab = Elab(Environment())
        >>> real = Const("Real", ())
        >>> child, x_fv = elab.with_let("x", real, Lit(1.0))
        >>> decl = child.state.lctx.find(x_fv.id)
        >>> decl.type == real
        True
        >>> decl.value == Lit(1.0)
        True
        """
        new_lctx, fv = self.state.lctx.push_let(name, type_, value)
        child = Elab.__new__(Elab)
        child.state = ElabState(
            env=self.state.env,
            lctx=new_lctx,
            mctx=self.state.mctx,
            errors=self.state.errors,
        )
        return child, fv

    def new_mvar(self,
                 name: str,
                 type_: Expr,
                 kind: MetaVarKind = MetaVarKind.NATURAL) -> MVar:
        """
        Create and register a metavariable.

        Parameters
        ----------
        name : str
            Display name, used to build the mvar's id and shown in
            error messages.
        type_ : Expr
            The metavariable's type.
        kind : MetaVarKind
            ``NATURAL`` (an ordinary elaboration hole),
            ``SYNTHETIC`` (a tactic-inserted placeholder), or
            ``SYNTHETIC_OPAQUE`` (an open tactic goal).

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Const, MVar
        >>> elab = Elab(Environment())
        >>> mv = elab.new_mvar("_hole", Const("Real", ()))
        >>> isinstance(mv, MVar)
        True
        """
        return self.state.mctx.mk_mvar(name, self.state.lctx, type_, kind=kind)

    def new_level_mvar(self) -> Level:
        """
        Create a fresh universe level metavariable.

        Unlike ``new_mvar``, this isn't registered against a
        ``MetaVarContext``. Level metavariables are identified
        by unique id, never solved through ``mctx``.

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.level import LMVar
        >>> elab = Elab(Environment())
        >>> lv = elab.new_level_mvar()
        >>> isinstance(lv, LMVar)
        True
        """
        from physika.core.level import LMVar
        import itertools
        lmvar_counter = itertools.count()

        uid = f"u.{next(lmvar_counter)}"
        return LMVar(uid)

    def unify(self, t1: Expr, t2: Expr) -> bool:
        """
        Unify t1 and t2, while solving MVars on the process.

        Returns True and mutates mctx on success and on failure return False
        and leaves mctx unchanged.

        Parameters
        ----------
        t1: Expr
            First term to check for definitional equality.
        t2 : Expr
            Second term to check for definitional equality.

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Const
        >>> elab = Elab(Environment())
        >>> elab.unify(Const("Real", ()), Const("Real", ()))
        True
        >>> elab.unify(Const("Real", ()), Const("Nat", ()))
        False
        """
        snap = self.state.mctx.save()
        self.state.mctx.depth += 1
        try:
            ok, _ = is_def_eq(
                t1,
                t2,
                self.state.env,
                self.state.lctx,
                self.state.mctx,
            )
        finally:
            self.state.mctx.depth -= 1
        if ok:
            # self.state.mctx was mutated in place by is_def_eq
            return True
        self.state.mctx.restore(snap)
        return False

    def infer_type(self, expr: Expr) -> Expr:
        """
        Infer the type of expr under the current (env, lctx, mctx).

        Raises ElabError for bad typed or open terms.

        Parameters
        ----------
        expr : Expr
            Elaborated CIC term to infer the type of. ``Const``/``FVar``
            it references must already be registered or in scope, or
            ``ElabError`` will be raised.

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Lit
        >>> elab = Elab(Environment())
        >>> elab.infer_type(Lit(1.0))
        Const(name='Real', levels=())
        """
        if isinstance(expr, MData):
            return self.infer_type(expr.expr)

        if isinstance(expr, Sort):
            return Sort(LSucc(expr.level))

        if isinstance(expr, BVar):
            raise ElabError(
                f"infer_type: loose BVar({expr.idx}) — term is not open-closed"
            )

        if isinstance(expr, FVar):
            fvar_decl = self.state.lctx.find(expr.id)
            if fvar_decl is None:
                raise ElabError(
                    f"infer_type: FVar '{expr.id.id}' not in local context")
            return self.state.lctx.fvar_type(expr)

        if isinstance(expr, MVar):
            decl = self.state.mctx.find_decl(expr.id)
            if decl is None:
                raise ElabError(
                    f"infer_type: MVar '{expr.id.id}' not declared in mctx")
            return decl.type

        if isinstance(expr, Const):
            ci = self.state.env.constants.get(expr.name)
            if ci is None:
                raise ElabError(f"infer_type: unknown constant '{expr.name}'")
            t = ci.type
            if ci.level_params and expr.levels:
                t = instantiate_level_params_in_expr(t, list(ci.level_params),
                                                     list(expr.levels))
            return t

        if isinstance(expr, App):
            fn_type = self.infer_type(expr.func)
            fn_type_whnf = whnf(fn_type, self.state.env, self.state.lctx,
                                self.state.mctx)
            if not isinstance(fn_type_whnf, ForallE):
                raise ElabError("infer_type: function has non-Pi type")

            arg_type = self.infer_type(expr.arg)
            if not self.unify(arg_type, fn_type_whnf.binder_type):
                raise ElabError(
                    "infer_type: argument type does not match expected "
                    f"parameter type (binder {fn_type_whnf.binder_name!r})")

            return instantiate1(fn_type_whnf.body, expr.arg)

        if isinstance(expr, Lam):
            dom = expr.binder_type

            inner, fv = self.with_local(expr.binder_name,
                                        dom,
                                        bi=expr.binder_info)
            body_open = instantiate1(expr.body, fv)
            cod_open = inner.infer_type(body_open)
            return inner.state.lctx.mk_forall([fv], cod_open)

        if isinstance(expr, ForallE):
            # whnf reduce each side to a Sort and take
            # its level
            dom_t = self.infer_type(expr.binder_type)
            dom_whnf = whnf(dom_t, self.state.env, self.state.lctx,
                            self.state.mctx)
            if not isinstance(dom_whnf, Sort):
                raise ElabError(
                    f"ForallE domain does not have a Sort type (got {type(dom_whnf).__name__})"  # noqa: E501
                )
            dom_sort = dom_whnf.level
            inner, fv = self.with_local(expr.binder_name, expr.binder_type)
            cod_open = instantiate1(expr.body, fv)
            cod_t = inner.infer_type(cod_open)
            cod_whnf = whnf(cod_t, inner.state.env, inner.state.lctx,
                            inner.state.mctx)
            if not isinstance(cod_whnf, Sort):
                raise ElabError(
                    f"ForallE codomain does not have a Sort type (got {type(cod_whnf).__name__})"  # noqa: E501
                )
            cod_sort = cod_whnf.level
            pi_level = mk_level_imax(dom_sort, cod_sort)
            return Sort(pi_level)

        if isinstance(expr, LetE):
            reduced = instantiate1(expr.body, expr.value)
            return self.infer_type(reduced)

        if isinstance(expr, Lit):
            if isinstance(expr.val, int):
                return Const("Nat", ())
            else:  # float
                return Const("Real", ())

        if isinstance(expr, Proj):
            try:
                return kernel.proj_field_type(
                    expr.type_name,
                    expr.idx,
                    expr.expr,
                    self.state.env,
                    self.state.lctx,
                    self.state.mctx,
                )
            except KernelException as exc:
                raise ElabError(f"infer_type: Proj failed: {exc}") from exc

        raise ElabError(
            f"infer_type: unhandled expression {type(expr).__name__}")

    def check(self, expr: Expr, expected: Expr) -> None:
        """
        Verify that expr have expected type. Raises ElabError if the check
        fails.

        Parameters
        ----------
        expr : Expr
            Elaborated CIC term to check.
        expected : Expr
            The type it must have (up to ``unify``).

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Lit, Const
        >>> elab = Elab(Environment())
        >>> elab.check(Lit(1.0), Const("Real", ()))
        >>> elab.check(Lit(1.0), Const("Nat", ()))  # noqa: E501
        Traceback (most recent call last):
            ...
        physika.core.elab.elab.ElabError: type mismatch: inferred type does not match expected type
        """
        inferred = self.infer_type(expr)
        if not self.unify(inferred, expected):
            raise ElabError(
                "type mismatch: inferred type does not match expected type")

    def elaborate(self, unified_ast: dict) -> dict:
        """Translate a Physika unified AST dict to CIC terms.

        First, converts each function definition in
        ``unified_ast["functions"]`` into a ``ForallE`` chain representing
        its dependent type in CIC. Dimension variables (e.g. ``n`` in
        ``ℝ[n]``) become implicit ``ForallE`` binders over ``Nat`` and
        function parameters become explicit ``ForallE`` binders. Then, each
        functions, class, classes methods bodies and top level programs
        (including funciton calls) are elaborated into CIC terms.

        Parameters
        ----------
        unified_ast : dict
            Output of ``build_unified_ast()``; must contain a
            ``"functions"`` key mapping names to function defs.

        Examples
        --------
        >>> from physika.core.environment import Environment
        >>> from physika.core.elab import Elab
        >>> from physika.core.expr import ForallE, Const, App, BVar, BinderInfo
        >>> elab = Elab(Environment())
        >>> # def id_real(x: ℝ): ℝ  →  Π (x : Real), Real
        >>> func_def = {"params": [("x", "ℝ")], "return_type": "ℝ",
        ...             "body": ("var", "x"), "statements": []}
        >>> result = elab.elaborate({"functions": {"id_real": func_def},
        ...                          "classes": {}, "program": []})
        >>> pi = result["functions"]["id_real"]
        >>> isinstance(pi, ForallE)
        True
        >>> pi.binder_name
        'x'
        >>> pi.binder_type == Const("Real", ())
        True
        >>> pi.body == Const("Real", ())
        True
        >>> pi.binder_info == BinderInfo.DEFAULT
        True
        """
        errors: list = []
        funcs_out: dict = {}
        resolved_bodies: dict = {}
        resolved_methods: dict = {}

        # Step #1: Register all symbols in the CIC env
        # Classes are registered as single constructor inductive
        # structures (Foo.mk), following what Lean 4 uses for
        # structures
        class_meta: dict = {}
        for name, class_def in unified_ast.get("classes", {}).items():
            fields = class_def.get("class_params", [])
            if f"{name}.mk" in self.state.env.constants:
                ii = self.state.env.inductives.get(name)
                if ii is not None:
                    class_meta[name] = (
                        ii.decl.num_params,
                        collect_dim_vars_ordered(fields, None),
                    )
                continue
            if name in self.state.env.constants:
                errors.append(
                    f"class '{name}' conflicts with an existing CIC name "
                    f"(a built-in, e.g. Vec/Nat/Real, or another "
                    f"declaration): name already registered")
                continue
            try:
                kind, ctor_type, num_params, dim_vars = (
                    elaborate_struct_kind_and_ctor(name, fields))

            except Exception as exc:
                errors.append(
                    f"class '{name}': could not elaborate fields: {exc}")
                continue
            ctor_name = f"{name}.mk"
            decl = InductiveDecl(
                name=name,
                level_params=(),
                num_params=num_params,
                type=kind,
                constructors=(Constructor(ctor_name, ctor_type), ),
                is_recursive=False,
            )
            ctor_ci = ConstantInfo(name=ctor_name,
                                   level_params=(),
                                   type=ctor_type,
                                   value=None)
            # No ι-reduction rules (rec_info=None)
            rec_ci = ConstantInfo(name=f"{name}.rec",
                                  level_params=(),
                                  type=_TYPE_0_LEVEL,
                                  value=None)
            try:
                self.state.env.add_inductive(
                    InductiveInfo(
                        decl=decl,
                        ctors={ctor_name: ctor_ci},
                        recursor=rec_ci,
                        rec_info=None,
                    ))
            except ValueError as exc:
                errors.append(f"class '{name}': {exc}")
                continue
            class_meta[name] = (num_params, dim_vars)

        # Method signatures registered as `ClassName.method` constants
        # with an explicit leading `this : ClassName` parameter.
        _pre_existing = {c.name for c in self.state.env.constants.values()}
        for name, class_def in unified_ast.get("classes", {}).items():
            if name not in class_meta:
                continue
            num_params, dim_vars = class_meta[name]
            for method in class_def.get("methods", []):
                m_name = method.get("name")
                qualified = f"{name}.{m_name}"
                if qualified in _pre_existing:
                    errors.append(
                        f"method '{qualified}' conflicts with an existing "
                        f"CIC name (a built-in axiom or another "
                        f"declaration): name already registered")
                    continue
                if qualified in self.state.env.constants:
                    continue
                try:
                    m_type = elaborate_method_type(name, num_params, dim_vars,
                                                   method)
                except Exception as exc:
                    errors.append(
                        f"In method '{qualified}': could not elaborate "
                        f"signature: {exc}")
                    continue
                self.state.env.add_constant(
                    ConstantInfo(name=qualified,
                                 level_params=(),
                                 type=m_type,
                                 value=None))

        for name, func_def in unified_ast.get("functions", {}).items():
            try:
                cic_type = elaborate_func_type(func_def, self)
            except ValueError as exc:
                errors.append(f"In function '{name}': {exc}")
                continue
            funcs_out[name] = cic_type
            if name not in self.state.env.constants:
                self.state.env.add_constant(
                    ConstantInfo(name=name,
                                 level_params=(),
                                 type=cic_type,
                                 value=None))

        # Step #2: Verify eeach function body with trusted kernel.
        # Elaborate the body (sequence of local `body_decl`, `body_assign`, etc
        # statements followed by a final return expression). Then,
        # verify the result with kernel using ``kernel.check``. An
        # unresolved MVar anywhere in the term is rejected.

        # Check mutial recursion
        mutual_cycle_members = check_mutual_recursion_termination(
            unified_ast.get("functions", {}), errors)

        for name, func_def in unified_ast.get("functions", {}).items():
            if name not in funcs_out:
                continue
            cic_type = funcs_out[name]
            snap = self.save()
            self.state.mctx.depth += 1
            errors_before = len(errors)
            try:
                inner, fvar_env, binder_order, return_type_cic = (
                    open_telescope(cic_type, self))
                body_node = func_def.get("body")
                statements = func_def.get("statements", [])
                has_terminal = body_node is not None or any(
                    isinstance(s, tuple) and s
                    and s[0] == "body_if_else_return" for s in statements)
                if has_terminal:
                    if name not in mutual_cycle_members:
                        check_recursive_termination(name,
                                                    func_def.get("params", []),
                                                    body_node, statements,
                                                    errors,
                                                    f"In function '{name}'")
                    body_cic, inner, cur_env, local_decls, fvar_names = elaborate_stmts_with_return(  # noqa: E501
                        statements, body_node, fvar_env, inner, errors,
                        f"In function '{name}'")
                    body_closed = inner.state.mctx.instantiate_mvars(body_cic)
                    return_type_for_check = return_type_cic
                    try:
                        body_type_for_unify = inner.infer_type(body_closed)
                        if inner.unify(body_type_for_unify, return_type_cic):
                            return_type_for_check = (
                                inner.state.mctx.instantiate_mvars(
                                    return_type_cic))
                    except Exception:
                        pass
                    try:
                        kernel.check(body_closed, return_type_for_check,
                                     inner.state.env, inner.state.lctx,
                                     inner.state.mctx)

                        solved_cic_type = inner.state.mctx.instantiate_mvars(
                            cic_type)
                        self.state.env.constants[name].type = solved_cic_type
                        funcs_out[name] = solved_cic_type
                        local_decls_closed = [
                            (nm, inner.state.mctx.instantiate_mvars(rhs))
                            for nm, rhs in local_decls
                        ]
                        synthetic_holes = find_synthetic_hole_names(
                            inner.state.mctx, body_closed,
                            *(rhs for _, rhs in local_decls_closed))
                        if synthetic_holes:
                            errors.append(
                                f"In function '{name}': an earlier error "
                                f"left unresolved placeholder(s) for "
                                f"{', '.join(synthetic_holes)}")
                        # check no MVars
                        if (not has_mvar(body_closed) and not any(
                                has_mvar(rhs) for _, rhs in local_decls_closed)
                                and len(errors) == errors_before):
                            # check binder args order
                            n_dim_vars = len(binder_order) - len(
                                func_def.get("params", []))
                            is_dim_var = [
                                i < n_dim_vars
                                for i in range(len(binder_order))
                            ]
                            compiled_names, param_order = resolve_binder_names(
                                binder_order, is_dim_var)
                            for _, fv in binder_order:
                                fvar_names[fv.id] = compiled_names[fv.id]
                            dim_rename = {
                                nm: compiled_names[fv.id]
                                for nm, fv in binder_order[:n_dim_vars]
                                if compiled_names[fv.id] != nm
                            }
                            resolved_bodies[name] = (
                                body_closed,
                                fvar_names,
                                local_decls_closed,
                                param_order,
                                dim_rename,
                            )
                    except KernelException as exc:
                        exp_s = error_type_str(return_type_for_check, inner)
                        errors.append(
                            f"In function '{name}': kernel check failed "
                            f"against declared return type {exp_s}: {exc}")
                    except Exception as exc:
                        errors.append(
                            f"In function '{name}': internal error during "
                            f"type checking: {exc}")
            except Exception as exc:
                errors.append(
                    f"In function '{name}': internal error during type "
                    f"checking: {exc}")
            finally:
                self.state.mctx.depth -= 1
                self.restore(snap)

        # Step 3: Verify each class method signature with kernel
        for cname, class_def in unified_ast.get("classes", {}).items():
            if cname not in class_meta:
                continue
            for method in class_def.get("methods", []):
                m_name = method.get("name")
                qualified = f"{cname}.{m_name}"
                method_ci = self.state.env.constants.get(qualified)
                if method_ci is None:
                    continue
                snap = self.save()
                self.state.mctx.depth += 1
                try:
                    inner, fvar_env, binder_order, return_type_cic = (
                        open_telescope(method_ci.type, self))
                    this_fvar = fvar_env.get("this")
                    body_fvar_env = fvar_env
                    if isinstance(this_fvar, FVar):
                        fnames = struct_field_names(cname, self.state.env)
                        if fnames:
                            field_env = {
                                fname: Proj(cname, idx, this_fvar)
                                for idx, fname in enumerate(fnames)
                            }
                            body_fvar_env = {**field_env, **fvar_env}
                    body_node = method.get("body")
                    statements = method.get("statements", [])
                    has_terminal = body_node is not None or any(
                        isinstance(s, tuple) and s
                        and s[0] == "body_if_else_return" for s in statements)
                    if has_terminal:
                        check_recursive_termination(
                            m_name, method.get("params", []), body_node,
                            statements, errors, f"In method '{qualified}'")
                        body_cic, inner, cur_env, local_decls, fvar_names = elaborate_stmts_with_return(  # noqa: E501
                            statements, body_node, body_fvar_env, inner,
                            errors, f"In method '{qualified}'")
                        body_closed = inner.state.mctx.instantiate_mvars(
                            body_cic)
                        try:
                            kernel.check(body_closed, return_type_cic,
                                         inner.state.env, inner.state.lctx,
                                         inner.state.mctx)
                            local_decls_closed = [
                                (nm, inner.state.mctx.instantiate_mvars(rhs))
                                for nm, rhs in local_decls
                            ]
                            synthetic_holes = find_synthetic_hole_names(
                                inner.state.mctx, body_closed,
                                *(rhs for _, rhs in local_decls_closed))
                            if synthetic_holes:
                                errors.append(
                                    f"In method '{qualified}': an earlier "
                                    f"error left unresolved placeholder(s) "
                                    f"for {', '.join(synthetic_holes)}")

                            if not has_mvar(body_closed) and not any(
                                    has_mvar(rhs)
                                    for _, rhs in local_decls_closed):

                                num_class_params, _ = class_meta[cname]
                                n_method_params = len(method.get("params", []))
                                n_method_dim_vars = (len(binder_order) -
                                                     num_class_params - 1 -
                                                     n_method_params)
                                is_dim_var = ([True] * num_class_params +
                                              [False] +
                                              [True] * n_method_dim_vars +
                                              [False] * n_method_params)
                                compiled_names, param_order = resolve_binder_names(  # noqa: E501
                                    binder_order, is_dim_var)
                                for _, fv in binder_order:
                                    fvar_names[fv.id] = compiled_names[fv.id]
                                dim_rename = {
                                    nm: compiled_names[fv.id]
                                    for (nm, fv
                                         ), dv in zip(binder_order, is_dim_var)
                                    if dv and compiled_names[fv.id] != nm
                                }
                                resolved_methods[qualified] = (
                                    body_closed,
                                    fvar_names,
                                    local_decls_closed,
                                    param_order,
                                    dim_rename,
                                )
                        except KernelException as exc:
                            exp_s = error_type_str(return_type_cic, inner)
                            errors.append(
                                f"In method '{qualified}': kernel check "
                                f"failed against declared return type "
                                f"{exp_s}: {exc}")
                        except Exception as exc:
                            errors.append(
                                f"In method '{qualified}': internal error "
                                f"during type checking: {exc}")
                except Exception as exc:
                    errors.append(
                        f"In method '{qualified}': internal error during "
                        f"type checking: {exc}")
                finally:
                    self.state.mctx.depth -= 1
                    self.restore(snap)

        # Step 4: Verify top level programs with kernel
        top_env: dict = {}
        cur_elab = self
        program_fvar_names: dict = {}
        resolved_program: dict = {}
        for idx, stmt in enumerate(unified_ast.get("program", [])):
            if not (isinstance(stmt, tuple) and stmt):
                continue
            tag = stmt[0]
            if tag == "decl" and len(stmt) >= 4:
                var_name, type_spec, expr_ast = stmt[1], stmt[2], stmt[3]
            elif tag == "assign" and len(stmt) >= 3:
                var_name, type_spec, expr_ast = stmt[1], None, stmt[2]
            elif tag == "expr" and len(stmt) >= 2:
                var_name, type_spec, expr_ast = None, None, stmt[1]
            else:
                continue
            known_decl_type = (typespec_to_cic_resolved(
                type_spec, top_env.get) if type_spec is not None else None)
            try:
                cic = elaborate_expr(expr_ast,
                                     top_env,
                                     cur_elab,
                                     errors,
                                     expected_type=known_decl_type)
            except Exception:
                continue
            if isinstance(cic, MVar):
                continue
            try:
                cic_closed = cur_elab.state.mctx.instantiate_mvars(cic)
                if has_mvar(cic_closed):
                    continue
                if known_decl_type is not None:
                    decl_type = known_decl_type
                    kernel.check(cic_closed, decl_type, cur_elab.state.env,
                                 cur_elab.state.lctx, cur_elab.state.mctx)
                else:
                    decl_type = kernel.infer_type(cic_closed,
                                                  cur_elab.state.env,
                                                  cur_elab.state.lctx,
                                                  cur_elab.state.mctx)
            except KernelException as exc:
                where = f"'{var_name}'" if var_name else "top-level expression"
                errors.append(f"At top level ({where}): {exc}")
                continue
            except Exception:
                continue
            if var_name is not None:
                cur_elab, fv = cur_elab.with_let(var_name, decl_type,
                                                 cic_closed)
                top_env = {**top_env, var_name: fv}
                program_fvar_names[fv.id] = var_name
            resolved_program[idx] = (var_name, cic_closed)

        return {
            "functions": funcs_out,
            "errors": errors,
            "resolved_bodies": resolved_bodies,
            "resolved_methods": resolved_methods,
            "resolved_program": resolved_program,
            "resolved_program_fvar_names": program_fvar_names,
        }

    def save(self) -> MetaVarContextState:
        """
        Snapshot mctx assignments when trying unification.

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Const, Lit
        >>> elab = Elab(Environment())
        >>> mv = elab.new_mvar("_h", Const("Real", ()))
        >>> snap = elab.save()
        """
        return self.state.mctx.save()

    def restore(self, snap: MetaVarContextState) -> None:
        """
        Restore mctx to a previous snapshot.

        Parameters
        ----------
        snap : MetaVarContextState
            A snapshot returned by ``save``.

        Example
        -------
        >>> from physika.core.elab.elab import Elab
        >>> from physika.core.environment import Environment
        >>> from physika.core.expr import Const, Lit
        >>> elab = Elab(Environment())
        >>> mv = elab.new_mvar("_h", Const("Real", ()))
        >>> snap = elab.save()
        >>> elab.unify(mv, Lit(1.0))
        True
        >>> elab.state.mctx.instantiate_mvars(mv) == Lit(1.0)
        True
        >>> elab.restore(snap)
        >>> elab.state.mctx.instantiate_mvars(mv) == mv
        True
        """
        self.state.mctx.restore(snap)
