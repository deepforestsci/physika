from typing import Callable, Dict, List, Optional, Tuple, Union

from physika.core.level import (
    LZero,
    LSucc,
)
from physika.core.expr import (
    App,
    BVar,
    Const,
    Expr,
    ForallE,
    Lit,
    MVar,
    NatLit,
    Sort,
    BinderInfo,
)
from physika.core.environment import Environment
from physika.utils.cic_utils.expr_utils import get_app_fn_args

_NAT_CONST = Const("Nat", ())
_REAL_CONST = Const("Real", ())
_VEC_CONST = Const("Vec", ())
_NAT_ADD = Const("Nat.add", ())
_NAT_MUL = Const("Nat.mul", ())
_NAT_SUB = Const("Nat.sub", ())
_TYPE_0_LEVEL = Sort(LSucc(LZero()))  # Type₀  (used as placeholder MVar type)


def dim_leaf_names(d: Union[int, str, tuple]) -> List[str]:
    """
    Returns symbolic dimension var names referenced in a dim entry.

    Parameters
    ----------
    d : Union[int, str, tuple]
        A dimension entry from a type spec. Can be an
        ``int`` literal size, a ``str`` symbolic dim var name, or a
        dimension arithmetic expression (dependent types).

    Examples
    --------
    >>> from physika.core.elab.dim_typespec import dim_leaf_names
    >>> dim_leaf_names(3)
    []
    >>> dim_leaf_names("n")
    ['n']
    >>> dim_leaf_names(("mul_dim", "n", 2))
    ['n']
    >>> dim_leaf_names(("add_dim_id", "n", "m"))
    ['n', 'm']
    """
    if isinstance(d, str):
        return [d]
    if isinstance(d, tuple) and len(d) == 3:
        tag = d[0]
        if tag in ("mul_dim", "add_dim", "sub_dim"):
            return dim_leaf_names(d[1])
        if tag in ("add_dim_id", "sub_dim_id", "mul_dim_id"):
            return dim_leaf_names(d[1]) + dim_leaf_names(d[2])
    return []


def collect_dim_vars_ordered(
        params: list, return_type: Optional[Union[str, tuple]]) -> List[str]:
    """
    Return symbolic dim var names for physika tensors in appearence order.

    Scans ``params`` and  ``return_type`` from left to right and records each
    new string dim variable avoiding repetition.

    Parameters
    ----------
    params: list
        List of function parameters with its name and type
    return_type: Tupe
        Function's return type

    Examples
    --------
    >>> from physika.core.elab.dim_typespec import collect_dim_vars_ordered
    >>> collect_dim_vars_ordered(
    ...     [("u", ("tensor", [("n", "invariant")])),
    ...      ("v", ("tensor", [("n", "invariant")]))],
    ...     "ℝ")
    ['n']
    >>> collect_dim_vars_ordered(
    ...     [("A", ("tensor", [("n", "invariant"), ("m", "invariant")]))],
    ...     ("tensor", [("n", "invariant")]))
    ['n', 'm']
    >>> # n reuses the explicit Nat param's own binder (common case)
    >>> collect_dim_vars_ordered([("n", "ℕ"), ("v", ("tensor", [("n", "invariant")]))], "ℝ")  # noqa: E501
    []
    """
    seen = set()
    result = []

    explicit_nat_param_names = {
        pname
        for pname, ts in params if ts in ("ℕ", "ℤ")
    }
    for ts in [ts for (_, ts) in params] + [return_type]:
        if isinstance(ts, tuple) and ts[0] == "tensor":
            for (d, _) in ts[1]:
                for name in dim_leaf_names(d):
                    if name not in seen and name not in explicit_nat_param_names:  # noqa: E501
                        seen.add(name)
                        result.append(name)
    return result


def dim_bvar(name: str, binder_names: List[str], depth: int) -> BVar:
    """
    Resolve a dimension variable to a ``BVar`` at the current depth.

    ``depth`` refers to binders that are already open at the point this type
    is being built. ``j < depth`` binder positions are in
    scope. Pi types can only depend on binders introduced
    before it.

    Raises an error if ``name`` isn't a binder, or when it names
    a real binder that hasn't been introduced yet.

    Parameters
    ----------
    name : str
        Dimension var name to resolve (``"n"``).
    binder_names : List[str]
        Binder's display name, outermost first.
    depth : int
        Number of binders already open at the point this type is being
        built.

    Examples
    --------
    >>> from physika.core.elab.dim_typespec import dim_bvar
    >>> dim_bvar("n", ["n", "v"], depth=1)
    BVar(idx=0)
    """
    try:
        j = binder_names.index(name)
    except ValueError:
        raise ValueError(
            f"dimension variable '{name}' is not a parameter of this "
            f"function/method/class") from None
    if j >= depth:
        raise ValueError(
            f"dimension variable '{name}' is used before it is declared")
    return BVar(depth - 1 - j)


def dim_to_cic_resolved(
        dim: Union[int, str, tuple],
        resolve: Callable[[str], Optional[Expr]]) -> Optional[Expr]:
    """
    Convert a Physika dimension entry to CIC ``Expr``, given a
    ``resolve`` callback mapping a symbolic dim name to its
    elaborated ``Expr`` (or ``None`` if unresolvable).

    Parameters
    ----------
    dim : Union[int, str, tuple]
        A dimension entry that can be an ``int`` literal, a ``str`` symbolic
        dim-var name, or a dim-arithmetic tuple.
    resolve : Callable[[str], Optional[Expr]]
        Maps a symbolic dim var to its elaborated ``Expr``
        (e.g. a ``BVar`` at some depth, or a ``local_env`` lookup), or
        ``None`` if that name doesn't resolve.

    Examples
    --------
    >>> from physika.core.elab.dim_typespec import dim_to_cic_resolved  # noqa: E501
    >>> dim_to_cic_resolved(3, lambda name: None)
    Lit(val=3)
    >>> dim_to_cic_resolved("n", lambda name: {"n": BVar(0)}.get(name))
    BVar(idx=0)
    >>> dim_to_cic_resolved(("mul_dim", "n", 2), lambda name: {"n": BVar(0)}.get(name))
    App(func=App(func=Const(name='Nat.mul', levels=()), arg=BVar(idx=0)), arg=Lit(val=2))
    >>> dim_to_cic_resolved("missing", lambda name: None) is None
    True
    """
    if isinstance(dim, int):
        return Lit(dim)  # type: ignore[arg-type]
    if isinstance(dim, str):
        return resolve(dim)
    if isinstance(dim, tuple) and len(dim) == 3:
        tag = dim[0]
        if tag in ("mul_dim", "add_dim", "sub_dim"):
            var_expr = resolve(dim[1])
            if var_expr is None:
                return None
            const_expr = Lit(dim[2])
            op = {
                "mul_dim": _NAT_MUL,
                "add_dim": _NAT_ADD,
                "sub_dim": _NAT_SUB
            }[tag]
            return App(App(op, var_expr), const_expr)
        if tag in ("add_dim_id", "sub_dim_id", "mul_dim_id"):
            v1 = resolve(dim[1])
            v2 = resolve(dim[2])
            if v1 is None or v2 is None:
                return None
            op = {
                "add_dim_id": _NAT_ADD,
                "sub_dim_id": _NAT_SUB,
                "mul_dim_id": _NAT_MUL
            }[tag]
            return App(App(op, v1), v2)
    return None


CANON_IDENTITY = {"Nat.add": 0, "Nat.mul": 1}


def nat_lit_int(e: Expr) -> Optional[int]:
    """
    Return a non negative integer value of a ``Nat`` literal.

    Parameters
    ----------
    e : Expr
        Expression to inspect.

    Examples
    --------
    >>> from physika.core.expr import Lit, NatLit, Const
    >>> from physika.core.elab.dim_typespec import nat_lit_int
    >>> nat_lit_int(Lit(3))
    3
    >>> nat_lit_int(Lit(NatLit(4)))
    4
    >>> nat_lit_int(Const("n", ())) is None
    True
    """
    if isinstance(e, Lit):
        v = e.val
        if isinstance(v, NatLit):
            return v.val
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return v
    return None


def flatten_nat_chain(e: Expr, op_name: str, acc: List[Expr]) -> None:
    """
    Flatten a nested ``op_name`` application chain into ``acc``.

    Walks ``e`` collecting leaves of a left and right nested chain of
    Nat operations (``Nat.add`` or ``Nat.mul``).

    Parameters
    ----------
    e : Expr
        Expression to be flattened.
    op_name : str
        `"Nat.add"`` or ``"Nat.mul"``chain operator.
    acc : List[Expr]
        Output list mutated in place.

    Examples
    --------
    >>> from physika.core.expr import App, BVar, Const
    >>> from physika.core.elab.dim_typespec import flatten_nat_chain
    >>> add = Const("Nat.add", ())
    >>> chain = App(App(add, App(App(add, BVar(0)), BVar(1))), BVar(2))
    >>> acc = []
    >>> flatten_nat_chain(chain, "Nat.add", acc)
    >>> acc
    [BVar(idx=0), BVar(idx=1), BVar(idx=2)]
    """
    if isinstance(e, App):
        head, args = get_app_fn_args(e)
        if (isinstance(head, Const) and head.name == op_name
                and len(args) == 2):
            flatten_nat_chain(args[0], op_name, acc)
            flatten_nat_chain(args[1], op_name, acc)
            return
    acc.append(canon_nat_shape(e))


def canon_nat_shape(expr: Expr) -> Expr:
    """
    Normalize a CIC ``Nat`` shape expression to a canonical form.
    ``canon_nat_shape``is useful when  two definitionally equal shapes
    become the same``Expr`.

    Examples
    --------
    >>> from physika.core.elab.dim_typespec import canon_nat_shape
    >>> from physika.core.expr import App, BVar, Const, Lit
    >>> add = Const("Nat.add", ())
    >>> canon_nat_shape(App(App(add, Lit(0)), BVar(0)))
    BVar(idx=0)
    >>> m_plus_n = App(App(add, BVar(1)), BVar(0))
    >>> n_plus_m = App(App(add, BVar(0)), BVar(1))
    >>> canon_nat_shape(m_plus_n) == canon_nat_shape(n_plus_m)
    True
    >>> canon_nat_shape(App(App(add, Lit(3)), Lit(1)))
    Lit(val=4)
    """
    if not isinstance(expr, App):
        return expr

    head, args = get_app_fn_args(expr)
    if not (isinstance(head, Const) and len(args) == 2
            and head.name in ("Nat.add", "Nat.mul", "Nat.sub")):
        return expr

    op = head.name

    if op == "Nat.sub":
        a = canon_nat_shape(args[0])
        b = canon_nat_shape(args[1])
        av, bv = nat_lit_int(a), nat_lit_int(b)
        if av is not None and bv is not None:
            return Lit(max(av - bv, 0))  # type: ignore[arg-type]
        if bv == 0:
            return a
        return App(App(head, a), b)

    # Nat.add / Nat.mul
    # flatten, fold literals, sort non-literals.
    terms: List[Expr] = []
    flatten_nat_chain(expr, op, terms)

    lit_acc = CANON_IDENTITY[op]
    non_lits: List[Expr] = []
    for t in terms:
        tv = nat_lit_int(t)
        if tv is None:
            non_lits.append(t)
        elif op == "Nat.add":
            lit_acc += tv
        else:
            lit_acc *= tv

    if op == "Nat.mul" and lit_acc == 0:
        return Lit(0)  # type: ignore[arg-type]

    non_lits.sort(key=repr)

    ordered: List[Expr] = list(non_lits)
    if lit_acc != CANON_IDENTITY[op] or not non_lits:
        ordered.append(Lit(lit_acc))  # type: ignore[arg-type]

    result = ordered[-1]
    for t in reversed(ordered[:-1]):
        result = App(App(head, t), result)
    return result


def typespec_to_cic_resolved(ts: Union[str, tuple],
                             resolve: Callable[[str], Optional[Expr]]) -> Expr:
    """Convert a Physika type spec ``ts`` to a CIC ``Expr``, given a
    ``resolve`` callback (similar as ``dim_to_cic_resolved``).

    Parameters
    ----------
    ts : Union[str, tuple]
        ``"ℝ"``/``"ℕ"``/``"ℤ"``, or a tagged tuple — ``("tensor",
        [...])``, ``("struct_type", name)``, or ``("tuple_type", [...])``.
    resolve : Callable[[str], Optional[Expr]]
        Forwarded to ``dim_to_cic_resolved`` for any dim var inside a
        ``"tensor"`` spec.

    Example
    -------
    >>> from physika.core.elab.dim_typespec import typespec_to_cic_resolved  # noqa: E501
    >>> typespec_to_cic_resolved("ℝ", lambda name: None)
    Const(name='Real', levels=())
    >>> from physika.core.expr import BVar
    >>> typespec_to_cic_resolved(("tensor", [("n", "invariant")]),
    ...                          lambda name: {"n": BVar(0)}.get(name))
    App(func=App(func=Const(name='Vec', levels=()), arg=Const(name='Real', levels=())), arg=BVar(idx=0))
    """
    if ts == "ℝ":
        return _REAL_CONST
    if ts in ("ℕ", "ℤ"):
        return _NAT_CONST
    if isinstance(ts, tuple) and ts[0] == "tensor":
        dims = ts[1]  # [(dim, variance), ...]
        if not dims:
            return _REAL_CONST
        # Build nested Vec types
        elem: Expr = _REAL_CONST
        for (d, _) in reversed(dims):
            d_expr = dim_to_cic_resolved(d, resolve)
            if d_expr is None:
                # unbound symbolic dim
                d_expr = Lit(0)  # type: ignore[arg-type]
            # Canonicalize the shape so definitionally-equal dims
            # (m+n vs n+m, 0+n vs n) become the identical Expr
            d_expr = canon_nat_shape(d_expr)
            elem = App(App(_VEC_CONST, elem), d_expr)
        return elem
    if isinstance(ts, tuple) and ts[0] == "struct_type":
        return Const(ts[1], ())
    if isinstance(ts, tuple) and ts[0] == "tuple_type":
        # `→ ℝ[n], ℝ[n]` (multi value return)
        types = ts[1]
        result = typespec_to_cic_resolved(types[-1], resolve)
        for t in reversed(types[:-1]):
            elem_t = typespec_to_cic_resolved(t, resolve)
            result = App(App(Const("Prod", ()), elem_t), result)
        return result
    # Fallback for unknown
    return _REAL_CONST


def bvar_resolver(
    binder_names: List[str],
    depth: int,
    return_only_mvars: Optional[Dict[str, "MVar"]] = None
) -> Callable[[str], Expr]:
    """
    Returns a function from binder name to ``Expr``. A dim var resolves
    to its ``BVar`` at the current depth, or to a ``return_only_mvars`` entry
    when it's a dim var with no Pi-binder.

    Parameters
    ----------
    binder_names : List[str]
        Binder's display name.
    depth : int
        Opened binders at the point the resolver is used.
    return_only_mvars : Optional[Dict[str, MVar]]
        Dim vars with no Pi-binder, resolved to their own MVar directly.

    Example
    -------
    >>> from physika.core.elab.dim_typespec import bvar_resolver
    >>> resolve = bvar_resolver(["n", "v"], depth=1)
    >>> resolve("n")
    BVar(idx=0)
    """

    def resolve(name: str) -> Expr:
        """
        Resolve one dim-var name to its MVar, else its BVar.

        Parameters
        ----------
        name: str
            Name of dim var.
        """
        if return_only_mvars and name in return_only_mvars:
            return return_only_mvars[name]
        return dim_bvar(name, binder_names, depth)

    return resolve


def typespec_to_cic(
        ts: Union[str, tuple],
        binder_names: List[str],
        depth: int,
        return_only_mvars: Optional[Dict[str, "MVar"]] = None) -> Expr:
    """
    Convert a Physika type spec to CIC ``Expr``.

    Parameters
    ----------
    ts : Union[str, tuple]
        ``"ℝ"``, ``"ℕ"``, ``("tensor", [...])`` or ``("struct_type", name)``.
    binder_names : List[str]
        Binder's display name.
    depth : int
        Opened binders at the point the resolver is used.
    return_only_mvars : Optional[Dict[str, MVar]]
        Dim vars with no Pi-binder, resolved to their own MVar directly.

    Example
    -------
    >>> from physika.core.elab.dim_typespec import typespec_to_cic  # noqa: E501
    >>> typespec_to_cic(("tensor", [("n", "invariant")]), ["n", "v"], depth=1)
    App(func=App(func=Const(name='Vec', levels=()), arg=Const(name='Real', levels=())), arg=BVar(idx=0))
    """
    return typespec_to_cic_resolved(
        ts, bvar_resolver(binder_names, depth, return_only_mvars))


def elaborate_func_type(func_def: dict, elab: object) -> Expr:
    """
    Build CIC ``ForallE`` expression for one Physika function definition.

    Dim variables are implicit (``BinderInfo.IMPLICIT``), parameters are
    explicit (``BinderInfo.DEFAULT``) and body is the return type.

    Parameters
    ----------
    func_def : dict
        Parsed function defintion
    elab : object
        Elaborator exposing ``new_mvar(name, type)``, used to add a fresh
        ``MVar`` for each return-only dim var.

    Example
    -------
    >>> from physika.core.elab.dim_typespec import elaborate_func_type  # noqa: E501
    >>> func_def = {"params": [("x", "ℝ")], "return_type": "ℝ"}
    >>> elaborate_func_type(func_def, object())
    ForallE(binder_name='x', binder_type=Const(name='Real', levels=()), body=Const(name='Real', levels=()), binder_info=<BinderInfo.DEFAULT: 1>)
    """
    params = func_def.get("params", [])  # [(name, type_spec), ...]
    return_type = func_def.get("return_type", "ℝ")

    dim_vars = collect_dim_vars_ordered(params, return_type)

    param_dim_names = set(collect_dim_vars_ordered(params, None))
    binder_dim_vars = [d for d in dim_vars if d in param_dim_names]
    return_only_mvars: Dict[str, "MVar"] = {
        d:
        elab.new_mvar(  # type: ignore[attr-defined]
            f"_ret_dim_{d}", _NAT_CONST)
        for d in dim_vars if d not in param_dim_names
    }

    binder_names = binder_dim_vars + [p[0] for p in params]
    total = len(binder_names)

    result = typespec_to_cic(return_type, binder_names, total,
                             return_only_mvars)

    # start in reverse order
    for i in range(total - 1, -1, -1):
        name = binder_names[i]
        if i < len(binder_dim_vars):
            btype: Expr = _NAT_CONST
            binfo = BinderInfo.IMPLICIT
        else:
            param_idx = i - len(binder_dim_vars)
            btype = typespec_to_cic(params[param_idx][1], binder_names, i,
                                    return_only_mvars)
            binfo = BinderInfo.DEFAULT
        result = ForallE(name, btype, result, binfo)

    return result


def elaborate_struct_kind_and_ctor(
    class_name: str, fields: List[Tuple[str, Union[str, tuple]]]
) -> Tuple[Expr, Expr, int, List[str]]:
    """
    Returns kind, ctor_type, num_params, dim_vars for a Physika class.

    Parameters
    ----------
    class_name : str
        Class's registered name (e.g. ``"Box"``).
    fields : List[Tuple[str, Union[str, tuple]]]
        field name and type for a class defined in Physika

    Example
    -------
    >>> from physika.core.elab.dim_typespec import elaborate_struct_kind_and_ctor  # noqa: E501
    >>> kind, ctor_type, num_params, dim_vars = (
    ...     elaborate_struct_kind_and_ctor("Box", [("x", "ℝ")]))
    >>> kind
    Sort(level=LSucc(pred=LZero()))
    >>> ctor_type
    ForallE(binder_name='x', binder_type=Const(name='Real', levels=()), body=Const(name='Box', levels=()), binder_info=<BinderInfo.DEFAULT: 1>)
    >>> num_params, dim_vars
    (0, [])
    """
    dim_vars = collect_dim_vars_ordered(fields, None)
    num_params = len(dim_vars)
    binder_names: List[str] = dim_vars + [f[0] for f in fields]
    total = len(binder_names)

    class_const = Const(class_name, ())

    kind: Expr = _TYPE_0_LEVEL
    for i in range(num_params - 1, -1, -1):
        kind = ForallE(dim_vars[i], _NAT_CONST, kind, BinderInfo.DEFAULT)

    result: Expr = class_const
    for i in range(num_params):
        result = App(result, BVar(total - 1 - i))

    for i in range(total - 1, -1, -1):
        if i < num_params:
            btype: Expr = _NAT_CONST
            binfo = BinderInfo.IMPLICIT
        else:
            fidx = i - num_params
            btype = typespec_to_cic(fields[fidx][1], binder_names, i)
            binfo = BinderInfo.DEFAULT
        result = ForallE(binder_names[i], btype, result, binfo)

    return kind, result, num_params, dim_vars


def struct_field_names(type_name: str,
                       env: Environment) -> Optional[List[str]]:
    """
    Return class field names derived from its constructor's ForallE chain.

    Parameters
    ----------
    type_name : str
        Class's registered name.
    env : Environment
        Environment where the class was registered.

    Example
    -------
    >>> from physika.core.elab.dim_typespec import (
    ...     elaborate_struct_kind_and_ctor, struct_field_names)
    >>> from physika.core.environment import (
    ...     ConstantInfo, Environment, InductiveInfo)
    >>> from physika.core.inductive import Constructor, InductiveDecl
    >>> kind, ctor_type, num_params, _ = elaborate_struct_kind_and_ctor(
    ...     "Box", [("n", "ℝ"), ("m", "ℝ")])
    >>> decl = InductiveDecl(
    ...     name="Box", level_params=(), num_params=num_params, type=kind,
    ...     constructors=(Constructor("Box.mk", ctor_type),),
    ...     is_recursive=False)
    >>> ctor_ci = ConstantInfo("Box.mk", (), ctor_type, None)
    >>> rec_ci = ConstantInfo("Box.rec", ("u",), kind, None)
    >>> env = Environment()
    >>> env.add_inductive(InductiveInfo(
    ...     decl=decl, ctors={"Box.mk": ctor_ci}, recursor=rec_ci))
    >>> struct_field_names("Box", env)
    ['n', 'm']
    """
    ii = env.inductives.get(type_name)
    if ii is None or len(ii.decl.constructors) != 1:
        return None
    tp: Expr = ii.decl.constructors[0].type
    for _ in range(ii.decl.num_params):
        if not isinstance(tp, ForallE):
            return None
        tp = tp.body
    names: List[str] = []
    while isinstance(tp, ForallE):
        names.append(tp.binder_name)
        tp = tp.body
    return names


def elaborate_method_type(class_name: str, num_class_params: int,
                          class_dim_vars: List[str], method_def: dict) -> Expr:
    """
    Returns CIC ``ForallE`` chain for a class method.

    ``this : ClassName {class dim vars}`` is an explicit parameter , followed by
    the method's implicit dim vars and explicit params.

    Parameters
    ----------
    class_name : str
        Class registered name.
    num_class_params : int
        Number of class dim vars ``this`` have in scope.
    class_dim_vars : List[str]
        Dim var names for impicit binders
    method_def : dict
        Parsed method definition.

    Example
    -------
    >>> from physika.core.elab.dim_typespec import struct_field_names  # noqa: E501
    >>> method_def = {"params": [("x", "ℝ")], "return_type": "ℝ"}
    >>> elaborate_method_type("Box", 0, [], method_def)
    ForallE(binder_name='this', binder_type=Const(name='Box', levels=()), body=ForallE(binder_name='x', binder_type=Const(name='Real', levels=()), body=Const(name='Real', levels=()), binder_info=<BinderInfo.DEFAULT: 1>), binder_info=<BinderInfo.DEFAULT: 1>)
    """
    params = method_def.get("params", [])
    return_type = method_def.get("return_type", "ℝ")
    method_dim_vars = collect_dim_vars_ordered(params, return_type)

    binder_names: List[str] = (list(class_dim_vars) + ["this"] +
                               method_dim_vars + [p[0] for p in params])
    total = len(binder_names)
    this_pos = num_class_params
    params_start = this_pos + 1 + len(method_dim_vars)

    result: Expr = typespec_to_cic(return_type, binder_names, total)

    for i in range(total - 1, -1, -1):
        if i < num_class_params:
            btype: Expr = _NAT_CONST
            binfo = BinderInfo.IMPLICIT
        elif i == this_pos:
            this_type: Expr = Const(class_name, ())
            for j in range(num_class_params):
                this_type = App(this_type, BVar(i - 1 - j))
            btype = this_type
            binfo = BinderInfo.DEFAULT
        elif i < params_start:
            btype = _NAT_CONST
            binfo = BinderInfo.IMPLICIT
        else:
            pidx = i - params_start
            btype = typespec_to_cic(params[pidx][1], binder_names, i)
            binfo = BinderInfo.DEFAULT
        result = ForallE(binder_names[i], btype, result, binfo)

    return result
