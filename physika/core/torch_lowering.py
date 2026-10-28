from typing import Any, Collection, Dict, List, Optional, Tuple, Union

from physika.core.expr import (
    App,
    Const,
    Expr,
    FVar,
    FVarId,
    FloatLit,
    ForallE,
    Lam,
    LetE,
    Lit,
    NatLit,
    Proj,
    BinderInfo,
)
from physika.utils.cic_utils.expr_utils import get_app_fn_args, instantiate1
from physika.core.environment import Environment
from physika.core.local_context import fresh_fvar_id
from physika.core.elab.elab import struct_field_names

BUILTIN_BINOP = {
    "Real.add": "+",
    "Real.sub": "-",
    "Real.mul": "*",
    "Real.div": "/",
    "Real.pow": "**",
    "Nat.add": "+",
    "Nat.sub": "-",
    "Nat.mul": "*",
    "Real.eqb": "==",
    "Real.neb": "!=",
    "Real.ltb": "<",
    "Real.gtb": ">",
    "Real.leb": "<=",
    "Real.geb": ">=",
    "Nat.eqb": "==",
    "Nat.neb": "!=",
    "Nat.ltb": "<",
    "Nat.gtb": ">",
    "Nat.leb": "<=",
    "Nat.geb": ">=",
}

BUILTIN_ARITY = {
    "Vec.tabulate": 3,
    "Vec.sum": 2,
    "Vec.get": 4,
    "Vec.cons": 4,
    "Vec.dot": 3,
    "Vec.vadd": 3,
    "Vec.vmul": 3,
    "Vec.scale": 3,
    "Mat.matmul": 5,
    "Mat.inv": 2,
    "Mat.madd": 4,
    "Mat.add_scalar": 4,
    "Fin.ofNat": 2,
    "Fin.zero": 1,
    "Fin.succ": 2,
    "Vec.foldl": 4,
    "Nat.rec": 4,
    "Real.neg": 1,
    "Nat.toReal": 1,
    "Nat.succ": 1,
    "Real.ite": 3,
    "Nat.ite": 3,
    "Vec.ite": 4,
    "Vec.zeros": 1,
    "Vec.concat": 4,
    "Mat.concat_rows": 5,
    **{
        k: 2
        for k in BUILTIN_BINOP
    },
}

_IN_PLACE_ASSIGN_TAGS = (
    "body_index_assign",
    "body_index_assign_nd",
    "loop_index_assign",
    "loop_index_assign_nd",
    "loop_index_pluseq",
)

# CIC terms that are not resolved
NO_RESOLVED_NAMES: Collection[str] = ()
LINEAR_DIM_TAGS = ("mul_dim", "add_dim", "sub_dim")
QUADRATIC_DIM_TAGS = ("mul_dim_id", )

# Dimension entry for dim variables
DimEntry = Union[int, str, Tuple[str, Any, Any]]


def body_mutates_in_place(func_def: Dict[str, Any]) -> bool:
    """
    Return ``True`` if a function's body reassigns over a variable (in place mutation).

    Parameters
    ----------
    func_def : Dict[str, Any]
        ``unified_ast["functions"]`` where func names are keys and statements
        list is walked recursevily..

    Examples
    --------
    >>> from physika.core.torch_lowering import body_mutates_in_place  # noqa: E501
    >>> body_mutates_in_place(
    ...     {"statements": [("body_index_assign", "x", ("num", 1), ("num", 3))]})
    True
    >>> body_mutates_in_place({"statements": [("return", ("var", "x"))]})
    False
    """

    def walk(node: Any) -> bool:
        """
        Recursively walk an AST node looking for an in-place assignment.

        Parameters
        ----------
        node : Any
            ASTNode, a list of nodes, or a leaf value.
        """
        if isinstance(node, tuple):
            if node and node[0] in _IN_PLACE_ASSIGN_TAGS:
                return True
            return any(walk(c) for c in node)
        if isinstance(node, list):
            return any(walk(c) for c in node)
        return False

    return walk(func_def.get("statements", []))


def implicit_binders(name: str, env: Environment) -> int:
    """
    Number of implicit Pi-binders registerd for ``name``.

    Parameters
    ----------
    name : str
        Constant name to look up in ``env``.
    env : Environment
        CIC environment ``name`` was checked against.

    Examples
    --------
    >>> from physika.core.environment import Environment
    >>> from physika.core.torch_lowering import implicit_binders
    >>> implicit_binders("not.registered", Environment())
    0
    """
    ci = env.constants.get(name)
    if ci is None:
        return 0
    n = 0
    tp = ci.type
    while isinstance(tp, ForallE) and tp.binder_info == BinderInfo.IMPLICIT:
        n += 1
        tp = tp.body
    return n


def dim_var_name(dim: DimEntry) -> Optional[str]:
    """
    Return the single symbolic dim-var name embedded in a dim entry, or
    ``None`` when there is not exactly one extractable variable.

    A concrete ``int`` (or unification variable), and a two-variable dim
    whose vars differ (``n*m``), both return ``None`` — neither yields a
    value recoverable from this one dimension alone.

    Parameters
    ----------
    dim : DimEntry
        A dimension entry (``int``, dim-var name, or derived-dim tuple).

    Examples
    --------
    >>> from physika.core.torch_lowering import dim_var_name
    >>> dim_var_name("n")
    'n'
    >>> dim_var_name(("mul_dim", "n", 2))
    'n'
    >>> dim_var_name(("mul_dim_id", "n", "m")) is None
    True
    >>> dim_var_name(3) is None
    True
    """
    if isinstance(dim, str):
        return dim
    if isinstance(dim, tuple) and len(dim) == 3 and dim[0] in LINEAR_DIM_TAGS:
        return dim[1]
    if (isinstance(dim, tuple) and len(dim) == 3
            and dim[0] in QUADRATIC_DIM_TAGS):
        _, v1, v2 = dim
        if v1 == v2:
            return v1
    return None


def apply_dim_rename(dim: DimEntry, rename: Dict[str, str]) -> DimEntry:
    """
    Substitute renamed dim-var names into a dim entry, leaving tags and
    constants unchanged.

    Needed when a dim var's compiled Python identifier differs from its
    source name (e.g. ``def f(v: ℝ[n], n: ℝ)`` compiles the *dim var*
    ``n`` to ``__dim_n`` to avoid a duplicate parameter — see
    ``resolve_binder_names`` in ``elab.py``). Call this before
    ``dim_var_name`` / ``dim_extraction_stmt``, which read a name
    straight out of the raw dim entry.

    Parameters
    ----------
    dim : DimEntry
        A dimension entry (``int``, dim-var name, or derived-dim tuple).
    rename : dict[str, str]
        Source dim-var name -> compiled identifier; only differing
        entries need to appear.


    Examples
    --------
    >>> from physika.core.torch_lowering import apply_dim_rename
    >>> apply_dim_rename("n", {"n": "__dim_n"})
    '__dim_n'
    >>> apply_dim_rename(("add_dim_id", "n", "m"), {"n": "__dim_n"})
    ('add_dim_id', '__dim_n', 'm')
    >>> apply_dim_rename(3, {"n": "__dim_n"})
    3
    """
    if not rename:
        return dim
    if isinstance(dim, str):
        return rename.get(dim, dim)
    if isinstance(dim, tuple) and len(dim) == 3:
        tag, a, b = dim
        if isinstance(a, str):
            a = rename.get(a, a)
        if isinstance(b, str):
            b = rename.get(b, b)
        return (tag, a, b)
    return dim


def dim_extraction_stmt(dim: DimEntry,
                        axis: int,
                        param_expr: str,
                        indent: str = "    ") -> Optional[str]:
    """
    Emit the Python assignment that recovers a dim variable from a tensor
    parameter's shape, applying the inverse of the arithmetic in ``dim``.

    A CIC-resolved function keeps its dim vars as trailing ``=None``
    parameters (``def f(x, n=None)``) so unresolved callers can still
    call it; this statement is the ``if n is None:`` fallback body that
    self-derives ``n`` from ``x.shape``.

    Parameters
    ----------
    dim : DimEntry
        A dimension entry (dim-var name or linear/square derived tuple).
    axis : int
        Shape axis to read from ``param_expr``.
    param_expr : str
        Python expression for the tensor (``"x"``, ``"self.weights"``).
    indent : str, default ``"    "``
        Leading whitespace for the emitted line.


    Examples
    --------
    >>> from physika.core.torch_lowering import dim_extraction_stmt
    >>> dim_extraction_stmt("n", 0, "x")
    '    n = int(x.shape[0])'
    >>> dim_extraction_stmt(("add_dim", "k", 1), 1, "X")
    '    k = int(X.shape[1]) - 1'
    >>> dim_extraction_stmt(4, 0, "x") is None
    True
    """
    base = f"int({param_expr}.shape[{axis}])"
    if isinstance(dim, str):
        return f"{indent}{dim} = {base}"
    if isinstance(dim, tuple) and len(dim) == 3 and dim[0] in LINEAR_DIM_TAGS:
        tag, var, const = dim
        if tag == "mul_dim":
            return f"{indent}{var} = {base} // {const}"
        if tag == "add_dim":
            return f"{indent}{var} = {base} - {const}"
        if tag == "sub_dim":
            return f"{indent}{var} = {base} + {const}"
    if (isinstance(dim, tuple) and len(dim) == 3
            and dim[0] in QUADRATIC_DIM_TAGS):
        _, v1, v2 = dim
        if v1 == v2:
            return (f"{indent}{v1} = "
                    f"int(int({param_expr}.shape[{axis}]) ** 0.5)")
    return None


def stack_elems(elem_srcs: List[str]) -> str:
    """
    Combine ``Vec`` literals into a differentiable ``torch.stack`` call.

    Parameters
    ----------
    elem_srcs : list[str]
        CIC term lowered to Python elements, outermost first.

    Examples
    --------
    >>> from physika.core.torch_lowering import stack_elems
    >>> stack_elems(["a", "1.0"])
    'torch.stack([torch.as_tensor(a), torch.as_tensor(1.0)])'
    >>> stack_elems([])
    'torch.tensor([])'
    """
    if not elem_srcs:
        return "torch.tensor([])"
    wrapped = ", ".join(f"torch.as_tensor({e})" for e in elem_srcs)
    return f"torch.stack([{wrapped}])"


def fold_gen(n_src: str, k_name: str, acc_name: str, step_src: str,
             init_src: str) -> str:
    """
    ``fold_gen`` is called when an accumulator is updated under a for loop.
    Fold is a reference to consume or collapse the accumulator at each step.
    ``fold_gen``emit fold over ``range(n)``. Used when lowering a CIC
    ``Vec.foldl`` axiom, and ``Nat.rec`` whose motive erases a shape change.

    Parameters
    ----------
    n_src : str
        Python lowered source for the iteration count.
    k_name: str
        Loop-index name
    acc_name : str
        Accumulator binder name
    step_src : str
        Loop body, referencing ``k_name`` and ``acc_name``.
    init_src : str
        Lowered source for the initial accumulator.

    Examples
    --------
    >>> from physika.core.torch_lowering import fold_gen
    >>> fold_gen("n", "k", "acc", "(acc + k)", "0.0")
    '(lambda acc: ([acc := (acc + k) for k in range(int(n))], acc)[1])(0.0)'
    """
    return (
        f"(lambda {acc_name}: ([{acc_name} := {step_src} "
        f"for {k_name} in range(int({n_src}))], {acc_name})[1])({init_src})")


def grad_gen(output_expr: Expr, var_expr: FVar, names: Dict[FVarId, str],
             env: Environment, resolved_names: Collection[str]) -> str:
    """
    Emit ``compute_grad()`` call for an elaborated CIC ``grad`` or
    ``Vec.grad`` term.

    Parameters
    ----------
    output_expr : Expr
        Elaborated expression (a function application).
    var_expr : FVar
        Variable to differentiate with respect to.
    names : dict[FVarId, str]
        Name for free variables in scope.
    env : Environment
        CIC environment the term was checked against.
    resolved_names : Collection[str]
        Names of CIC resolved bodies.

    Examples
    --------
    >>> from physika.core.torch_lowering import grad_gen
    >>> from physika.core.expr import App, Const, FVar, FVarId
    >>> from physika.core.environment import Environment
    >>> x = FVar(FVarId("x0"))
    >>> grad_gen(App(Const("f", ()), x), x, {x.id: "x"}, Environment(), ())
    'compute_grad(lambda _dx: f(_dx), x)'
    """
    g_head, _ = get_app_fn_args(output_expr)
    if isinstance(g_head, Const) and g_head.name in ("grad", "Vec.grad"):
        raise NotImplementedError(
            "torch_lowering: nested grad(grad(...)) is not supported yet")
    src_name = names.get(var_expr.id, "x")
    param = f"_d{src_name}"
    child_names = dict(names)
    child_names[var_expr.id] = param
    body_src = lower(output_expr, child_names, env, resolved_names)
    var_src = lower(var_expr, names, env, resolved_names)
    return f"compute_grad(lambda {param}: {body_src}, {var_src})"


def lower_vec_literal(expr: Expr, names: Dict[FVarId, str], env: Environment,
                      resolved_names: Collection[str]) -> Optional[List[str]]:
    """
    Flatten a ``Vec.cons`` or ``Vec.nil`` chain to a list, or return
    ``None`` if ``expr`` is not such a chain.

    An elaborated array literal is a chain of ``Vec.cons``. Each scalar element
    is lowered to a int or float type. A nested element is lowered from CIC to
    torch code using a ``torch.stack`` call.

    Parameters
    ----------
    expr : Expr
        Candidate ``Vec.cons``or ``Vec.nil`` application chain.
    names : dict[FVarId, str]
        Name for free variables in scope.
    env : Environment
        CIC environment the term was checked against.
    resolved_names : Collection[str]
        Names whose bodies were CIC-resolved.

    Examples
    --------
    >>> from physika.core.torch_lowering import lower_vec_literal
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.environment import Environment
    >>> real = Const("Real", ())
    >>> chain = App(App(App(App(Const("Vec.cons", ()), real), Lit(1)),
    ...                 Lit(1.0)),
    ...             App(Const("Vec.nil", ()), real))
    >>> lower_vec_literal(chain, {}, Environment(), ())
    ['1.0']
    """
    head, args = get_app_fn_args(expr)
    if isinstance(head, Const) and head.name == "Vec.nil" and len(args) == 1:
        return []
    if isinstance(head, Const) and head.name == "Vec.cons" and len(args) == 4:
        _, _, hd, tl = args
        rest = lower_vec_literal(tl, names, env, resolved_names)
        if rest is None:
            return None
        hd_head, _ = get_app_fn_args(hd)
        if isinstance(hd_head,
                      Const) and hd_head.name in ("Vec.cons", "Vec.nil"):
            inner = lower_vec_literal(hd, names, env, resolved_names)
            if inner is None:
                return None
            hd_src = stack_elems(inner)
        else:
            hd_src = lower(hd, names, env, resolved_names)
        return [hd_src] + rest
    return None


def lower(expr: Expr, names: Dict[FVarId, str], env: Environment,
          resolved_names: Collection[str]) -> str:
    """
    Lower a kernel verified CIC expression to a Python/Pytorch
    source string.

    Recursively lower CIC terms of ``expr``'s constructor. Variables are
    transformed into regular python variable names, ``Proj`` are converted
    into field access, and a funciton application is matched against the
    builtin axiom vocabulary (``BUILTIN_ARITY``, ``BUILTIN_BINOP``, or
    ``Vec.*`` / ``Real.*`` / ``Nat.rec`` cases) before calling
    ``lower_call`` for user functions, methods, or constructors.

    Parameters
    ----------
    expr : Expr
        Closed and elaborated CIC term to lower. Any free variable must be in
        ``names``.
    names : dict[FVarId, str]
        Name for free variables in scope.
    env : Environment
        CIC environment the term was checked against.
    resolved_names : Collection[str]
        Names whose bodies were CIC-resolved.

    Examples
    --------
    >>> from physika.core.torch_lowering import lower
    >>> from physika.core.expr import Lit
    >>> from physika.core.environment import Environment
    >>> lower(Lit(3), {}, Environment(), ())
    '3'
    """
    if isinstance(expr, FVar):
        if expr.id not in names:
            raise NotImplementedError(f"FVar {expr.id!r} has no source name ")
        return names[expr.id]

    if isinstance(expr, Lit):
        val = expr.val
        if isinstance(val, (NatLit, FloatLit)):
            val = val.val
        return repr(val)

    if isinstance(expr, Proj):
        if expr.type_name == "OfNat" and expr.idx == 0:
            inst_head, inst_args = get_app_fn_args(expr.expr)
            if (isinstance(inst_head, Const)
                    and inst_head.name in ("instOfNatNat", "instOfNatReal")
                    and len(inst_args) == 1 and isinstance(inst_args[0], Lit)):
                n_val = inst_args[0].val
                if isinstance(n_val, (NatLit, FloatLit)):
                    n_val = n_val.val
                return repr(int(n_val))
            raise NotImplementedError(f"Unrecognized OfNat instance shape "
                                      f"{expr.expr!r}")
        obj_src = lower(expr.expr, names, env, resolved_names)
        if expr.type_name == "Prod":
            return f"{obj_src}[{expr.idx}]"
        fields = struct_field_names(expr.type_name, env)
        if fields is None or expr.idx >= len(fields):
            raise NotImplementedError(f"Cannot resolve field {expr.idx} of "
                                      f"'{expr.type_name}'")
        return f"{obj_src}.{fields[expr.idx]}"

    if isinstance(expr, LetE):
        fresh = FVar(fresh_fvar_id(expr.binder_name))
        opened_body = instantiate1(expr.body, fresh)
        child_names = dict(names)
        child_names[fresh.id] = expr.binder_name
        value_src = lower(expr.value, names, env, resolved_names)
        body_src = lower(opened_body, child_names, env, resolved_names)
        return f"(lambda {expr.binder_name}={value_src}: {body_src})()"

    if isinstance(expr, Lam):
        fresh = FVar(fresh_fvar_id(expr.binder_name))
        opened = instantiate1(expr.body, fresh)
        child_names = dict(names)
        child_names[fresh.id] = expr.binder_name
        body_src = lower(opened, child_names, env, resolved_names)
        return f"torch.as_tensor({body_src}) for {expr.binder_name} in {{range_expr}}"  # noqa: E501

    if isinstance(expr, (Const, App)):
        head, args = get_app_fn_args(expr)
        if isinstance(head, Const) and head.name in ("Vec.cons", "Vec.nil"):
            elems = lower_vec_literal(expr, names, env, resolved_names)
            if elems is not None:
                return stack_elems(elems)
        if isinstance(head, Const) and head.name in BUILTIN_ARITY:
            name = head.name
            arity = BUILTIN_ARITY[name]
            if len(args) != arity:
                raise NotImplementedError(
                    f"'{name}' applied to {len(args)} args, expected {arity}")
            if name == "Vec.tabulate":
                # args[0] is implicit type argument
                elem_type_arg = args[0]
                n_src = lower(args[1], names, env, resolved_names)
                fn_src = lower(args[2], names, env, resolved_names)
                stack_src = (
                    f"torch.stack([{fn_src.format(range_expr=f'range(int({n_src}))')}])"  # noqa: E501
                )
                if isinstance(elem_type_arg,
                              Const) and elem_type_arg.name == "Real":
                    return f"{stack_src}.float()"
                return stack_src
            if name == "Vec.sum":
                return f"torch.sum({lower(args[1], names, env, resolved_names)})"  # noqa: E501
            if name == "Vec.get":
                # args[0] is the implicit {α} type argument — skipped.
                vec_src = lower(args[2], names, env, resolved_names)
                idx_src = lower(args[3], names, env, resolved_names)
                return f"{vec_src}[int({idx_src})]"
            if name == "Vec.cons":
                hd_src = lower(args[2], names, env, resolved_names)
                tl_src = lower(args[3], names, env, resolved_names)
                return f"torch.cat([torch.as_tensor([{hd_src}]).float(), {tl_src}])"  # noqa: E501
            if name == "Vec.dot":
                return (
                    f"torch.dot({lower(args[1], names, env, resolved_names)}, "
                    f"{lower(args[2], names, env, resolved_names)})")
            if name == "Vec.vadd":
                return (f"({lower(args[1], names, env, resolved_names)} + "
                        f"{lower(args[2], names, env, resolved_names)})")
            if name == "Vec.vmul":
                return (f"({lower(args[1], names, env, resolved_names)} * "
                        f"{lower(args[2], names, env, resolved_names)})")
            if name == "Vec.scale":
                return (f"({lower(args[1], names, env, resolved_names)} * "
                        f"{lower(args[2], names, env, resolved_names)})")
            if name == "Mat.matmul":
                return (
                    f"torch.matmul({lower(args[3], names, env, resolved_names)}, "  # noqa: E501
                    f"{lower(args[4], names, env, resolved_names)})")
            if name == "Mat.inv":
                # args[0] is the implicit {n} dimension — skipped.
                return (f"torch.linalg.inv("
                        f"{lower(args[1], names, env, resolved_names)})")
            if name in ("Mat.madd", "Mat.add_scalar"):
                return (f"({lower(args[2], names, env, resolved_names)} + "
                        f"{lower(args[3], names, env, resolved_names)})")
            if name == "Fin.ofNat":
                return lower(args[1], names, env, resolved_names)
            if name == "Fin.zero":
                # index 0 at runtime
                return "0"
            if name == "Fin.succ":
                # one past k index at runtime
                return f"(1 + {lower(args[1], names, env, resolved_names)})"
            if name == "Vec.foldl":
                n_src = lower(args[1], names, env, resolved_names)
                f_expr = args[2]
                if not isinstance(f_expr, Lam):
                    raise NotImplementedError(
                        "Vec.foldl's function must be a literal"
                        " two-argument Lam")
                k_fresh = FVar(fresh_fvar_id(f_expr.binder_name))
                inner_lam = instantiate1(f_expr.body, k_fresh)
                if not isinstance(inner_lam, Lam):
                    raise NotImplementedError(
                        "Vec.foldl's function must be a literal"
                        "two-argument Lam")
                acc_fresh = FVar(fresh_fvar_id(inner_lam.binder_name))
                step_body = instantiate1(inner_lam.body, acc_fresh)
                child_names = dict(names)
                child_names[k_fresh.id] = f_expr.binder_name
                child_names[acc_fresh.id] = inner_lam.binder_name
                step_src = lower(step_body, child_names, env, resolved_names)
                init_src = lower(args[3], names, env, resolved_names)
                return fold_gen(n_src, f_expr.binder_name,
                                inner_lam.binder_name, step_src, init_src)
            if name == "Nat.rec":
                # args[0] -> motive
                # args[1] -> base
                # args[2] -> curried Lam k => Lam acc => body
                #  args[3] -> n
                base_src = lower(args[1], names, env, resolved_names)
                f_expr = args[2]
                if not isinstance(f_expr, Lam):
                    raise NotImplementedError(
                        "Nat.rec's step function must be "
                        "a literal two-argument curried Lam")
                k_fresh = FVar(fresh_fvar_id(f_expr.binder_name))
                inner_lam = instantiate1(f_expr.body, k_fresh)
                if not isinstance(inner_lam, Lam):
                    raise NotImplementedError(
                        "Nat.rec's step function must be "
                        "a literal two-argument curried Lam")
                acc_fresh = FVar(fresh_fvar_id(inner_lam.binder_name))
                step_body = instantiate1(inner_lam.body, acc_fresh)
                child_names = dict(names)
                child_names[k_fresh.id] = f_expr.binder_name
                child_names[acc_fresh.id] = inner_lam.binder_name
                step_src = lower(step_body, child_names, env, resolved_names)
                n_src = lower(args[3], names, env, resolved_names)
                return fold_gen(n_src, f_expr.binder_name,
                                inner_lam.binder_name, step_src, base_src)
            if name == "Real.neg":
                return f"(-{lower(args[0], names, env, resolved_names)})"
            if name == "Nat.toReal":
                return f"float({lower(args[0], names, env, resolved_names)})"
            if name == "Nat.succ":
                return f"({lower(args[0], names, env, resolved_names)} + 1)"
            if name in ("Real.ite", "Nat.ite"):
                cond_src = lower(args[0], names, env, resolved_names)
                then_src = lower(args[1], names, env, resolved_names)
                else_src = lower(args[2], names, env, resolved_names)
                return f"({then_src} if {cond_src} else {else_src})"
            if name == "Vec.ite":
                # args[0] is n
                cond_src = lower(args[1], names, env, resolved_names)
                then_src = lower(args[2], names, env, resolved_names)
                else_src = lower(args[3], names, env, resolved_names)
                return f"({then_src} if {cond_src} else {else_src})"
            if name == "Vec.zeros":
                n_src = lower(args[0], names, env, resolved_names)
                return f"torch.zeros({n_src})"
            if name == "Vec.concat":
                # args[0], args[1] are m, n
                u_src = lower(args[2], names, env, resolved_names)
                v_src = lower(args[3], names, env, resolved_names)
                return f"torch.cat([{u_src}, {v_src}])"
            if name == "Mat.concat_rows":
                a_src = lower(args[3], names, env, resolved_names)
                b_src = lower(args[4], names, env, resolved_names)
                return f"torch.cat([{a_src}, {b_src}])"
            if name in BUILTIN_BINOP:
                op = BUILTIN_BINOP[name]
                return (f"({lower(args[0], names, env, resolved_names)} {op} "
                        f"{lower(args[1], names, env, resolved_names)})")
            raise NotImplementedError(f"Unhandled builtin '{name}'")

        if isinstance(head, Const) and head.name in ("log", "exp", "cos",
                                                     "sin", "sqrt", "abs"):
            if len(args) != 1:
                raise NotImplementedError(
                    f"'{head.name}' applied to {len(args)} "
                    "args, expected 1")
            arg_src = lower(args[0], names, env, resolved_names)
            if head.name == "abs":
                # ``abs`` must preserve a complex dtype -- ``.float()`` would
                # silently drop the imaginary part (and its gradient). It
                # accepts integer tensors directly, so a plain
                # ``torch.as_tensor`` is enough.
                return f"torch.abs(torch.as_tensor({arg_src}))"
            return f"torch.{head.name}(torch.as_tensor({arg_src}).float())"

        if isinstance(head, Const) and head.name == "grad":
            if len(args) != 2:
                raise NotImplementedError(
                    f"'grad' applied to {len(args)} args, expected 2")
            if not isinstance(args[1], FVar):
                raise NotImplementedError(
                    "grad's second argument must be the "
                    "variable to differentiate with respect to")
            return grad_gen(args[0], args[1], names, env, resolved_names)

        if isinstance(head, Const) and head.name == "Vec.grad":
            if len(args) != 4:
                raise NotImplementedError(
                    f"'Vec.grad' applied to {len(args)} args, expected 4")
            call_expr, var_expr = args[2], args[3]
            _, inner_args = get_app_fn_args(call_expr)
            if not isinstance(var_expr, FVar) or var_expr not in inner_args:
                raise NotImplementedError(
                    "Vec.grad's differentiation variable "
                    "must be one of the called function's own arguments")
            return grad_gen(call_expr, var_expr, names, env, resolved_names)

        if isinstance(head, Const):
            return lower_call(head.name, args, names, env, resolved_names)

        raise NotImplementedError(f"Unhandled application head {head!r}")

    raise NotImplementedError(f"Unhandled node {expr!r}")


def lower_call(name: str, args: List[Expr], names: Dict[FVarId, str],
               env: Environment, resolved_names: Collection[str]) -> str:
    """
    Lower a call to a user function, method or class constructor.

    Implicit dimension variable arguments are kept if ``name`` is in
    ``resolved_names``. For example, when its body was CIC elaborated and
    compiled with ``dim_var=None`` parameter.

    Parameters
    ----------
    name : str
        The callee's CIC Const name (i.e. ``"f"``, ``"Vec2D.dot"``, etc).
    args : list[Expr]
        Function arguments in CIC order (implicit dim vars first).
    names : dict[FVarId, str]
        Name for free variable in scope.
    env : Environment
        CIC environment that the term was checked against.
    resolved_names : Collection[str]
        Names whose bodies were resolved during CIC elaboration.

    Examples
    --------
    >>> from physika.core.torch_lowering import lower_call
    >>> from physika.core.expr import Lit
    >>> from physika.core.environment import Environment
    >>> lower_call("f", [Lit(1), Lit(2)], {}, Environment(), ())
    'f(1, 2)'
    """
    # class constructor case
    if name.endswith(".mk"):
        n_implicit = implicit_binders(name, env)
        explicit_args = args[n_implicit:]
        arg_srcs = [
            lower(a, names, env, resolved_names) for a in explicit_args
        ]
        class_name = name[:-len(".mk")]
        if class_name == "Prod":
            return f"({', '.join(arg_srcs)})"
        return f"{class_name}({', '.join(arg_srcs)})"

    callee_resolved = name in resolved_names

    if "." in name:
        # method call
        n_implicit = implicit_binders(name, env)
        if len(args) <= n_implicit:
            raise NotImplementedError(
                f"method call '{name}' have more implicit binders"
                " than method args.")
        receiver_src = lower(args[n_implicit], names, env, resolved_names)
        implicit_args = args[:n_implicit] if callee_resolved else []
        explicit_args = args[n_implicit + 1:]
        arg_srcs = (
            [lower(a, names, env, resolved_names) for a in explicit_args] +
            [lower(a, names, env, resolved_names) for a in implicit_args])
        class_name, method_name = name.split(".", 1)
        return f"{receiver_src}.{method_name}({', '.join(arg_srcs)})"

    # function call
    n_implicit = implicit_binders(name, env)
    implicit_args = args[:n_implicit] if callee_resolved else []
    explicit_args = args[n_implicit:]
    arg_srcs = ([lower(a, names, env, resolved_names) for a in explicit_args] +
                [lower(a, names, env, resolved_names) for a in implicit_args])
    return f"{name}({', '.join(arg_srcs)})"


def lower_function_body(
        body: Expr,
        names: Dict[FVarId, str],
        env: Environment,
        local_decls: Optional[List[Tuple[str, Expr]]] = None,
        indent: str = "    ",
        resolved_names: Collection[str] = NO_RESOLVED_NAMES) -> str:
    """
    Recursively lower an elaborated and verified CIC term for a function or
    method body to Python code.

    Parameters
    ----------
    body : Expr
        Verified body expression including its return.
    names : dict[FVarId, str]
        Name for free variable in scope (params, ``this``).
    env : Environment
        CIC environment the term was checked against.
    local_decls : list[tuple[str, Expr]], optional
        ``(variable_name, verified_rhs)`` for each local declaration.
    indent : str, default ``"    "``
        Whitespace indentation for emitted lines.
    resolved_names : Collection[str]
        Names whose signature carries the ``dim_var=None`` parameters.


    Examples
    --------
    >>> from physika.core.torch_lowering import lower_function_body
    >>> from physika.core.expr import Lit
    >>> from physika.core.environment import Environment
    >>> lower_function_body(Lit(0), {}, Environment(), [("x", Lit(1))])
    '    x = 1\\n    return 0'
    """
    lines = []
    for var_name, rhs_expr in (local_decls or []):
        lines.append(
            f"{indent}{var_name} = {lower(rhs_expr, names, env, resolved_names)}"  # noqa: E501
        )
    lines.append(f"{indent}return {lower(body, names, env, resolved_names)}")
    return "\n".join(lines)


def lower_expr(expr: Expr,
               names: Dict[FVarId, str],
               env: Environment,
               resolved_names: Collection[str] = NO_RESOLVED_NAMES) -> str:
    """
    Generates Python code from a elaborated and kernel verified expression.

    Used for a top level program statements.

    Parameters
    ----------
    expr : Expr
        Verified top level expression.
    names : dict[FVarId, str]
        Source name for free variables in scope.
    env : Environment
        CIC environment the term was checked against.
    resolved_names : Collection[str]
        Names whose signature carries the non dimension variable parameters.

    Examples
    --------
    >>> from physika.core.torch_lowering import lower_expr
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.environment import Environment
    >>> lower_expr(App(App(Const("Real.add", ()), Lit(1.0)), Lit(2.0)),
    ...            {}, Environment(), ())
    '(1.0 + 2.0)'
    """
    return lower(expr, names, env, resolved_names)
