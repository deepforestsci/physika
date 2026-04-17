from typing import Any, Callable, Optional, Tuple, Union
from physika.utils.types import Substitution, Type, TVar, TDim, TTensor, TInstance, TFunc, TScalar, T_NAT, T_REAL, T_COMPLEX, new_dim  # noqa: E501
from physika.utils.ast_utils import ASTNode


class ExprContext:
    """
    Data class represeting the context in which an expression is being inferred.

    Passed to every ``expr_*`` handler so each handler receives
    the full typing environment.

    Attributes
    ----------
    env : dict
        Maps variable names to their current ``Type``.
    s : Substitution
        The substitution dictionary accumulated so far. Handler functions thread ``s``
        so each step registers any bindings made by previous steps.
    func_env : dict
        Maps user defined function names to ``(param_types, return_type)``.
    class_env : dict
        Maps class names to their definition dicts (``class_params``,
        ``return_type``, ...).
    add_error : Callable
        Error callback.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, T_REAL
    >>> from physika.utils.types import Substitution
    >>> errors = []
    >>> ctx = ExprContext(env={"x": T_REAL}, s=Substitution(), func_env={}, class_env={}, add_error=errors.append)  # noqa: E501
    >>> ctx.env
    {'x': ℝ}
    >>> ctx.s
    {}
    """

    def __init__(self, env: dict, s: Substitution, func_env: dict,
                 class_env: dict, add_error: Callable) -> None:
        self.env = env
        self.s: Substitution = s
        self.func_env: dict = func_env
        self.class_env: dict = class_env
        self.add_error: Callable = add_error


def expr_num(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    The type of a numeric literal is always ``ℝ``.

    Parameters
    ----------
    node : tuple
        AST node of the form ``("num", value)`` where *value* is an
        ``int`` or ``float``.
    ctx : ExprContext
        Current inference context.

    Returns
    -------
    tuple[Type, Substitution]
        Always ``(T_REAL, ctx.s)``.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_num, T_REAL
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_num(("num", 3.14), ctx)
    >>> t
    ℝ
    """
    return T_REAL, ctx.s


def expr_imaginary(node: Any,
                   ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of the imaginary unit ``i``.

    Inside ``for i : Fin(n)`` body ``i`` is bound as a loop index (``ℝ``). But
    at the top level it is the imaginary unit ``ℂ``.

    Parameters
    ----------
    node : tuple
        AST node ``("imaginary",)``.
    ctx : ExprContext
        Current inference context.  When ``"i"`` is present in ``ctx.env``
        the loop variable shadows the imaginary unit.

    Returns
    -------
    tuple[Type, Substitution]
        ``(T_REAL, ctx.s)`` when ``"i"`` is a live loop variable;
        ``(T_COMPLEX, ctx.s)`` otherwise.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_imaginary, T_REAL, T_COMPLEX
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_imaginary(("imaginary",), ctx)
    >>> t
    ℂ
    >>> ctx_loop = ExprContext({"i": T_REAL}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_imaginary(("imaginary",), ctx_loop)  # loop var shadows ℂ
    >>> t
    ℝ
    """
    if "i" in ctx.env:
        # Loop variable shadows the imaginary unit inside for-expr bodies.
        return ctx.s.apply(ctx.env["i"]), ctx.s
    return T_COMPLEX, ctx.s


def expr_var(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Look up a variable in the current environment.

    Returns ``(None, s)`` when the variable is not yet in scope.

    Parameters
    ----------
    node : tuple
        AST node ``("var", name)`` where *name* is the variable name.
    ctx : ExprContext
        Current inference context.  ``ctx.env`` looks for *name* and
        ``ctx.s`` is applied to the result to expose any resolved
        unification bindings.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(resolved_type, ctx.s)`` when *name* is in scope.
        ``(None, ctx.s)`` otherwise.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_var
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_var(("var", "x"), ctx)
    >>> t
    ℝ[3]
    >>> t, _= expr_var(("var", "y"), ctx)  # not in scope
    >>> t is None
    True
    """
    t = ctx.env.get(node[1])
    # Apply pending substitutions so the caller sees the most-resolved type.
    return (ctx.s.apply(t), ctx.s) if t is not None else (None, ctx.s)


def expr_array(node: Any,
               ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of an array element ``[e0, e1, ..., en]``.

    All elements are inferred and unified pairwise. A type error is reported
    if any two elements have incompatible types. An empty literal produces
    ``ℝ[0]``.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("array", elements)`` where *elements* is a
        list of AST expression nodes, one for each array element.
    ctx : ExprContext
        Current inference context.  ``ctx.s`` is threaded through each
        element inference so later elements see bindings from earlier ones.
        Type errors are registered via ``ctx.add_error``.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(TTensor(dims), updated_s)`` where ``dims[0]`` is the number of
        elements and any additional dims come from the element type (nested
        arrays).  Returns ``(make_tensor([0]), ctx.s)`` for an empty literal.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_array
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_array(("array", [("num", 1.0), ("num", 2.0), ("num", 3.0)]), ctx)
    >>> t
    ℝ[3]
    >>> t, _= expr_array(("array", []), ctx)  # empty literal
    >>> t
    ℝ[0]
    >>> nested = [("array", [("num", 1.0), ("num", 2.0)]), ("array", [("num", 3.0), ("num", 4.0)])]  # noqa: E501
    >>> t, _= expr_array(("array", nested), ctx)  # ℝ[2,2]
    >>> t
    ℝ[2,2]
    """
    from physika.utils.type_checker_utils import make_tensor, unify

    # Example node: ("array", [("num", 1.0), ("num", 2.0), ("num", 3.0)])
    elements = node[1]
    if not elements:
        return make_tensor([0]), ctx.s  # empty array literal

    elem_types = []
    cur = ctx.s
    for e in elements:
        et, cur = infer_expr(e, ctx.env, cur, ctx.func_env, ctx.class_env,
                             ctx.add_error)
        elem_types.append(et)

    # Unify element types pairwise to find a common type for the whole array.
    base = elem_types[0]
    for i, et in enumerate(elem_types[1:], 1):
        if base is not None and et is not None:
            try:
                cur = unify(base, et, cur)
                base = cur.apply(base)
            except TypeError as e:
                ctx.add_error(
                    f"Inconsistent array element types at index {i}: {e}")
    n = len(elements)
    # For a nested array, prepend the outer length as a new leading dimension.
    if isinstance(base, TTensor):
        return TTensor(((n, "invariant"), ) + base.dims), cur
    return make_tensor([n]), cur


def expr_index(node: Any,
               ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of a 1D index expression ``arr[idx]`` from ``arr``'s shape.

    A indexed 1D array produces ``ℝ``. A higher rank tensor
    produces a tensor of the remaining dimensions.  The index expression
    is itself inferred and unified against the leading dimension to
    propagate any symbolic dim bindings. Errors are reported via
    ``ctx.add_error`` when:

    - ``arr`` is not in scope — returns ``(None, ctx.s)``.
    - ``arr`` is a scalar (cannot be indexed).
    - The index type is incompatible with the leading dimension size.

    Parameters
    ----------
    node : tuple
        AST node of the form ``("index", arr_name, idx_expr)`` where
        *idx_expr* is any expression node whose type will be unified
        with the leading dimension.
    ctx : ExprContext
        Current inference context.  If *arr_name* is not in ``ctx.env``,
        we get a `None`` result.  ``ctx.s`` is updated with any new dim
        bindings produced by unifying the index type.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(T_REAL, s)`` when the array is 1D (scalar element).
        ``(TTensor(shape[1:]), s)`` when the array has rank > 1 (row slice).
        ``(None, ctx.s)`` when ``arr_name`` is not in scope.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_index, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"v": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_index(("index", "v", ("num", 0.0)), ctx)  # ℝ[3][0] → ℝ
    >>> t
    ℝ
    >>> ctx2 = ExprContext({"A": TTensor(((3, "invariant"), (4, "invariant")))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_index(("index", "A", ("num", 0.0)), ctx2)  # ℝ[3,4][0] → ℝ[4]  # noqa: E501
    >>> t
    ('tensor', [(4, 'invariant')])
    """
    from physika.utils.type_checker_utils import get_tensor_shape, unify_dim, make_tensor_type  # noqa: E501

    # example node: ("index", "A", ("num", 0.0))
    arr_name = node[1]
    if arr_name not in ctx.env:
        return None, ctx.s

    # Apply pending substitutions to get latest resolved type of arr_name
    arr_t = ctx.s.apply(ctx.env[arr_name])
    shape = get_tensor_shape(arr_t)

    if shape is None:  # arr_name is a scalar
        ctx.add_error(f"Cannot index scalar '{arr_name}'")
        return None, ctx.s

    # Infer the index expression type and unify it with the leading dimension.
    idx_t, s = infer_expr(node[2], ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                          ctx.add_error)
    # Unify the index against the leading dimension to bind symbolic dims.
    if isinstance(idx_t, (TVar, TDim, str, int)) and shape:
        try:
            s = unify_dim(idx_t, shape[0], s)
        except TypeError as e:
            ctx.add_error(f"Index mismatch for '{arr_name}': {e}")

    # Return the leading dimension: ℝ[n] → ℝ, ℝ[n,m] → ℝ[m], etc
    if len(shape) == 1:
        return (T_REAL, s)
    else:
        return (make_tensor_type(shape[1:]), s)


def expr_indexN(node: Any,
                ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of a nD index expression ``arr[i0, i1, ...]``.

    A generalisation of ''expr_index'' to an arbitrary number of indices.
    Each index expression is inferred and unified against the corresponding
    leading dimension of ``arr``. The visited dimensions are stripped
    from the front of the shape.

    Fully indexing an ND tensor produces ``ℝ`` and partial indexing produces
    a lower rank tensor than the original.

    Parameters
    ----------
    node : Any
        AST node of the form ``("indexN", arr_name, idx_exprs)`` *idx_exprs*
        is a list of expression nodes.
    ctx : ExprContext
        Current inference context.  ``ctx.env`` must contain *arr_name*.
        ``ctx.s`` is threaded through each index inference and updated with
        any new dimension bindings from unification.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(TTensor(shape[n_idx:]), s)`` for partial indexing — when the
        number of indices is strictly less than the tensor rank, yielding a
        lower-rank tensor of the remaining dimensions.
        ``(T_REAL, s)`` when fully indexed (index count equals rank).
        ``(None, s)`` when ``arr_name`` is not in scope, resolves to a
        scalar, or is over-indexed (index count exceeds rank); an error is
        reported via ``ctx.add_error`` in the over-indexed case.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_indexN, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"T": TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_indexN(("indexN", "T", [("num", 0.0), ("num", 1.0)]), ctx)  # ℝ[2,3,4][0,1] → ℝ[4]  # noqa: E501
    >>> t
    ℝ[4]
    >>> t, _= expr_indexN(("indexN", "T", [("num", 0.0), ("num", 1.0), ("num", 2.0)]), ctx)  # fully indexed → ℝ  # noqa: E501
    >>> t
    ℝ
    """
    from physika.utils.type_checker_utils import get_tensor_shape, unify_dim, make_tensor  # noqa: E501

    # example node: ("indexN", "A", [i_expr, k_expr])
    arr_name, idx_exprs = node[1], node[2]

    # Apply pending substitutions to get latest resolved type of arr_name
    arr_t = ctx.s.apply(ctx.env[arr_name]) if arr_name in ctx.env else None
    shape = get_tensor_shape(arr_t)

    s = ctx.s
    # Unify each index expression against the corresponding leading dimension
    for idx_expr, dim in zip(idx_exprs, shape or []):
        idx_t, s = infer_expr(idx_expr, ctx.env, s, ctx.func_env,
                              ctx.class_env, ctx.add_error)
        if isinstance(idx_t, (TVar, TDim, str, int)):
            try:
                s = unify_dim(idx_t, dim, s)
            except TypeError as e:
                ctx.add_error(f"Index mismatch for '{arr_name}[...]: {e}")

    n_idx = len(idx_exprs)
    # arr is scalar (catched by expr_index as well)
    # or not in scope (runtime error before type checking)
    if shape is None:
        return None, s
    # overindexed expression
    if n_idx > len(shape):
        ctx.add_error(
            f"Over-indexed '{arr_name}': {n_idx} indices for a rank-{len(shape)} tensor"  # noqa: E501
        )
        return None, s

    # fully indexed
    if n_idx == len(shape):
        return T_REAL, s
    # partial indexed
    return make_tensor(shape[n_idx:]), s


def expr_chain_index(node: Any,
                     ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of a chained index expression ``A[i][k]``.

    The inner expression (``A[i]``) is inferred first, yielding an
    intermediate tensor type. This handler then looks one more leading dimension
    from that result. ``A[i][k]`` is equivalent to ``A[i, k]`` for 2-D arrays.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("chain_index", inner_expr)`` where
        *inner_expr* is an expression node.
    ctx : ExprContext
        Current inference context passed unchanged to the inner expression
        inference.  The returned substitution reflects any new bindings
        produced while inferring *inner_expr*.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(None, s)`` when the inner expression itself failed to type-check.
        ``(T_REAL, s)`` when the inner expression is 0-D or 1-D (scalar result).
        ``(TTensor(inner_shape[1:]), s)`` when the inner expression has rank > 1.

    Raises (via add_error)
    ----------------------
    - Inner expression typed as a scalar (``T_REAL``) — chaining ``[k]``
      onto a scalar is over-indexing.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_chain_index, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"A": TTensor(((3, "invariant"), (4, "invariant")))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> inner = ("index", "A", ("num", 0.0))  # A[0] → ℝ[4]
    >>> t, _= expr_chain_index(("chain_index", inner), ctx)  # A[0][k] → ℝ
    >>> t
    ℝ
    """
    from physika.utils.type_checker_utils import get_tensor_shape, make_tensor
    # Infer the type of the inner sub-expression first.
    obj_t, cur = infer_expr(node[1], ctx.env, ctx.s, ctx.func_env,
                            ctx.class_env, ctx.add_error)

    # Inner inference failed entirely
    if obj_t is None:
        return None, cur

    shape = get_tensor_shape(obj_t)
    if shape is None and isinstance(
            obj_t, tuple) and len(obj_t) == 2 and obj_t[0] == "tensor":
        shape = [d for d, _ in obj_t[1]]

    # Inner expression is already a scalar
    # chaining [k] is over-indexing.
    if shape is None:
        ctx.add_error("Chain index applied to a scalar: cannot index a scalar")
        return None, cur

    # 1D Tensor
    if len(shape) <= 1:
        return T_REAL, cur

    # Higher-rank Tensor (peel one dimension)
    return make_tensor(shape[1:]), cur


def expr_slice(node: Any,
               ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of a slice expression ``arr[start:end]``.

    When both bounds are numeric literals the result length is computed
    as ``end − start`` (end-exclusive).  For higher-rank tensors only
    the leading dimension is sliced and remaining dims are preserved.

    When either bound is a non-literal expression (e.g. a loop variable)
    a fresh symbolic dimension ``TDim("δN")`` is introduced for the sliced
    leading dimension so that the rank and trailing dims are still preserved.

    When both bounds are literals, the following are reported as static
    semantic errors:

    - negative start or end
    - end < start
    - end == start
    - start >= leading dimension (start out of bounds)
    - end > leading dimension (end out of bounds)

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("slice", arr_name, start_expr, end_expr)``
        where *start_expr* and *end_expr* are expression nodes.
    ctx : ExprContext
        Current inference context.  ``ctx.env`` must contain *arr_name*;
        ``ctx.s`` is returned unchanged.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(TTensor([end - start]), ctx.s)`` for a 1D array with literal
        bounds.
        ``(TTensor([end - start] + shape[1:]), ctx.s)`` for a higher-rank
        array with literal bounds.
        ``(TTensor([TDim("δN")] + shape[1:]), ctx.s)`` when either bound is
        dynamic — a fresh symbolic dimension replaces the sliced leading dim.
        ``(None, ctx.s)`` when ``arr_name`` is not in scope.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_slice, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"v": TTensor(((6, "invariant"),))}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_slice(("slice", "v", ("num", 0.0), ("num", 3.0)), ctx)  # v[0:3] → ℝ[3]
    >>> t
    ℝ[3]
    >>> ctx2 = ExprContext({"A": TTensor(((3, "invariant"), (4, "invariant")))}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_slice(("slice", "A", ("num", 0.0), ("num", 2.0)), ctx2)  # A[0:2] → ℝ[2,4]  # noqa: E501
    >>> t
    ℝ[2,4]
    """
    from physika.utils.type_checker_utils import get_tensor_shape, make_tensor
    from physika.utils.types import VarCounter

    # example node: ("slice", "A", ("num", 0.0), ("num", 2.0))
    arr_name = node[1]
    if arr_name not in ctx.env:
        return None, ctx.s

    # Apply pending substitutions to get latest resolved type of arr_name
    arr_t = ctx.s.apply(ctx.env[arr_name])
    shape = get_tensor_shape(arr_t)

    # If arr_name is a scalar or not in scope, we cannot slice it. Return None.
    if shape is None:
        return None, ctx.s

    start, end = node[2], node[3]
    if (isinstance(start, tuple) and start[0] == "num"
            and isinstance(end, tuple) and end[0] == "num"):
        s_val, e_val = int(start[1]), int(end[1])

        # Static semantic checks.
        if s_val < 0:
            ctx.add_error(
                f"Slice start {s_val} is negative in '{arr_name}[{s_val}:{e_val}]'"  # noqa: E501
            )
            return None, ctx.s
        if e_val < 0:
            ctx.add_error(
                f"Slice end {e_val} is negative in '{arr_name}[{s_val}:{e_val}]'"  # noqa: E501
            )
            return None, ctx.s
        if e_val < s_val:
            ctx.add_error(f"Slice end {e_val} is less than start {s_val} "
                          f"in '{arr_name}[{s_val}:{e_val}]'")
            return None, ctx.s
        if e_val == s_val:
            ctx.add_error(
                f"Empty slice '{arr_name}[{s_val}:{e_val}]': start equals end")
            return None, ctx.s
        if s_val >= shape[0]:
            ctx.add_error(
                f"Slice start {s_val} is out of bounds for '{arr_name}' "
                f"with leading dimension {shape[0]}")
            return None, ctx.s
        if e_val > shape[0]:
            ctx.add_error(
                f"Slice end {e_val} is out of bounds for '{arr_name}' "
                f"with leading dimension {shape[0]}")
            return None, ctx.s

        length = e_val - s_val
        if len(shape) == 1:
            return make_tensor([length]), ctx.s
        else:
            return make_tensor([length] + list(shape[1:])), ctx.s

    # introduce a fresh symbolic dimension for the sliced
    # leading dim so the rank and trailing dims are still correct.
    fresh_dim = VarCounter().new_dim()
    return make_tensor([fresh_dim] + list(shape[1:])), ctx.s


def expr_add_sub(node: Any,
                 ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the result type of an addition or subtraction ``t1 +/− t2``.

    Both operand types are inferred with the substitution, so bindings
    made while inferring the left operand are visible when inferring the
    right. For two tensor operands their shapes are unified to catch
    mismatches. The result shape equals the unified shape.  When one
    operand is scalar (``ℝ``) and the other is a tensor, broadcasting
    applies and the tensor shape is returned.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("add", left_expr, right_expr)`` or
        ``("sub", left_expr, right_expr)``.
    ctx : ExprContext
        Current inference context.  ``ctx.s`` is threaded through both
        operand inferences and updated with any new unification bindings.
        Shape mismatch errors are registered via ``ctx.add_error``.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(tensor_type, s)`` when either operand is a tensor, the tensor
        shape is the broadcast result (unified shape for tensor+tensor and
        tensor shape for tensor+scalar).
        ``(T_REAL, s)`` when both operands are scalars.
        ``(None, s)`` when both operand types could not be determined.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_add_sub, T_REAL, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),)), "y": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_add_sub(("add", ("var", "x"), ("var", "y")), ctx)  # ℝ[3] + ℝ[3] → ℝ[3]
    >>> t
    ℝ[3]
    >>> t, _= expr_add_sub(("add", ("var", "x"), ("num", 1.0)), ctx)  # ℝ[3] + ℝ → ℝ[3]
    >>> t
    ℝ[3]
    >>> t, _= expr_add_sub(("sub", ("num", 1.0), ("num", 2.0)), ctx)  # ℝ - ℝ → ℝ  # noqa: E501
    >>> t
    ℝ
    """
    from physika.utils.type_checker_utils import broadcast_op, unify, type_to_str  # noqa: E501

    op = node[0]  # "add" or "sub"
    t1, s = infer_expr(node[1], ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                       ctx.add_error)
    t2, s = infer_expr(node[2], ctx.env, s, ctx.func_env, ctx.class_env,
                       ctx.add_error)

    # Apply accumulated substitutions before shape comparison.
    from typing import cast
    t1 = s.apply(cast(Type, t1)) if t1 is not None else t1
    t2 = s.apply(cast(Type, t2)) if t2 is not None else t2
    if isinstance(t1, TTensor) and isinstance(t2, TTensor):
        try:
            s = unify(t1, t2, s)
            t1 = s.apply(t1)  # apply any new dim bindings in t1
        except TypeError:
            ctx.add_error(
                f"Shape mismatch in {op}: {type_to_str(t1)} vs {type_to_str(t2)}"  # noqa: E501
            )
    # Scalar + Tensor → Tensor
    # Scalar + Scalar → Scalar.
    return broadcast_op(t1, t2), s


def expr_mul(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of multiplication ``t1 * t2``.

    As same as ``expr_add_sub``, shapes must match for tensor operands, and
    follows broadcast rules.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("mul", left_expr, right_expr)``.
    ctx : ExprContext
        Current inference context.  ``ctx.s`` is threaded through both
        operand inferences and updated with any new unification bindings.
        Shape mismatch errors are registered via ``ctx.add_error``.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(TTensor, s)`` when either operand is a tensor. The tensor
        shape is the broadcast result.
        ``(T_REAL, s)`` when both operands are scalars.
        ``(None, s)`` when both operand types could not be determined.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_mul, T_REAL, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _=expr_mul(("mul", ("var", "x"), ("num", 2.0)), ctx)  # ℝ[3] * ℝ → ℝ[3]
    >>> t
    ℝ[3]
    >>> t, _= expr_mul(("mul", ("num", 2.0), ("num", 3.0)), ctx)  # ℝ * ℝ → ℝ
    >>> t
    ℝ
    """
    from physika.utils.type_checker_utils import unify, type_to_str, broadcast_op  # noqa: E501

    t1, s = infer_expr(node[1], ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                       ctx.add_error)
    t2, s = infer_expr(node[2], ctx.env, s, ctx.func_env, ctx.class_env,
                       ctx.add_error)

    # Apply accumulated substitutions before shape comparison.
    if t1:
        t1 = s.apply(t1)

    if t2:
        t2 = s.apply(t2)

    if isinstance(t1, TTensor) and isinstance(t2, TTensor):
        try:
            s = unify(t1, t2, s)
            t1 = s.apply(t1)  # register any new dim bindings in t1
        except TypeError:
            ctx.add_error(
                f"Shape mismatch in mul: {type_to_str(t1)} vs {type_to_str(t2)}"  # noqa: E501
            )
    return broadcast_op(t1, t2), s


def expr_div(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the result type of division ``t1 / t2``.

    Valid cases:

    - **tensor / scalar** broadcasts and result has the shape of ``t1``.
    - **scalar / scalar** result is ``ℝ``.
    - **tensor / tensor** must match dimensions for element-wise operation,
      else a type error is reported.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("div", numerator_expr, denominator_expr)``.
    ctx : ExprContext
        Current inference context.  ``ctx.s`` is threaded through both
        operand inferences and updated with any new unification bindings.
        Shape mismatch errors are registered via ``ctx.add_error``.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(unified_tensor_type, s)`` for tensor / tensor.
        ``(tensor_type, s)`` for tensor / scalar (broadcast).
        ``(T_REAL, s)`` for scalar / scalar.
        ``(None, s)`` when the numerator type could not be determined.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_div, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),)), "y": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_div(("div", ("var", "x"), ("num", 2.0)), ctx)  # ℝ[3] / ℝ → ℝ[3]
    >>> t
    ℝ[3]
    >>> t, _= expr_div(("div", ("num", 6.0), ("num", 2.0)), ctx)  # ℝ / ℝ → ℝ
    >>> t
    ℝ
    >>> t, _= expr_div(("div", ("var", "x"), ("var", "y")), ctx)  # ℝ[3] / ℝ[3] → ℝ[3]
    >>> t
    ℝ[3]
    >>> errors = []
    >>> ctx2 = ExprContext({"x": TTensor(((3, "invariant"),)), "z": TTensor(((2, "invariant"),))}, Substitution(), {}, {}, errors.append)  # noqa: E501
    >>> t, _= expr_div(("div", ("var", "x"), ("var", "z")), ctx2)  # ℝ[3] / ℝ[2] → error
    >>> errors
    ['Shape mismatch in div: ℝ[3] vs ℝ[2]']
    """
    from physika.utils.type_checker_utils import unify, type_to_str, broadcast_op  # noqa: E501

    t1, s = infer_expr(node[1], ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                       ctx.add_error)
    t2, s = infer_expr(node[2], ctx.env, s, ctx.func_env, ctx.class_env,
                       ctx.add_error)

    if t1:
        t1 = s.apply(t1)

    if t2:
        t2 = s.apply(t2)

    if isinstance(t1, TTensor) and isinstance(t2, TTensor):
        try:
            s = unify(t1, t2, s)
            t1 = s.apply(t1)
        except TypeError:
            ctx.add_error(
                f"Shape mismatch in div: {type_to_str(t1)} vs {type_to_str(t2)}"  # noqa: E501
            )
    return broadcast_op(t1, t2), s


def expr_matmul(node: Any,
                ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the result type of matrix multiplication ``t1 @ t2``.

    ``matmul_op`` handles supported rank combinations.
    The inner dimensions must agree and a type error is reported otherwise.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("matmul", left_expr, right_expr)``.
    ctx : ExprContext
        Current inference context.  ``ctx.s`` is threaded through both
        operand inferences and updated with any new substitution bindings.
        Shape incompatibility errors are reported via ``ctx.add_error``.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(T_REAL, s)`` for a dot product (vector @ vector).
        ``(TTensor([m, p]), s)`` for a mat-mat product (ℝ[m,n] @ ℝ[n,p]).
        ``(None, s)`` when either operand type could not be determined or
        shapes are incompatible.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_matmul, TTensor
    >>> from physika.utils.types import Substitution
    >>> env = {"A": TTensor(((2, "invariant"), (3, "invariant"))), "B": TTensor(((3, "invariant"), (4, "invariant")))}  # noqa: E501
    >>> ctx = ExprContext(env, Substitution(), {}, {}, [].append)
    >>> t, _= expr_matmul(("matmul", ("var", "A"), ("var", "B")), ctx)  # ℝ[2,3] @ ℝ[3,4] → ℝ[2,4]
    >>> t
    ℝ[2,4]
    >>> env2 = {"u": TTensor(((3, "invariant"),)), "v": TTensor(((3, "invariant"),))}  # noqa: E501
    >>> ctx2 = ExprContext(env2, Substitution(), {}, {}, [].append)
    >>> t, _= expr_matmul(("matmul", ("var", "u"), ("var", "v")), ctx2)  # ℝ[3] @ ℝ[3] → ℝ (dot)  # noqa: E501
    >>> t
    ℝ
    """
    from physika.utils.type_checker_utils import matmul_op
    t1, s = infer_expr(node[1], ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                       ctx.add_error)
    t2, s = infer_expr(node[2], ctx.env, s, ctx.func_env, ctx.class_env,
                       ctx.add_error)
    # Apply accumulated substitutions so symbolic dims are as resolved as possible.  # noqa: E501
    if t1:
        t1 = s.apply(t1)

    if t2:
        t2 = s.apply(t2)

    return matmul_op(t1, t2, ctx.add_error), s


def expr_pow(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """Infer the result type of exponentiation ``t1 ** t2``.

    The result has the same shape as the base. The exponent type is inferred
    to accumulate substitutions but does not affect the output shape.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("pow", base_expr, exp_expr)``.
    ctx : ExprContext
        Current inference context.  Only the base expression is used to
        determine the result type; ``ctx.s`` is threaded through the base
        inference and the result has the base's type after substitution.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(base_type, s)``. Same type as the base expression after applying
        the accumulated substitution.
        ``(None, s)`` when the base type could not be determined.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_pow, TTensor
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_pow(("pow", ("var", "x"), ("num", 2.0)), ctx)  # ℝ[3] ** ℝ → ℝ[3]  # noqa: E501
    >>> t
    ℝ[3]
    >>> t, _= expr_pow(("pow", ("num", 2.0), ("num", 3.0)), ctx)  # ℝ ** ℝ → ℝ
    >>> t
    ℝ
    """
    t1, s = infer_expr(node[1], ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                       ctx.add_error)

    if t1:
        t1 = s.apply(t1)

    return t1, s


def expr_neg(node: Any,
             ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the result type of negation ``-expr``.

    Negation is shape-preserving and inferred through the
    single operand sub-expression.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("neg", operand_expr)``.
    ctx : ExprContext
        Current inference context passed unchanged to the operand inference.
        The returned substitution reflects any new bindings produced while
        inferring the operand.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        Exactly the result of inferring the operand: ``(operand_type, s)``.
        ``(None, s)`` when the operand type could not be determined.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_neg, TTensor, T_REAL
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_neg(("neg", ("var", "x")), ctx)  # -ℝ[3] → ℝ[3]
    >>> t
    ℝ[3]
    >>> t, _= expr_neg(("neg", ("num", 1.0)), ctx)  # -ℝ → ℝ
    >>> t
    ℝ
    """
    return infer_expr(node[1], ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                      ctx.add_error)


def expr_call(node: Any,
              ctx: ExprContext) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the result type of a function call.

    Resolution order:

    1. Built-in elementwise functions (``exp``, ``sin``, ``cos``, ``sqrt``,
       ``abs``, ``tanh``, ``log``, ``real``, ``imag``) — preserve the shape
       of their first argument.
    2. Built-in reduction (``sum``) → ``ℝ``.
    3. ``grad(f, x)`` → same shape as ``x``.
    4. User-defined functions in ``func_env`` — arity and argument types
       are checked; each argument is unified against its declared parameter
       type; the declared return type is returned.


    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("call", func_name, arg_list)`` where
        *func_name* is a ``str`` and *arg_list* is a list of expression nodes.
    ctx : ExprContext
        Current inference context.  ``ctx.s`` is threaded through all
        argument inferences and updated with unification bindings from
        parameter matching.  Arity and type mismatch errors are reported via
        ``ctx.add_error``.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(result_type, s)`` where *result_type* depends on the solved
        path and type.
        ``(None, s)`` when the call target is unrecognised.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_call, T_REAL, TTensor  # noqa: E501
    >>> from physika.utils.types import Substitution
    >>> ctx = ExprContext({"x": TTensor(((3, "invariant"),))}, Substitution(), {}, {}, [].append)  # noqa: E501
    >>> t, _= expr_call(("call", "sin", [("var", "x")]), ctx)  # element-wise
    >>> t
    ℝ[3]
    >>> t, _= expr_call(("call", "sum", [("var", "x")]), ctx)  # ℝ
    >>> t
    ℝ
    >>> t, _= expr_call(("call", "grad", [("num", 1.0), ("var", "x")]), ctx)
    >>> t
    ℝ[3]
    >>> func_env = {"f": ([TTensor(((3, "invariant"),))], TTensor(((3, "invariant"),)))}  # noqa: E501
    >>> ctx2 = ExprContext({"x": TTensor(((3, "invariant"),))}, Substitution(), func_env, {}, [].append)  # noqa: E501
    >>> t, _= expr_call(("call", "f", [("var", "x")]), ctx2)
    >>> t
    ℝ[3]
    """
    from physika.utils.type_checker_utils import from_typespec, unify

    func_name, args = node[1], node[2]
    # Infer all argument types first
    arg_types = []
    s = ctx.s
    for a in args:
        at, s = infer_expr(a, ctx.env, s, ctx.func_env, ctx.class_env,
                           ctx.add_error)
        arg_types.append(at)

    # Built-in functions
    elementwise_ops = ("exp", "log", "sin", "cos", "sqrt", "abs", "tanh",
                       "real", "imag")
    if func_name in elementwise_ops:
        # Element-wise ops preserve the shape of their argument
        if arg_types:
            return arg_types[0], s
        else:
            return T_REAL, s

    if func_name in ("sum"):
        return T_REAL, s
    if func_name == "grad":
        # TODO: We should add support to get the tangent space type (e.g. Tₓ)
        return (arg_types[1] if len(arg_types) >= 2 else None), s

    # User defined functions
    if func_name in ctx.func_env:
        param_types, ret_type = ctx.func_env[func_name]
        if len(args) != len(param_types):
            ctx.add_error(
                f"Function '{func_name}' expects {len(param_types)} args, got {len(args)}"  # noqa: E501
            )
        else:
            for i, (pt, at) in enumerate(zip(param_types, arg_types)):
                # Convert raw typespec to Type when needed.
                if not isinstance(pt,
                                  (TVar, TScalar, TTensor, TFunc, TInstance)):
                    pt = from_typespec(pt)
                if pt is not None and at is not None:
                    try:
                        s = unify(pt, at, s)
                    except TypeError as e:
                        ctx.add_error(f"Arg {i} of '{func_name}': {e}")
        # return the inferred type (no errors catched during unifcation)
        if not isinstance(ret_type, (TVar, TScalar, TTensor, TFunc, TInstance,
                                     type(None))):  # noqa: E125
            ret = from_typespec(ret_type)
        else:
            ret = ret_type
        return ret, s

    # Unknown call
    return None, s


def expr_for_expr(
        node: Any, ctx: ExprContext,
        new_dim: Callable[[], TDim]) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of for-expression ``for i : ℕ(n) → body``.

    The loop variable ``i`` is bound as ``ℕ`` inside the body.
    The result type has explicit ``ℕ(n)`` size as its leading dimension.
    When the body is itself a tensor (implicit for loops) the result is same
    rank. Nested for-exprs produce multi-dimensional tensors.

    Parameters
    ----------
    node : ASTNode
        AST node of the form ``("for_expr", loop_var, size_expr, body_expr)``
        where *size_expr* is typically ``("num", n)`` literal, and *body_expr*
        is the body expression.
    ctx : ExprContext
        Current inference context.  The body is inferred with an extended
        environment that passes *loop_var* as ``ℕ``.  ``ctx.s`` is threaded
        through the size and body inferences.
    new_dim : Callable[[], TDim]
        Fresh symbolic dimension variables, used when *size_expr*
        is not a numeric literal.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(TTensor(((n, "invariant"),) + body_dims), s)`` when the body has a
        tensor type, prepends the outer ``ℕ(n)`` dimension.
        ``(TTensor(((n, "invariant"),)), s)`` when the body is scalar (``ℝ``).
        The leading dimension is a concrete ``int`` for literal sizes or a
        fresh ``TDim`` for dynamic sizes.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_for_expr
    >>> from physika.utils.types import Substitution, new_dim
    >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_for_expr(("for_expr", "i", ("num", 3.0), ("imaginary")), ctx, new_dim)  #  ℝ[3]  # noqa: E501
    >>> t
    ℝ[3]
    >>> inner_body = ("array", [("num", 1.0), ("num", 2.0)])  #  ℝ[2]
    >>> t, _= expr_for_expr(("for_expr", "i", ("num", 4.0), inner_body), ctx, new_dim)  # ℝ[4,2]  # noqa: E501
    >>> t
    ℝ[4,2]
    """
    from physika.utils.type_checker_utils import make_tensor
    loop_var, size_expr, body_expr = node[1], node[2], node[3]
    # Infer the size expression for substitution.
    _, s = infer_expr(size_expr, ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                      ctx.add_error)

    # Extend env dict with the loop variable bound as ℕ.
    body_t, s = infer_expr(node=body_expr,
                           env={
                               **ctx.env, loop_var: T_NAT
                           },
                           s=s,
                           func_env=ctx.func_env,
                           class_env=ctx.class_env,
                           add_error=ctx.add_error)

    # Use a concrete dim when the size is a literal, otherwise a fresh TDim.
    outer_dim: Union[int, TDim]
    if isinstance(size_expr, tuple) and size_expr[0] == "num":
        outer_dim = int(size_expr[1])
    else:
        outer_dim = new_dim()

    # Prepend the outer dimension:
    # scalar body → ℝ[n]
    # tensor body → ℝ[n, etc]
    # for 1-level for loops body_t ==  ℕ -> True
    if isinstance(body_t, TTensor):
        return TTensor(((outer_dim, "invariant"), ) + body_t.dims), s
    return make_tensor([outer_dim]), s


def expr_for_expr_range(
        node: Any, ctx: ExprContext,
        new_dim: Callable[[], TDim]) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of a range for-expression ``for i : ℕ(start, end) → body``

    Is similar to ``expr_for_expr`` but the size is derived from numeric
    ``start`` and ``end`` bounds. When bounds are non literal, a fresh
    symbolic dimension is introduced.

    Parameters
    ----------
    node : ASTNode
        AST node of the form
        ``("for_expr_range", loop_var, start_expr, end_expr, body_expr)``
        where  *start_expr* and *end_expr* are num nodes (``("num", value)``).
    ctx : ExprContext
        Current inference context.  The body is inferred with an extended
        environment that passes *loop_var* as ``ℕ``.  ``ctx.s`` is
        threaded through the body inference.
    new_dim : Callable[[], TDim]
        Fresh symbolic dimension variables, used when
        *start_expr* or *end_expr* is not a numeric literal.

    Returns
    -------
    tuple[Optional[Type], Substitution]
        ``(TTensor(((end - start, "invariant"),) + body_dims), s)`` when both
        bounds are literals and the body has a tensor type.
        ``(TTensor(((end - start, "invariant"),)), s)`` when both bounds are
        literals and the body is scalar.
        ``(TTensor(((TDim, "invariant"), ...)), s)`` when either bound is
        dynamic, a fresh symbolic dimension replaces the outer size.

    Examples
    --------
    >>> from physika.utils.infer_expr import ExprContext, expr_for_expr_range
    >>> from physika.utils.types import Substitution, new_dim
    >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
    >>> t, _= expr_for_expr_range(("for_expr_range", "i", ("num", 0.0), ("num", 4.0), ("imaginary")), ctx, new_dim)
    >>> t  # ℝ[4]
    ℝ[4]
    >>> inner_body = ("array", [("num", 0.0), ("num", 1.0), ("num", 2.0)])  # body produces ℝ[3]
    >>> t, _= expr_for_expr_range(("for_expr_range", "k", ("num", 0.0), ("num", 2.0), inner_body), ctx, new_dim)  # noqa: E501
    >>> t  # ℝ[2,3]
    ℝ[2,3]
    """
    from physika.utils.type_checker_utils import make_tensor

    _, loop_var, start_expr, end_expr, body_expr = node

    # Bind loop variable as ℕ
    body_t, s = infer_expr(node=body_expr,
                           env={
                               **ctx.env, loop_var: T_NAT
                           },
                           s=ctx.s,
                           func_env=ctx.func_env,
                           class_env=ctx.class_env,
                           add_error=ctx.add_error)

    outer_dim: Union[int, TDim]
    if (isinstance(start_expr, tuple) and start_expr[0] == "num"
            and isinstance(end_expr, tuple) and end_expr[0] == "num"):
        outer_dim = int(end_expr[1]) - int(start_expr[1])
    else:
        outer_dim = new_dim()

    # Case nested for-loops
    if isinstance(body_t, TTensor):
        return TTensor(((outer_dim, "invariant"), ) + body_t.dims), s
    return make_tensor([outer_dim]), s


EXPR_DISPATCH: dict = {
    "num": expr_num,
    "var": expr_var,
    "imaginary": expr_imaginary,
    "array": expr_array,
    "index": expr_index,
    "indexN": expr_indexN,
    "chain_index": expr_chain_index,
    "slice": expr_slice,
    "add": expr_add_sub,
    "sub": expr_add_sub,
    "mul": expr_mul,
    "div": expr_div,
    "matmul": expr_matmul,
    "pow": expr_pow,
    "neg": expr_neg,
    "call": expr_call,
    "for_expr": expr_for_expr,
    "for_expr_range": expr_for_expr_range,
}


def infer_expr(
    node: ASTNode,
    env: dict,
    s: Substitution,
    func_env: dict,
    class_env: dict,
    add_error: Callable,
) -> Tuple[Optional[Type], Substitution]:
    """
    Infer the type of an expression AST node.

    This is the main entry point for expression level type inference.
    The substitution *s* is threaded through every recursive call so that
    unification bindings made by sub-expressions are visible to the next ones.
    Dispatch is driven by the string tag in ``node[0]`` via ``EXPR_DISPATCH``.

    Parameters
    ----------
    node : ASTNode
        The expression node to type-check.  Accepted shapes:

        * ``None``                        — missing/optional sub-expression.
        * ``int`` or ``float``            — bare numeric literal (from the parser).
        * ``("num",    value)``           — explicit numeric literal node.
        * ``("var",    name)``            — variable reference.
        * ``("imaginary",)``              — the imaginary unit ``i``.
        * ``("array",  elems)``           — array literal ``[e0, e1, …]``.
        * ``("index",  arr, idx)``        — 1-D subscript ``arr[idx]``.
        * ``("indexN", arr, idxs)``       — N-D subscript ``arr[i,j,…]``.
        * ``("chain_index", base, idx)``  — chained subscript ``A[i][k]``.
        * ``("slice",  arr, start, end)`` — slice ``arr[start:end]``.
        * ``("add",    l, r)``            — addition ``l + r``.
        * ``("sub",    l, r)``            — subtraction ``l - r``.

    env : dict
        Maps variable names (``str``) to their current inferred ``Type``.
        Populated by the statement level inferencer before calling this function.
    s : Substitution
        The substitution dictionary accumulated so far.  Each call returns a
        substitution that the caller should thread forward.
    func_env : dict
        Maps user defined function names to ``(param_types, return_type)`` pairs.
    class_env : dict
        Maps class names to their definition dicts (``class_params``,
        ``return_type``, etc), used for constructor call inference.
    add_error : Callable
        Error reporting callback (``errors.append``).

    Returns
    -------
    tuple[Optional[Type], Substitution]
        A pair ``(t, s2)`` where:

        * ``t``  is the inferred ``Type`` or ``None`` when the node cannot
          be typed.
        * ``s2`` is the updated substitution that includes any new unification
          bindings produced while inferring *node*.

    Examples
    --------
    >>> from physika.utils.infer_expr import infer_expr, T_REAL
    >>> from physika.utils.types import Substitution, TTensor
    >>> errors = []
    >>> # Numeric literal always infers ℝ
    >>> t, _ = infer_expr(3.14, {}, Substitution(), {}, {}, errors.append)
    >>> t
    ℝ
    >>> # Variable reference resolved from env
    >>> t, _ = infer_expr(("var", "x"), {"x": T_REAL}, Substitution(), {}, {}, errors.append)
    >>> t
    ℝ
    >>> # Array literal of two scalars is type ℝ[2]
    >>> t, _ = infer_expr(("array", [("num", 1.0), ("num", 2.0)]), {}, Substitution(), {}, {}, errors.append)
    >>> t
    ℝ[2]
    >>> # Unknown tag error recorded, returns None
    >>> t, _ = infer_expr(("unknown", 42), {}, Substitution(), {}, {}, errors.append)  # noqa: E501
    >>> t is None
    True
    >>> errors[-1]
    'Unknown expression type: unknown'
    """

    if node is None:
        return None, s

    if not isinstance(node, tuple):
        if isinstance(node, (int, float)):
            return T_REAL, s
        else:
            return None, s

    ctx = ExprContext(env=env,
                      s=s,
                      func_env=func_env,
                      class_env=class_env,
                      add_error=add_error)
    # Dispatch to the appropriate expr_* handler based on the AST tag.
    handler = EXPR_DISPATCH.get(node[0])
    if handler is not None:
        if node[0] in ("for_expr", "for_expr_range"):
            return handler(node, ctx, new_dim)
        return handler(node, ctx)

    # No handler registered for this tag — report and return unknown type.
    add_error(f"Unknown expression type: {node[0]}")
    return None, s
