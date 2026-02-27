from __future__ import annotations

from typing import Any, Callable, List, Tuple, Union


# Type aliases used throughout this module.
TypeSpec = Any       # "ℝ", "ℕ", ("tensor", [...]), ("func_type", ...), None
ASTExpr = Any        # tagged tuple, scalar, or None
ErrorFn = Callable[[str], None]
InferFn = Callable[..., TypeSpec]


def type_to_str(t: TypeSpec) -> str:
    """Convert a Physika type spec to a readable string.

    Parameters
    ----------
    t : TypeSpec
        A Physika type: ``"ℝ"``, ``"ℕ"``, ``"ℂ"``,
        ``("tensor", [(dim, variance), ...])``,
        ``("func_type", input, output)``, or ``None``.

    Returns
    -------
    str
        A readable representation, e.g. ``"ℝ"``, ``"ℝ[3]"``,
        ``"ℝ[2,3]"``, ``"(ℝ) → ℝ"``, or ``"unknown"`` for ``None``.

    Examples
    --------
    >>> from utils.type_checker_utils import type_to_str
    >>> type_to_str("ℝ")
    'ℝ'
    >>> type_to_str(("tensor", [(3, "invariant")]))
    'ℝ[3]'
    >>> type_to_str(("tensor", [(2, "invariant"), (3, "invariant")]))
    'ℝ[2,3]'
    >>> type_to_str(None)
    'unknown'
    """
    if t is None:
        return "unknown"
    if isinstance(t, str):
        return t
    if isinstance(t, tuple):
        if t[0] == "tensor":
            dims = t[1]
            if len(dims) == 1:
                return f"ℝ[{dims[0][0]}]"
            else:
                dim_strs = [str(d[0]) for d in dims]
                return f"ℝ[{','.join(dim_strs)}]"
        elif t[0] == "func_type":
            return f"({t[1]}) → {type_to_str(t[2])}"
    return str(t)


def get_shape(t: TypeSpec) -> List[int] | None:
    """Extract the shape dimensions from a type spec.

    Parameters
    ----------
    t : TypeSpec
        A Physika type.

    Returns
    -------
    list[int] or None
        The list of dimension sizes for tensor types, or ``None``
        for scalars (``"ℝ"``, ``"ℕ"``) and unrecognised types.

    Examples
    --------
    >>> from utils.type_checker_utils import get_shape
    >>> get_shape(("tensor", [(3, "invariant")]))
    [3]
    >>> get_shape(("tensor", [(2, "invariant"), (3, "invariant")]))
    [2, 3]
    >>> get_shape("ℝ") is None
    True
    """
    if t in ("ℝ", "ℕ"):
        return None
    if isinstance(t, tuple) and t[0] == "tensor":
        return [d[0] for d in t[1]]
    return None


def make_tensor_type(shape: list[int] | None) -> TypeSpec:
    """Build a tensor type spec from a shape list.

    Parameters
    ----------
    shape : list[int] or None
        Dimension sizes.  ``None`` or empty list produces ``"ℝ"``
        (scalar).

    Returns
    -------
    TypeSpec
        ``"ℝ"`` for scalars, or ``("tensor", [(d, "invariant"), ...])``
        for tensors.

    Examples
    --------
    >>> from utils.type_checker_utils import make_tensor_type
    >>> make_tensor_type(None)
    'ℝ'
    >>> make_tensor_type([3])
    ('tensor', [(3, 'invariant')])
    >>> make_tensor_type([2, 3])
    ('tensor', [(2, 'invariant'), (3, 'invariant')])
    """
    if shape is None or len(shape) == 0:
        return "ℝ"
    return ("tensor", [(d, "invariant") for d in shape])


def dims_compatible(d1: int | str, d2: int | str) -> bool:
    """Check if two dimension values are compatible.

    Two dimensions are compatible if is a symbolic string (e.g. "M", "N")
    or if they are equal integers.

    Parameters
    ----------
    d1 : int or str
        First dimension.
    d2 : int or str
        Second dimension.

    Returns
    -------
    bool
        ``True`` if the dimensions are compatible.

    Examples
    --------
    >>> from utils.type_checker_utils import dims_compatible
    >>> dims_compatible(3, 3)
    True
    >>> dims_compatible("M", 16)
    True
    >>> dims_compatible("M", "M")
    True
    >>> dims_compatible("M", "N")
    True
    >>> dims_compatible(3, 4)
    False
    """
    if isinstance(d1, str) or isinstance(d2, str):
        return True  # at least one symbolic — accept any pairing
    return d1 == d2


def types_compatible(t1: TypeSpec, t2: TypeSpec) -> bool:
    """Check whether two Physika types are compatible.

    Two types are compatible if either is ``None`` (unknown), they are
    equal, both are numeric scalars (``"ℝ"`` / ``"ℕ"``), or they are
    tensors with the same shape.

    Parameters
    ----------
    t1 : TypeSpec
        First type.
    t2 : TypeSpec
        Second type.

    Returns
    -------
    bool
        ``True`` if the types are compatible, ``False`` otherwise.

    Examples
    --------
    >>> from utils.type_checker_utils import types_compatible
    >>> types_compatible("ℝ", "ℝ")
    True
    >>> types_compatible("ℝ", "ℕ")
    True
    >>> types_compatible("ℝ", ("tensor", [(3, "invariant")]))
    False
    >>> types_compatible(None, "ℝ")
    True
    """
    if t1 is None or t2 is None:
        return True
    if t1 == t2:
        return True
    if t1 in ("ℝ", "ℕ") and t2 in ("ℝ", "ℕ"):
        return True
    s1 = get_shape(t1)
    s2 = get_shape(t2)
    if s1 is None or s2 is None:
        return s1 is None and s2 is None
    if len(s1) != len(s2):
        return False
    return all(dims_compatible(d1, d2) for d1, d2 in zip(s1, s2))


def shapes_broadcast_compatible(
    s1: Union[List[int] | None],
    s2: Union[List[int] | None],
    allow_scalar_broadcast: bool = False,
) -> Union[Tuple[List[int] | None, bool]]:
    """Check whether two shapes are broadcast-compatible.

    Parameters
    ----------
    s1 : list[int] or None
        Shape of the left operand (``None`` means scalar).
    s2 : list[int] or None
        Shape of the right operand (``None`` means scalar).
    allow_scalar_broadcast : bool, default False
        If ``True``, a scalar (``None``) is compatible with any
        tensor shape, and the tensor shape is returned as the result.

    Returns
    -------
    tuple[list[int] | None, bool]
        A ``(result_shape, ok)`` pair.  ``ok`` is ``True`` if the
        shapes are compatible; ``result_shape`` is the broadcasted
        shape, or ``None`` if incompatible.

    Examples
    --------
    >>> from utils.type_checker_utils import shapes_broadcast_compatible
    >>> shapes_broadcast_compatible([3], [3])
    ([3], True)
    >>> shapes_broadcast_compatible(None, [3], allow_scalar_broadcast=True)
    ([3], True)
    >>> shapes_broadcast_compatible([2], [3])
    (None, False)
    """
    if s1 is None and s2 is None:
        return None, True

    if s1 == s2:
        return s1, True

    if allow_scalar_broadcast:
        if s1 is None:
            return s2, True
        if s2 is None:
            return s1, True

    # Allow shapes that differ only in symbolic vs concrete dimensions
    if s1 is not None and s2 is not None and len(s1) == len(s2):
        if all(dims_compatible(d1, d2) for d1, d2 in zip(s1, s2)):
            return s1, True  # Return first shape (may contain symbolic dims)

    return None, False


def get_line_info(stmt: ASTExpr) -> int | None:
    """Extract the source line number from a statement AST node.

    Each statement type stores the line number at a different index.
    This function knows the layout for ``decl``, ``assign``, ``expr``,
    and ``for_loop``.

    Parameters
    ----------
    stmt : ASTExpr
        A program-level statement tuple, or ``None``.

    Returns
    -------
    int or None
        The source line number, or ``None`` if it cannot be
        determined.

    Examples
    --------
    >>> from utils.type_checker_utils import get_line_info
    >>> get_line_info(("decl", "x", "ℝ", ("num", 3.0), 7))
    7
    >>> get_line_info(("expr", ("num", 1.0), 3))
    3
    >>> get_line_info(None) is None
    True
    """
    if stmt is None:
        return None
    op = stmt[0]
    if op == "decl" and len(stmt) >= 5:
        return stmt[4]
    elif op == "assign" and len(stmt) >= 4:
        return stmt[3]
    elif op == "expr" and len(stmt) >= 3:
        return stmt[2]
    elif op == "for_loop" and len(stmt) >= 5:
        return stmt[4]
    return None


def type_infer(
    op: str,
    expr: ASTExpr,
    type_env: dict[str, TypeSpec],
    local_env: dict[str, TypeSpec] | None,
    add_error: ErrorFn,
    infer_type: InferFn,
    func_env: dict[str, tuple[list[TypeSpec], TypeSpec | None]],
    class_env: dict[str, dict[str, Any]],
) -> TypeSpec:
    """Infer the Physika type of an AST expression by its tag.

    This is the main dispatch function called by
    ``TypeChecker.infer_type``.  It pattern-matches on the expression
    tag (*op*) and recursively infers operand types using *infer_type*
    (which is ``TypeChecker.infer_type`` itself, passed as a callback).

    Handles: ``num``, ``var``, ``array``, ``index``, ``slice``,
    ``add``/``sub``, ``mul``, ``div``, ``matmul``, ``pow``, ``neg``,
    ``call``, ``call_index``, ``string``, ``imaginary``.

    Parameters
    ----------
    op : str
        The expression tag (first element of the AST tuple).
    expr : ASTExpr
        The full AST expression tuple.
    type_env : dict[str, TypeSpec]
        Program-level variable type environment.
    local_env : dict[str, TypeSpec] or None
        Local (function-scope) type environment.
    add_error : ErrorFn
        Callback to record a type error message.
    infer_type : InferFn
        Recursive type inference callback (``TypeChecker.infer_type``).
    func_env : dict[str, tuple[list[TypeSpec], TypeSpec | None]]
        Registered function signatures ``{name: (param_types, return_type)}``.
    class_env : dict[str, dict[str, Any]]
        Registered class AST definitions.

    Returns
    -------
    TypeSpec
        The inferred type, or ``None`` if it cannot be determined.

    Examples
    --------
    >>> from utils.type_checker_utils import type_infer
    >>> type_infer("num", ("num", 3.0), {}, {}, lambda m: None,
    ...            lambda e, le=None: "ℝ", {}, {})
    'ℝ'
    """
    if local_env is None:
        local_env = {}

    if op == "num":
            return "ℝ"

    elif op == "var":
            var_name = expr[1]
            if var_name in local_env:
                return local_env[var_name]
            if var_name in type_env:
                return type_env[var_name]
            return None

    elif op == "array":
        elements = expr[1]
        if not elements:
            return ("tensor", [(0, "invariant")])

        def infer_array_shape(arr_node):
            """Recursively infer the shape of a nested array."""
            if not isinstance(arr_node, tuple) or arr_node[0] != "array":
                return []  # Scalar element

            elems = arr_node[1]
            if not elems:
                return [0]

            outer_dim = len(elems)

                # Check if elements are nested arrays
            first_elem = elems[0]
            if isinstance(first_elem, tuple) and first_elem[0] == "array":
                    # Get inner shape from first element
                inner_shape = infer_array_shape(first_elem)

                    # Verify all elements have the same shape
                for i, elem in enumerate(elems):
                    if not isinstance(elem, tuple) or elem[0] != "array":
                        add_error(f"Inconsistent array shape at index {i}: {elem} vs {inner_shape}")
                        return None

                return [outer_dim] + inner_shape
            else:
                    # Leaf level - all elements should be scalars
                for i, elem in enumerate(elems):
                    if isinstance(elem, tuple) and elem[0] == "array":
                        add_error(f"Inconsistent nesting at index {i}: expected scalar, got array")
                        return None
                return [outer_dim]

        shape = infer_array_shape(expr)
        if shape is None:
            return None
        return ("tensor", [(d, "invariant") for d in shape])

    elif op == "index":
            var_name = expr[1]
            var_type = local_env.get(var_name) or type_env.get(var_name)
            if var_type is None:
                return None
            shape = get_shape(var_type)
            if shape is None:
                add_error(f"Cannot index scalar '{var_name}'")
                return None
            if len(shape) == 1:
                return "ℝ"  # Indexing 1D array gives scalar
            else:
                # Indexing multi-dim array gives sub-array
                return make_tensor_type(shape[1:])

    elif op == "slice":
            var_name = expr[1]
            start_expr = expr[2]
            end_expr = expr[3]
            var_type = local_env.get(var_name) or type_env.get(var_name)
            if var_type is None:
                return None
            shape = get_shape(var_type)
            if shape is None:
                add_error(f"Cannot slice scalar '{var_name}'")
                return None

            # Try to compute slice length
            start_val = None
            end_val = None
            if isinstance(start_expr, tuple) and start_expr[0] == "num":
                start_val = int(start_expr[1])
            if isinstance(end_expr, tuple) and end_expr[0] == "num":
                end_val = int(end_expr[1])

            if start_val is not None and end_val is not None:
                # Physika uses inclusive end, so length is end - start + 1
                slice_len = end_val - start_val + 1
                if len(shape) == 1:
                    return ("tensor", [(slice_len, "invariant")])
                else:
                    new_shape = [slice_len] + shape[1:]
                    return make_tensor_type(new_shape)
            return None  # Cannot determine slice length

    elif op in ("add", "sub"):
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            left_shape = get_shape(left_type)
            right_shape = get_shape(right_type)

            # Allow scalar broadcasting for add/sub (e.g. tensor + 1.0)
            result_shape, ok = shapes_broadcast_compatible(left_shape, right_shape, allow_scalar_broadcast=True)
            if not ok:
                add_error(f"Shape mismatch in {op}: {type_to_str(left_type)} vs {type_to_str(right_type)}")
                return None
            return make_tensor_type(result_shape)

    elif op == "mul":
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            left_shape = get_shape(left_type)
            right_shape = get_shape(right_type)

            # Allow scalar multiplication with tensors
            result_shape, ok = shapes_broadcast_compatible(left_shape, right_shape, allow_scalar_broadcast=True)
            if not ok:
                add_error(f"Shape mismatch in multiplication: {type_to_str(left_type)} vs {type_to_str(right_type)}")
                return None
            return make_tensor_type(result_shape)

    elif op == "div":
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            # Division by scalar is always ok
            right_shape = get_shape(right_type)
            if right_shape is not None:
                left_shape = get_shape(left_type)
                if left_shape != right_shape:
                    add_error(f"Element-wise division requires matching shapes: {type_to_str(left_type)} vs {type_to_str(right_type)}")
            return left_type

    elif op == "matmul":
            left_type = infer_type(expr[1], local_env)
            right_type = infer_type(expr[2], local_env)
            left_shape = get_shape(left_type)
            right_shape = get_shape(right_type)

            if left_shape is None or right_shape is None:
                # Scalar matmul - treat as regular mul
                return left_type if right_shape is None else right_type

            # Matrix multiplication dimension check
            if len(left_shape) == 1 and len(right_shape) == 1:
                # Vector dot product: (n,) @ (n,) -> scalar
                if not dims_compatible(left_shape[0], right_shape[0]):
                    add_error(f"Vector dot product dimension mismatch: {left_shape[0]} vs {right_shape[0]}")
                return "ℝ"
            elif len(left_shape) == 2 and len(right_shape) == 1:
                # Matrix-vector: (m,n) @ (n,) -> (m,)
                if not dims_compatible(left_shape[1], right_shape[0]):
                    add_error(f"Matrix-vector multiplication dimension mismatch: {left_shape[1]} vs {right_shape[0]}")
                return make_tensor_type([left_shape[0]])
            elif len(left_shape) == 1 and len(right_shape) == 2:
                # Vector-matrix: (m,) @ (m,n) -> (n,)
                if not dims_compatible(left_shape[0], right_shape[0]):
                    add_error(f"Vector-matrix multiplication dimension mismatch: {left_shape[0]} vs {right_shape[0]}")
                return make_tensor_type([right_shape[1]])
            elif len(left_shape) == 2 and len(right_shape) == 2:
                # Matrix-matrix: (m,k) @ (k,n) -> (m,n)
                if not dims_compatible(left_shape[1], right_shape[0]):
                    add_error(f"Matrix multiplication dimension mismatch: {left_shape[1]} vs {right_shape[0]}")
                return make_tensor_type([left_shape[0], right_shape[1]])

            return None

    elif op == "pow":
            left_type = infer_type(expr[1], local_env)
            # Power typically returns same shape as base
            return left_type

    elif op == "neg":
            return infer_type(expr[1], local_env)

    elif op == "call":
            func_name = expr[1]
            args = expr[2]

            # Built-in functions
            if func_name in ("exp", "log", "sin", "cos", "sqrt", "abs", "tanh"):
                if args:
                    return infer_type(args[0], local_env)
                return "ℝ"
            elif func_name == "sum":
                return "ℝ"  # Sum reduces to scalar
            elif func_name in ("real", "imag"):
                return "ℝ"
            elif func_name == "grad":
                # grad returns gradient with same shape as input
                if len(args) >= 2:
                    return infer_type(args[1], local_env)
                return None
            elif func_name == "solve":
                return None  # Solve returns tuple, type depends on equations
            elif func_name == "train":
                # train returns the same instance type as its first argument
                if args:
                    return infer_type(args[0], local_env)
                return None
            elif func_name == "evaluate":
                return "ℝ"  # evaluate returns a scalar loss

            # User-defined function
            if func_name in func_env:
                _, return_type = func_env[func_name]
                return return_type

            # Class constructor
            if func_name in class_env:
                class_def = class_env[func_name]
                class_params = class_def["class_params"]
                # Check argument count
                if len(args) != len(class_params):
                    add_error(
                        f"Class '{func_name}' expects {len(class_params)} arguments, got {len(args)}"
                    )
                else:
                    # Check each argument type against the declared parameter type
                    for i, ((param_name, param_type), arg_expr) in enumerate(zip(class_params, args)):
                        arg_type = infer_type(arg_expr, local_env)
                        if arg_type is not None and param_type is not None:
                            if not types_compatible(param_type, arg_type):
                                add_error(
                                    f"Type mismatch for parameter '{param_name}' of class '{func_name}': "
                                    f"expected {type_to_str(param_type)}, got {type_to_str(arg_type)}"
                                )
                return ("instance", func_name)

            # Instance call (e.g., net2([1.0, 2.0]) where net2 is an instance)
            var_type = type_env.get(func_name) or (local_env.get(func_name) if local_env else None)
            if isinstance(var_type, tuple) and var_type[0] == "instance":
                class_name = var_type[1]
                if class_name in class_env:
                    class_def = class_env[class_name]
                    lambda_params = class_def["lambda_params"]
                    return_type = class_def.get("return_type")
                    # Check argument count
                    if len(args) != len(lambda_params):
                        add_error(
                            f"Instance of '{class_name}' expects {len(lambda_params)} argument(s), got {len(args)}"
                        )
                    else:
                        # Check each argument type against the lambda parameter type
                        for (param_name, param_type), arg_expr in zip(lambda_params, args):
                            arg_type = infer_type(arg_expr, local_env)
                            if arg_type is not None and param_type is not None:
                                if not types_compatible(param_type, arg_type):
                                    add_error(
                                        f"Type mismatch for parameter '{param_name}' of '{class_name}': "
                                        f"expected {type_to_str(param_type)}, got {type_to_str(arg_type)}"
                                    )
                    return return_type

            return None

    elif op == "call_index":
            func_name = expr[1]
            args = expr[2]
            index = expr[3]

            if func_name == "grad":
                # grad returns gradient vector, indexing gives scalar
                return "ℝ"

            return None

    elif op == "string":
            return "string"

    elif op == "imaginary":
            return "ℂ"  # Complex type

    return None


def check_branch(
    branch_stmts: list[ASTExpr],
    infer_type: InferFn,
) -> None:
    """Type-check the expressions inside a if/else branch.

    Iterates over ``for_assign``, ``for_pluseq``, and ``for_call`` AST nodes,
    calling *infer_type* on every expression so that shape mismatches and
    unknown-variable errors are catched.

    Variables assigned inside the branch are not added to the
    type environment.

    Parameters
    ----------
    branch_stmts:
        List of ``for_assign``, ``for_pluseq``, or ``for_call`` AST tuples
        from an ``if_else`` or ``if_only`` program statement.
    infer_type:
        Type inference callback.

    Examples
    --------
    >>> from utils.type_checker_utils import check_branch
    >>> inferred = []
    >>> check_branch(
    ...     [("for_assign", "z", ("num", 1.0))],
    ...     lambda expr, le=None: inferred.append(expr) or "ℝ",
    ... )
    >>> inferred
    [('num', 1.0)]
    """
    for s in branch_stmts:
        if s is None:
            continue
        branch_op = s[0]
        if branch_op == "for_assign":
            infer_type(s[2])
        elif branch_op == "for_pluseq":
            infer_type(s[2])
        elif branch_op == "for_call":
            for arg in s[2]:
                infer_type(arg)


def statement_check(
    op: str,
    stmt: ASTExpr,
    infer_type: InferFn,
    add_error: ErrorFn,
    type_env: dict[str, TypeSpec],
    check_statement: Callable[[ASTExpr], None],
) -> None:
    """Check a single program-level statement for type errors.

    Dispatches on the statement tag (*op*) and validates declared
    types against inferred types.  Called by
    ``TypeChecker.check_statement``.

    Parameters
    ----------
    op : str
        The statement tag: ``"decl"``, ``"assign"``, ``"expr"``,
        or ``"for_loop"``.
    stmt : ASTExpr
        The full statement AST tuple.
    infer_type : InferFn
        Type inference callback (``TypeChecker.infer_type``).
    add_error : ErrorFn
        Callback to record a type error message.
    type_env : dict[str, TypeSpec]
        Program-level variable type environment (mutated for
        ``decl`` and ``assign``).
    check_statement : Callable[[ASTExpr], None]
        Recursive callback for checking nested statements
        (e.g. ``for_loop`` bodies).

    Examples
    --------
    >>> from utils.type_checker_utils import statement_check
    >>> env = {}
    >>> statement_check("decl", ("decl", "x", "ℝ", ("num", 3.0), 1),
    ...                 lambda e, le=None: "ℝ", lambda m: None, env,
    ...                 lambda s: None)
    >>> env["x"]
    'ℝ'
    """
    if op == "decl":
        name = stmt[1]
        type_spec = stmt[2]
        expr = stmt[3]
        inferred = infer_type(expr)
        if type_spec and inferred and not types_compatible(type_spec, inferred):
            add_error(
                f"Type mismatch for '{name}': declared as {type_to_str(type_spec)}, "
                f"got {type_to_str(inferred)}"
            )
        type_env[name] = type_spec if type_spec else inferred

    elif op == "assign":
        name = stmt[1]
        expr = stmt[2]
        inferred = infer_type(expr)
        type_env[name] = inferred

    elif op == "expr":
        expr = stmt[1]
        infer_type(expr)

    elif op == "for_loop":
        loop_var = stmt[1]
        body_stmts = stmt[2]
        type_env[loop_var] = "ℕ"
        for body_stmt in body_stmts:
            if body_stmt is None:
                continue
            check_statement(body_stmt)

    elif op in ("if_else", "if_only"):
        cond = stmt[1]
        then_stmts = stmt[2]
        else_stmts = stmt[3] if op == "if_else" else []

        # Type-check both sides of the condition
        if isinstance(cond, tuple) and len(cond) == 3:
            infer_type(cond[1])
            infer_type(cond[2])

        check_branch(then_stmts, infer_type)
        check_branch(else_stmts, infer_type)
