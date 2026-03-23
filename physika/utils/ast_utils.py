from __future__ import annotations

import re
from typing import Any, Callable, Literal, Union, cast
from physika.utils.print_utils import print_unified_ast

# AST TYPE DEFINITIONS
# The parser produces a tree of tagged tuples.  Every non-leaf node is a
# tuple whose first element is a string (tag) and whose remaining elements are
# other nodes, lists of nodes, or scalar leaves.
#
# The type aliases below make the tag vocabulary explicit so that mypy
# can flag typos, missing branches, and wrong argument types.

# Tag literals (every valid first element of an AST tuple)

ExprTag = Literal[
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "matmul",  # binary arithmetic
    "neg",  # unary arithmetic
    "num",
    "var",  # literals / references
    "string",
    "equation_string",  # string literals
    "array",  # [elem, ...]
    "index",
    "slice",  # arr[i], arr[a:b]
    "call",
    "call_index",  # f(x), f(x)[i]
    "imaginary",  # complex unit  i
]

StmtTag = Literal[
    "decl",  # x : R = expr          -> (tag, name, type, expr, lineno)
    "assign",  # x = expr              -> (tag, name, expr, lineno)
    "expr",  # expr                  -> (tag, expr, lineno)
    "func_def",  # def f(...)            -> (tag, name)
    "class_def",  # class C(...)          -> (tag, name)
    "for_loop",  # for i: ...          -> (tag, var, [body], [arrays], lineno)
    "for_loop_range",  # for i: ℕ(n) / ℕ(s,e): -> (tag, var, start, end,
    #                                                 [body], lineno)
]

BodyStmtTag = Literal[
    "body_assign",  # x = expr          inside function body
    "body_decl",  # x : T = expr      inside function body
    "body_tuple_unpack",  # a, b = expr       inside function body
    "body_for",  # for k: ...        for-loop inside function body
    "body_zeros_decl",  # C : ℝ[n,o]        type annotation for an accumulation
    #                                           target
    "body_for_accum",  # for i j k: ...    accumulation loop. emits
    #                                          torch.stack
    #                                          per target
    "loop_assign",  # x = expr          inside for-loop body
    "loop_pluseq",  # x += expr         inside for-loop body
    "loop_index_pluseq",  # C[i,...] += expr  nD accumulation inside for-loop
    #                                             body
    "for_assign",  # x = expr          program-level for body
    "for_pluseq",  # x += expr         program-level for body
    "for_call",  # f(x)              program-level for body
]

TypeTag = Literal[
    "func_type",  # R -> R                -> (tag, input_type, output_type)
    "tangent",  # T_x M                -> (tag, point_id, manifold_type)
    "tensor",  # R[3,3]               -> (tag, [(dim, variance), ...])
]

ASTTag = Union[ExprTag, StmtTag, BodyStmtTag, TypeTag]

# Composite node type
ASTNode = Union[
    tuple[Any, ...],  # tagged nodes: ("add", left, right), ("num", 1.0), ...
    list["ASTNode"],  # child sequences: function args, loop bodies, ...
    str,  # identifiers, string literal values
    int,  # integer values (line numbers, dimension sizes)
    float,  # numeric literal values
    None,  # empty / no-op placeholder
]


def ast_uses_solve(node: ASTNode) -> bool:
    """Check whether an AST subtree contains a call to ``solve``.

    Recursively walks *node* looking for ``("call", "solve", ...)``.

    Parameters
    ----------
    node : ASTNode
        A tagged tuple, list, or scalar leaf of an AST.

    Returns
    -------
    bool
        ``True`` if a ``("call", "solve", ...)`` node exists anywhere
        in the subtree, ``False`` otherwise.

    Examples
    --------
    >>> from physika.utils.ast_utils import ast_uses_solve
    >>> ast_uses_solve(("call", "solve", [("var", "eq1"), ("var", "eq2")]))
    True
    >>> ast_uses_solve(("add", ("num", 1.0), ("var", "x")))
    False
    """
    if not isinstance(node, (tuple, list)):
        return False
    if isinstance(node, tuple) and len(node) >= 2:
        if node[0] == "call" and node[1] == "solve":
            return True
        return any(
            ast_uses_solve(child) for child in node[1:]
            if isinstance(child, (tuple, list)))
    if isinstance(node, list):
        return any(ast_uses_solve(item) for item in node)
    return False


def ast_uses_func(node: ASTNode, func_name: str) -> bool:
    """Check whether an AST subtree contains a call to *func_name*.

    Recursively walks *node* looking for both ``("call", func_name, ...)``
    and ``("call_index", func_name, ..., idx)`` nodes.  Used during calling of
    ``from_ast_to_torch`` to decide which runtime helpers need to be imported.

    Parameters
    ----------
    node : ASTNode
        A tagged tuple, list, or scalar leaf of an AST.
    func_name : str
        The function identifier to search for (e.g. ``"train"``,
        ``"grad"``, ``"simulate"``).

    Returns
    -------
    bool
        ``True`` if a matching call node exists anywhere in the
        subtree, ``False`` otherwise.

    Examples
    --------
    >>> from physika.utils.ast_utils import ast_uses_func
    >>> ast_uses_func(("call", "train", [("var", "model")]), "train")
    True
    >>> ast_uses_func(("call_index", "grad", [("var", "H")], ("num", 0.0)), "grad")  # noqa: E501
    True
    >>> ast_uses_func(("add", ("num", 1.0), ("num", 2.0)), "train")
    False
    """
    if not isinstance(node, (tuple, list)):
        return False
    if isinstance(node, tuple) and len(node) >= 2:
        if node[0] == "call" and node[1] == func_name:
            return True
        if node[0] == "call_index" and node[1] == func_name:
            return True
        return any(
            ast_uses_func(child, func_name) for child in node[1:]
            if isinstance(child, (tuple, list)))
    if isinstance(node, list):
        return any(ast_uses_func(item, func_name) for item in node)
    return False


def collect_grad_targets(node: ASTNode, targets: set[str]) -> None:
    """
    Collect variable names used as differentiation targets in ``grad()`` calls.

    Recursively walks *node* looking for ``("call", "grad", [output, input])``
    patterns and extracts the second argument (the differentiation variable)
    when it is a ``("var", name)`` node.  The collected names are added to
    *targets* so that ``generate_statement`` can initialise those variables
    with ``requires_grad=True``.

    Parameters
    ----------
    node : ASTNode
        A tagged tuple, list, or scalar leaf of an AST.
    targets : set[str]
        Mutable set to add target variable names into.  Modified in
        place; not returned.

    Examples
    --------
    >>> from physika.utils.ast_utils import collect_grad_targets
    >>> targets = set()
    >>> stmt = ("expr", ("call", "grad", [("var", "H"), ("var", "t")]))
    >>> collect_grad_targets(stmt, targets)
    >>> targets
    {'t'}
    """
    if not isinstance(node, (tuple, list)):
        return
    if isinstance(node, tuple) and len(node) >= 2:
        if node[0] == "call" and node[1] == "grad" and len(node) >= 3:
            args = node[2]
            if len(args) >= 2 and isinstance(args[1],
                                             tuple) and args[1][0] == "var":
                targets.add(args[1][1])
        for child in node[1:]:
            if isinstance(child, (tuple, list)):
                collect_grad_targets(child, targets)
    elif isinstance(node, list):
        for item in node:
            collect_grad_targets(item, targets)


def replace_class_params(code: str, class_params: list[tuple[str,
                                                             ASTNode]]) -> str:
    """
    Replace class parameter references with ``self.param`` in generated code.

    Rewrites bare parameter names inside the generated ``forward`` and
    ``loss`` method bodies.  Applies regex substitutions for three
    contexts: function calls (``f(`` -> ``self.f(``), array indexing
    (``W[`` -> ``self.W[``), and standalone references inside
    parenthesised expressions.

    Parameters
    ----------
    code : str
        The generated Python source string to transform.
    class_params : list[tuple[str, ASTNode]]
        List of ``(name, type_spec)`` pairs from the class definition.
        Only the names are used; type specs are ignored.

    Returns
    -------
    str
        A new string with class parameter names prefixed by ``self.``
        in the appropriate syntactic contexts.

    Examples
    --------
    >>> from physika.utils.ast_utils import replace_class_params
    >>> replace_class_params("(W @ x + b)", [("W", "ℝ"), ("b", "ℝ")])
    '(self.W @ x + self.b)'
    """
    for cp_name, _ in class_params:
        # Replace function calls: f(...) -> self.f(...)
        code = re.sub(rf'\b{cp_name}\(', f'self.{cp_name}(', code)
        # Replace array indexing: W[...] -> self.W[...]
        code = re.sub(rf'\b{cp_name}\[', f'self.{cp_name}[', code)
        # Replace standalone references in expressions
        code = re.sub(rf'\(({cp_name})\s', r'(self.\1 ', code)
        code = re.sub(rf'\s({cp_name})\)', r' self.\1)', code)
        code = re.sub(rf'\(({cp_name})\)', r'(self.\1)', code)
        # Catch remaining word references not already prefixed
        code = re.sub(rf'(?<!self\.)\b{cp_name}\b', f'self.{cp_name}', code)
    return code


def _is_loop_var(expr: ASTNode, var: str) -> bool:
    """Return True if `expr` represents the loop variable.

    Handles both the ``("var", name)`` form and the special
    ``("imaginary",)`` form, which is used when the loop variable is
    named ``"i"`` (since the lexer emits ``IMAGINARY`` for the token ``i``).

    Parameters
    ----------
    expr : ASTNode
        An AST expression node to test.
    var : str
        The loop variable name to match against.

    Returns
    -------
    bool
        ``True`` if *expr* refers to the loop variable *var*.

    Examples
    --------
    from physika.utils.ast_utils import _is_loop_var
    >>> _is_loop_var(("var", "k"), "k")
    True
    >>> _is_loop_var(("imaginary",), "i")
    True
    >>> _is_loop_var(("var", "j"), "k")
    False
    """
    return ((isinstance(expr, tuple) and expr[0] == "var" and expr[1] == var)
            or (var == "i" and isinstance(expr, tuple)
                and expr[0] == "imaginary"))


def _decompose_chain(expr: ASTNode) -> tuple[str | None, list[ASTNode]]:
    """Decompose a chain-index or 1-D index node into (array_name, [idx_expr, ...]).

    Recursively walks left-associative ``("chain_index", base, idx)``
    nodes back to the underlying array name and collects all index
    expressions in order. The base case is 1-D indexing ``("index", arr, idx)``.

    Parameters
    ----------
    expr : ASTNode
        A ``("chain_index", ...)`` or ``("index", ...)`` node, or any
        other node (returns ``(None, [])`` for unrecognised shapes).

    Returns
    -------
    array_name : str or None
        The name of the array being indexed, or ``None`` if the
        expression is not a recognised index form.
    idx_exprs : list of ASTNode
        Index expressions in outermost-to-innermost order, matching the
        dimension order of the underlying array.

    Examples
    --------
    >>> from physika.utils.ast_utils import _decompose_chain
    >>> _decompose_chain(("index", "A", ("var", "i")))
    ('A', [('var', 'i')])
    >>> _decompose_chain(("chain_index", ("index", "A", ("var", "i")), ("var", "k")))  # noqa: E501
    ('A', [('var', 'i'), ('var', 'k')])
    """
    if not isinstance(expr, tuple):
        return None, []
    if expr[0] == "index":
        _, arr, idx = expr
        if isinstance(arr, str):
            return arr, [idx]
        return None, []
    if expr[0] == "chain_index":
        base_name, base_idxs = _decompose_chain(expr[1])
        return base_name, base_idxs + [expr[2]]
    return None, []


def _infer_range(var: str, expr: ASTNode, skip: str) -> str | None:
    """Walk an AST expression and return a ``shape`` string for *var*.

    Searches the expression tree for array-index nodes where *var*
    appears as a subscript, then returns the corresponding
    shape as string.  The array named *skip* is excluded from
    the search as it is the accumulation target being defined.

    Handles ``("indexN", arr, [idx, ...])``,
    ``("index", arr, idx)`` (1-D indexing), and
    ``("chain_index", ...)`` (chained bracket indexing A[i][k]).

    Parameters
    ----------
    var : str
        The loop variable whose range we want to determine.
    expr : ASTNode
        The RHS AST expression to search.
    skip : str
        Name of the tensor being accumulated into.

    Returns
    -------
    str or None
        A Python expression such as ``"A.shape[0]"`` giving the loop
        range, or ``None`` if no suitable index access was found.

    Examples
    --------
    >>> from physika.utils.ast_utils import _infer_range
    >>> rhs = ("indexN", "A", [("var", "i"), ("var", "k")])
    >>> _infer_range("i", rhs, "C")
    'A.shape[0]'
    >>> _infer_range("k", rhs, "C")
    'A.shape[1]'
    """
    if not isinstance(expr, tuple):
        return None
    op = expr[0]
    if op == "indexN":
        arr = expr[1]
        if arr != skip:
            for dim, ie in enumerate(expr[2]):
                if _is_loop_var(ie, var):
                    return f"{arr}.shape[{dim}]"
    elif op == "index":
        _, arr, ie = expr
        if isinstance(arr, str) and arr != skip:
            if _is_loop_var(ie, var):
                return f"{arr}.shape[0]"
    elif op == "chain_index":
        base_name, idx_exprs = _decompose_chain(expr)
        if base_name and base_name != skip:
            for dim, ie in enumerate(idx_exprs):
                if _is_loop_var(ie, var):
                    return f"{base_name}.shape[{dim}]"
    for child in expr[1:]:
        if isinstance(child, tuple):
            r = _infer_range(var, child, skip)
            if r is not None:
                return r
    return None


def _lhs_var_name(expr: ASTNode) -> str | None:
    """Extract the loop-variable name from an LHS index expression.

    Used to classify which loop variables appear as output dimensions
    (LHS indices of ``T[i, j]``).

    Parameters
    ----------
    expr : ASTNode
        An index expression from the LHS of a ``loop_index_pluseq``
        node.

    Returns
    -------
    str or None
        The variable name, or
        ``None`` if the expression is not a plain variable reference.

    Examples
    --------
    >>> from physika.utils.ast_utils import _lhs_var_name
    >>> _lhs_var_name(("var", "j"))
    'j'
    >>> _lhs_var_name(("imaginary",))
    'i'
    >>> _lhs_var_name(("num", 0.0))
    """
    if isinstance(expr, tuple) and expr[0] == "var":
        return expr[1]
    if isinstance(expr, tuple) and expr[0] == "imaginary":
        return "i"
    return None


def ast_to_torch_expr(node: ASTNode,
                      indent: int = 0,
                      current_loop_var: str | set[str] | None = None) -> str:
    """Convert an AST expression node to a PyTorch source code string.

    Recursively translates a Physika AST subtree into a valid
    Python/PyTorch expression string.  Handles arithmetic operators,
    array construction, indexing, slicing, function calls (mapping
    builtins like ``sin`` to ``torch.sin``), and complex numbers.

    This is the core of the string-codegen path used by
    ``generate_function``, ``generate_class``, and
    ``generate_statement``.

    Parameters
    ----------
    node : ASTNode
        AST expression node (tagged tuple) or a scalar leaf.
    indent : int, default 0
        Current indentation level.  Reserved for future use.
    current_loop_var : str or None, default None
        When set, an ``("imaginary",)`` node whose loop variable is
        ``"i"`` will emit the loop variable name instead of
        ``torch.tensor(1j)``, disambiguating the complex unit from
        the loop index.

    Returns
    -------
    str
        A torch Python expression string corresponding to the given ASTNode.

    Examples
    --------
    >>> from physika.utils.ast_utils import ast_to_torch_expr
    >>> ast_to_torch_expr(("add", ("num", 1.0), ("var", "x")))
    '(1.0 + x)'
    >>> ast_to_torch_expr(("call", "sin", [("var", "theta")]))
    'torch.sin(theta)'
    >>> ast_to_torch_expr(("array", [("num", 1.0), ("num", 2.0)]))
    'torch.tensor([1.0, 2.0])'
    """
    if not isinstance(node, tuple):
        return repr(node)

    op = node[0]

    if op == "num":
        val = node[1]
        if isinstance(val, float) and val == int(val):
            return f"{val}"
        return repr(val)

    elif op == "var":
        return node[1]

    elif op == "add":
        left = ast_to_torch_expr(node[1], indent, current_loop_var)
        right = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"({left} + {right})"

    elif op == "sub":
        left = ast_to_torch_expr(node[1], indent, current_loop_var)
        right = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"({left} - {right})"

    elif op == "mul":
        left = ast_to_torch_expr(node[1], indent, current_loop_var)
        right = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"({left} * {right})"

    elif op == "div":
        left = ast_to_torch_expr(node[1], indent, current_loop_var)
        right = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"({left} / {right})"

    elif op == "matmul":
        left = ast_to_torch_expr(node[1], indent, current_loop_var)
        right = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"({left} @ {right})"

    elif op == "pow":
        left = ast_to_torch_expr(node[1], indent, current_loop_var)
        right = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"({left} ** {right})"

    elif op == "neg":
        val = ast_to_torch_expr(node[1], indent, current_loop_var)
        return f"(-{val})"

    elif op == "array":
        elements = node[1]
        # Check if this is a nested array (contains other arrays)
        has_nested = any(
            isinstance(e, tuple) and e[0] == "array" for e in elements)
        if has_nested:
            # For nested arrays, generate list-of-lists and wrap in
            # torch.tensor
            def array_to_list(arr_node):
                if isinstance(arr_node, tuple) and arr_node[0] == "array":
                    inner = [array_to_list(e) for e in arr_node[1]]
                    return f"[{', '.join(inner)}]"
                else:
                    return ast_to_torch_expr(arr_node, indent,
                                             current_loop_var)

            inner_lists = [array_to_list(e) for e in elements]
            return f"torch.tensor([{', '.join(inner_lists)}])"
        else:
            all_numeric = all(
                isinstance(e, tuple) and
                (e[0] == "num" or (e[0] == "neg" and isinstance(e[1], tuple)
                                   and e[1][0] == "num")) for e in elements)
            elem_strs = [
                ast_to_torch_expr(e, indent, current_loop_var)
                for e in elements
            ]
            if all_numeric:
                return f"torch.tensor([{', '.join(elem_strs)}])"
            else:
                # Elements may be tensors (e.g., x[1], sin(x[0]))
                # use torch.stack
                wrapped = [f"torch.as_tensor({s}).float()" for s in elem_strs]
                return f"torch.stack([{', '.join(wrapped)}])"

    elif op == "index":
        var_name = node[1]
        idx = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"{var_name}[int({idx})]"

    elif op == "slice":
        var_name = node[1]
        start = ast_to_torch_expr(node[2], indent, current_loop_var)
        end = ast_to_torch_expr(node[3], indent, current_loop_var)
        # Convert to int if needed
        start_int = f"int({start})" if "." in start else start
        end_int = f"int({end})+1" if "." in end else f"{end}+1"
        return f"{var_name}[{start_int}:{end_int}]"

    elif op == "chain_index":
        obj = ast_to_torch_expr(node[1], indent, current_loop_var)
        idx = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"{obj}[int({idx})]"

    elif op == "indexN":
        arr = node[1]
        idx_codes = [
            f"int({ast_to_torch_expr(e, indent, current_loop_var)})"
            for e in node[2]
        ]
        return f"{arr}[{', '.join(idx_codes)}]"

    elif op == "call":
        func_name = node[1]
        args = node[2]
        arg_strs = [
            ast_to_torch_expr(arg, indent, current_loop_var) for arg in args
        ]

        # Map built-in functions to PyTorch equivalents
        torch_funcs = {
            "exp": "torch.exp",
            "log": "torch.log",
            "sin": "torch.sin",
            "cos": "torch.cos",
            "sqrt": "torch.sqrt",
            "abs": "torch.abs",
            "sum": "torch.sum",
            "mean": "torch.mean",
            "real": "torch.real",
        }

        if func_name in torch_funcs:
            return f"{torch_funcs[func_name]}({', '.join(arg_strs)})"
        elif func_name == "grad":
            # grad(output, input) -> compute_grad(output, input)
            return f"compute_grad({', '.join(arg_strs)})"
        else:
            return f"{func_name}({', '.join(arg_strs)})"

    elif op == "call_index":
        # Indexed function call: func(args)[index]
        func_name = node[1]
        args = node[2]
        index_ast = node[3]
        arg_strs = [
            ast_to_torch_expr(arg, indent, current_loop_var) for arg in args
        ]
        idx = ast_to_torch_expr(index_ast, indent, current_loop_var)

        if func_name == "grad":
            # grad(output, input)[i] -> compute_grad(output, input)[i]
            return f"compute_grad({', '.join(arg_strs)})[int({idx})]"
        else:
            return f"{func_name}({', '.join(arg_strs)})[int({idx})]"

    elif op == "imaginary":
        # If we're inside a for-expr whose loop var is 'i', emit 'i'.
        # current_loop_var may be a string (single var) or set (nested vars).
        active = (current_loop_var if isinstance(current_loop_var, set) else
                  (set({current_loop_var}) if current_loop_var else set()))
        if "i" in active:
            return "i"
        # Use torch.tensor(1j) so it can be used with torch.exp
        return "torch.tensor(1j)"

    elif op == "for_expr":
        # active_vars accumulates all enclosing loop var names
        # to handle nested loops
        loop_var = node[1]
        size_expr = node[2]
        body_expr = node[3]
        outer_active = (
            current_loop_var if isinstance(current_loop_var, set) else
            (set({current_loop_var}) if current_loop_var else set()))
        active_vars = outer_active | set({loop_var})
        n_code = ast_to_torch_expr(size_expr, indent, outer_active or None)
        body_code = ast_to_torch_expr(body_expr, indent, active_vars)
        tmp = f"_fi_{loop_var}"
        return (f"torch.stack(["
                f"{body_code} "
                f"for {tmp} in range(int({n_code})) "
                f"for {loop_var} in [torch.tensor(float({tmp}))]])")

    elif op == "for_expr_range":
        # for i : ℕ(start, end) → body  — range(start, end), end-exclusive
        loop_var = node[1]
        start_expr = node[2]
        end_expr = node[3]
        body_expr = node[4]
        outer_active = (
            current_loop_var if isinstance(current_loop_var, set) else
            (set({current_loop_var}) if current_loop_var else set()))
        active_vars = outer_active | set({loop_var})
        start_code = ast_to_torch_expr(start_expr, indent, outer_active
                                       or None)
        end_code = ast_to_torch_expr(end_expr, indent, outer_active or None)
        body_code = ast_to_torch_expr(body_expr, indent, active_vars)
        tmp = f"_fi_{loop_var}"
        return (f"torch.stack(["
                f"{body_code} "
                f"for {tmp} in range(int({start_code}), int({end_code})) "
                f"for {loop_var} in [torch.tensor(float({tmp}))]])")

    elif op == "equation_string":
        return repr(node[1])

    elif op == "string":
        # Equation string literal
        return repr(node[1])

    return f"/* unknown: {node} */"


def condition_to_expr(cond: ASTNode,
                      current_loop_var: str | set[str] | None = None) -> str:
    """Convert a condition AST node to a Python boolean expression string.

    Parameters
    ----------
    cond : tuple[str, ...]
        A condition tuple like ``("cond_eq", left, right)``.
    current_loop_var : str or set, optional
        Active loop variable(s) for disambiguating the imaginary token ``i``.

    Returns
    -------
    str
        A Python boolean expression (e.g. ``"n == 0.0"``).

    Examples
    --------
    >>> from physika.utils.ast_utils import condition_to_expr
    >>> condition_to_expr(("cond_eq", ("var", "n"), ("num", 0.0)))
    'n == 0.0'
    >>> condition_to_expr(("cond_lt", ("var", "x"), ("num", 1.0)))
    'x < 1.0'
    """
    op_map = {
        "cond_eq": "==",
        "cond_neq": "!=",
        "cond_lt": "<",
        "cond_gt": ">",
        "cond_leq": "<=",
        "cond_geq": ">=",
    }
    cond_t = cast(tuple[Any, ...], cond)
    op = cond_t[0]
    left = ast_to_torch_expr(cond_t[1], current_loop_var=current_loop_var)
    right = ast_to_torch_expr(cond_t[2], current_loop_var=current_loop_var)
    return f"{left} {op_map[op]} {right}"


def emit_func_loop_body(
    loop_body: list,
    indent_level: int,
    lines: list[str],
    loop_var,
) -> None:
    """Emit code lines for a list of ``func_loop_stmt`` AST nodes.

    Recurse for nested ``loop_for_range``, ``loop_if``, and ``loop_if_else``
    nodes, extending ``loop_var`` with each new inner variable.

    ``ast_to_torch_expr`` resolves the imaginary-unit token ``i`` to the
    correct Python name instead of ``torch.tensor(1j)``.

    Parameters
    ----------
    loop_body : list[ASTNode]
        ``func_loop_stmt`` nodes. ``None`` entries are skipped.
        Supported tags:
        - ``loop_assign``
        - ``loop_pluseq``
        - ``loop_index_pluseq``
        - ``loop_for_range``
        - ``loop_if``
        - ``loop_if_else``
    indent_level : int
        Current indentation depth. Each level adds 4 spaces.
    lines : list[str]
        Output list. Source lines are appended.
    loop_var : str or set[str]
        Active loop variable name(s).  Grows as inner loops are entered.
    """
    prefix = "    " * indent_level
    active = loop_var if isinstance(
        loop_var, set) else ({loop_var} if loop_var else set())
    for loop_stmt in loop_body:
        if loop_stmt is None:
            continue
        tag = loop_stmt[0]
        if tag == "loop_assign":
            _, var_name, expr = loop_stmt
            lines.append(
                f"{prefix}{var_name} = {ast_to_torch_expr(expr, current_loop_var=active)}"  # noqa: E501
            )
        elif tag == "loop_pluseq":
            _, var_name, expr = loop_stmt
            lines.append(
                f"{prefix}{var_name} = {var_name} + {ast_to_torch_expr(expr, current_loop_var=active)}"  # noqa: E501
            )
        elif tag == "loop_index_pluseq":
            _, arr_name, idx_list, rhs = loop_stmt
            idx_codes = [
                ast_to_torch_expr(e, current_loop_var=active) for e in idx_list
            ]
            rhs_code = ast_to_torch_expr(rhs, current_loop_var=active)
            lines.append(
                f"{prefix}{arr_name}[{', '.join(f'int({c})' for c in idx_codes)}] += {rhs_code}"  # noqa: E501
            )
        elif tag == "loop_for_range":
            _, inner_var, start_expr, end_expr, inner_body = loop_stmt
            start_code = ast_to_torch_expr(start_expr, current_loop_var=active)
            end_code = ast_to_torch_expr(end_expr, current_loop_var=active)
            lines.append(
                f"{prefix}for {inner_var} in range(int({start_code}), int({end_code})):"  # noqa: E501
            )
            emit_func_loop_body(inner_body, indent_level + 1, lines,
                                active | {inner_var})
        elif tag == "loop_if":
            _, cond, then_body = loop_stmt
            lines.append(
                f"{prefix}if {condition_to_expr(cond, current_loop_var=active)}:"  # noqa: E501
            )
            emit_func_loop_body(then_body, indent_level + 1, lines, active)
        elif tag == "loop_if_else":
            _, cond, then_body, else_body = loop_stmt
            lines.append(
                f"{prefix}if {condition_to_expr(cond, current_loop_var=active)}:"  # noqa: E501
            )
            emit_func_loop_body(then_body, indent_level + 1, lines, active)
            lines.append(f"{prefix}else:")
            emit_func_loop_body(else_body, indent_level + 1, lines, active)


# Code generators (function / class / statement)
def emit_body_stmts(
    stmts: list[ASTNode],
    indent_level: int,
    lines: list[str],
    known_vars: list[str],
    equation_vars: set[str],
    generate_solve_call: Callable[[ASTNode], str],
    scalar_only: bool = False,
    expr_fn=ast_to_torch_expr,
    _equation_vars: set[str] | None = None,
) -> None:
    """Recursively emit Python code lines for a function body.

    Converts a sequence of ``body_decl``, ``body_assign``,
    ``body_tuple_unpack``, ``body_if_return``,
    ``body_if_else_return``, ``body_if_else``, or
    ``body_if`` AST nodes into indented Python source lines
    and appends them to `lines`.


    Parameters
    ----------
    stmts: list[ASTNode]
        Sequence of ``body_decl``, ``body_assign``, ``body_tuple_unpack``,
        ``body_if_return``, ``body_if_else_return``, ``body_if_else``, or
        ``body_if`` AST tuples to emit.  ``None`` entries are skipped.
    indent_level: int
        Nesting depth. 1 if directly inside the function body,
        `indent_level` is 2 if inside an if/else branch, etc.).
        Each level adds four spaces.
    lines: list[str]
        Output list; generated source lines are appended here.
    known_vars: list[str]
        Running list of variable names in scope.
        Extended in place when new locals are declared.
    equation_vars: set[str]
        Set of variable names bound to equation strings (used to exclude
        them from ``solve()`` keyword arguments).  Updated in place.
    generate_solve_call: Callable[[ASTNode], str]
        Callable that converts an expression AST to a Python string,
        expanding ``solve(...)`` calls with the current `known_vars`.
    expr_fn : callable, optional
        Expression code-generator; defaults to ``ast_to_torch_expr``.
    _equation_vars : set, optional
        Internal — tracks variables bound to equation strings so they are
        excluded from ``solve()`` keyword arguments.  Pass ``None`` (default)
        to create a fresh set for this call.

    Examples
    --------
    >>> from physika.utils.ast_utils import emit_body_stmts
    >>> from physika.utils.ast_utils import ast_to_torch_expr
    >>> lines = []
    >>> known_vars = ["x"]
    >>> equation_vars = set()
    >>> emit_body_stmts(
    ...     [("body_assign", "y", ("mul", ("var", "x"), ("num", 2.0)))],
    ...     1, lines, known_vars, equation_vars, ast_to_torch_expr,
    ... )
    >>> lines
    ['    y = (x * 2.0)']
    """
    if expr_fn is None:
        expr_fn = ast_to_torch_expr
    if _equation_vars is None:
        _equation_vars = set()

    prefix = "    " * indent_level
    for stmt in stmts:
        if not isinstance(stmt, tuple):
            continue
        stmt_op = stmt[0]
        if stmt_op == "body_decl":
            _, var_name, var_type, expr = stmt
            if isinstance(expr, tuple) and expr[0] == "string":
                equation_vars.add(var_name)
            expr_code = generate_solve_call(expr)
            lines.append(f"{prefix}{var_name} = {expr_code}")
            known_vars.append(var_name)
        elif stmt_op == "body_assign":
            _, var_name, expr = stmt
            expr_code = generate_solve_call(expr)
            lines.append(f"{prefix}{var_name} = {expr_code}")
            known_vars.append(var_name)
        elif stmt_op == "body_tuple_unpack":
            _, var_names, expr = stmt
            expr_code = generate_solve_call(expr)
            lines.append(f"{prefix}{', '.join(var_names)} = {expr_code}")
            known_vars.extend(var_names)
        elif stmt_op == "body_if_return":
            _, cond, return_expr = stmt
            cond_code = condition_to_expr(cond)
            return_code = ast_to_torch_expr(return_expr)
            lines.append(f"{prefix}if {cond_code}:")
            lines.append(f"{prefix}    return {return_code}")
        elif stmt_op == "body_if_else_return":
            _, cond, then_expr, else_expr = stmt
            cond_code = condition_to_expr(cond)
            then_code = ast_to_torch_expr(then_expr)
            else_code = ast_to_torch_expr(else_expr)
            if scalar_only:
                # Scalar functions: use Python if/else so recursion works
                lines.append(f"{prefix}if {cond_code}:")
                lines.append(f"{prefix}    return {then_code}")
                lines.append(f"{prefix}else:")
                lines.append(f"{prefix}    return {else_code}")
            else:
                # Vector functions: use torch.where for elementwise
                # differentiability
                lines.append(
                    f"{prefix}return torch.where(torch.as_tensor({cond_code}), {then_code}, {else_code})"  # noqa: E501
                )
        elif stmt_op == "body_if_else":
            _, cond, then_stmts, else_stmts = stmt
            cond_code = condition_to_expr(cond)
            lines.append(f"{prefix}if {cond_code}:")
            emit_body_stmts(then_stmts, indent_level + 1, lines, known_vars,
                            equation_vars, generate_solve_call, scalar_only)
            lines.append(f"{prefix}else:")
            emit_body_stmts(else_stmts, indent_level + 1, lines, known_vars,
                            equation_vars, generate_solve_call, scalar_only)
        elif stmt_op == "body_if":
            _, cond, then_stmts = stmt
            cond_code = condition_to_expr(cond)
            lines.append(f"{prefix}if {cond_code}:")
            emit_body_stmts(then_stmts, indent_level + 1, lines, known_vars,
                            equation_vars, generate_solve_call, scalar_only)
        elif stmt_op == "body_for":
            _, loop_var, loop_body, indexed_arrays = stmt
            if indexed_arrays:
                lines.append(
                    f"{prefix}for {loop_var} in range(len({indexed_arrays[0]})):"  # noqa: E501
                )
            else:
                lines.append(f"{prefix}for {loop_var} in range(n):")
            emit_func_loop_body(loop_body, indent_level + 1, lines, loop_var)
        elif stmt_op == "body_for_range":
            _, loop_var, start_expr, end_expr, loop_body = stmt
            start_code = ast_to_torch_expr(start_expr)
            end_code = ast_to_torch_expr(end_expr)
            lines.append(
                f"{prefix}for {loop_var} in range(int({start_code}), int({end_code})):"  # noqa: E501
            )
            emit_func_loop_body(loop_body, indent_level + 1, lines, loop_var)
        elif stmt_op == "body_zeros_decl":
            # Type annotation for an accumulation target.
            # The paired body_for_accum emits the `torch.stack` expression that
            # defines the tensor.
            pass

        elif stmt_op == "body_for_accum":
            # Generates one differentiable torch.stack per += target.
            # Emits one `name = torch.stack(...)`
            # line per unique += target tensor.
            _, loop_vars, loop_body = stmt
            active = set(loop_vars)

            # Collect all unique accumulation targets
            accums: dict = {}
            for s in loop_body:
                if s and s[0] == "loop_index_pluseq":
                    _, name, idx_list, rhs = s
                    if name not in accums:
                        accums[name] = (idx_list, rhs)

            if not accums:
                raise ValueError(
                    "body_for_accum has no loop_index_pluseq statement")

            # Generate one differentiable torch.stack expression per target
            # tensor
            for tensor_name, (lhs_idx_exprs, rhs_expr) in accums.items():
                ranges = {
                    v:
                    _infer_range(v, rhs_expr, tensor_name)
                    or f"# range unknown for {v}"
                    for v in loop_vars
                }
                lhs_vars = [
                    n for n in (_lhs_var_name(e) for e in lhs_idx_exprs) if n
                ]
                reduction_vars = [v for v in loop_vars if v not in lhs_vars]
                rhs_code = ast_to_torch_expr(rhs_expr, current_loop_var=active)
                inner_expr = rhs_code
                for rv in reversed(reduction_vars):
                    inner_expr = (f"torch.sum(torch.stack([{inner_expr}"
                                  f" for {rv} in range({ranges[rv]})]))")
                for ov in reversed(lhs_vars):
                    inner_expr = (f"torch.stack([{inner_expr}"
                                  f" for {ov} in range({ranges[ov]})])")
                lines.append(f"{prefix}{tensor_name} = {inner_expr}")


def generate_function(name: str, func_def: dict[str, Any]) -> str:
    """Generate a Python/PyTorch function definition from a function AST.

    Translates a Physika function (params, body statements, return
    expression) into a valid Python function definition string.

    If the function body contains a ``solve()`` call, local known-variable
    tracking is used to pass all in-scope variables as keyword
    arguments to ``solve``.

    Parameters
    ----------
    name : str
        The function identifier (e.g. ``"sigma"``, ``"U"``).
    func_def : dict[str, ASTNode]
        A dict from ``unified_ast["functions"]`` with keys
        ``"params"`` (list of ``(name, type)`` pairs), ``"body"``
        (return expression AST), and optionally ``"statements"``
        (list of body statement ASTs).

    Returns
    -------
    str
        A multi-line Python source string containing the complete
        function definition.

    Examples
    --------
    >>> from physika.utils.ast_utils import generate_function
    >>> func_def = {
    ...     "params": [("x", "ℝ")],
    ...     "body": ("call", "exp", [("var", "x")]),
    ...     "statements": [],
    ... }
    >>> print(generate_function("f", func_def))
    def f(x):
        return torch.exp(x)
    """
    params = func_def["params"]
    body = func_def["body"]
    statements = func_def.get("statements", [])

    # Build parameter list
    param_strs = []
    param_names = []
    for param_name, param_type in params:
        param_strs.append(f"{param_name}")
        param_names.append(param_name)

    lines = [f"def {name}({', '.join(param_strs)}):"]

    # Track known variables (params + locals)
    known_vars = list(param_names)

    # Track equation string variable names
    equation_vars: set[str] = set()

    # Helper to generate solve call with known variables
    # (kept local: it accumulates known_vars/equation_vars as statements
    # are processed)
    def generate_solve_call(expr):
        if isinstance(expr,
                      tuple) and expr[0] == "call" and expr[1] == "solve":
            args = expr[2]
            arg_strs = [ast_to_torch_expr(arg) for arg in args]
            # Add known variables as keyword arguments (exclude equation vars)
            kw_strs = [
                f"{v}={v}" for v in known_vars if v not in equation_vars
            ]
            return f"solve({', '.join(arg_strs)}, {', '.join(kw_strs)})"
        return ast_to_torch_expr(expr)

    # Use if/else (not torch.where) when all params are scalars
    # — allows recursion
    scalar_only = all(pt == "\u211d" for _, pt in params)

    # Generate body statements
    emit_body_stmts(statements, 1, lines, known_vars, equation_vars,
                    generate_solve_call, scalar_only)

    # Generate for-loop body
    if func_def.get("has_loop"):
        init_stmts = func_def.get("init_stmts", [])
        loop_var = func_def.get("loop_var", "k")
        indexed_arrays = func_def.get("loop_indexed_arrays", [])
        loop_body = func_def.get("loop_body", [])

        # Emit pre-loop initialisation
        for stmt in init_stmts:
            if stmt is None:
                continue
            if stmt[0] == "init_assign":
                _, var_name, expr = stmt
                expr_code = ast_to_torch_expr(expr)
                lines.append(f"    {var_name} = {expr_code}")

        # Emit loop header — range inferred from the first indexed array
        if indexed_arrays:
            lines.append(
                f"    for {loop_var} in range(len({indexed_arrays[0]})):")
        else:
            lines.append(f"    for {loop_var} in range(n):")

        # Emit loop body statements
        for stmt in loop_body:
            if stmt is None:
                continue
            if stmt[0] == "loop_assign":
                _, var_name, expr = stmt
                expr_code = ast_to_torch_expr(expr, current_loop_var=loop_var)
                lines.append(f"        {var_name} = {expr_code}")
            elif stmt[0] == "loop_pluseq":
                _, var_name, expr = stmt
                expr_code = ast_to_torch_expr(expr, current_loop_var=loop_var)
                lines.append(f"        {var_name} = {var_name} + {expr_code}")

    # Generate return statement only when there is a final expression
    if body is not None:
        body_code = ast_to_torch_expr(body)
        lines.append(f"    return {body_code}")

    return "\n".join(lines)


def emit_for_stmts(
    stmts: list[ASTNode],
    indent: int = 4,
    loop_var: str | set[str] | None = None,
) -> list[str]:
    """Emit Python code for a top-level for-loop or if-else branch body.

    Handles ``for_assign``, ``for_pluseq``, ``for_index_assign``,
    ``for_call``, and nested ``for_loop`` / ``for_loop_range`` nodes.
    Recurses for nested loops, increasing indentation by 4 spaces per level.

    Parameters
    ----------
    stmts: list[ASTNode]
        List of ``for_assign``, ``for_pluseq``, ``for_index_assign``,
        ``for_call``, ``for_loop`` or ``for_loop_range``  AST nodes.
    indent: int
        Integer representing the whitespace in emitted line.
    loop_var: str or None
        Enclosing loop variable name, forwarded to ``ast_to_torch_expr``.

    Returns
    -------
    list[str]
        Python code lines .

    Examples
    --------
    >>> from physika.utils.ast_utils import emit_for_stmts
    >>> stmts = [("for_assign", "z", ("mul", ("var", "a"), ("var", "b")))]
    >>> emit_for_stmts(stmts, 4)
    ['    z = (a * b)']
    """
    prefix = " " * indent
    result = []
    for s in stmts:
        if not isinstance(s, tuple):
            continue
        body_op = s[0]
        if body_op == "for_assign":
            _, var_name, expr = s
            result.append(
                f"{prefix}{var_name} = {ast_to_torch_expr(expr, current_loop_var=loop_var)}"  # noqa: E501
            )
        elif body_op == "for_pluseq":
            _, var_name, expr = s
            result.append(
                f"{prefix}{var_name} = {var_name} + {ast_to_torch_expr(expr, current_loop_var=loop_var)}"  # noqa: E501
            )
        elif body_op == "for_index_assign":
            _, arr_name, idx_expr, rhs_expr = s
            idx_code = ast_to_torch_expr(idx_expr, current_loop_var=loop_var)
            rhs_code = ast_to_torch_expr(rhs_expr, current_loop_var=loop_var)
            result.append(f"{prefix}{arr_name}[int({idx_code})] = {rhs_code}")
        elif body_op == "for_call":
            _, func_name, arg_asts = s
            arg_strs = [
                ast_to_torch_expr(arg, current_loop_var=loop_var)
                for arg in arg_asts
            ]
            result.append(f"{prefix}{func_name}({', '.join(arg_strs)})")
        elif body_op == "for_loop_range":
            inner_var = s[1]
            start_code = ast_to_torch_expr(s[2], current_loop_var=loop_var)
            end_code = ast_to_torch_expr(s[3], current_loop_var=loop_var)
            inner_body = s[4]
            result.append(
                f"{prefix}for {inner_var} in range(int({start_code}), int({end_code})):"  # noqa: E501
            )
            # Accumulate all active loop vars so inner body can reference
            # outer vars (e.g. 'i')
            outer_vars = loop_var if isinstance(
                loop_var, set) else ({loop_var} if loop_var else set())
            inner_loop_var = outer_vars | {inner_var}
            result.extend(
                emit_for_stmts(inner_body, indent + 4, inner_loop_var))
        elif body_op == "for_loop":
            inner_var = s[1]
            inner_body = s[2]
            indexed_arrays = s[3]
            if indexed_arrays:
                result.append(
                    f"{prefix}for {inner_var} in range(len({indexed_arrays[0]})):"  # noqa: E501
                )
            else:
                result.append(f"{prefix}for {inner_var} in range(n):")
            outer_vars = loop_var if isinstance(
                loop_var, set) else ({loop_var} if loop_var else set())
            result.extend(
                emit_for_stmts(inner_body, indent + 4,
                               outer_vars | {inner_var}))
        elif body_op == "for_if":
            _, cond, then_body = s
            cond_code = condition_to_expr(cond, current_loop_var=loop_var)
            result.append(f"{prefix}if {cond_code}:")
            result.extend(emit_for_stmts(then_body, indent + 4, loop_var))
        elif body_op == "for_if_else":
            _, cond, then_body, else_body = s
            cond_code = condition_to_expr(cond, current_loop_var=loop_var)
            result.append(f"{prefix}if {cond_code}:")
            result.extend(emit_for_stmts(then_body, indent + 4, loop_var))
            result.append(f"{prefix}else:")
            result.extend(emit_for_stmts(else_body, indent + 4, loop_var))
    return result


def generate_class(name: str, class_def: dict[str, ASTNode]) -> str:
    """Generate an ``nn.Module`` subclass from a class AST entry.

    Translates a Physika class into a Python class string with
    ``__init__`` (wrapping tensor params as ``nn.Parameter``),
    ``forward`` (the lambda body, with optional loop), and an
    optional ``loss`` method.  Class parameter references in the
    forward/loss bodies are rewritten to ``self.param`` via
    ``replace_class_params``.

    Parameters
    ----------
    name : str
        The class identifier (e.g. ``"OneLayerNet"``).
    class_def : dict[str, ASTNode]
        A dict from ``unified_ast["classes"]`` with keys:

        * ``"class_params"`` — list of ``(name, type)`` pairs.
        * ``"lambda_params"`` — list of ``(name, type)`` pairs.
        * ``"body"`` — forward return expression AST.
        * ``"has_loop"`` (optional) — whether forward contains a loop.
        * ``"loop_var"``, ``"loop_body"`` (optional) — loop details.
        * ``"has_loss"``, ``"loss_body"``, ``"loss_params"`` (optional).

    Returns
    -------
    str
        A multi-line Python source string containing the complete
        ``nn.Module`` subclass definition.

    Examples
    --------
    >>> from physika.utils.ast_utils import generate_class
    >>> class_def = {
    ...     "class_params": [("w", "ℝ")],
    ...     "lambda_params": [("x", "ℝ")],
    ...     "body": ("mul", ("var", "w"), ("var", "x")),
    ...     "has_loop": False, "has_loss": False,
    ... }
    >>> code = generate_class("Linear", class_def)
    >>> "class Linear(nn.Module):" in code
    True
    """
    class_params: list[tuple[str, ASTNode]] = cast(list[tuple[str, ASTNode]],
                                                   class_def["class_params"])
    lambda_params: list[tuple[str, ASTNode]] = cast(list[tuple[str, ASTNode]],
                                                    class_def["lambda_params"])
    body = class_def["body"]
    statements: list[ASTNode] = cast(list[ASTNode],
                                     class_def.get("statements", []))
    has_loop = class_def.get("has_loop", False)
    loop_var = class_def.get("loop_var")
    loop_body: list[ASTNode] = cast(list[ASTNode],
                                    class_def.get("loop_body", []))
    has_loss = class_def.get("has_loss", False)
    loss_body = class_def.get("loss_body")

    lines = [f"class {name}(nn.Module):"]

    # __init__ method
    init_params = ", ".join([p[0] for p in class_params])
    lines.append(f"    def __init__(self, {init_params}):")
    lines.append("        super().__init__()")
    for param_name, param_type in class_params:
        # Check if this is a tensor type that should be a parameter
        is_tensor = False
        if isinstance(param_type, tuple) and param_type[0] == "tensor":
            is_tensor = True
        elif param_type == "\u211d":
            is_tensor = True  # Scalar could be a learnable parameter

        if is_tensor:
            # Handle both tensors and scalars
            lines.append(
                f"        self.{param_name} = nn.Parameter(torch.tensor({param_name}).float() if not isinstance({param_name}, torch.Tensor) else {param_name}.clone().detach().float())"  # noqa: E501
            )
        else:
            # Non-tensor (like function 'f' or int 'n')
            lines.append(f"        self.{param_name} = {param_name}")

    # forward method (lambda)
    lambda_param_names = [p[0] for p in lambda_params]
    lines.append("")
    lines.append(f"    def forward(self, {', '.join(lambda_param_names)}):")
    # Convert inputs to tensors
    for pname, ptype in lambda_params:
        if ptype == "\u211d" or ptype == "\u2115" or (isinstance(
                ptype, tuple) and ptype[0] == "tensor"):
            lines.append(f"        {pname} = torch.as_tensor({pname}).float()")

    # Generate forward body statements (multi-statement lambda body)
    if statements:
        stmt_lines: list[str] = []
        known_vars = [p[0] for p in lambda_params]
        equation_vars: set[str] = set()
        scalar_only = all(pt == "\u211d" for _, pt in lambda_params)
        emit_body_stmts(statements, 2, stmt_lines, known_vars, equation_vars,
                        ast_to_torch_expr, scalar_only)
        for line in stmt_lines:
            lines.append(replace_class_params(line, class_params))

    # Generate loop if present
    if has_loop and loop_body:
        lines.append(
            f"        n = int(self.n) if hasattr(self, 'n') else self.{class_params[-1][0]}.shape[0] if hasattr(self.{class_params[-1][0]}, 'shape') else 2"  # noqa: E501
        )
        lines.append(f"        for {loop_var} in range(n):")
        for stmt in loop_body:
            if isinstance(stmt, tuple) and stmt[0] == "loop_assign":
                var_name = stmt[1]
                expr = stmt[2]
                expr_code = ast_to_torch_expr(expr)
                expr_code = replace_class_params(expr_code, class_params)
                lines.append(f"            {var_name} = {expr_code}")

    # Generate return
    body_code = ast_to_torch_expr(body)
    body_code = replace_class_params(body_code, class_params)
    lines.append(f"        return {body_code}")

    # loss method if present
    if has_loss and loss_body:
        loss_params: list[tuple[str, ASTNode]] = cast(
            list[tuple[str, ASTNode]],
            class_def.get("loss_params", [("y", "\u211d"),
                                          ("target", "\u211d")]))
        loss_param_names = [p[0] for p in loss_params]
        loss_stmts: list[ASTNode] = cast(list[ASTNode],
                                         class_def.get("loss_statements", []))

        # Check if loss uses grad — also scan loss body statements
        loss_uses_grad = ast_uses_func(loss_body, "grad")
        if not loss_uses_grad:
            for s in loss_stmts:
                if ast_uses_func(s, "grad"):
                    loss_uses_grad = True
                    break

        if loss_uses_grad and lambda_param_names:
            # Add the input parameter (x) to loss params
            input_param = lambda_param_names[0]  # typically 'x'
            lines.append("")
            lines.append(
                f"    def loss(self, {', '.join(loss_param_names)}, {input_param}):"  # noqa: E501
            )
        else:
            lines.append("")
            lines.append(f"    def loss(self, {', '.join(loss_param_names)}):")

        # Emit loss body statements
        for stmt in loss_stmts:
            if not isinstance(stmt, tuple):
                continue
            stmt_op = stmt[0]
            if stmt_op == "body_decl":
                _, var_name, var_type, expr = stmt
                expr_code = ast_to_torch_expr(expr)
                expr_code = replace_class_params(expr_code, class_params)
                lines.append(f"        {var_name} = {expr_code}")
            elif stmt_op == "body_assign":
                _, var_name, expr = stmt
                expr_code = ast_to_torch_expr(expr)
                expr_code = replace_class_params(expr_code, class_params)
                lines.append(f"        {var_name} = {expr_code}")
            elif stmt_op == "body_tuple_unpack":
                _, var_names, expr = stmt
                expr_code = ast_to_torch_expr(expr)
                expr_code = replace_class_params(expr_code, class_params)
                lines.append(f"        {', '.join(var_names)} = {expr_code}")

        loss_code = ast_to_torch_expr(loss_body)
        loss_code = replace_class_params(loss_code, class_params)
        lines.append(f"        return {loss_code}")

    return "\n".join(lines)


def generate_statement(stmt: ASTNode,
                       grad_target_vars: set[str]) -> str | None:
    """Generate a PyTorch code string for a program-level statement.

    Handles ``decl`` (variable declaration), ``assign`` (reassignment),
    ``expr`` (bare expression — wrapped in ``physika_print`` unless it
    is a side-effect call like ``simulate``/``animate``), ``for_loop``,
    and skips ``func_def``/``class_def`` (already emitted by
    ``from_ast_to_torch``).

    Variables whose names appear in *grad_target_vars* are initialised
    with ``requires_grad=True`` so that ``grad()`` can differentiate
    through them.

    Parameters
    ----------
    stmt : ASTNode
        An AST statement tuple (e.g.
        ``("decl", name, type, expr, lineno)``) or ``None``.
    grad_target_vars : set[str]
        Variable names used as differentiation targets in ``grad()``
        calls.  Collected by ``collect_grad_targets`` during the
        analysis pass.

    Returns
    -------
    str or None
        A Python source string for the statement, or ``None`` if the
        statement should be skipped (``func_def``, ``class_def``, or
        ``None`` input).

    Examples
    --------
    >>> from physika.utils.ast_utils import generate_statement
    >>> generate_statement(("decl", "x", "ℝ", ("num", 3.0), 1), set())
    'x = 3.0'
    >>> generate_statement(("decl", "t", "ℝ", ("num", 0.0), 2), {"t"})
    't = torch.tensor(0.0, requires_grad=True)'
    >>> generate_statement(("expr", ("var", "x"), 0), set())
    'physika_print(x)'
    """
    if not isinstance(stmt, tuple):
        return None

    op = stmt[0]

    if op == "decl":
        name = stmt[1]
        type_spec = stmt[2]
        expr = stmt[3]
        expr_code = ast_to_torch_expr(expr)
        # Variables used as grad targets need to be tensors with requires_grad
        if name in grad_target_vars and type_spec == "\u211d":
            return f"{name} = torch.tensor({expr_code}, requires_grad=True)"
        return f"{name} = {expr_code}"

    elif op == "assign":
        name = stmt[1]
        expr = stmt[2]
        expr_code = ast_to_torch_expr(expr)
        return f"{name} = {expr_code}"

    elif op == "expr":
        expr = stmt[1]
        expr_code = ast_to_torch_expr(expr)
        # Don't wrap side-effect-only calls in physika_print
        if isinstance(expr,
                      tuple) and expr[0] == "call" and expr[1] in ("simulate",
                                                                   "animate"):
            return expr_code
        return f"physika_print({expr_code})"

    elif op == "func_def":
        return None  # Already generated

    elif op == "class_def":
        return None  # Already generated

    elif op == "for_loop":
        # For loop: ("for_loop", loop_var, body_statements,
        # indexed_arrays[, lineno])
        loop_var = stmt[1]
        body_statements = stmt[2]
        indexed_arrays = stmt[3]
        if indexed_arrays:
            header = f"for {loop_var} in range(len({indexed_arrays[0]})):"
        else:
            header = f"for {loop_var} in range(n):  # TODO: determine n"
        lines = [header] + emit_for_stmts(body_statements, 4, loop_var)
        return "\n".join(lines)

    elif op == "for_loop_range":
        # Explicit-range for loop: ("for_loop_range", loop_var, start_expr,
        # end_expr, body_stmts, lineno)
        loop_var = stmt[1]
        start_code = ast_to_torch_expr(stmt[2])
        end_code = ast_to_torch_expr(stmt[3])
        body_statements = stmt[4]
        lines = [
            f"for {loop_var} in range(int({start_code}), int({end_code})):"
        ]
        lines += emit_for_stmts(body_statements, 4, loop_var)
        return "\n".join(lines)

    elif op in ("if_else", "if_only"):
        cond = stmt[1]
        then_stmts = stmt[2]
        cond_code = condition_to_expr(cond)

        branch_lines = [f"if {cond_code}:"]
        branch_lines.extend(emit_for_stmts(then_stmts))

        if op == "if_else":
            else_stmts = stmt[3]
            branch_lines.append("else:")
            branch_lines.extend(emit_for_stmts(else_stmts))

        return "\n".join(branch_lines)

    return f"# Unknown: {stmt}"


def build_unified_ast(
    program_ast: list[ASTNode],
    symbol_table: dict[str, dict[str, Any]],
    print_ast: bool = False,
) -> dict[str, Any]:
    """Build a unified AST combining definitions and program statements.

    Merges the flat ``program_ast`` (list of statement tuples produced
    by the parser) with the ``symbol_table`` (function and class
    definitions accumulated during parsing) into a single dict with
    three sections: ``"functions"``, ``"classes"``, and ``"program"``.

    Parameters
    ----------
    program_ast : list[ASTNode]
        The list of top-level statement AST tuples returned by
        ``parser.parse()``.
    symbol_table : dict[str, dict[str, Any]]
        The parser's symbol table mapping names to
        ``{"type": "function"|"class", "value": ...}`` entries.
    print_ast : bool, default False
        If ``True``, print the unified AST to stdout for debugging.

    Returns
    -------
    dict[str, dict[str, ASTNode] | list[ASTNode]]
        A dict with keys:

        * ``"functions"`` — ``{name: func_def, ...}``
        * ``"classes"`` — ``{name: class_def, ...}``
        * ``"program"`` — ``[stmt, ...]``

    Examples
    --------
    >>> from physika.utils.ast_utils import build_unified_ast
    >>> ast = [("expr", ("num", 42.0), 1)]
    >>> sym = {}
    >>> unified = build_unified_ast(ast, sym)
    >>> unified["program"]
    [('expr', ('num', 42.0), 1)]
    >>> unified["functions"]
    {}
    """
    unified: dict[str, Any] = {"functions": {}, "classes": {}, "program": []}

    # Extract functions and classes from symbol table
    for name, entry in symbol_table.items():
        if entry["type"] == "function":
            unified["functions"][name] = entry["value"]
        elif entry["type"] == "class":
            unified["classes"][name] = entry["value"]

    # Add program statements
    for stmt in program_ast:
        if stmt is not None:
            unified["program"].append(stmt)

    if print_ast:
        print("\n=== UNIFIED AST ===")
        print(print_unified_ast(unified))

    return unified
