from __future__ import annotations

import re
from typing import Any, Literal, Union # TYPECHECKING
from utils.print_utils import print_unified_ast


# AST TYPE DEFINITIONS
# The parser produces a tree of tagged tuples.  Every non-leaf node is a
# tuple whose first element is a string (tag) and whose remaining elements are
# other nodes, lists of nodes, or scalar leaves.
#
# The type aliases below make the tag vocabulary explicit so that mypy
# can flag typos, missing branches, and wrong argument types.

# Tag literals (every valid first element of an AST tuple)

ExprTag = Literal[
    "add", "sub", "mul", "div", "pow", "matmul",   # binary arithmetic
    "neg",                                           # unary arithmetic
    "num", "var",                                    # literals / references
    "string", "equation_string",                     # string literals
    "array",                                         # [elem, ...]
    "index", "slice",                                # arr[i], arr[a:b]
    "call", "call_index",                            # f(x), f(x)[i]
    "imaginary",                                     # complex unit  i
]

StmtTag = Literal[
    "decl",       # x : R = expr          -> (tag, name, type, expr, lineno)
    "assign",     # x = expr              -> (tag, name, expr, lineno)
    "expr",       # expr                  -> (tag, expr, lineno)
    "func_def",   # def f(...)            -> (tag, name)
    "class_def",  # class C(...)          -> (tag, name)
    "for_loop",   # for i: ...            -> (tag, var, [body], [arrays], lineno)
]

BodyStmtTag = Literal[
    "body_assign",       # x = expr          inside function body
    "body_decl",         # x : T = expr      inside function body
    "body_tuple_unpack", # a, b = expr       inside function body
    "loop_assign",       # x = expr          inside for-loop body
    "loop_pluseq",       # x += expr         inside for-loop body
    "init_assign",       # x = expr          pre-loop initialisation
    "for_assign",        # x = expr          program-level for body
    "for_pluseq",        # x += expr         program-level for body
    "for_call",          # f(x)              program-level for body
]

TypeTag = Literal[
    "func_type",   # R -> R                -> (tag, input_type, output_type)
    "tangent",     # T_x M                -> (tag, point_id, manifold_type)
    "tensor",      # R[3,3]               -> (tag, [(dim, variance), ...])
]

ASTTag = Union[ExprTag, StmtTag, BodyStmtTag, TypeTag]

# Composite node type 
ASTNode = Union[
        tuple[str, ...],     # tagged nodes: ("add", left, right), ("num", 1.0), ...
        list["ASTNode"],     # child sequences: function args, loop bodies, ...
        str,                 # identifiers, string literal values
        int,                 # integer values (line numbers, dimension sizes)
        float,               # numeric literal values
        None,                # empty / no-op placeholder
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
    >>> from utils.ast_utils import ast_uses_solve
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
        return any(ast_uses_solve(child) for child in node[1:] if isinstance(child, (tuple, list)))
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
    >>> from utils.ast_utils import ast_uses_func
    >>> ast_uses_func(("call", "train", [("var", "model")]), "train")
    True
    >>> ast_uses_func(("call_index", "grad", [("var", "H")], ("num", 0.0)), "grad")
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
        return any(ast_uses_func(child, func_name) for child in node[1:] if isinstance(child, (tuple, list)))
    if isinstance(node, list):
        return any(ast_uses_func(item, func_name) for item in node)
    return False


def collect_grad_targets(node: ASTNode, targets: set[str]) -> None:
    """Collect variable names used as differentiation targets in ``grad()`` calls.

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
    >>> from utils.ast_utils import collect_grad_targets
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
            if len(args) >= 2 and isinstance(args[1], tuple) and args[1][0] == "var":
                targets.add(args[1][1])
        for child in node[1:]:
            if isinstance(child, (tuple, list)):
                collect_grad_targets(child, targets)
    elif isinstance(node, list):
        for item in node:
            collect_grad_targets(item, targets)


def replace_class_params(code: str, class_params: list[tuple[str, ASTNode]]) -> str:
    """Replace class parameter references with ``self.param`` in generated code.

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
    >>> from utils.ast_utils import replace_class_params
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
    return code


def ast_to_torch_expr(node: ASTNode, indent: int = 0, current_loop_var: str | None = None) -> str:
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
    >>> from utils.ast_utils import ast_to_torch_expr
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

    elif op == "intdiv":
        left = ast_to_torch_expr(node[1], indent, current_loop_var)
        right = ast_to_torch_expr(node[2], indent, current_loop_var)
        return f"({left} // {right})"

    elif op == "neg":
        val = ast_to_torch_expr(node[1], indent, current_loop_var)
        return f"(-{val})"

    elif op == "array":
        elements = node[1]
        # Check if this is a nested array (contains other arrays)
        has_nested = any(isinstance(e, tuple) and e[0] == "array" for e in elements)
        if has_nested:
            # For nested arrays, generate list-of-lists and wrap in torch.tensor
            def array_to_list(arr_node):
                if isinstance(arr_node, tuple) and arr_node[0] == "array":
                    inner = [array_to_list(e) for e in arr_node[1]]
                    return f"[{', '.join(inner)}]"
                else:
                    return ast_to_torch_expr(arr_node, indent, current_loop_var)
            inner_lists = [array_to_list(e) for e in elements]
            return f"torch.tensor([{', '.join(inner_lists)}])"
        else:
            def _is_numeric_literal(e: ASTNode) -> bool:
                """Check whether an AST node is a number.

                Returns ``True`` for ``("num", value)`` nodes and for
                ``("neg", ("num", value))``.

                Parameters
                ----------
                e : ASTNode
                    An AST expression node (array element).

                Returns
                -------
                bool
                    ``True`` if *e* is a numeric literal or a negated
                    numeric literal; ``False`` otherwise.
                """
                if isinstance(e, tuple) and e[0] == "num":
                    return True
                if isinstance(e, tuple) and e[0] == "neg":
                    return _is_numeric_literal(e[1])
                return False
            all_numeric = all(_is_numeric_literal(e) for e in elements)
            elem_strs = [ast_to_torch_expr(e, indent, current_loop_var) for e in elements]
            if all_numeric:
                return f"torch.tensor([{', '.join(elem_strs)}])"
            else:
                # Elements may be tensors (e.g., x[1], sin(x[0])) — use torch.stack
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

    elif op == "step_slice":
        var_name = node[1]
        start = node[2]
        step = node[3]
        return f"{var_name}[{start}::{step}]"

    elif op == "call":
        func_name = node[1]
        args = node[2]
        arg_strs = [ast_to_torch_expr(arg, indent, current_loop_var) for arg in args]

        # Map built-in functions to PyTorch equivalents
        torch_funcs = {
            "exp": "torch.exp",
            "log": "torch.log",
            "sin": "torch.sin",
            "cos": "torch.cos",
            "tanh": "torch.tanh",
            "sqrt": "torch.sqrt",
            "abs": "torch.abs",
            "sum": "torch.sum",
            "mean": "torch.mean",
            "real": "torch.real",
            "arange": "torch.arange",
        }

        if func_name in torch_funcs:
            return f"{torch_funcs[func_name]}({', '.join(arg_strs)})"
        elif func_name == "grad":
            # grad(output, input) -> compute_grad(output, input)
            return f"compute_grad({', '.join(arg_strs)})"
        elif func_name == "len":
            return f"len({arg_strs[0]})"
        elif func_name == "cat":
            # cat(a, b) -> torch.cat([a, b])
            return f"torch.cat([{', '.join(arg_strs)}])"
        else:
            return f"{func_name}({', '.join(arg_strs)})"

    elif op == "call_index":
        # Indexed function call: func(args)[index]
        func_name = node[1]
        args = node[2]
        index_ast = node[3]
        arg_strs = [ast_to_torch_expr(arg, indent, current_loop_var) for arg in args]
        idx = ast_to_torch_expr(index_ast, indent, current_loop_var)

        if func_name == "grad":
            # grad(output, input)[i] -> compute_grad(output, input)[i]
            return f"compute_grad({', '.join(arg_strs)})[int({idx})]"
        else:
            return f"{func_name}({', '.join(arg_strs)})[int({idx})]"

    elif op == "imaginary":
        # If we're inside a for loop with loop var 'i', use the loop var
        if current_loop_var == "i":
            return "i"
        # Use torch.tensor(1j) so it can be used with torch.exp
        return "torch.tensor(1j)"

    elif op == "equation_string":
        return repr(node[1])

    elif op == "string":
        # Equation string literal
        return repr(node[1])

    return f"/* unknown: {node} */"



def condition_to_expr(cond: ASTNode) -> str:
    """Convert a condition AST node to a Python boolean expression string.

    Parameters
    ----------
    cond : ASTNode
        A condition tuple like ``("cond_eq", left, right)``.

    Returns
    -------
    str
        A Python boolean expression (e.g. ``"n == 0.0"``).

    Examples
    --------
    >>> from utils.ast_utils import condition_to_expr
    >>> condition_to_expr(("cond_eq", ("var", "n"), ("num", 0.0)))
    'n == 0.0'
    >>> condition_to_expr(("cond_lt", ("var", "x"), ("num", 1.0)))
    'x < 1.0'
    """
    op_map = {
        "cond_eq": "==", "cond_neq": "!=",
        "cond_lt": "<",  "cond_gt": ">",
        "cond_leq": "<=", "cond_geq": ">=",
    }
    op = cond[0]
    left = ast_to_torch_expr(cond[1])
    right = ast_to_torch_expr(cond[2])
    return f"{left} {op_map[op]} {right}"


# Code generators (function / class / statement)
def generate_function(name: str, func_def: dict[str, ASTNode]) -> str:
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
    >>> from utils.ast_utils import generate_function
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
    equation_vars = set()

    # Helper to generate solve call with known variables
    # (kept local: it accumulates known_vars/equation_vars as statements are processed)
    def generate_solve_call(expr):
        if isinstance(expr, tuple) and expr[0] == "call" and expr[1] == "solve":
            args = expr[2]
            arg_strs = [ast_to_torch_expr(arg) for arg in args]
            # Add known variables as keyword arguments (exclude equation vars)
            kw_strs = [f"{v}={v}" for v in known_vars if v not in equation_vars]
            return f"solve({', '.join(arg_strs)}, {', '.join(kw_strs)})"
        return ast_to_torch_expr(expr)

    def emit_body_stmts(stmts: list[ASTNode], indent_level: int) -> None:
        """Recursively emit body statements at a given indentation level.

        Appends generated Python source lines to the ``lines``
        list from ``generate_function``.

        Parameters
        ----------
        stmts : list[ASTNode]
            Sequence of body-statement AST tuples to emit.
        indent_level : int
            Nesting depth (1 = directly inside the function body,
            2 = inside an if/else branch, etc.).  Each level adds
            four spaces of indentation.
        """
        prefix = "    " * indent_level
        for stmt in stmts:
            if stmt is None:
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
                lines.append(f"{prefix}if {cond_code}:")
                lines.append(f"{prefix}    return {then_code}")
                lines.append(f"{prefix}else:")
                lines.append(f"{prefix}    return {else_code}")
            elif stmt_op == "body_if_else":
                _, cond, then_stmts, else_stmts = stmt
                cond_code = condition_to_expr(cond)
                lines.append(f"{prefix}if {cond_code}:")
                emit_body_stmts(then_stmts, indent_level + 1)
                lines.append(f"{prefix}else:")
                emit_body_stmts(else_stmts, indent_level + 1)
            elif stmt_op == "body_if":
                _, cond, then_stmts = stmt
                cond_code = condition_to_expr(cond)
                lines.append(f"{prefix}if {cond_code}:")
                emit_body_stmts(then_stmts, indent_level + 1)

    # Generate body statements
    emit_body_stmts(statements, 1)

    # Generate return statement only when there is a final expression
    # (body-only functions have body=None; their returns live inside if/else branches)
    if body is not None:
        body_code = ast_to_torch_expr(body)
        lines.append(f"    return {body_code}")

    return "\n".join(lines)


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
    >>> from utils.ast_utils import generate_class
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
    class_params = class_def["class_params"]
    lambda_params = class_def["lambda_params"]
    body = class_def["body"]
    statements = class_def.get("statements", [])
    has_loop = class_def.get("has_loop", False)
    loop_var = class_def.get("loop_var")
    loop_body = class_def.get("loop_body", [])
    has_loss = class_def.get("has_loss", False)
    loss_body = class_def.get("loss_body")

    lines = [f"class {name}(nn.Module):"]

    # __init__ method
    init_params = ", ".join([p[0] for p in class_params])
    lines.append(f"    def __init__(self, {init_params}):")
    lines.append(f"        super().__init__()")
    for param_name, param_type in class_params:
        # Check if this is a tensor type that should be a parameter
        is_tensor = False
        if isinstance(param_type, tuple) and param_type[0] == "tensor":
            is_tensor = True
        elif param_type == "\u211d":
            is_tensor = True  # Scalar could be a learnable parameter

        if is_tensor:
            # Handle both tensors and scalars
            lines.append(f"        self.{param_name} = nn.Parameter(torch.tensor({param_name}).float() if not isinstance({param_name}, torch.Tensor) else {param_name}.clone().detach().float())")
        else:
            # Non-tensor (like function 'f' or int 'n')
            lines.append(f"        self.{param_name} = {param_name}")

    # forward method (lambda)
    lambda_param_names = [p[0] for p in lambda_params]
    lines.append(f"")
    lines.append(f"    def forward(self, {', '.join(lambda_param_names)}):")
    # Convert inputs to tensors
    for pname, ptype in lambda_params:
        if ptype == "\u211d" or ptype == "\u2115" or (isinstance(ptype, tuple) and ptype[0] == "tensor"):
            lines.append(f"        {pname} = torch.as_tensor({pname}).float()")

    # Generate forward body statements (multi-statement lambda body)
    for stmt in statements:
        if stmt is None:
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

    # Generate loop if present
    if has_loop and loop_body:
        lines.append(f"        n = int(self.n) if hasattr(self, 'n') else self.{class_params[-1][0]}.shape[0] if hasattr(self.{class_params[-1][0]}, 'shape') else 2")
        lines.append(f"        for {loop_var} in range(n):")
        for stmt in loop_body:
            if stmt and stmt[0] == "loop_assign":
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
        loss_params = class_def.get("loss_params", [("y", "\u211d"), ("target", "\u211d")])
        loss_param_names = [p[0] for p in loss_params]
        loss_stmts = class_def.get("loss_statements", [])

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
            lines.append(f"")
            lines.append(f"    def loss(self, {', '.join(loss_param_names)}, {input_param}):")
        else:
            lines.append(f"")
            lines.append(f"    def loss(self, {', '.join(loss_param_names)}):")

        # Emit loss body statements
        for stmt in loss_stmts:
            if stmt is None:
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


def generate_statement(stmt: ASTNode, grad_target_vars: set[str]) -> str | None:
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
    >>> from utils.ast_utils import generate_statement
    >>> generate_statement(("decl", "x", "ℝ", ("num", 3.0), 1), set())
    'x = 3.0'
    >>> generate_statement(("decl", "t", "ℝ", ("num", 0.0), 2), {"t"})
    't = torch.tensor(0.0, requires_grad=True)'
    >>> generate_statement(("expr", ("var", "x"), 0), set())
    'physika_print(x)'
    """
    if stmt is None:
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
        if isinstance(expr, tuple) and expr[0] == "call" and expr[1] in ("simulate", "animate"):
            return expr_code
        return f"physika_print({expr_code})"

    elif op == "func_def":
        return None  # Already generated

    elif op == "class_def":
        return None  # Already generated

    elif op == "for_loop":
        # For loop: ("for_loop", loop_var, body_statements, indexed_arrays[, lineno])
        loop_var = stmt[1]
        body_statements = stmt[2]
        indexed_arrays = stmt[3]
        lines = []
        # Use first indexed array to get length
        if indexed_arrays:
            arr_name = indexed_arrays[0]
            lines.append(f"for {loop_var} in range(len({arr_name})):")
        else:
            lines.append(f"for {loop_var} in range(n):  # TODO: determine n")

        for body_stmt in body_statements:
            if body_stmt is None:
                continue
            body_op = body_stmt[0]
            if body_op == "for_assign":
                _, var_name, expr = body_stmt
                expr_code = ast_to_torch_expr(expr, current_loop_var=loop_var)
                lines.append(f"    {var_name} = {expr_code}")
            elif body_op == "for_pluseq":
                _, var_name, expr = body_stmt
                expr_code = ast_to_torch_expr(expr, current_loop_var=loop_var)
                lines.append(f"    {var_name} = {var_name} + {expr_code}")
            elif body_op == "for_call":
                _, func_name, arg_asts = body_stmt
                arg_strs = [ast_to_torch_expr(arg, current_loop_var=loop_var) for arg in arg_asts]
                lines.append(f"    {func_name}({', '.join(arg_strs)})")

        return "\n".join(lines)

    elif op in ("if_else", "if_only"):
        cond = stmt[1]
        then_stmts = stmt[2]
        cond_code = condition_to_expr(cond)

        def emit_for_stmts(stmts, prefix):
            result = []
            for s in stmts:
                if s is None:
                    continue
                body_op = s[0]
                if body_op == "for_assign":
                    _, var_name, expr = s
                    result.append(f"{prefix}{var_name} = {ast_to_torch_expr(expr)}")
                elif body_op == "for_pluseq":
                    _, var_name, expr = s
                    result.append(f"{prefix}{var_name} = {var_name} + {ast_to_torch_expr(expr)}")
                elif body_op == "for_call":
                    _, func_name, arg_asts = s
                    arg_strs = [ast_to_torch_expr(arg) for arg in arg_asts]
                    result.append(f"{prefix}{func_name}({', '.join(arg_strs)})")
            return result

        branch_lines = [f"if {cond_code}:"]
        branch_lines.extend(emit_for_stmts(then_stmts, "    "))

        if op == "if_else":
            else_stmts = stmt[3]
            branch_lines.append("else:")
            branch_lines.extend(emit_for_stmts(else_stmts, "    "))

        return "\n".join(branch_lines)

    return f"# Unknown: {stmt}"


def build_unified_ast(
    program_ast: list[ASTNode],
    symbol_table: dict[str, dict[str, Any]],
    print_ast: bool = False,
) -> dict[str, Union[dict[str, ASTNode], list[ASTNode]]]:
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
    >>> from utils.ast_utils import build_unified_ast
    >>> ast = [("expr", ("num", 42.0), 1)]
    >>> sym = {}
    >>> unified = build_unified_ast(ast, sym)
    >>> unified["program"]
    [('expr', ('num', 42.0), 1)]
    >>> unified["functions"]
    {}
    """
    unified = {
        "functions": {},
        "classes": {},
        "program": []
    }

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
