from typing import Dict, Union, List

from utils.ast_utils import (
    ast_uses_solve, ast_uses_func, collect_grad_targets,
    generate_function, generate_class, generate_statement,
)


def from_ast_to_torch(
    unified_ast: Dict[str, Union[Dict, List]],
    print_code: bool = True
) -> str:
    """Convert a unified AST into a complete, executable Python/PyTorch source string.

    This conversion is done in two passes:

    1. **Analysis pass** — walks the AST to determine which ``runtime.py``
       helpers (``solve``, ``train``, ``evaluate``, ``compute_grad``,
       ``simulate``, ``animate``, etc) are referenced, and collects variables
       used as ``grad()`` differentiation targets.
    2. **Code-generation pass** — uses ``generate_function``,
       ``generate_class``, and ``generate_statement`` (from
       ``utils.ast_utils``) to emit Python source for each AST entry,
       preceded by import header.

    The returned string is ready to be executed with ``exec()``.

    Parameters
    ----------
    unified_ast : Dict[str, Union[Dict, List]]
        The unified AST dict produced by ``build_unified_ast()``, with keys:

        * ``"functions"`` — ``Dict[str, dict]`` mapping function names to
          their AST definitions (params, body, statements).
        * ``"classes"`` — ``Dict[str, dict]`` mapping class names to their
          AST definitions (class_params, lambda_params, body, loss_body, …).
        * ``"program"`` — ``List[tuple]`` of top-level statement AST nodes
          (decl, assign, expr, for_loop, func_def, class_def).
    print_code : bool, default True
        If ``True``, print the generated code.

    Returns
    -------
    str :
        A complete Python/PyTorch source string containing ``import``
        statements, function definitions, ``nn.Module`` class definitions,
        and program-level statements.  Variables that appear as ``grad()``
        targets are initialised with ``requires_grad=True``.

    Examples
    --------
    >>> # Example #1: simple expression
    >>> unified_ast = {
    ...     "functions": {},
    ...     "classes": {},
    ...     "program": [("expr", ("num", 42.0), 1)],
    ... }
    >>> code = from_ast_to_torch(unified_ast, print_code=False)
    >>> "import torch" in code
    True
    >>> "physika_print(42.0)" in code
    True
    >>> print(code)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    <BLANKLINE>
    from runtime import physika_print
    <BLANKLINE>
    # === Program ===
    physika_print(42.0)
    >>> # Example #2: function definition and call
    >>> unified_ast = {
    ...     "functions": {
    ...         "f": {"params": [("x", "ℝ")], "body": ("call", "exp", [("var", "x")]), "statements": []},
    ...     },
    ...     "classes": {},
    ...     "program": [("expr", ("call", "f", [("num", 1.0)]), 2)],
    ... }
    >>> code = from_ast_to_torch(unified_ast, print_code=False)
    >>> "def f(x):" in code
    True
    >>> "torch.exp(x)" in code
    True
    >>> print(code)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    <BLANKLINE>
    from runtime import physika_print
    <BLANKLINE>
    # === Functions ===
    def f(x):
        return torch.exp(x)
    <BLANKLINE>
    # === Program ===
    physika_print(f(1.0))
    """
    code_lines = []

    # Analysis pass: determine which helpers are needed
    needs_solve = any(ast_uses_solve(stmt) for stmt in unified_ast["program"])
    for func_def in unified_ast["functions"].values():
        if ast_uses_solve(func_def.get("body")) or any(ast_uses_solve(s) for s in func_def.get("statements", [])):
            needs_solve = True
            break

    needs_train = any(ast_uses_func(stmt, "train") for stmt in unified_ast["program"])
    needs_evaluate = any(ast_uses_func(stmt, "evaluate") for stmt in unified_ast["program"])
    needs_simulate = any(ast_uses_func(stmt, "simulate") for stmt in unified_ast["program"])
    needs_animate = any(ast_uses_func(stmt, "animate") for stmt in unified_ast["program"])

    # Collect variables used as differentiation targets in grad() calls
    grad_target_vars = set()
    # Collect from top-level program statements
    for stmt in unified_ast["program"]:
        collect_grad_targets(stmt, grad_target_vars)

    # Collect from function bodies and statements
    for func_def in unified_ast["functions"].values():
        collect_grad_targets(func_def.get("body"), grad_target_vars)
        for s in func_def.get("statements", []):
            collect_grad_targets(s, grad_target_vars)

    # Check for grad usage in classes and program statements
    needs_grad = False
    for class_def in unified_ast["classes"].values():
        if ast_uses_func(class_def.get("loss_body"), "grad"):
            needs_grad = True
            break
        if ast_uses_func(class_def.get("body"), "grad"):
            needs_grad = True
            break
        if any(ast_uses_func(s, "grad") for s in class_def.get("statements", [])):
            needs_grad = True
            break
        if any(ast_uses_func(s, "grad") for s in class_def.get("loss_statements", [])):
            needs_grad = True
            break
    if not needs_grad:
        for stmt in unified_ast["program"]:
            if ast_uses_func(stmt, "grad"):
                needs_grad = True
                break

    # Code generation 

    # Header
    code_lines.append("import torch")
    code_lines.append("import torch.nn as nn")
    code_lines.append("import torch.optim as optim")
    if needs_solve:
        code_lines.append("import re")
    code_lines.append("")

    # Import helpers from runtime.py
    imports = ["from runtime import physika_print"]
    if needs_solve:
        imports.append("from runtime import solve")
    if needs_train:
        imports.append("from runtime import train")
    if needs_evaluate:
        imports.append("from runtime import evaluate")
    if needs_grad:
        imports.append("from runtime import compute_grad")
    if needs_simulate:
        imports.append("from runtime import simulate")
    if needs_animate:
        imports.append("from runtime import animate")
    code_lines.append("\n".join(imports))
    code_lines.append("")

    # Generate functions
    if unified_ast["functions"]:
        code_lines.append("# === Functions ===")
        for name, func_def in unified_ast["functions"].items():
            code_lines.append(generate_function(name, func_def))
            code_lines.append("")

    # Generate classes
    if unified_ast["classes"]:
        code_lines.append("# === Classes ===")
        for name, class_def in unified_ast["classes"].items():
            code_lines.append(generate_class(name, class_def))
            code_lines.append("")

    # Generate program statements
    code_lines.append("# === Program ===")
    for stmt in unified_ast["program"]:
        stmt_code = generate_statement(stmt, grad_target_vars)
        if stmt_code:
            code_lines.append(stmt_code)

    # Join all code
    generated_code = "\n".join(code_lines)

    if print_code:
        print("\n=== Physika generated Pytorch code ===")
        print(generated_code)
        print("=== End Pytorch code ===\n")

    return generated_code
