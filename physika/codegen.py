from typing import Dict, Optional, Set, Any
from physika.core.torch_lowering import lower_expr, body_mutates_in_place
from physika.core.environment import Environment
from physika.core.expr import FVarId
from physika.utils.ast_utils import (ast_uses_solve, ast_uses_func,
                                     collect_grad_targets, generate_function,
                                     generate_statement, ast_uses_sympy,
                                     ast_to_torch_expr)
from physika.features.classes import generate_class
from physika.elf import REGISTRY
from physika.units import update_unit_env

# fallback to ast codegen
AST_CODEGEN_STMT_TAGS = ("body_for", "body_for_range", "body_for_accum")


def from_ast_to_torch(unified_ast: Dict[str, Any],
                      print_code: bool = True,
                      resolved_bodies: Optional[Dict[str, Any]] = None,
                      resolved_methods: Optional[Dict[str, Any]] = None,
                      resolved_program: Optional[Dict[int, Any]] = None,
                      resolved_program_fvar_names: Optional[Dict[FVarId,
                                                                 str]] = None,
                      cic_env: Optional[Environment] = None,
                      func_sigs: Optional[Dict[str, Any]] = None) -> str:
    """Convert a unified AST into a complete, executable Python/PyTorch
    source string.

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


    Functions, methods and top level program declarations that has
    been CIC elaborated, verified and lowered to torch code skips the AST walk
    (since there Physika CIC already verified it). However, any unresolved term
    , not present in resolved arguments, fallbacks to ast to torch codegen
    path.

    Parameters
    ----------
    unified_ast : Dict[str, Any]
        The unified AST dict produced by ``build_unified_ast()``, with keys:

        * ``"functions"`` — ``Dict[str, dict]`` mapping function names to
          their AST definitions (params, body, statements).
        * ``"classes"`` — ``Dict[str, dict]`` mapping class names to their
          AST definitions (class_params, lambda_params, body, loss_body, …).
        * ``"program"`` — ``List[tuple]`` of top-level statement AST nodes
          (decl, assign, expr, for_loop, func_def, class_def).
    print_code : bool, default True
        If ``True``, print the generated code.
    resolved_bodies : Dict[str, Any], optional
        Function name to verified CIC term (plus fvar names, local decls,
        param order, dim renames) disctionary.
    resolved_methods : Dict[str, Any], optional
        Class method to verified CIC term mapping.
    resolved_program : Dict[int, Any], optional
        ``unified_ast["program"]`` to ``(var_name, CIC elborated and verified
        term)``.
    resolved_program_fvar_names : Dict[str, Any], optional
        ``FVarId`` to source name mapping.
    cic_env : Any, optional
        ``Environment`` used during elaboration used to lower the resolved CIC
        terms.
    func_sigs : Dict[str, Any], optional
        Function name to ``(param units, return units)`` mapping.

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
    >>> from physika.codegen import from_ast_to_torch
    >>> unified_ast = {
    ...     "functions": {},
    ...     "classes": {},
    ...     "program": [("expr", ("num", 42.0), 1)],
    ... }
    >>> code = from_ast_to_torch(unified_ast, print_code=False)
    >>> "import torch" in code
    True
    >>> "print(42.0)" in code
    True
    >>> print(code)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from physika.runtime import DEVICE
    <BLANKLINE>
    from physika.runtime import print
    <BLANKLINE>
    # === Program ===
    print(42.0)
    >>> # Example #2: function definition and call
    >>> unified_ast = {
    ...     "functions": {
    ...         "f": {"params": [("x", "ℝ")], "body": ("call", "exp",
    ...         [("var", "x")]), "statements": []},
    ...     },
    ...     "classes": {},
    ...     "program": [("expr", ("call", "f", [("num", 1.0)]), 2)],
    ... }
    >>> code = from_ast_to_torch(unified_ast, print_code=False)
    >>> "def f(x):" in code
    True
    >>> "torch.exp" in code
    True
    >>> print(code)  # noqa: E501
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from physika.runtime import DEVICE
    <BLANKLINE>
    from physika.runtime import print
    <BLANKLINE>
    # === Functions ===
    def f(x):
        return torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    <BLANKLINE>
    # === Program ===
    print(f(1.0))
    """
    code_lines = []

    # Analysis pass: determine which helpers are needed
    needs_solve = any(ast_uses_solve(stmt) for stmt in unified_ast["program"])
    for func_def in unified_ast["functions"].values():
        if ast_uses_solve(func_def.get("body")) or any(
                ast_uses_solve(s) for s in func_def.get("statements", [])):
            needs_solve = True
            break

    needs_train = any(
        ast_uses_func(stmt, "train") for stmt in unified_ast["program"])
    needs_evaluate = any(
        ast_uses_func(stmt, "evaluate") for stmt in unified_ast["program"])
    needs_simulate = any(
        ast_uses_func(stmt, "simulate") for stmt in unified_ast["program"])
    needs_animate = any(
        ast_uses_func(stmt, "animate") for stmt in unified_ast["program"])
    needs_sympy = any(ast_uses_sympy(stmt) for stmt in unified_ast["program"])

    # Collect variables used as differentiation targets in grad() calls
    grad_target_vars: Set[str] = set()
    for stmt in unified_ast["program"]:
        collect_grad_targets(stmt, grad_target_vars)

    # Check for grad usage in classes and program statements
    needs_grad = False
    for class_def in unified_ast["classes"].values():
        if ast_uses_func(class_def.get("loss_body"), "grad"):
            needs_grad = True
            break
        if ast_uses_func(class_def.get("body"), "grad"):
            needs_grad = True
            break
        if any(
                ast_uses_func(s, "grad")
                for s in class_def.get("statements", [])):
            needs_grad = True
            break
        if any(
                ast_uses_func(s, "grad")
                for s in class_def.get("loss_statements", [])):
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
    code_lines.append("from physika.runtime import DEVICE")
    if needs_solve:
        code_lines.append("import re")
    code_lines.append("")

    # Import helpers from runtime.py
    imports = ["from physika.runtime import print"]
    if needs_solve:
        imports.append("from physika.runtime import solve")
    if needs_train:
        imports.append("from physika.runtime import train")
    if needs_evaluate:
        imports.append("from physika.runtime import evaluate")
    if needs_grad:
        imports.append("from physika.runtime import compute_grad")
    if needs_simulate:
        imports.append("from physika.runtime import simulate")
    if needs_animate:
        imports.append("from physika.runtime import animate")
    if needs_sympy:
        imports.append("import sympy as sp")
    code_lines.append("\n".join(imports))
    code_lines.append("")

    # fall back to raw-AST codegen for functions instead that have statement
    # reassignemnts in their bodies
    if resolved_bodies:
        kept: Dict[str, Any] = {}
        for name, term in resolved_bodies.items():
            func_def = unified_ast["functions"].get(name, {})
            param_order = term[3]
            dependent = len(param_order) > len(func_def.get("params", []))
            ast_only = body_mutates_in_place(func_def) or any(
                isinstance(s, tuple) and s and s[0] in AST_CODEGEN_STMT_TAGS
                for s in func_def.get("statements", []))
            if dependent and not ast_only:
                kept[name] = term
        resolved_bodies = kept

    resolved_methods = {}
    # dependent type binders are added explictly as solved during CIC
    # elaborationn to the function call
    if resolved_program:
        _dep = set(resolved_bodies or {})
        resolved_program = {
            i: entry
            for i, entry in resolved_program.items() if any(
                ast_uses_func(unified_ast["program"][i], d) for d in _dep)
        }

    # merge solved cic terms duting elaboration for names lookup
    resolved_names = set(resolved_bodies or {}) | set(resolved_methods or {})

    # Generate functions
    if unified_ast["functions"]:
        code_lines.append("# === Functions ===")
        for name, func_def in unified_ast["functions"].items():
            resolved = None
            # if resolved_bodies ot resolved_methods are empty, fallback to
            # standard codegen (no torch lowring from cic)
            if resolved_bodies is not None and name in resolved_bodies:
                body, fvar_names, local_decls, param_order, dim_rename = resolved_bodies[  # noqa: E501
                    name]
                resolved = (body, fvar_names, local_decls, param_order,
                            dim_rename, cic_env)
            try:
                fn_code = generate_function(name,
                                            func_def,
                                            resolved=resolved,
                                            resolved_names=resolved_names)
            except Exception as e:
                # Body is CIC verified but torch lowering failed
                # fall back to the raw-AST codegen
                if resolved is not None:
                    print(f"  (CIC: lowering 'def {name}' failed ({e}); ")
                fn_code = generate_function(name, func_def)
            code_lines.append(fn_code)
            code_lines.append("")

    # Generate classes
    if unified_ast["classes"]:
        # class methods env from CIC elaboration including inferred types
        # and dependent bindings
        resolved_methods_with_env = {
            qualified:
            (body, fvar_names, local_decls, param_order, dim_rename, cic_env)
            for qualified,
            (body, fvar_names, local_decls, param_order,
             dim_rename) in (resolved_methods or {}).items()
        }
        code_lines.append("# === Classes ===")
        for name, class_def in unified_ast["classes"].items():
            if REGISTRY.features != []:
                node = ("class_def", name, class_def)
                class_code = REGISTRY.dispatch_forward(
                    "class_def", node, to_expr=ast_to_torch_expr)
                assert class_code is not None
                code_lines.append(class_code)
            else:
                code_lines.append(
                    generate_class(name,
                                   class_def,
                                   resolved_methods=resolved_methods_with_env,
                                   resolved_names=resolved_names))
            code_lines.append("")

    # Generate program statements
    code_lines.append("# === Program ===")

    unit_env: Dict[str, Any] = {}
    for idx, stmt in enumerate(unified_ast["program"]):
        resolved_expr_code = None
        # lookup if there are any CIC elaborated terms from top level program
        entry = (resolved_program or {}).get(idx)
        if entry is not None and cic_env is not None:
            # if there are CIC elaborated and verifeid terms, lower them to
            # torch code
            _, resolved_cic = entry
            try:
                resolved_expr_code = lower_expr(resolved_cic,
                                                resolved_program_fvar_names
                                                or {},
                                                cic_env,
                                                resolved_names=resolved_names)
            except Exception as e:
                # statement is CIC verified but torch lowering failed
                # fall back to the raw-AST codegen
                print(f"  (CIC: lowering a program statement failed ({e}); ")
                resolved_expr_code = None
        # else follows regular codegen
        stmt_code = generate_statement(stmt,
                                       grad_target_vars,
                                       resolved_expr_code=resolved_expr_code,
                                       unit_env=unit_env,
                                       func_sigs=func_sigs)
        if stmt_code:
            code_lines.append(stmt_code)
        update_unit_env(stmt, unit_env, func_sigs)

    # Join all code
    generated_code = "\n".join(code_lines)

    if print_code:
        print("\n=== Physika generated Pytorch code ===")
        print(generated_code)
        print("=== End Pytorch code ===\n")

    return generated_code
