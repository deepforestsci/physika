from __future__ import annotations

from typing import Any, Optional, Dict, Union, List

from utils.type_checker_utils import (
    type_to_str,
    types_compatible,
    get_line_info,
    type_infer,
    statement_check,
)

# Type aliases used in annotations throughout this module.
ASTExpr = Any       # tagged tuple, scalar, or None
TypeSpec = Any       # "ℝ", "ℕ", ("tensor", [...]), ("func_type", ...), None
UnifiedAST = Dict[str, Union[Dict, List]]


class TypeChecker:
    """Type checker for Physika unified ASTs.

    Passes through each function, class, and program statement in the unified AST
    and validates declared types against inferred types. Error messages are
    accumulated in ``self.errors`` and returned by :meth:`run`.

    The checker maintains three environments:

    * ``type_env`` — program variable types (populated as ``decl``
      and ``assign`` statements are processed).
    * ``func_env`` — registered function signatures ``(param_types, return_type)``.
    * ``class_env`` — registered class AST definitions (used by
      ``type_infer`` to validate constructor and instance calls).

    Parameters
    ----------
    unified_ast : UnifiedAST
        The unified AST dict produced by ``build_unified_ast()``, with
        keys ``"functions"``, ``"classes"``, and ``"program"``.

    Examples
    --------
    >>> from type_checker import TypeChecker
    >>> unified_ast = {
    ...     "functions": {},
    ...     "classes": {},
    ...     "program": [("decl", "x", "ℝ", ("num", 3.0), 1)],
    ... }
    >>> checker = TypeChecker(unified_ast)
    >>> errors = checker.run()
    >>> errors
    []
    """

    def __init__(self, unified_ast: UnifiedAST) -> None:
        self.unified_ast = unified_ast
        self.errors: list[str] = []
        self.type_env: dict[str, TypeSpec] = {}
        self.func_env: dict[str, tuple[list[TypeSpec], TypeSpec | None]] = {}
        self.class_env: dict[str, dict] = {}
        self.current_line: list[int | None] = [None]

    def add_error(self, msg: str) -> None:
        """Record a type error, prefixed with the current source line.

        Parameters
        ----------
        msg : str
            Human-readable error description (e.g.
            ``"Type mismatch for 'x': declared as ℝ, got ℝ[3]"``).
        """
        if self.current_line is not None:
            self.errors.append(f"Line {self.current_line}: {msg}")
        else:
            self.errors.append(msg)


    def infer_type(self, expr: ASTExpr, local_env: Optional[dict[str, TypeSpec]] = None) -> TypeSpec:
        """Infer the Physika type of an AST expression.

        Scalar literals (``int``, ``float``) map to ``"ℝ"``.  Tagged
        tuples are dispatched to ``type_infer`` (in
        ``utils.type_checker_utils``), which handles operators, calls,
        arrays, indexing, etc.

        Parameters
        ----------
        expr : ASTExpr
            An AST expression node — a tagged tuple like
            ``("add", left, right)``, a numeric literal, or ``None``.
        local_env : dict[str, TypeSpec] or None
            Local type environment (e.g. function parameters).  Merged
            with ``self.type_env`` for name lookups.  Defaults to ``{}``.

        Returns
        -------
        TypeSpec
            The inferred type: ``"ℝ"``, ``"ℕ"``, ``"ℂ"``,
            ``("tensor", [(dim, variance), ...])``, ``("instance", name)``,
            or ``None`` if the type cannot be determined.

        Examples
        --------
        >>> from type_checker import TypeChecker
        >>> checker = TypeChecker({"functions": {}, "classes": {}, "program": []})
        >>> checker.infer_type(("num", 3.0))
        'ℝ'
        >>> checker.infer_type(("array", [("num", 1.0), ("num", 2.0)]))
        ('tensor', [(2, 'invariant')])
        """
        if local_env is None:
            local_env = {}

        if not isinstance(expr, tuple):
            if isinstance(expr, (int, float)):
                return "ℝ"
            return None

        op = expr[0]
        return type_infer(op, expr, self.type_env, local_env, self.add_error, self.infer_type, self.func_env, self.class_env)


    def check_statement(self, stmt: ASTExpr) -> None:
        """Check a single program-level statement for type errors.

        Extracts the source line number (for error context), then
        delegates to ``statement_check`` which handles ``decl``,
        ``assign``, ``expr``, and ``for_loop`` statement types.

        Parameters
        ----------
        stmt : ASTExpr
            A program-level AST statement tuple, or ``None``.

        Examples
        --------
        >>> from type_checker import TypeChecker
        >>> checker = TypeChecker({"functions": {}, "classes": {}, "program": []})
        >>> checker.check_statement(("decl", "x", "ℝ", ("num", 3.0), 1))
        >>> checker.errors
        []
        >>> checker.type_env["x"]
        'ℝ'
        """
        if stmt is None:
            return

        op = stmt[0]
        line = get_line_info(stmt)
        self.current_line[0] = line
        statement_check(op, stmt, self.infer_type, self.add_error, self.type_env, self.check_statement)


    def check_function(self, name: str, func_def: dict[str, Any]) -> None:
        """Check a function definition for type errors.

        Builds a local type environment from the function's parameters,
        registers the function signature in ``self.func_env``, then
        checks each body statement and the return expression for type
        consistency.

        Parameters
        ----------
        name : str
            The function identifier (e.g. ``"sigma"``).
        func_def : dict[str, Any]
            A dict from ``unified_ast["functions"]`` with keys
            ``"params"`` (list of ``(name, type)`` pairs), ``"body"``
            (return expression AST), optionally ``"statements"`` (list
            of body statement ASTs) and ``"return_type"``.

        Examples
        --------
        >>> from type_checker import TypeChecker
        >>> checker = TypeChecker({"functions": {}, "classes": {}, "program": []})
        >>> func_def = {
        ...     "params": [("x", "ℝ")],
        ...     "body": ("var", "x"),
        ...     "statements": [],
        ...     "return_type": "ℝ",
        ... }
        >>> checker.check_function("identity", func_def)
        >>> checker.errors
        []
        >>> checker.func_env["identity"]
        (['ℝ'], 'ℝ')
        """
        params = func_def["params"]
        body = func_def["body"]
        statements = func_def.get("statements", [])
        return_type = func_def.get("return_type")

        # Build local environment from parameters
        local_env = {}
        param_types = []
        for param_name, param_type in params:
            local_env[param_name] = param_type
            param_types.append(param_type)

        # Register function signature
        self.func_env[name] = (param_types, return_type)

        # Check statements in function body
        for stmt in statements:
            if stmt is None:
                continue
            stmt_op = stmt[0]
            if stmt_op == "body_decl":
                _, var_name, var_type, expr = stmt
                inferred = self.infer_type(expr, local_env)
                if var_type and inferred and not types_compatible(var_type, inferred):
                    self.errors.append(
                        f"In function '{name}': type mismatch for '{var_name}': "
                        f"declared as {type_to_str(var_type)}, got {type_to_str(inferred)}"
                    )
                local_env[var_name] = var_type if var_type else inferred
            elif stmt_op == "body_assign":
                _, var_name, expr = stmt
                inferred = self.infer_type(expr, local_env)
                local_env[var_name] = inferred
            elif stmt_op == "body_tuple_unpack":
                _, var_names, expr = stmt
                for var_name in var_names:
                    local_env[var_name] = None  # Type unknown from unpack

        # Check return expression
        body_type = self.infer_type(body, local_env)
        if return_type and body_type and not types_compatible(return_type, body_type):
            self.errors.append(
                f"Function '{name}' return type mismatch: declared {type_to_str(return_type)}, "
                f"but body has type {type_to_str(body_type)}"
            )

    def check_class(self, name: str, class_def: dict[str, Any]) -> None:
        """Check a class definition for type errors.

        Registers the class in ``self.class_env``, builds a local type
        environment from class and lambda parameters, then validates the
        forward body and (if present) the loss body.

        Parameters
        ----------
        name : str
            The class identifier (e.g. ``"HamiltonianNet"``).
        class_def : dict[str, Any]
            A dict from ``unified_ast["classes"]`` with keys
            ``"class_params"``, ``"lambda_params"``, ``"body"``, and
            optionally ``"return_type"``, ``"loss_body"``,
            ``"loss_params"``.

        Examples
        --------
        >>> from type_checker import TypeChecker
        >>> checker = TypeChecker({"functions": {}, "classes": {}, "program": []})
        >>> class_def = {
        ...     "class_params": [("w", "ℝ")],
        ...     "lambda_params": [("x", "ℝ")],
        ...     "body": ("mul", ("var", "w"), ("var", "x")),
        ... }
        >>> checker.check_class("Linear", class_def)
        >>> checker.errors
        []
        """
        class_params = class_def["class_params"]
        lambda_params = class_def["lambda_params"]
        body = class_def["body"]
        return_type = class_def.get("return_type")
        loss_body = class_def.get("loss_body")
        loss_params = class_def.get("loss_params", [])

        # Register class
        self.class_env[name] = class_def

        # Build local environment from class params and lambda params
        local_env = {}
        for param_name, param_type in class_params:
            local_env[param_name] = param_type
        for param_name, param_type in lambda_params:
            local_env[param_name] = param_type

        # Check forward body
        body_type = self.infer_type(body, local_env)
        if return_type and body_type and not types_compatible(return_type, body_type):
            self.errors.append(
                f"Class '{name}' forward return type mismatch: declared {type_to_str(return_type)}, "
                f"but body has type {type_to_str(body_type)}"
            )

        # Check loss body if present
        if loss_body:
            loss_env = dict(local_env)
            for param_name, param_type in loss_params:
                loss_env[param_name] = param_type
            self.infer_type(loss_body, loss_env)


    # Main entry point
    def run(self) -> list[str]:
        """Run the type checker over the entire unified AST.

        Checks functions, classes, and program-level statements.

        Returns
        -------
        list[str]
            A list of human-readable error messages.  Empty if no type
            errors were found.

        Examples
        --------
        >>> from type_checker import TypeChecker
        >>> unified_ast = {
        ...     "functions": {},
        ...     "classes": {},
        ...     "program": [("expr", ("add", ("num", 1.0), ("num", 2.0)), 1)],
        ... }
        >>> TypeChecker(unified_ast).run()
        []
        """
        for name, func_def in self.unified_ast["functions"].items():
            self.check_function(name, func_def)

        for name, class_def in self.unified_ast["classes"].items():
            self.check_class(name, class_def)

        for stmt in self.unified_ast["program"]:
            if stmt and stmt[0] not in ("func_def", "class_def"):
                self.check_statement(stmt)

        return self.errors


def type_check(unified_ast: UnifiedAST) -> list[str]:
    """Run the Physika type checker on a unified AST.

    Convenience wrapper that creates a :class:`TypeChecker` and calls
    :meth:`~TypeChecker.run`.

    Parameters
    ----------
    unified_ast : UnifiedAST
        The unified AST dict produced by ``build_unified_ast()``.

    Returns
    -------
    List[str]
        A list of error messages.  Empty if the program
        is have no type errors.

    Examples
    --------
    >>> from type_checker import type_check
    >>> type_check({"functions": {}, "classes": {}, "program": []})
    []
    >>> # I need to add more robust examples
    """
    checker = TypeChecker(unified_ast)
    return checker.run()
