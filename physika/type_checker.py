from typing import Any, Dict

from physika.utils.types import (
    check_function,
    check_statement,
    check_class,
    TInstance,
    counter,
)

# Type aliases used in annotations throughout this module.
ASTExpr = Any  # tagged tuple, scalar, or None
TypeSpec = Any  # "ℝ", "ℕ", ("tensor", [...]), ("func_type", ...), None
UnifiedAST = Dict[str, Any]


class TypeChecker:
    """
    Type checker for Physika programs.

    The Hindley-Milner algorithm rests on two main steps:

    **Substitution** performs a mapping ``{αN: concrete_type}`` from unknown
    type variables (TVar `αN` or TDim `δN`) to valid types.  When ``unify``
    determines that ``αN`` must equal some type ``T``, it extends the
    substitution with the binding ``αN → T``.  Calling ``s.apply(t)``
    replaces every bound type variable in ``t`` with its mapped type,
    following chains of bindings until a concrete type is reached.

    **Unification** (``unify(t1, t2, s)``) uses the accumulated version
    of ``s`` that makes ``s.apply(t1) == s.apply(t2)``.  If either side is a
    free ``TVar``, a new binding is added to ``s``.
    If both sides are concrete types of the same constructor
    (arrays, matrices, tensors as ``TTensor`` types), their components are
    unified recursively. If the shapes are incompatible
    (e.g. ``ℝ`` vs. ``ℝ[3]``), the mismatch is recorded as a type error.

    Physika's type checker performs three passes over the unified AST:

    1. **Signature registration**: All function and class signatures are
       stored in ``func_env`` and ``class_env`` before any body is examined.
       Class constructors are stored in ``func_env`` as
       ``(field_types, TInstance(name))``.

    2. **Body checking** (``check_function``, ``check_class``): For each
       ``def`` and ``class``, ``infer_stmts`` walks statements in order,
       threading ``s`` through every expression to build a local type
       environment.  The return expression is inferred and unified against
       the declared return type. A mismatch is recorded as an error prefixed
       with the function or class name.

    3. **Program statement checking** (``check_statement``): Top-level
       stmts nodes are checked in source order.
       The line number is read from the last element of each statement tuple
       and prepended to error messages.

    Type mismatches are accumulated in ``self.errors`` as plain strings.

    Parameters
    ----------
    unified_ast : dict
        The unified AST dict produced by ``build_unified_ast()``, with keys
        ``"functions"``, ``"classes"``, and ``"program"``.

    Examples
    --------
    >>> # Example 1
    >>> # No errors
    >>> from physika.type_checker import TypeChecker
    >>> ast = {
    ...     "functions": {},
    ...     "classes": {},
    ...     "program": [("decl", "x", "ℝ", ("num", 1.0), 1)],
    ... }
    >>> TypeChecker(ast).run()
    []
    >>> # Example 2
    >>> # function called with wrong number of arguments:
    >>> fdef = {
    ...     "params": [("x", "ℝ"), ("y", "ℝ")],
    ...     "statements": [],
    ...     "body": ("add", ("var", "x"), ("var", "y")),
    ...     "return_type": "ℝ",
    ... }
    >>> ast = {
    ...     "functions": {"add2": fdef},
    ...     "classes": {},
    ...     "program": [("expr", ("call", "add2", [("num", 1.0)]), 3)],
    ... }
    >>> TypeChecker(ast).run()
    ["Line 3: Function 'add2' expects 2 args, got 1"]
    """

    def __init__(self, unified_ast: dict) -> None:
        self.unified_ast = unified_ast
        self.errors: list[str] = []
        self.type_env: dict = {}
        self.func_env: dict = {}
        self.class_env: dict = {}

    def run(self) -> list[str]:
        """Run type inference over the full unified AST.

        Three passes:

        1. Register all function and class signatures.
        2. Check function bodies
        3. Check class bodies.
        4. Check top-level statements.

        Returns
        -------
        list[str]
            Accumulated type error messages.  Empty if the program is
            well-typed.

        Examples
        --------
        >>> # No errors
        >>> from physika.type_checker import TypeChecker
        >>> ast = {
        ...     "functions": {},
        ...     "classes": {},
        ...     "program": [("decl", "x", "ℝ", ("num", 1.0), 1)],
        ... }
        >>> TypeChecker(ast).run()
        []
        """
        counter.reset()
        for name, fdef in self.unified_ast["functions"].items():
            params = fdef["params"]
            self.func_env[name] = ([pt for _, pt in params],
                                   fdef.get("return_type"))

        for name, cdef in self.unified_ast["classes"].items():
            all_fields = (list(cdef.get("class_params", [])) +
                          list(cdef.get("fields", [])))
            methods = cdef.get("methods", [])
            self.class_env[name] = {
                "fields": all_fields,
                "methods": {
                    m["name"]: {
                        "params": m.get("params", []),
                        "return_type": m.get("return_type")
                    }
                    for m in methods
                },
            }
            self.func_env[name] = (
                [pt for _, pt in all_fields],
                TInstance(name),
            )

        for name, fdef in self.unified_ast["functions"].items():
            check_function(name, fdef, self.func_env, self.class_env,
                           self.errors.append)

        for name, cdef in self.unified_ast["classes"].items():
            check_class(name, cdef, self.func_env, self.class_env,
                        self.errors.append)

        for stmt in self.unified_ast["program"]:
            if stmt and stmt[0] not in ("func_def", "class_def"):
                check_statement(stmt, self.type_env, self.func_env,
                                self.class_env, self.errors.append)

        return self.errors
