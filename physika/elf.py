from typing import Any, Callable, Optional


class ELF:
    """
    Easy Language Feature (ELF).
    
    Base class for Physika-ELFs.

    Subclass and override the four rule methods.  
    
    Return an empty dict for rules that are not needed by the new feature.
    """

    # feature name
    name: str = ""

    def parser_rules(self) -> list[Callable]:
        """Return PLY ``p_`` functions that define the grammar for this feature."""
        return []

    def type_rules(self) -> dict[str, Callable]:
        """Return a mapping of AST op tag to type inference function."""
        return {}

    def forward_rules(self) -> dict[str, Callable]:
        """Return a mapping of AST op tag to code generation function."""
        return {}

    def backward_rules(self) -> dict[str, Callable]:
        """
        Return a mapping of AST op tag to differentiation rule function.

        Rules on how to compute gradients towards a fully differentiable feature.
        """
        return {}



class FeatureRegistry:
    """
    Collects ``ELF`` instances and drives the
    dispatch chains in the parser, type checker, and code generation.
    """

    def __init__(self) -> None:
        self._features: list[ELF] = []
        self._type_dispatch:      dict[str, Callable] = {}
        self._forward_dispatch:   dict[str, Callable] = {}
        self._backward_dispatch:  dict[str, Callable] = {}
        self._call_type_dispatch: dict[str, Callable] = {}
        self._call_emit_dispatch: dict[str, Callable] = {}


    def register(self, feature: ELF) -> None:
        """Register a feature and merge its rule tables into the registry."""
        self._features.append(feature)
        self._type_dispatch.update(feature.type_rules())
        self._forward_dispatch.update(feature.forward_rules())
        self._backward_dispatch.update(feature.backward_rules())
        for fname, (type_fn, emit_fn) in feature.call_rules().items():
            self._call_type_dispatch[fname] = type_fn
            self._call_emit_dispatch[fname] = emit_fn


    def install_parser_rules(self, module: Any) -> None:
        """
        Inject all grammar rules so that
        PLY's ``yacc.yacc()`` can build the parser.


        Example
        -------
        >>> from physika.feature import REGISTRY
        >>> REGISTRY.install_parser_rules(sys.modules[__name__])
        >>> parser = yacc.yacc()
        """
        for feature in self._features:
            for rule_fn in feature.parser_rules():
                setattr(module, rule_fn.__name__, rule_fn)


    def has_type_rule(self, op: str) -> bool:
        return op in self._type_dispatch

    def dispatch_type(
        self,
        op: str,
        node: Any,
        env: dict,
        s: Any,
        func_env: dict,
        class_env: dict,
        add_error: Callable,
        infer_expr: Callable,
    ):
        """
        Dispatch to the type rule for *op*, if one is registered.

        Returns ``(None, s)`` if no rule is registered for *op*.

        Should be integrated at the end of ``infer_expr`` in ``types.py``.
        """
        fn = self._type_dispatch.get(op)
        if fn is None:
            return None, s
        return fn(node, env, s, func_env, class_env, add_error, infer_expr)


    def has_forward_rule(self, op: str) -> bool:
        return op in self._forward_dispatch

    def dispatch_forward(self, op: str, node: Any, **ctx) -> Optional[str]:
        """
        Dispatch to the forward (code generation) rule for *op*.

        Returns ``None`` if no rule is registered.

        Integrate at the end of ``ast_to_torch_expr`` in ``ast_utils.py``, and 
        at the end of ``emit_body_stmts``.
        """
        fn = self._forward_dispatch.get(op)
        if fn is None:
            return None
        return fn(node, **ctx)

    # Backward rules (I need to think more about generalizing gradient computation rules)

    def has_backward_rule(self, op: str) -> bool:
        return op in self._backward_dispatch

    def dispatch_backward(
        self, op: str, node: Any, grad_output: Any, **ctx
    ) -> Optional[str]:
        """
        Dispatch to the backward (differentiation) rule for *op*.

        Returns ``None`` if no rule is registered.

        Reserved for future use — not yet wired into the runtime.
        """
        fn = self._backward_dispatch.get(op)
        if fn is None:
            return None
        return fn(node, grad_output, **ctx)



"""
    Example:
    --------------

    from physika.feature import LanguageFeature, REGISTRY

    class WhileLoopFeature(LanguageFeature):
        name = "while_loop"

        def parser_rules(self):
            def p_while(p):
                \"\"\"statement : WHILE expr COLON NEWLINE INDENT stmts DEDENT\"\"\"
                p[0] = ("while_loop", p[2], p[6])
            return [p_while]

        def type_rules(self):
            def check(node, env, s, func_env, class_env, add_error, infer_expr):
                _, cond, body = node
                cond_t, s = infer_expr(cond, env, s, func_env, class_env, add_error)
                return None, s
            return {"while_loop": check}

        def forward_rules(self):
            def emit(node, to_expr, **_):
                _, cond, body = node
                return f"while {to_expr(cond)}:\\n    pass"
            return {"while_loop": emit}

    REGISTRY.register(WhileLoopFeature())
"""