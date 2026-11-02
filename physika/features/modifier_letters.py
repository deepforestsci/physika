from physika.elf import ELF
from typing import Callable, Optional
from physika.utils.types import TTensor, Substitution


def make_parser_rules():
    """
    PLY grammar functions for Modifier letters.
    """

    def p_factor_transpose(p):
        """factor : ID TRANSPOSE"""
        # Transpose a tensor (top-level)
        # Example:
        # Aᵀ
        # Parameters:
        #    p[1] - variable name
        #    p[2] - transpose modifier (ᵀ)
        p[0] = ("transpose", ("var", p[1]))

    def p_func_factor_transpose(p):
        """func_factor : ID TRANSPOSE"""
        # Transpose a tensor (function-level)
        # Example:
        # Aᵀ
        # Parameters:
        #    p[1] - variable name
        #    p[2] - transpose modifier (ᵀ)
        p[0] = ("transpose", ("var", p[1]))

    return [p_factor_transpose, p_func_factor_transpose]


class ModifierLetters(ELF):
    """
    Modifier letters implemented as ELF subclass.

    Supported symbols
    -----------------
    Transpose - ᵀ

    Examples
    --------
    >>> from physika.lexer import lexer
    >>> from physika.parser import parser, symbol_table
    >>> from physika.utils.ast_utils import build_unified_ast
    >>> from physika.codegen import from_ast_to_torch
    >>> def run_phyk(src):
    ...     symbol_table.clear()
    ...     lexer.lexer.lineno = 1
    ...     ast = build_unified_ast(parser.parse(src, lexer=lexer), symbol_table)  # noqa
    ...     exec(from_ast_to_torch(ast, print_code=False), {})

    >>> # simple matrix
    >>> src = '''
    ... A: ℝ[2, 3] = [[1, 2, 3], [4, 5, 6]]
    ... Aᵀ
    ... '''

    >>> # Execute code and verify outputs
    >>> run_phyk(src)
    [[1, 4], [2, 5], [3, 6]] ∈ ℝ[3,2]
    """

    name = "modifier_letters"

    def lexer_rules(self) -> dict:
        """
        Adds ``TRANSPOSE`` as TOKEN along with ``t_TRANSPOSE`` as
        token function.

        Examples
        --------
        >>> from physika.features.modifier_letters import ModifierLetters
        >>> rules = ModifierLetters().lexer_rules()
        >>> rules["tokens"]
        ['TRANSPOSE']
        >>> rules["reserved"]
        {'physika': 'PHYSIKA'}
        >>> rules["token_funcs"][0].__name__
        't_TRANSPOSE'
        """

        def t_TRANSPOSE(t):
            # Syntax: Aᵀ
            r"ᵀ"
            return t

        return {
            "reserved": {
                "physika": "PHYSIKA"
            },
            "tokens": ["TRANSPOSE"],
            "token_funcs": [t_TRANSPOSE]
        }

    def parser_rules(self) -> list:
        """
        Override ``parser_rules`` handler for new grammar rules.

        Returns
        -------
        list
            List of PLY grammar functions to be injected
            into ``physika.parser``.

        Examples
        --------
        >>> from physika.features import ModifierLetters
        >>> rules = ModifierLetters().parser_rules()
        >>> len(rules)
        2
        >>> rules[0].__name__
        'p_factor_transpose'
        """
        return make_parser_rules()

    def type_rules(self) -> dict:
        """
        Registers one type-checking handler for ``transpose`` named
        as ``expr_transpose``.

        Returns
        -------
        dict
            Dispatch table mapping ``"transpose"`` AST
            tags to their type inference handlers.

        Examples
        --------
        >>> from physika.features import ModifierLetters
        >>> rules = ModifierLetters().type_rules()
        >>> sorted(rules.keys())
        ['transpose']
        """

        def expr_transpose(
                node: tuple, env: dict, s: Substitution, func_env: dict,
                class_env: dict, add_error: Callable, infer_expr: Callable
        ) -> tuple[Optional[TTensor], Substitution]:
            """
            Type rules for ``transpose`` AST node.

            Parameters
            ----------
            node : tuple
                AST node of the form ``(("transpose", ("var", "x"))``.
            env : dict
                Current type environment mapping variable names to their types.
            s : Substitution
                Substitution dict containing bindings accumulated so far.
            func_env : dict
                Function definitions available in scope.
            class_env : dict
                Class definitions mapping class names to their parameters and
                variables.
            add_error : Callable[[str], None]
                Callback to register a type error message.
            infer_expr : Callable
                Type inference function for sub-expressions.

            Returns
            -------
            TTensor
                Returns a tensor since it uses `torch.transpose`.

            Examples
            --------
            >>> from physika.features import ModifierLetters
            >>> from physika.utils.types import Substitution, TTensor
            >>> rules = ModifierLetters().type_rules()
            >>> transpose = rules["transpose"]
            >>> s = Substitution()
            >>> errors = []
            >>> env = {"x": TTensor((3, 4))}
            >>> node = ("transpose", ("var", "x"))
            >>> def infer_expr(node, env, s, func_env, class_env, add_error):
            ...     return env[node[1]], s
            >>> t, _ = transpose(node, env, s, {}, {}, errors.append, infer_expr)  # noqa
            >>> isinstance(t, TTensor)
            True
            >>> t.dims
            (4, 3)
            >>> errors
            []
            """
            # AST: ("transpose", ("var", "x"))
            operand = node[1]

            operand_type, s = infer_expr(
                operand,
                env,
                s,
                func_env,
                class_env,
                add_error,
            )
            if operand_type is None:
                return None, s

            operand_type = s.apply(operand_type)

            if not isinstance(operand_type, TTensor):
                add_error("Transpose requires a tensor")
                return None, s

            if len(operand_type.dims) != 2:
                add_error(f"Transpose requires a 2D tensor, "
                          f"got {len(operand_type.dims)} dimensions")
                return None, s

            return TTensor((operand_type.dims[1], operand_type.dims[0])), s

        return {"transpose": expr_transpose}

    def forward_rules(self) -> dict:
        """
        Code generation handlers.

        Returns
        -------
        dict
            Dictionary containg code generation handlers.

        Examples
        --------
        >>> from physika.features import ModifierLetters
        >>> rules = ModifierLetters().forward_rules()
        >>> rules.keys()
        dict_keys(['transpose'])
        """

        def emit_transpose(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit a superscript ᵀ used with variable.

            Parameters
            ----------
            node : tuple
                AST node.
            to_expr : Callable
                Recursive codegen function that converts an AST node to a
                Python code string.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism;
                not used directly.

            Returns
            -------
            str
                Pytorch transpose wrapper code.

            Examples
            --------
            >>> from physika.features import ModifierLetters
            >>> rules = ModifierLetters().forward_rules()
            >>> emit = rules["transpose"]
            >>> node = ("transpose", ("var", "A"))
            >>> emit(node, lambda n: n[1])
            'torch.transpose(A, 0, 1)'
            """
            var = node[1][1]
            return f"torch.transpose({var}, 0, 1)"

        return {"transpose": emit_transpose}
