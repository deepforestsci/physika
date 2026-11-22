from physika.elf import ELF
from typing import Callable


def make_parser_rules():
    """
    PLY grammer functions for Physika dictionary.
    """

    def p_type_dict(p):
        """type_spec : DICT LBRACKET type_spec COMMA type_spec RBRACKET"""
        # base dictionary type syntax
        #   Dict[key_type, value_type]
        # Parameters:
        #   p[3] - key type
        #   p[5] - value type
        # Returns:
        #   ("dict_type", key_type_spec, value_type_spec)
        p[0] = ("dict_type", p[3], p[5])

    def p_factor_dict_empty(p):
        """factor : LBRACE RBRACE"""
        # An empty dictionary type
        # Returns:
        #   ("dict", [])
        p[0] = ("dict", [])

    def p_factor_dict(p):
        """factor : LBRACE dict_items RBRACE"""
        # A dictionary syntax with key value
        # pairs
        # Parameters:
        #   p[2] - dictionary items
        # Returns:
        #   ("dict", dict_items)
        p[0] = ("dict", p[2])

    def p_func_factor_dict_empty(p):
        """func_factor : LBRACE RBRACE"""
        # An empty dictionary type (inside function)
        # Returns:
        #   ("dict", [])
        p[0] = ("dict", [])

    def p_func_factor_dict(p):
        """func_factor : LBRACE dict_items RBRACE"""
        # A dictionary syntax with key value (inside function)
        # pairs
        # Parameters:
        #   p[2] - dictionary items
        # Returns:
        #   ("dict", dict_items)
        p[0] = ("dict", p[2])

    def p_dict_items_single(p):
        """dict_items : dict_item"""
        # A single dictionary item.
        # Parameters:
        #   p[1] - parser key-value pair
        # Returns:
        #   Recursive list which returns dict_item.
        p[0] = [p[1]]

    def p_dict_items_multi(p):
        """dict_items : dict_item COMMA dict_items
                  | dict_item NEWLINE dict_items"""
        # multiple dictionary items separated by commas.
        # Parameters:
        #   p[1] - first dictionary item
        #   p[3] - second dictionary item
        # Returns:
        # A single list containing all dictionary items.
        p[0] = [p[1]] + p[3]

    def p_dict_item(p):
        """dict_item : expr COLON expr"""
        # key-value pair (key : value)
        # Parameters:
        #   p[1] - key expression
        #   p[3] - value expression
        # Returns:
        #   (key_expr, value_expr)
        p[0] = (p[1], p[3])

    def p_dict_items_newline(p):
        """dict_items : NEWLINE dict_items
                    | dict_items NEWLINE"""
        # helper parser rules to allow NEWLINE
        # {
        #   key : value,
        #   key : value
        # }
        if len(p) == 3:
            p[0] = p[2] if isinstance(p[2], list) else p[1]

    return [
        p_type_dict, p_factor_dict_empty, p_factor_dict,
        p_func_factor_dict_empty, p_func_factor_dict, p_dict_item,
        p_dict_items_single, p_dict_items_multi, p_dict_items_newline
    ]


class DictionaryFeature(ELF):
    """
    Physika dictionary dtype implmeented as ELF subclass.

    ``DictionaryFeature`` injects rules via ``REGISTRY`` at
    parser and code generator.

    **Parser rules**
    Nine PLY grammer functions (see ``make_parser_rules``) handle
    dictionary declarations, empty dictionary declaration, and also
    support for indentation (BLANK spaces) for declaring key value
    pairs.

    **Forward rules**
    one code-generation handler named as ``emit_dict`` emits for ``dict``
    AST node.

    Physika dictionary dtype is fully differentiable (values), where values
    gets wrapped around ``torch.Tensor``.

    Physika syntax example (see ``examples/physika_dictionary.phyk``)::

        d: Dict[ℝ, ℝ | ℕ] = {
            0: 5.43,
            1: 7.8,
            2: 50
        }

    Examples
    --------
    >>> from physika.lexer import lexer
    >>> from physika.parser import parser, symbol_table
    >>> from physika.utils.ast_utils import build_unified_ast
    >>> from physika.codegen import from_ast_to_torch
    >>> def run_phyk(src):
    ...     symbol_table.clear()
    ...     lexer.lexer.lineno = 1
    ...     ast = build_unified_ast(parser.parse(src, lexer=lexer),
    ...                             symbol_table)
    ...     exec(from_ast_to_torch(ast, print_code=False), {})

    >>> # Physika dictionary example
    >>> src = '''
    ... d: Dict[ℝ, ℝ | ℕ] = {
    ...     0: 5.43,
    ...     1: 7.8,
    ...     2: 50
    ... }
    ... d
    ... '''

    >>> # Execute code and verify outputs
    >>> run_phyk(src)
    {0: 5.43, 1: 7.8, 2: 50} ∈ dict
    """
    name = "dictionary"

    def parser_rules(self) -> list:
        """
        Override ``parser_rules`` handler for new grammar rules.

        Returns
        -------
        list
            List of PLY grammar functions to be injected into
            ``physika.parser``.

        Examples
        --------
        >>> from physika.features import DictionaryFeature
        >>> rules = DictionaryFeature().parser_rules()
        >>> len(rules)
        7
        >>> rules[0].__name__
        'p_type_dict'
        """
        return make_parser_rules()

    def forward_rules(self) -> dict:
        """
        One code-generation handler as ``emit_dict`` emits a complete
        dictionary in key-value format (key : value).

        Returns
        -------
        dict
            Dictionary containg code generation handler.

        Examples
        --------
        >>> from physika.features import DictionaryFeature
        >>> from physika.utils.ast_utils import ast_to_torch_expr
        >>> rules = DictionaryFeature().forward_rules()
        >>> len(rules)
        1

        >>> # physika code:
        >>> # d: Dict[ℝ, ℝ | ℕ] = {0: 5, 1: 7, 2: 3}
        >>> node = (
        ...     "dict",
        ...     [
        ...         (("num", 0), ("num", 5)),
        ...         (("num", 1), ("num", 7)),
        ...         (("num", 2), ("num", 3)),
        ...     ]
        ... )
        >>> rules["dict"](node, ast_to_torch_expr)
        '{0: 5, 1: 7, 2: 3}'
        """

        def emit_dict(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit code for dictionary expression.

            Parameters
            ----------
            node : tuple
                ``("dict", entries)`` where ``entries`` is a list
                of ``(key_expr, value_expr)`` pairs.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit each key and value
                expression.

            Examples
            --------
            >>> from physika.features import DictionaryFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = DictionaryFeature().forward_rules()
            >>> len(rules)
            1

            >>> # physika code:
            >>> # d: Dict[ℝ, ℝ | ℕ] = {0: 5, 1: 7, 2: 3}
            >>> node = (
            ...     "dict",
            ...     [
            ...         (("num", 0), ("num", 5)),
            ...         (("num", 1), ("num", 7)),
            ...         (("num", 2), ("num", 3)),
            ...     ]
            ... )
            >>> rules["dict"](node, ast_to_torch_expr)
            '{0: 5, 1: 7, 2: 3}'
            """
            entries = node[1]

            items = []
            for k, v in entries:
                key = to_expr(k)
                value = to_expr(v)
                items.append(f"{key}: {value}")

            return "{" + ", ".join(items) + "}"

        return {"dict": emit_dict}
