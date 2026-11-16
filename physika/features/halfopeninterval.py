from physika.elf import ELF
from typing import Callable
from physika.utils.ast_utils import emit_for_stmts


def make_parser_rules():
    """
    PLY grammar functions for half open for loops.
    """

    def p_right_open_for_range(p):
        """statement : FOR ID ELEMENTOF LBRACKET func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT for_body DEDENT"""  # noqa
        # Right open interval for loop syntax (top level).
        # Example:
        #   for i \in [0, 4):
        #       total += i
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — for_body statements
        # Returns:
        #   ("for_loop_right_open", var, start_expr, end_expr, loop_body, lineno)  # noqa
        p[0] = ("for_loop_right_open", p[2], p[5], p[7], p[12], p.lineno(1))

    def p_for_statement_right_open_range(p):
        """for_statement : FOR ID ELEMENTOF LBRACKET func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT for_body DEDENT"""  # noqa
        # Right open interval for loop nested syntax (top level).
        # for i \in [start, end):
        # Example:
        #   for i \in [0, 4):
        #       for j \in [0, i):
        #           total += j
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — for_body statements
        # Returns:
        #   ("for_loop_right_open", var, start_expr, end_expr, loop_body, lineno)  # noqa
        p[0] = ("for_loop_right_open", p[2], p[5], p[7], p[12], p.lineno(1))

    def p_left_open_for_range(p):
        """statement : FOR ID ELEMENTOF LPAREN func_expr COMMA func_expr RBRACKET COLON NEWLINE INDENT for_body DEDENT"""  # noqa
        # Left open interval for loop syntax (top level).
        # for i \in (start, end]:
        # Example:
        #   for i \in (0, 4]:
        #       total += i
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — for_body statements
        # Returns:
        #   ("for_loop_left_open", var, start_expr, end_expr, loop_body, lineno)  # noqa
        p[0] = ("for_loop_left_open", p[2], p[5], p[7], p[12], p.lineno(1))

    def p_for_statement_left_open_for_range(p):
        """for_statement : FOR ID ELEMENTOF LPAREN func_expr COMMA func_expr RBRACKET COLON NEWLINE INDENT for_body DEDENT"""  # noqa
        # Left open interval for loop nested syntax (top level).
        # for i \in (start, end]:
        # Example:
        #   for i \in (0, 4]:
        #       for j \in (0, i]:
        #           total += j
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — for_body statements
        # Returns:
        #   ("for_loop_left_open", var, start_expr, end_expr, loop_body, lineno)  # noqa
        p[0] = ("for_loop_left_open", p[2], p[5], p[7], p[12], p.lineno(1))

    def p_func_body_right_open_for_range(p):
        """func_body_stmt : FOR ID ELEMENTOF LBRACKET func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa
        # Right open interval for loop in a function body.
        # for i \in [start, end):
        # Example:
        #   for i \in [0, n):
        #       total += i
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — func_loop_body statements
        # Returns:
        #   ("body_for_right_open", var, start_expr, end_expr, loop_body)
        p[0] = ("body_for_right_open", p[2], p[5], p[7], p[12])

    def p_func_body_left_open_for_range(p):
        """func_body_stmt : FOR ID ELEMENTOF LPAREN func_expr COMMA func_expr RBRACKET COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa
        # Left open interval for loop in a function body.
        # for i \in (start, end]:
        # Example:
        #   for i \in (0, n]:
        #       total += i
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — func_loop_body statements
        # Returns:
        #   ("body_for_left_open", var, start_expr, end_expr, loop_body)
        p[0] = ("body_for_left_open", p[2], p[5], p[7], p[12])

    def p_func_body_stmt_right_open_for_range(p):
        """func_loop_stmt : FOR ID ELEMENTOF LBRACKET func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa
        # Right open interval for loop nested inside function.
        # for i \in [start, end):
        # Example:
        #   for i \in [0, n):
        #       for j \in [0, i):
        #           total += j
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — func_loop_body statements
        # Returns:
        #   ("body_for_right_open", var, start_expr, end_expr, loop_body)
        p[0] = ("body_for_right_open", p[2], p[5], p[7], p[12])

    def p_func_body_stmt_left_open_for_range(p):
        """func_loop_stmt : FOR ID ELEMENTOF LPAREN func_expr COMMA func_expr RBRACKET COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa
        # Left open interval for loop nested inside function.
        # for i \in (start, end]:
        # Example:
        #   for i \in (0, n]:
        #       for j \in (0, i]:
        #           total += j
        # Parameters:
        # p[2] — loop variable name
        # p[6] — start expression
        # p[8] — end expression
        # p[13] — func_loop_body statements
        # Returns:
        #   ("body_for_left_open", var, start_expr, end_expr, loop_body)
        p[0] = ("body_for_left_open", p[2], p[5], p[7], p[12])

    return [
        p_right_open_for_range, p_for_statement_right_open_range,
        p_left_open_for_range, p_for_statement_left_open_for_range,
        p_func_body_right_open_for_range, p_func_body_left_open_for_range,
        p_func_body_stmt_right_open_for_range,
        p_func_body_stmt_left_open_for_range
    ]


class HalfOpenIntervalLoop(ELF):
    """
    Half open interval loop implemented as ELF subclass.

    (a, b] = {x ∈ ℕ | a < x <= b},
    [a, b) = {x ∈ ℕ | a <= x < b}. 

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

    >>> src = '''
    ... for i ∈ [0, 4):
    ...     x = i
    ... print(x)
    ... '''

    >>> run_phyk(src)
    3 ∈ ℝ
    """
    name = "half_open_inteval_loop"

    def lexer_rules(self) -> dict:
        """
        Adds ``ELEMENTOF`` as token with ``t_ELEMENTOF`` as token function.

        Examples
        --------
        >>> from physika.features.halfopeninterval import HalfOpenIntervalLoop
        >>> rules = HalfOpenIntervalLoop().lexer_rules()
        >>> rules["tokens"]
        ['ELEMENTOF']
        >>> rules["reserved"]
        {'physika': 'PHYSIKA'}
        >>> rules["token_funcs"][0].__name__
        't_ELEMENTOF'
        """

        def t_ELEMENTOF(t):
            r"∈"
            return t

        return {
            "reserved": {
                "physika": "PHYSIKA",
            },
            "tokens": ["ELEMENTOF"],
            "token_funcs": [t_ELEMENTOF]
        }

    def parser_rules(self) -> list:
        """
        Override ``parser_rules`` handler for new grammer rules.

        Returns
        -------
        list
            List of PLY grammer functions to be injected
            into ``physika.parser``.

        Examples
        --------
        >>> from physika.features import HalfOpenIntervalLoop
        >>> rules = HalfOpenIntervalLoop().parser_rules()
        >>> len(rules)
        8
        >>> rules[0].__name__
        'p_right_open_for_range'
        """
        return make_parser_rules()

    def forward_rules(self) -> dict:
        """
        Code generation handler to emit both left-open and right-open
        code.

        Returns
        -------
        dict
            Dictionary containg code generation handlers.

        Examples
        --------
        >>> from physika.features import HalfOpenIntervalLoop
        >>> rules = HalfOpenIntervalLoop().forward_rules()
        >>> len(rules)
        2
        >>> rules["for_loop_right_open"].__name__
        'emit_loop_right_open'
        >>> rules["for_loop_left_open"].__name__
        'emit_loop_left_open'
        """

        def emit_loop_right_open(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit code for ``for_loop_right_open`` AST node.

            Parameters
            ----------
            node : tuple
                AST node of the form
                ``("for_loop_right_open", loop_var, start_expr, end_expr, body_stmts, lineno)``.  # noqa
            to_expr : Callable
                Recursive codegen function that converts an AST expression to a
                Python code string.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism;
                not used directly.

            Returns
            -------
            str
                Python for loop code using ``range(start, end)``, where the
                start is included and the end is excluded.

            Examples
            --------
            >>> node = (
            ...     "for_loop_right_open", "i",
            ...     ("num", 0), ("num", 4), [], 1
            ... )
            >>> emit(node, lambda n: str(n[1]) if n[0] == "num" else n[1])
            'for i in range(int(0), int(4)):'
            """
            loop_var = node[1]
            start_code = to_expr(node[2])
            end_code = to_expr(node[3])
            body_stmts = node[4]

            indent = ctx.get("indent", 0)
            prefix = " " * indent

            lines = [
                f"{prefix}for {loop_var} in range(int({start_code}), int({end_code})):"  # noqa
            ]

            # Generate the loop body.
            body_code = emit_for_stmts(body_stmts, indent + 4, loop_var)
            lines.extend(body_code)

            return "\n".join(lines)

        def emit_loop_left_open(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit code for ``for_loop_left_open`` AST node.

            Parameters
            ----------
            node : tuple
                AST node of the form
                ``("for_loop_left_open", loop_var, start_expr, end_expr, body_stmts, lineno)``.  # noqa
            to_expr : Callable
                Recursive codegen function that converts an AST expression to a
                Python code string.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism;
                not used directly.

            Returns
            -------
            str
                Python for loop code using ``range(start + 1, end + 1)``, where
                the start is excluded and the end is included.

            Examples
            --------
            >>> node = (
            ...     "for_loop_left_open", "i",
            ...     ("num", 0), ("num", 4), [], 1
            ... )
            >>> emit(node, lambda n: str(n[1]) if n[0] == "num" else n[1])
            'for i in range(int(0) + 1, int(4) + 1):'
            """
            loop_var = node[1]
            start_code = to_expr(node[2])
            end_code = to_expr(node[3])
            body_stmts = node[4]

            indent = ctx.get("indent", 0)
            prefix = " " * indent

            lines = [
                f"{prefix}for {loop_var} in range(int({start_code}) + 1, int({end_code}) + 1):"  # noqa
            ]

            lines += emit_for_stmts(body_stmts, indent + 4, loop_var)
            return "\n".join(lines)

        return {
            "for_loop_right_open": emit_loop_right_open,
            "for_loop_left_open": emit_loop_left_open
        }
