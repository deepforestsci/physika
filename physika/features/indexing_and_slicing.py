from physika.elf import ELF
from typing import Callable


def make_parser_rules():
    """
    PLY grammar functions for Physika indexing and slicing syntax.
    """

    def p_factor_index(p):
        """factor : ID LBRACKET NUMBER RBRACKET"""
        # Numeric literal indexing of array at program level
        # Example:
        # arr[0]
        # arr[2]
        # Parameters:
        # p[1] — array name
        # p[3] — numeric literal index
        # Returns:
        #   ("index", name, ("num", index))
        p[0] = ("index", p[1], ("num", p[3]))

    def p_factor_index_var(p):
        """factor : ID LBRACKET ID RBRACKET"""
        # Variable indexing of array at program level
        # Example:
        # arr[i]
        # arr[m]
        # Parameters:
        # p[1] — array name
        # p[3] — variable name used as index
        # Returns:
        #   ("index", name, ("var", index))
        p[0] = ("index", p[1], ("var", p[3]))

    def p_factor_indexN(p):
        """factor : ID LBRACKET multi_index_list RBRACKET"""
        # N-dimensional comma indexing at program level.
        # Reuses the multi_index_list shared with function bodies,
        # which requires at least one comma.
        # Example:
        #   u0[1, 2]
        #   T[0, 1, 2]
        # Parameters:
        # p[1] — array name
        # p[3] — list of index expressions built by multi_index_list
        # Returns:
        #   ("indexN", name, [idx_expr, ...])
        p[0] = ("indexN", p[1], p[3])

    def p_func_factor_index(p):
        """func_factor : ID LBRACKET func_expr RBRACKET"""
        # Tensor indexing: W[i]
        p[0] = ("index", p[1], p[3])

    def p_multi_index_item_index(p):
        """multi_index_item : func_expr"""
        p[0] = ("index_item", p[1])

    def p_multi_index_item_slice(p):
        """multi_index_item : func_expr COLON func_expr
                            | func_expr COLON
                            | COLON func_expr
                            | COLON"""
        if len(p) == 4:
            p[0] = ("slice_item", p[1], p[3])
        elif len(p) == 3 and p[1] == ":":
            p[0] = ("slice_item", None, p[2])
        elif len(p) == 3:
            p[0] = ("slice_item", p[1], None)
        else:
            p[0] = ("slice_item", None, None)

    def p_multi_index_list_single(p):
        """multi_index_list : multi_index_item"""
        p[0] = [p[1]]

    def p_multi_index_list_base(p):
        """multi_index_list : multi_index_item COMMA multi_index_item"""
        # Base case: 2 comma-separated index expressions.
        # Requires at least one COMMA, so there is no conflict with 1
        # index rule.
        # Example:
        #   A[i, k]
        # Parameters:
        # p[1] — first index expression
        # p[3] — second index expression
        # Returns:
        #   [p[1], p[3]]
        p[0] = [p[1], p[3]]

    def p_multi_index_list_extend(p):
        """multi_index_list : multi_index_list COMMA multi_index_item"""
        # Extend an existing index list by one more dimension.
        # Example:
        #   T[i, j, k]
        # Parameters:
        # p[1] — existing index list
        # p[3] — next index expression
        # Returns:
        #   p[1] + [p[3]]
        p[0] = p[1] + [p[3]]

    def p_func_factor_indexN(p):
        """func_factor : ID LBRACKET multi_index_list RBRACKET"""
        # N-dimensional comma indexing inside a function body.
        # Example:
        #   A[i, k]
        #   T[i, j, k]
        # Parameters:
        # p[1] — array name
        # p[3] — list of index expressions (length ≥ 2)
        # Returns:
        #   ("indexN", name, [idx_exprs])
        p[0] = ("indexN", p[1], p[3])

    def p_for_statement_index_assign_nd(p):
        """for_statement : ID LBRACKET loop_index_list RBRACKET EQUALS func_expr NEWLINE"""  # noqa
        # nd Indexed assignment inside top-level for-body
        # Example:
        # # 1d array
        # for i:
        #   arr1d[i] = 1
        # # 2d array
        # for i:
        #   for j:
        #       arr2d[i, j] = 1
        # Parameters:
        # p[1] — array name
        # p[3] — list of index expressions
        # p[5] — right hand side expression
        # Returns:
        #   ("for_index_assign_nd", arr_name, [idx_exprs], rhs)
        p[0] = ("for_index_assign_nd", p[1], p[3], p[6])

    def p_statement_index_assign(p):
        """statement : ID LBRACKET ID RBRACKET EQUALS expr NEWLINE
                    | ID LBRACKET NUMBER RBRACKET EQUALS expr NEWLINE"""
        # Indexed assignment of array at program level
        # Example:
        # arr1d = [1, 2, 3]
        # arr1d[1] = 2
        # m: R = 1
        # arr1d[m] = 3
        # Parameters:
        # p[1] — array name
        # p[3] — index expression/number
        # p[5] — right hand side expression/number
        p[0] = ("index_assign", p[1], p[3], p[6], p.lineno(1))

    def p_statement_index_assign_nd(p):
        """statement : ID LBRACKET multi_index_list RBRACKET EQUALS expr NEWLINE"""  # noqa
        # Indexed assignment of array at program level
        # Example:
        # arr2d = [[1, 1], [1, 1]]
        # arr2d[1, 1] = 2
        # m: R = 1
        # arr1d[m, m] = 3
        # Parameters:
        # p[1] — array name
        # p[3] — index expression/number
        # p[5] — right hand side expression/number
        p[0] = ("index_assign_nd", p[1], p[3], p[6], p.lineno(1))

    def p_loop_index_list_single(p):
        """loop_index_list : func_expr"""
        # Single-element index list.
        # Base case for nD subscript accumulation.
        # Parameters:
        # p[1] — index expression
        # Returns:
        #   [p[1]]
        p[0] = [("index_item", p[1])]

    def p_loop_index_list_multi(p):
        """loop_index_list : loop_index_list COMMA func_expr"""
        # Extend index list.
        # Parameters:
        # p[1] — existing index list
        # p[3] — next index expression
        # Returns:
        #   p[1] + [p[3]]
        p[0] = p[1] + [("index_item", p[3])]

    def p_func_loop_stmt_index_pluseq(p):
        """func_loop_stmt : ID LBRACKET loop_index_list RBRACKET PLUSEQ func_expr NEWLINE"""  # noqa: E501
        # N-dimensional indexed accumulation statement inside a loop body.
        # Example:
        #   C[i, j]    += A[i, k] * B[k, j]
        #   T[i, j, l] += A[i, k] * B[k, j, l]
        # Parameters:
        # p[1] — tensor name
        # p[3] — list of index expressions
        # p[6] — right-hand side expression
        # Returns:
        #   ("loop_index_pluseq", name, [idx_exprs], rhs)
        p[0] = ("loop_index_pluseq", p[1], p[3], p[6])

    def p_func_loop_stmt_index_assign_nd(p):
        """func_loop_stmt : ID LBRACKET loop_index_list RBRACKET EQUALS func_expr NEWLINE"""  # noqa
        # nd Indexed assignment inside a loop body
        # Example:
        # # 1d array
        # for i:
        #   arr1d[i] = 1
        # # 2d array
        # for i:
        #   for j:
        #       arr2d[i, j] = 1
        # Parameters:
        # p[1] — array name
        # p[3] — list of index expressions
        # p[5] — right hand side expression
        # Returns:
        #   ("loop_index_assign_nd", arr_name, [idx_exprs], rhs)
        p[0] = ("loop_index_assign_nd", p[1], p[3], p[6])

    def p_func_body_stmt_index_assign(p):
        """func_body_stmt : ID LBRACKET func_expr RBRACKET EQUALS func_expr NEWLINE"""  # noqa
        # Indexed assignment of array inside function body
        # Example:
        # def update_1d_array(x: R[m]): R[m]:
        #   x[1] = 3
        #   return x
        # arr1d = [1, 2, 3]
        # update_1d_array(arr1d)
        # Parameters:
        # p[1] — array name
        # p[3] — index expression/number
        # p[5] — right hand side expression/number
        p[0] = ("body_index_assign", p[1], p[3], p[6])

    def p_func_body_stmt_index_assign_nd(p):
        """func_body_stmt : ID LBRACKET multi_index_list RBRACKET EQUALS func_expr NEWLINE"""  # noqa
        # nd Indexed assignment of array inside function body
        # Example:
        # def update_2d_array(x: R[m, n]): R[m, n]:
        #   x[1, 1] = 3
        #   return x
        # 2d_array: R[2 , 2] = [[1, 1], [1, 1]]
        # update_2d_array(2d_array)
        # Parameters:
        # p[1] — array name
        # p[3] — index list of numbers/expressions
        # p[5] — right hand side expression/number
        p[0] = ("body_index_assign_nd", p[1], p[3], p[6])

    return [
        p_factor_index, p_factor_index_var, p_factor_indexN,
        p_func_factor_index, p_multi_index_item_index,
        p_multi_index_item_slice, p_multi_index_list_single,
        p_multi_index_list_base, p_multi_index_list_extend,
        p_func_factor_indexN, p_for_statement_index_assign_nd,
        p_statement_index_assign, p_statement_index_assign_nd,
        p_loop_index_list_single, p_loop_index_list_multi,
        p_func_loop_stmt_index_pluseq, p_func_loop_stmt_index_assign_nd,
        p_func_body_stmt_index_assign, p_func_body_stmt_index_assign_nd
    ]


class IndexingandSlicing(ELF):
    """
    Physika Indexing and Slicing support implemented as ELF subclass.

    ``IndexingandSlicing`` injects rules via ``REGISTRY`` at parser,
    type checker and code generator.

    **Parser rules**
    Nineteen PLY grammer functions (see ``make_parser_rules``) which
    handles indexing and slicing.

    **Forward rules**
    Nine code-generation handlers are defined which handles
    N-dimensional indexing and slicing rules for Top-level program,
    Function-level programs, rules inside loops and also support
    indexing/slicing assignment.

    Physika syntax example (see ``examples/example_slicing.phyk``)::

        y: ℝ[2, 2] = [
            [1, 2],
            [3, 4]
        ]
        y[:, 0]
        y[0, 1]
        y[1, :]

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

    >>> # Physika Indexing and Slicing example
    >>> src = '''
    ... x: ℝ[5] = [1, 2, 3, 4, 5]
    ... x[0]
    ... x[1:3]
    ... x[3:]
    ... '''

    >>> # Execute code and verify outputs
    >>> run_phyk(src)
    1 ∈ ℝ
    [2, 3] ∈ ℝ[2]
    [4, 5] ∈ ℝ[2]
    """
    name = "Indexing_and_Slicing"

    def parser_rules(self) -> list:
        """
        Override ``parser_rules`` handler for new grammer rules.

        Nineteen PLY grammer functions (see ``make_parser_rules``) which
        handles indexing and slicing.

        Returns
        -------
        list
            List of PLY grammer functions to be injected into
            ``physika.parser``.

        Examples
        --------
        >>> from physika.features import IndexingandSlicing
        >>> rules = IndexingandSlicing().parser_rules()
        >>> len(rules)
        19
        >>> rules[0].__name__
        'p_factor_index'
        """
        return make_parser_rules()

    def forward_rules(self) -> dict:
        """
        Nine code-generation handlers are defined which handles
        N-dimensional indexing and slicing rules for Top-level
        program, Function-level programs, rules inside loops and
        also support indexing/slicing assignment.

        Returns
        -------
        dict
            Dictionary containing code generation handlers.

        Examples
        --------
        >>> from physika.features import IndexingandSlicing
        >>> from physika.utils.ast_utils import ast_to_torch_expr
        >>> rules = IndexingandSlicing().forward_rules()

        >>> # Physika code:
        >>> # x: ℝ[5] = [1, 2, 3, 4, 5]
        >>> # x[1]
        >>> node = ("index", "x", ("num", 1))
        >>> rules["index"](node, ast_to_torch_expr)
        'x[int(1)]'
        """

        def emit_index(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit code for single-dimensional tensor indexing.

            Parameters
            ----------
            node : tuple
                ``("index", array_name, index_expr)``.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit the index expression.

            Returns
            -------
            str
                Pytorch indexing expression.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = ("index", "x", ("num", 1))
            >>> rules["index"](node, ast_to_torch_expr)
            'x[int(1)]'
            """
            var_name = node[1]
            idx = to_expr(node[2])
            return f"{var_name}[int({idx})]"

        def emit_indexN(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit code for N-dimensinal indexing and slicing.

            Parameters
            ----------
            node : tuple
                ``("indexN", array_name, index_list)`` where
                ``index_list`` contains ``"index_item"`` and/or
                ``"slice_item"`` AST nodes.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit index and slice
                expressions.

            Returns
            -------
            str
                Pytorch indexing or slicing expression.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "indexN",
            ...     "A",
            ...     [
            ...         ("index_item", ("num", 1)),
            ...         ("index_item", ("num", 2)),
            ...     ],
            ... )
            >>> rules["indexN"](node, ast_to_torch_expr)
            'A[int(1), int(2)]'
            """
            arr = node[1]
            parts = []
            for item in node[2]:
                if item[0] == "index_item":
                    idx = to_expr(item[1])
                    parts.append(f"int({idx})")
                elif item[0] == "slice_item":
                    start = (to_expr(item[1]) if item[1] is not None else "")
                    end = (to_expr(item[2]) if item[2] is not None else "")
                    parts.append(f"{start}:{end}")
            return f"{arr}[{', '.join(parts)}]"

        def emit_for_index_assign_nd(node: tuple, to_expr: Callable,
                                     **ctx) -> str:
            """
            Emit code for indexed tensor assignment inside loop body.

            Parameters
            ----------
            node : tuple
                ``("loop_index_assign_nd", array_name, index_list, rhs_expr)``
                where ``index_list`` contains the index expressions for each
                tensor dimension.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit index expressions and the
                right-hand side expression.
            
            Returns
            -------
            str
                PyTorch indexed assignment statement.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "loop_index_assign_nd",
            ...     "A",
            ...     [("var", "i"), ("var", "j")],
            ...     ("num", 5),
            ... )
            >>> rules["loop_index_assign_nd"](node, ast_to_torch_expr, current_loop_var={"i", "j"})  # noqa
            'A[int(i), int(j)] = 5'
            """
            _, arr_name, idx_list, rhs_expr = node
            indices = ", ".join(f"int({to_expr(idx)})" for idx in idx_list)
            rhs_code = to_expr(rhs_expr)
            return f"{arr_name}[{indices}] = {rhs_code}"

        def emit_index_assign(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit code for one-dimensional indexed assignment.

            Parameters
            ----------
            node : tuple
                ``("index_assign", array_name, index_expr, value_expr)``.
                The index expression may be either an AST node or a variable
                name represented as a string.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit the index and value
                expressions.

            Returns
            -------
            str
                PyTorch indexed assignment statement.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "index_assign",
            ...     "x",
            ...     ("num", 1),
            ...     ("num", 10),
            ... )
            >>> rules["index_assign"](node, ast_to_torch_expr)
            'x[int(1)] = 10'
            """
            name, idx, val = node[1], node[2], node[3]
            idx_code = to_expr(
                ("var", idx)) if isinstance(idx, str) else to_expr(idx)
            return f"{name}[int({idx_code})] = {to_expr(val)}"

        def emit_index_assign_nd(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit code for N-dimensional indexed or sliced assignment.

            Parameters
            ----------
            node : tuple
                ``("index_assign_nd", array_name, index_list, value_expr)``
                where ``index_list`` contains ``"index_item"`` and/or
                ``"slice_item"`` AST nodes.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit index, slice, and
                value expressions.

            Returns
            -------
            str
                PyTorch indexed or sliced assignment statement.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "index_assign_nd",
            ...     "A",
            ...     [
            ...         ("index_item", ("num", 1)),
            ...         ("index_item", ("num", 2)),
            ...     ],
            ...     ("num", 5),
            ... )
            >>> rules["index_assign_nd"](node, ast_to_torch_expr)
            'A[int(1), int(2)] = 5'
            """
            name, indices, val = node[1], node[2], node[3]
            parts = []
            for item in indices:
                if item[0] == "index_item":
                    idx = to_expr(item[1])
                    parts.append(f"int({idx})")

                elif item[0] == "slice_item":
                    start = (to_expr(item[1]) if item[1] is not None else "")
                    end = (to_expr(item[2]) if item[2] is not None else "")
                    parts.append(f"{start}:{end}")
            idx_code = ", ".join(parts)
            return f"{name}[{idx_code}] = {to_expr(val)}"

        def emit_loop_index_pluseq(node: tuple, to_expr: Callable,
                                   **ctx) -> str:
            """
            Emit code for indexed plus equals ``+=`` assignment
            inside loop body.

            Parameters
            ----------
            node : tuple
                ``("loop_index_pluseq", array_name, index_list, rhs_expr)``
                where ``index_list`` contains ``"index_item"`` and/or
                ``"slice_item"`` AST nodes.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit index, slice, and
                right-hand side expressions.

            Returns
            -------
            str
                PyTorch indexed ``+=`` assignment statement.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "loop_index_pluseq",
            ...     "C",
            ...     [
            ...         ("index_item", ("var", "i")),
            ...         ("index_item", ("var", "j")),
            ...     ],
            ...     ("var", "v"),
            ... )
            >>> rules["loop_index_pluseq"](node, ast_to_torch_expr)
            'C[int(i), int(j)] += v'
            """
            _, arr_name, idx_list, rhs = node
            parts = []
            for item in idx_list:
                if item[0] == "index_item":
                    idx = to_expr(item[1])
                    parts.append(f"int({idx})")
                elif item[0] == "slice_item":
                    start = (to_expr(item[1]) if item[1] is not None else "")
                    end = (to_expr(item[2]) if item[2] is not None else "")
                    parts.append(f"{start}:{end}")
            rhs_code = to_expr(rhs)
            return f"{arr_name}[{', '.join(parts)}] += {rhs_code}"

        def emit_loop_index_assign_nd(node: tuple, to_expr: Callable,
                                      **ctx) -> str:
            """
            Emit code for N-dimensional indexed assignment inside loop body.

            Parameters
            ----------
            node : tuple
                ``("loop_index_assign_nd", array_name, index_list, rhs_expr)``
                where ``index_list`` contains the index expressions for each
                tensor dimension.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit index expressions and the
                right-hand side expression.
            **ctx
                Additional context passed to the emitter. The
                ``current_loop_var`` entry is forwarded to ``to_expr`` so
                loop variables are emitted correctly.

            Returns
            -------
            str
                PyTorch indexed assignment statement.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "loop_index_assign_nd",
            ...     "A",
            ...     [("var", "i"), ("var", "j")],
            ...     ("var", "v"),
            ... )
            >>> rules["loop_index_assign_nd"](
            ...     node,
            ...     ast_to_torch_expr,
            ...     current_loop_var={"i", "j"},
            ... )
            'A[int(i), int(j)] = v'
            """
            _, arr_name, idx_list, rhs = node
            indices = ", ".join(
                f"int({to_expr(idx, current_loop_var = ctx['current_loop_var'])})"  # noqa
                for idx in idx_list)
            rhs_code = to_expr(rhs, current_loop_var=ctx["current_loop_var"])
            return f"{arr_name}[{indices}] = {rhs_code}"

        def emit_body_index_assign(node: tuple, to_expr: Callable,
                                   **ctx) -> str:
            """
            Emit code for one-dimensional indexed assignment
            inside function body.

            Parameters
            ----------
            node : tuple
                ``("body_index_assign", array_name, index_expr, value_expr)``.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit the index and value
                expressions.

            Returns
            -------
            str
                PyTorch indexed assignment statement.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "body_index_assign",
            ...     "x",
            ...     ("num", 1),
            ...     ("num", 42),
            ... )
            >>> rules["body_index_assign"](node, ast_to_torch_expr)
            'x[int((1)] = 42'
            """
            _, name, idx, val = node
            idx_code = to_expr(idx)
            val_code = to_expr(val)
            return f"{name}[int({idx_code})] = {val_code}"

        def emit_body_index_assign_nd(node: tuple, to_expr: Callable,
                                      **ctx) -> str:
            """
            Emit code for N-dimensional indexed or sliced assignment
            inside function body.

            Parameters
            ----------
            node : tuple
                ``("body_index_assign_nd", array_name, index_list, value_expr)``  # noqa
                where ``index_list`` contains ``"index_item"`` and/or
                ``"slice_item"`` AST nodes.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit index, slice, and
                value expressions.

            Returns
            -------
            str
                PyTorch indexed or sliced assignment statement.

            Examples
            --------
            >>> from physika.features import IndexingandSlicing
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = IndexingandSlicing().forward_rules()
            >>> node = (
            ...     "body_index_assign_nd",
            ...     "A",
            ...     [
            ...         ("index_item", ("num", 1)),
            ...         ("index_item", ("num", 2)),
            ...     ],
            ...     ("num", 5),
            ... )
            >>> rules["body_index_assign_nd"](node, ast_to_torch_expr)
            'A[int(1), int(2)] = 5'
            """
            _, name, indices, val = node
            parts = []
            for item in indices:
                if item[0] == "index_item":
                    idx = to_expr(item[1])
                    parts.append(f"int({idx})")

                elif item[0] == "slice_item":
                    start = to_expr(item[1]) if item[1] is not None else ""
                    end = to_expr(item[2]) if item[2] is not None else ""
                    parts.append(f"{start}:{end}")
            idx_code = ", ".join(parts)
            val_code = to_expr(val)
            return f"{name}[{idx_code}] = {val_code}"

        return {
            "index": emit_index,
            "indexN": emit_indexN,
            "index_assign": emit_index_assign,
            "index_assign_nd": emit_index_assign_nd,
            "for_index_assign_nd": emit_for_index_assign_nd,
            "loop_index_pluseq": emit_loop_index_pluseq,
            "loop_index_assign_nd": emit_loop_index_assign_nd,
            "body_index_assign": emit_body_index_assign,
            "body_index_assign_nd": emit_body_index_assign_nd
        }
