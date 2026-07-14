from physika.elf import ELF
from typing import Callable


def make_parser_rules():

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

    return [
        p_factor_index, p_factor_index_var, p_factor_indexN,
        p_func_factor_index, p_multi_index_item_index,
        p_multi_index_item_slice, p_multi_index_list_single,
        p_multi_index_list_base, p_multi_index_list_extend,
        p_func_factor_indexN, p_for_statement_index_assign_nd
    ]


class Indexing(ELF):
    """
    """
    name = "Indexing"

    def parser_rules(self) -> list:
        return make_parser_rules()

    def forward_rules(self) -> dict:

        def emit_index(node: tuple, to_expr: Callable, **ctx) -> str:
            var_name = node[1]
            idx = to_expr(node[2])
            return f"{var_name}[int({idx})]"

        def emit_indexN(node: tuple, to_expr: Callable, **ctx) -> str:
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
            _, arr_name, idx_list, rhs_expr = node
            indices = ", ".join(f"int({to_expr(idx)})" for idx in idx_list)
            rhs_code = to_expr(rhs_expr)
            return f"{arr_name}[{indices}] = {rhs_code}"

        return {
            "index": emit_index,
            "indexN": emit_indexN,
            "for_index_assign_nd": emit_for_index_assign_nd
        }
