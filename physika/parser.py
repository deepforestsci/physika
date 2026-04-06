import ply.yacc as yacc
from physika.lexer import tokens  # noqa: F401
from physika.utils.parser_utils import find_indexed_arrays

symbol_table: dict[str, dict] = {}
print_separator: bool = False

# PARSER
precedence = (
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "DIVIDE"),
    ("left", "MATMUL"),
    ("right", "POWER"),
    (
        "left", "LBRACKET"
    ),  # parse [ before doing arithmetic operations for multiple iter indexing
    ("right", "EQUALS"),
)


# Program
def p_program(p):
    """program : statements"""
    p[0] = p[1]


def p_statements_multi(p):
    """statements : statements statement"""
    p[0] = p[1] + ([] if p[2] is None else [p[2]])


def p_statements_single(p):
    """statements : statement"""
    p[0] = [] if p[1] is None else [p[1]]


# Types
def p_type_scalar(p):
    """type_spec : TYPE"""
    if p[1] == "ℤ":
        p[0] = "ℤ"
    elif p[1] == "ℕ":
        p[0] = "ℕ"
    else:
        p[0] = "ℝ"


def p_type_function(p):
    """type_spec : TYPE ARROW TYPE"""
    # Function type: R → R
    p[0] = ("func_type", p[1], p[3])


def p_type_tangent(p):
    """type_spec : TANGENT ID TYPE"""
    # T_x M notation for tangent space at point x of manifold M
    p[0] = ("tangent", p[2], p[3])


def p_type_tensor(p):
    """type_spec : TYPE LBRACKET dimension_list RBRACKET"""
    p[0] = ("tensor", p[3])


def p_dimension_list_single(p):
    """dimension_list : dimension_spec"""
    p[0] = [p[1]]


def p_dimension_list_multi(p):
    """dimension_list : dimension_spec COMMA dimension_list"""
    p[0] = [p[1]] + p[3]


def p_dimension_contravariant(p):
    """dimension_spec : PLUS NUMBER"""
    p[0] = (int(p[2]), "contravariant")


def p_dimension_covariant(p):
    """dimension_spec : MINUS NUMBER"""
    p[0] = (int(p[2]), "covariant")


def p_dimension_invariant(p):
    """dimension_spec : NUMBER"""
    p[0] = (int(p[1]), "invariant")


def p_dimension_invariant_id(p):
    """dimension_spec : ID"""
    # Symbolic dimension variable (e.g. M, K, hidden) — kept as string
    p[0] = (p[1], "invariant")


def p_dimension_type_as_symbol(p):
    """dimension_spec : TYPE"""
    # N lexes as TYPE(ℕ); map back to the ASCII letter so user intent is
    # preserved
    mapping = {"ℕ": "N", "ℝ": "R", "ℤ": "Z"}
    p[0] = (mapping.get(p[1], p[1]), "invariant")


# Statements
def p_statement_function(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT"""  # noqa: E501
    name, params, return_type, body = p[2], p[4], p[7], p[12]

    func_def = {
        "params": params,
        "return_type": return_type,
        "body": body,
        "has_loop": False,
        "statements": []
    }

    symbol_table[name] = {"type": "function", "value": func_def}
    p[0] = ("func_def", name)


def p_statement_function_with_body(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT func_body_stmts RETURN func_expr NEWLINE DEDENT"""  # noqa: E501
    # def funcname(params): return_type:
    #     stmt1
    #     stmt2
    #     return expr
    name = p[2]
    params = p[4]
    return_type = p[7]
    body_stmts = p[11]
    final_expr = p[13]

    func_def = {
        "params": params,
        "return_type": return_type,
        "body": final_expr,
        "has_loop": False,
        "statements": body_stmts
    }

    symbol_table[name] = {"type": "function", "value": func_def}
    p[0] = ("func_def", name)


def p_statement_function_body_only(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT func_body_stmts DEDENT"""  # noqa: E501
    # Function with return statements that live inside if/else branches.
    # Used when every code path ends with a return inside a conditional
    # Example:
    #   def f(x: ℝ) : ℝ:
    #       if x > 0:
    #           return x
    #       else:
    #           return -x

    # Parameters:
    # p[2]  — function name (ID)
    # p[4]  — parameter list [(name, type)]
    # p[7]  — return type
    # p[11] — list of func_body_stmt nodes
    # (body_assign, body_if_else_return, etc.)
    # Returns:
    #  ("func_def", name) + symbol_table entry

    name = p[2]
    params = p[4]
    return_type = p[7]
    body_stmts = p[11]

    func_def = {
        "params": params,
        "return_type": return_type,
        "body":
        None,  # No final return expression; returns are inside body stmts
        "has_loop": False,
        "statements": body_stmts
    }

    symbol_table[name] = {"type": "function", "value": func_def}
    p[0] = ("func_def", name)


def p_func_body_stmts_single(p):
    """func_body_stmts : func_body_stmt"""
    p[0] = [p[1]] if p[1] else []


def p_func_body_stmts_multi(p):
    """func_body_stmts : func_body_stmts func_body_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])


def p_func_body_stmt_assign(p):
    """func_body_stmt : ID EQUALS func_expr NEWLINE"""
    # Simple assignment: x = expr
    p[0] = ("body_assign", p[1], p[3])


def p_func_body_stmt_decl(p):
    """func_body_stmt : ID COLON type_spec EQUALS func_expr NEWLINE"""
    # Typed declaration: x : R = expr
    p[0] = ("body_decl", p[1], p[3], p[5])


def p_func_body_stmt_zeros_decl(p):
    """func_body_stmt : ID COLON type_spec NEWLINE"""
    # Type annotation for an accumulation target.
    # Example:
    # A : ℝ[m, n]

    p[0] = ("body_zeros_decl", p[1], p[3])


def p_func_body_stmt_tuple_unpack(p):
    """func_body_stmt : ID COMMA ID EQUALS func_expr NEWLINE"""
    # Tuple unpacking: a, b = expr
    p[0] = ("body_tuple_unpack", [p[1], p[3]], p[5])


def p_func_body_stmt_tuple_unpack_three(p):
    """func_body_stmt : ID COMMA ID COMMA ID EQUALS func_expr NEWLINE"""
    # Tuple unpacking: a, b, c = expr
    p[0] = ("body_tuple_unpack", [p[1], p[3], p[5]], p[7])


def p_func_body_stmt_empty(p):
    """func_body_stmt : NEWLINE"""
    p[0] = None


def p_func_body_stmt_if_return(p):
    """func_body_stmt : IF condition COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT"""  # noqa: E501
    # Early-return, no else branch.
    # if cond:
    #   return expr

    # Parameters:
    # p[2]  — condition node (cond_gt, cond_lt, …)
    # p[7]  — return expression
    # Returns:
    #  ("body_if_return", cond, return_expr)
    p[0] = ("body_if_return", p[2], p[7])


def p_func_body_stmt_if_else_return(p):
    """func_body_stmt : IF condition COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT ELSE COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT"""  # noqa: E501
    # Both branches return immediately.

    # Parameters:
    # p[2]  — condition node
    # p[7]  — then-branch return expression
    # p[15] — else-branch return expression
    # Returns:
    #  ("body_if_else_return", cond, then_expr, else_expr)
    p[0] = ("body_if_else_return", p[2], p[7], p[15])


def p_func_body_stmt_if_else(p):
    """func_body_stmt : IF condition COLON NEWLINE INDENT func_body_stmts DEDENT ELSE COLON NEWLINE INDENT func_body_stmts DEDENT"""  # noqa: E501
    # if/else block with statements in both branches.
    # Used when branches assign to a variable and the return is at the end of
    # the function.
    # Example:
    #   if x > 0:
    #       y = x * x
    #   else:
    #       y = -x

    # Parameters:
    # p[2]  — condition node
    # p[6]  — then-branch statement list
    # p[12] — else-branch statement list
    # Returns:
    #  ("body_if_else", cond, then_stmts, else_stmts)
    p[0] = ("body_if_else", p[2], p[6], p[12])


def p_func_body_stmt_if_only(p):
    """func_body_stmt : IF condition COLON NEWLINE INDENT func_body_stmts DEDENT"""  # noqa: E501
    # One-sided conditional: executes then_stmts only when cond is True.
    # Example:
    #   if y < -1:
    #       y = -1
    # Parameters:
    # p[2] — condition node
    # p[6] — then-branch statement list
    # Returns:
    #  ("body_if", cond, then_stmts)
    p[0] = ("body_if", p[2], p[6])


# Conditions:
#   Boolean comparisons between two func_expr values.
#   Each rule produces a 3-tuple: (tag, left_expr, right_expr).
#   condition_to_expr() in ast_utils.py converts these to Python
#   operator strings.

# Parameters:
#   p[1] — left-hand expression
#   p[3] — right-hand expression.


def p_loop_var_list_single(p):
    """loop_var_list : ID"""
    # Single loop variable: for i.
    # Parameters:
    # p[1] — loop variable name
    # Returns:
    #  [name]
    p[0] = [p[1]]


def p_loop_var_list_multi(p):
    """loop_var_list : loop_var_list ID"""
    # Append next loop variable: for i j k.
    # Parameters:
    # p[1] — existing variable list
    # p[2] — next variable name
    # Returns:
    #  [name, ...]
    p[0] = p[1] + [p[2]]


def p_func_body_stmt_for_accum(p):
    """func_body_stmt : FOR loop_var_list COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa: E501
    # Accumulation loop with multiple loop variables.
    # Example:
    # A : ℝ[2, 2] = [[1, 2], [3, 4]]
    # B : ℝ[2, 2] = [[0, 1], [1, 0]]
    # C : ℝ[2, 2]
    # for i j k:
    #     C[i, j] += A[i, k] * B[k, j]
    # Parameters:
    # p[2] — loop variable list [i, j, k]
    # p[6] — loop body statements
    # Returns:
    #  ("body_for_accum", loop_vars, loop_body)
    loop_vars = p[2]
    loop_body = p[6]
    p[0] = ("body_for_accum", loop_vars, loop_body)


def p_func_body_stmt_for_range(p):
    """func_body_stmt : FOR ID COLON TYPE LPAREN func_expr RPAREN NEWLINE INDENT func_loop_body DEDENT
                      | FOR ID COLON TYPE LPAREN func_expr RPAREN COLON NEWLINE INDENT func_loop_body DEDENT
                      | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN NEWLINE INDENT func_loop_body DEDENT
                      | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa: E501
    # Explicit range for loop in a function body.
    # for i: ℕ(end) or for i: ℕ(start, end)
    # Example:
    #   for i: ℕ(n): implicit start of 0.
    #       total += i
    #   for i: ℕ(low, high):
    #       total += arr[i]
    # Parameters:
    # p[2] — loop variable name
    # p[6] — end expression (1-arg form) or start expression (2-arg form)
    # p[8] — end expression (2-arg form only)
    # p[10..13] — func_loop_body statements
    #             (position shifts with each alternative)
    # Returns:
    #   ("body_for_range", var, start_expr, end_expr, loop_body)
    zero = ("num", 0)
    n = len(p)
    if n == 12:
        p[0] = ("body_for_range", p[2], zero, p[6], p[10])
    elif n == 13:
        p[0] = ("body_for_range", p[2], zero, p[6], p[11])
    elif n == 14:
        p[0] = ("body_for_range", p[2], p[6], p[8], p[12])
    else:
        p[0] = ("body_for_range", p[2], p[6], p[8], p[13])


def p_func_body_stmt_for(p):
    """func_body_stmt : FOR ID COLON NEWLINE INDENT func_loop_body DEDENT"""
    # for k:
    #     loop_body
    loop_var = p[2]
    loop_body = p[6]
    indexed_arrays = []
    for stmt in loop_body:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))
    p[0] = ("body_for", loop_var, loop_body, indexed_arrays)


def p_condition_eq(p):
    """condition : func_expr EQEQ func_expr"""
    # Example:
    # left == right
    # Returns:
    #   ("cond_eq", left, right)
    p[0] = ("cond_eq", p[1], p[3])


def p_condition_neq(p):
    """condition : func_expr NEQ func_expr"""
    # Example:
    # left != right
    # Returns:
    #   ("cond_neq", left, right)
    p[0] = ("cond_neq", p[1], p[3])


def p_condition_lt(p):
    """condition : func_expr LT func_expr"""
    # Example:
    # left < right
    # Returns:
    #   ("cond_lt", left, right)
    p[0] = ("cond_lt", p[1], p[3])


def p_condition_gt(p):
    """condition : func_expr GT func_expr"""
    # Example:
    # left > right
    # Returns:
    #   ("cond_gt", left, right)
    p[0] = ("cond_gt", p[1], p[3])


def p_condition_leq(p):
    """condition : func_expr LEQ func_expr"""
    # Example:
    # left <= right
    # Returns:
    #   ("cond_leq", left, right)
    p[0] = ("cond_leq", p[1], p[3])


def p_condition_geq(p):
    """condition : func_expr GEQ func_expr"""
    # Example:
    # left >= right
    # Returns:
    #   ("cond_geq", left, right)
    p[0] = ("cond_geq", p[1], p[3])


def p_statement_function_with_loop(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT func_init FOR ID COLON NEWLINE INDENT func_loop_body DEDENT NEWLINE RETURN func_expr DEDENT"""  # noqa: E501
    # def funcname(params): return_type:
    #     init_stmts
    #     for i:
    #         loop_body
    #     return expr
    name = p[2]
    params = p[4]
    return_type = p[7]
    init_stmts = p[11]  # After first INDENT
    loop_var = p[13]  # FOR ID
    loop_body = p[17]  # After second INDENT
    final_expr = p[21]  # After RETURN

    # Find arrays indexed by loop var to determine iteration count at runtime
    indexed_arrays = []
    for stmt in loop_body:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    func_def = {
        "params": params,
        "return_type": return_type,
        "has_loop": True,
        "init_stmts": init_stmts,
        "loop_var": loop_var,
        "loop_indexed_arrays": indexed_arrays,  # Store for runtime inference
        "loop_body": loop_body,
        "body": final_expr
    }

    symbol_table[name] = {"type": "function", "value": func_def}
    p[0] = ("func_def", name)


def p_func_init_empty(p):
    """func_init : """
    p[0] = []


def p_func_init_multi(p):
    """func_init : func_init func_init_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])


def p_func_init_stmt_assign(p):
    """func_init_stmt : ID EQUALS func_expr NEWLINE"""
    p[0] = ("init_assign", p[1], p[3])


def p_func_init_stmt_empty(p):
    """func_init_stmt : NEWLINE"""
    p[0] = None


def p_func_loop_body_empty(p):
    """func_loop_body : """
    p[0] = []


def p_func_loop_body_multi(p):
    """func_loop_body : func_loop_body func_loop_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])


def p_func_loop_stmt_assign(p):
    """func_loop_stmt : ID EQUALS func_expr NEWLINE"""
    # Assignment inside a loop body
    # Example:
    # for i:
    #   cur = arr[i]
    # Parameters:
    # p[1] — variable name
    # p[3] — right hand side expression
    # Returns:
    #   ("loop_assign", name, rhs)
    p[0] = ("loop_assign", p[1], p[3])


def p_func_loop_stmt_index_assign(p):
    """func_loop_stmt : ID LBRACKET func_expr RBRACKET EQUALS func_expr NEWLINE"""  # noqa: E501
    # Indexed assignment inside a loop body
    # Example:
    # for i:
    #   arr[i] = 1
    # Parameters:
    # p[1] — array name
    # p[3] — index expression (iter var)
    # p[5] — right hand side expression
    # Returns:
    #   ("loop_index_assign", arr_name, idx_expr, rhs)
    p[0] = ("loop_index_assign", p[1], p[3], p[6])


def p_func_loop_stmt_index_assign_2d(p):
    """func_loop_stmt : ID LBRACKET loop_index_list RBRACKET EQUALS func_expr NEWLINE"""  # noqa: E501
    # 2d Indexed assignment inside a loop body
    # Example:
    # for i:
    #   for j:
    #       arr[i, j] = 1
    # Parameters:
    # p[1] — array name
    # p[3] — list of index expressions
    # p[5] — right hand side expression
    # Returns:
    #   ("loop_index_assign_2d", arr_name, [idx_exprs], rhs)
    p[0] = ("loop_index_assign_2d", p[1], p[3], p[6])


def p_func_loop_stmt_pluseq(p):
    """func_loop_stmt : ID PLUSEQ func_expr NEWLINE"""
    # Accumulation inside a loop body.
    # Example:
    #   total += arr[i]
    # Parameters:
    # p[1] — variable name
    # p[3] — right hand side expression
    # Returns:
    #   ("loop_pluseq", name, rhs)
    p[0] = ("loop_pluseq", p[1], p[3])


def p_loop_index_list_single(p):
    """loop_index_list : func_expr"""
    # Single-element index list.
    # Base case for nD subscript accumulation.
    # Parameters:
    # p[1] — index expression
    # Returns:
    #   [p[1]]
    p[0] = [p[1]]


def p_loop_index_list_multi(p):
    """loop_index_list : loop_index_list COMMA func_expr"""
    # Extend index list.
    # Parameters:
    # p[1] — existing index list
    # p[3] — next index expression
    # Returns:
    #   p[1] + [p[3]]
    p[0] = p[1] + [p[3]]


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


def p_func_loop_stmt_if(p):
    """func_loop_stmt : IF condition COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa: E501
    # One side conditional inside a loop body.
    # Example:
    # for i:
    #   if arr[i] > 0.0:
    #       total += arr[i]
    # Parameters:
    # p[2] — condition expression
    # p[6] — then branch in loop body
    # Returns:
    #   ("loop_if", condition, then_body)
    p[0] = ("loop_if", p[2], p[6])


def p_func_loop_stmt_if_else(p):
    """func_loop_stmt : IF condition COLON NEWLINE INDENT func_loop_body DEDENT ELSE COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa: E501
    # If-else conditional inside a loop body.
    # Example:
    # for i:
    #   if arr[i] > 0.0:
    #       total += arr[i]
    #   else:
    #       total += 0.0 - arr[i]
    # Parameters:
    # p[2]  — condition expression
    # p[6]  — then-branch loop body
    # p[12] — else-branch loop body
    # Returns:
    #   ("loop_if_else", condition, then_body, else_body)
    p[0] = ("loop_if_else", p[2], p[6], p[12])


def p_func_loop_stmt_for_range(p):
    """func_loop_stmt : FOR ID COLON TYPE LPAREN func_expr RPAREN NEWLINE INDENT func_loop_body DEDENT
                      | FOR ID COLON TYPE LPAREN func_expr RPAREN COLON NEWLINE INDENT func_loop_body DEDENT
                      | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN NEWLINE INDENT func_loop_body DEDENT
                      | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT func_loop_body DEDENT"""  # noqa: E501
    # Explicit range nested for loop inside a loop body.
    # Inner loop iterates over an explicit/variable range.
    # Example:
    # for i: ℕ(n):
    #   for j: ℕ(i, 10):
    #       total += j
    # Parameters:
    # p[2] — loop variable name
    # p[6] — end expression (1-arg form) or start expression (2-arg form)
    # p[8] — end expression (2-arg form only)
    # p[10..13] — func_loop_body statements
    #             (position shifts with each alternative)
    # Returns:
    #   ("loop_for_range", var, start_expr, end_expr, loop_body)
    zero = ("num", 0)
    n = len(p)
    if n == 12:
        p[0] = ("loop_for_range", p[2], zero, p[6], p[10])
    elif n == 13:
        p[0] = ("loop_for_range", p[2], zero, p[6], p[11])
    elif n == 14:
        p[0] = ("loop_for_range", p[2], p[6], p[8], p[12])
    else:
        p[0] = ("loop_for_range", p[2], p[6], p[8], p[13])


def p_func_loop_stmt_empty(p):
    """func_loop_stmt : NEWLINE"""
    p[0] = None


def p_statement_class(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         return expr
    class_name = p[2]
    class_params = p[4]  # Parameters for the class (like weights)
    lambda_params = p[12]  # Parameters for the lambda (like input x)
    return_type = p[15]
    body = p[20]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": body,
        "has_loop": False,
        "has_loss": False
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_statement_class_with_loss(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         return forward_expr
    #     def loss(params) → R:
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    body = p[20]
    loss_params = p[26]
    loss_body = p[34]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": body,
        "has_loop": False,
        "has_loss": True,
        "loss_params": loss_params,
        "loss_body": loss_body
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_statement_class_with_body(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         stmt1
    #         ...
    #         return expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    body_stmts = p[19]
    final_expr = p[21]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": final_expr,
        "statements": body_stmts,
        "has_loop": False,
        "has_loss": False
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_statement_class_with_body_and_loss(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts RETURN func_expr NEWLINE DEDENT DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         stmt1
    #         ...
    #         return forward_expr
    #     def loss(params) → R:
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    body_stmts = p[19]
    final_expr = p[21]
    loss_params = p[27]
    loss_body = p[35]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": final_expr,
        "statements": body_stmts,
        "has_loop": False,
        "has_loss": True,
        "loss_params": loss_params,
        "loss_body": loss_body
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_statement_class_with_loss_body(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         return forward_expr
    #     def loss(params) → R:
    #         stmt1
    #         ...
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    body = p[20]
    loss_params = p[26]
    loss_stmts = p[33]
    loss_body = p[35]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": body,
        "has_loop": False,
        "has_loss": True,
        "loss_params": loss_params,
        "loss_statements": loss_stmts,
        "loss_body": loss_body
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_statement_class_with_body_and_loss_body(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts RETURN func_expr NEWLINE DEDENT DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         stmt1; ...; return forward_expr
    #     def loss(params) → R:
    #         stmt1; ...; return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    body_stmts = p[19]
    final_expr = p[21]
    loss_params = p[27]
    loss_stmts = p[34]
    loss_body = p[36]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "body": final_expr,
        "statements": body_stmts,
        "has_loop": False,
        "has_loss": True,
        "loss_params": loss_params,
        "loss_statements": loss_stmts,
        "loss_body": loss_body
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_statement_class_with_loop(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT FOR ID COLON NEWLINE INDENT lambda_loop_body DEDENT RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         for i:
    #             x = expr
    #         return final_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]  # After first INDENT
    return_type = p[15]
    loop_var = p[20]  # FOR ID
    loop_body = p[24]  # After third INDENT
    final_expr = p[27]  # After DEDENT RETURN

    # Find arrays indexed by loop var
    indexed_arrays = []
    for stmt in loop_body:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "has_loop": True,
        "has_loss": False,
        "loop_var": loop_var,
        "loop_indexed_arrays": indexed_arrays,
        "loop_body": loop_body,
        "body": final_expr
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_statement_class_with_loop_and_loss(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT FOR ID COLON NEWLINE INDENT lambda_loop_body DEDENT RETURN func_expr NEWLINE DEDENT DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT RETURN func_expr NEWLINE DEDENT DEDENT"""  # noqa: E501
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         for i:
    #             x = expr
    #         return final_expr
    #     def loss(params) → R:
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[12]
    return_type = p[15]
    loop_var = p[20]
    loop_body = p[24]
    final_expr = p[27]  # After DEDENT RETURN
    loss_params = p[33]  # params after LPAREN
    loss_body = p[41]

    # Find arrays indexed by loop var
    indexed_arrays = []
    for stmt in loop_body:
        if stmt:
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "has_loop": True,
        "has_loss": True,
        "loop_var": loop_var,
        "loop_indexed_arrays": indexed_arrays,
        "loop_body": loop_body,
        "body": final_expr,
        "loss_params": loss_params,
        "loss_body": loss_body
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)


def p_lambda_loop_body_empty(p):
    """lambda_loop_body : """
    p[0] = []


def p_lambda_loop_body_multi(p):
    """lambda_loop_body : lambda_loop_body lambda_loop_stmt"""
    p[0] = p[1] + ([p[2]] if p[2] else [])


def p_lambda_loop_stmt_assign(p):
    """lambda_loop_stmt : ID EQUALS func_expr NEWLINE"""
    p[0] = ("loop_assign", p[1], p[3])


def p_lambda_loop_stmt_empty(p):
    """lambda_loop_stmt : NEWLINE"""
    p[0] = None


def p_statement_decl(p):
    """statement : ID COLON type_spec EQUALS expr NEWLINE"""
    # NEWLINE forces to parse the full expression on one line before reducing
    # the statement.
    name = p[1]
    type_spec = p[3]
    expr_ast = p[5]  # This is now an AST, not an evaluated value

    # Return AST node for declaration (evaluation happens later)
    # Include line number for error reporting
    p[0] = ("decl", name, type_spec, expr_ast, p.lineno(1))


def p_statement_assign(p):
    """statement : ID EQUALS expr"""
    name = p[1]
    expr_ast = p[3]  # This is now an AST

    # Return AST node for assignment (evaluation happens later)
    # Include line number for error reporting
    p[0] = ("assign", name, expr_ast, p.lineno(1))


def p_statement_expr(p):
    """statement : expr"""
    # Return AST node for standalone expression (evaluation happens later)
    # Include line number for error reporting
    p[0] = ("expr", p[1], p.lineno(1))


def p_statement_empty(p):
    """statement : NEWLINE"""
    global print_separator
    if p[1] >= 2:
        print_separator = True
    p[0] = None


def p_statement_if_else(p):
    """statement : IF condition COLON NEWLINE INDENT for_body DEDENT ELSE COLON NEWLINE INDENT for_body DEDENT"""  # noqa: E501
    # Program-level if/else (body uses for_statement rules)
    p[0] = ("if_else", p[2], p[6], p[12])


def p_statement_if_only(p):
    """statement : IF condition COLON NEWLINE INDENT for_body DEDENT"""
    # Program-level if without else
    p[0] = ("if_only", p[2], p[6])


def p_statement_for_range(p):
    """statement : FOR ID COLON TYPE LPAREN func_expr RPAREN NEWLINE INDENT for_body DEDENT
                 | FOR ID COLON TYPE LPAREN func_expr RPAREN COLON NEWLINE INDENT for_body DEDENT
                 | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN NEWLINE INDENT for_body DEDENT
                 | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT for_body DEDENT"""  # noqa: E501
    # Top-level for loop with explicit range: for i: ℕ(end) or
    # for i: ℕ(start, end).
    # Example:
    #   for i: ℕ(10):
    #       total += i
    #   for i: ℕ(start, end):
    #       total += i
    # Parameters:
    # p[2]        — loop variable name
    # p[6]        — end expression (1-arg form) or start expression
    #             (2-arg form)
    # p[8]        — end expression (2-arg form only)
    # p[10..13]   — for_body statements
    # p.lineno(1) — source line number of the FOR keyword, for error reporting
    # Returns:
    #   ("for_loop_range", var, start_expr, end_expr, body_stmts, lineno)
    zero = ("num", 0)
    n = len(p)
    if n == 12:  # ℕ(n) no colon
        p[0] = ("for_loop_range", p[2], zero, p[6], p[10], p.lineno(1))
    elif n == 13:  # ℕ(n):
        p[0] = ("for_loop_range", p[2], zero, p[6], p[11], p.lineno(1))
    elif n == 14:  # ℕ(start, end) no colon
        p[0] = ("for_loop_range", p[2], p[6], p[8], p[12], p.lineno(1))
    else:  # ℕ(start, end):
        p[0] = ("for_loop_range", p[2], p[6], p[8], p[13], p.lineno(1))


def p_statement_for(p):
    """statement : FOR ID COLON NEWLINE INDENT for_body DEDENT"""
    # Top-level for loop over a single variable.
    # The iteration count is inferred when generating torch code from array
    # usage inside the body.
    # Example:
    #   for i:
    #       total += arr[i]
    # Parameters:
    # p[2] — loop variable name
    # p[6] — list of for_statement nodes inside the loop body
    # Returns:
    #   ("for_loop", loop_var, body_statements, indexed_arrays, lineno)
    loop_var = p[2]
    body_statements = p[6]

    # Arrays length determines the iteration count.
    indexed_arrays = []
    for stmt in body_statements:
        if stmt:

            # lhs of indexed-assignment statements (b[i] = expr) added directly
            if stmt[0] == "for_index_assign":
                indexed_arrays.append(stmt[1])
            # Arrays read via loop variable (arr[i]) found recursively
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))

    # Return AST node - iteration count will be determined during evaluation
    # Include line number for error reporting
    p[0] = ("for_loop", loop_var, body_statements, indexed_arrays, p.lineno(1))


def p_for_body_empty(p):
    """for_body : """
    p[0] = []


def p_for_body_multi(p):
    """for_body : for_body for_statement"""
    p[0] = p[1] + ([p[2]] if p[2] is not None else [])


def p_for_statement_index_assign(p):
    """for_statement : ID LBRACKET func_expr RBRACKET EQUALS func_expr NEWLINE"""  # noqa: E501
    # Indexed assignment statement inside a top-level for loop body.
    # Example:
    #   for i:
    #       b[i] = 1
    # Parameters:
    # p[1] — array name
    # p[3] — index expression (iter var)
    # p[6] — right-hand side expression
    # Returns:
    #   ("for_index_assign", arr_name, idx_expr, rhs_expr)
    p[0] = ("for_index_assign", p[1], p[3], p[6])


def p_for_statement_index_assign_2d(p):
    """for_statement : ID LBRACKET func_expr COMMA func_expr RBRACKET EQUALS func_expr NEWLINE"""  # noqa: E501
    # 2d Indexed assignment statement inside a top-level for loop body.
    # Example:
    #   for i:
    #       for j:
    #           b[1, j] = 1
    # Parameters:
    # p[1] — array name
    # p[3] — first index expression (iter var)
    # p[5] — second index expression (iter var)
    # p[8] — right-hand side expression
    # Returns:
    #   ("for_index_assign_2d", arr_name, first_idx_expr, second_idx_expr, rhs_expr)    # noqa: E501
    p[0] = ("for_index_assign_2d", p[1], p[3], p[5], p[8])


def p_for_statement_assign(p):
    """for_statement : ID EQUALS func_expr NEWLINE"""
    # Store as AST to be evaluated later
    p[0] = ("for_assign", p[1], p[3])


def p_for_statement_pluseq(p):
    """for_statement : ID PLUSEQ func_expr NEWLINE"""
    # Store as AST: x += expr becomes x = x + expr
    p[0] = ("for_pluseq", p[1], p[3])


def p_for_statement_call(p):
    """for_statement : ID LPAREN func_args RPAREN NEWLINE"""
    # Store function call as AST
    p[0] = ("for_call", p[1], p[3])


def p_for_statement_for_range(p):
    """for_statement : FOR ID COLON TYPE LPAREN func_expr RPAREN NEWLINE INDENT for_body DEDENT
                     | FOR ID COLON TYPE LPAREN func_expr RPAREN COLON NEWLINE INDENT for_body DEDENT
                     | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN NEWLINE INDENT for_body DEDENT
                     | FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN COLON NEWLINE INDENT for_body DEDENT"""  # noqa: E501
    # Explicit range for loop nested inside a top level for body.
    # Same four variants as p_statement_for_range.
    # Example:
    # for i: ℕ(10):
    #   for j: ℕ(i, 10):
    #       total += j
    # Parameters:
    # p[2]        — loop variable name
    # p[6]        — end expression (1-arg form) or start expression
    #             (2-arg form)
    # p[8]        — end expression (2-arg form only)
    # p[10..13]   — for_body statements
    # p.lineno(1) — source line number of the FOR keyword, for error reporting
    # Returns:
    #   ("for_loop_range", var, start_expr, end_expr, body_stmts, lineno)
    zero = ("num", 0)
    n = len(p)
    if n == 12:
        p[0] = ("for_loop_range", p[2], zero, p[6], p[10], p.lineno(1))
    elif n == 13:
        p[0] = ("for_loop_range", p[2], zero, p[6], p[11], p.lineno(1))
    elif n == 14:
        p[0] = ("for_loop_range", p[2], p[6], p[8], p[12], p.lineno(1))
    else:
        p[0] = ("for_loop_range", p[2], p[6], p[8], p[13], p.lineno(1))


def p_for_statement_if_only(p):
    """for_statement : IF condition COLON NEWLINE INDENT for_body DEDENT"""
    # One-sided if inside a top-level for body or if branch.
    # Example:
    #   for i:
    #       if arr[i] > 0.0:
    #           total += arr[i]
    # Parameters:
    # p[2] — condition node
    # p[6] — then-branch for_body statements
    # Returns:
    #   ("for_if", condition, then_body)
    p[0] = ("for_if", p[2], p[6])


def p_for_statement_if_else(p):
    """for_statement : IF condition COLON NEWLINE INDENT for_body DEDENT ELSE COLON NEWLINE INDENT for_body DEDENT"""  # noqa: E501
    # If-else inside a top-level for body or if branch.
    # Example:
    #   for i:
    #       if arr[i] > 0.0:
    #           total += arr[i]
    #       else:
    #           total += 0.0 - arr[i]
    # Parameters:
    # p[2]  — condition node
    # p[6]  — then-branch for_body statements
    # p[12] — else-branch for_body statements
    # Returns:
    #   ("for_if_else", condition, then_body, else_body)
    p[0] = ("for_if_else", p[2], p[6], p[12])


def p_for_statement_for(p):
    """for_statement : FOR ID COLON NEWLINE INDENT for_body DEDENT"""
    # Implicit-range for loop nested inside a for_body.
    # Range is inferred from arrays indexed by loop_var in the body.
    # Example:
    #   for i:
    #       for j:
    #           total += arr[j]
    # Parameters:
    # p[2] — loop variable name
    # p[6] — for_body statements
    # Returns:
    #   ("for_loop", loop_var, body_stmts, indexed_arrays, lineno)
    loop_var = p[2]
    body_stmts = p[6]
    indexed_arrays = []
    for stmt in body_stmts:
        if stmt:
            if stmt[0] == "for_index_assign":
                indexed_arrays.append(stmt[1])
            indexed_arrays.extend(find_indexed_arrays(stmt, loop_var))
    p[0] = ("for_loop", loop_var, body_stmts, indexed_arrays, p.lineno(1))


def p_for_statement_empty(p):
    """for_statement : NEWLINE"""
    p[0] = None


# Parameters
def p_params_empty(p):
    """params : """
    p[0] = []


def p_params_single(p):
    """params : ID COLON type_spec"""
    p[0] = [(p[1], p[3])]


def p_params_multi(p):
    """params : ID COLON type_spec COMMA params"""
    p[0] = [(p[1], p[3])] + p[5]


# Function Expression (AST nodes)
def p_func_expr_plus(p):
    """func_expr : func_expr PLUS func_term"""
    p[0] = ("add", p[1], p[3])


def p_func_expr_minus(p):
    """func_expr : func_expr MINUS func_term"""
    p[0] = ("sub", p[1], p[3])


def p_func_expr_term(p):
    """func_expr : func_term"""
    p[0] = p[1]


def p_func_term_times(p):
    """func_term : func_term TIMES func_power
                 | func_term DIVIDE func_power
                 | func_term INTDIV func_power
                 | func_term MATMUL func_power"""
    if p[2] == "*":
        p[0] = ("mul", p[1], p[3])
    elif p[2] == "/":
        p[0] = ("div", p[1], p[3])
    elif p[2] == "//":
        p[0] = ("intdiv", p[1], p[3])
    else:  # @
        p[0] = ("matmul", p[1], p[3])


def p_func_term_power(p):
    """func_term : func_power"""
    p[0] = p[1]


def p_func_power_pow(p):
    """func_power : func_factor POWER func_power"""
    p[0] = ("pow", p[1], p[3])


def p_func_power_neg(p):
    """func_power : MINUS func_power"""
    p[0] = ("neg", p[2])


def p_func_power_factor(p):
    """func_power : func_factor"""
    p[0] = p[1]


def p_func_factor_number(p):
    """func_factor : NUMBER"""
    p[0] = ("num", p[1])


def p_func_factor_id(p):
    """func_factor : ID"""
    p[0] = ("var", p[1])


def p_func_factor_group(p):
    """func_factor : LPAREN func_expr RPAREN"""
    p[0] = p[2]


def p_func_factor_call(p):
    """func_factor : ID LPAREN func_args RPAREN"""
    p[0] = ("call", p[1], p[3])


def p_func_factor_index(p):
    """func_factor : ID LBRACKET func_expr RBRACKET"""
    # Tensor indexing: W[i]
    p[0] = ("index", p[1], p[3])


def p_multi_index_list_base(p):
    """multi_index_list : func_expr COMMA func_expr"""
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
    """multi_index_list : multi_index_list COMMA func_expr"""
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


def p_func_factor_call_index(p):
    """func_factor : ID LPAREN func_args RPAREN LBRACKET func_expr RBRACKET"""
    # Indexing a function call result: grad(H, x)[0]
    p[0] = ("call_index", p[1], p[3], p[6])


def p_func_factor_chain_index(p):
    """func_factor : func_factor LBRACKET func_expr RBRACKET"""
    # Chain indexing: A[i][k], B[k][j], etc.
    p[0] = ("chain_index", p[1], p[3])


def p_func_factor_step_slice(p):
    """func_factor : ID LBRACKET NUMBER COLON COLON NUMBER RBRACKET"""
    # Step slice: x[0::2]  (start::step, no stop)
    p[0] = ("step_slice", p[1], int(p[3]), int(p[6]))


def p_func_factor_array(p):
    """func_factor : LBRACKET func_elements RBRACKET"""
    # Array literal in function body: [1.0, 2.0, 3.0]
    p[0] = ("array", p[2])


def p_func_factor_string(p):
    """func_factor : STRING"""
    # String literal (for equations): 'x0 = a + b'
    p[0] = ("string", p[1])


def p_func_factor_imaginary(p):
    """func_factor : IMAGINARY"""
    # Imaginary unit i
    p[0] = ("imaginary", )


def p_func_factor_for_expr_range(p):
    """func_factor : FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN ARROW func_expr"""  # noqa: E501
    # Ranged for inside a function body with explicit start and end.
    # Iterates the loop variable over range(start, end) with exclusive end.
    # Example:
    #   for i : ℕ(1, n) -> cos(i * 0.5)
    # Parameters:
    # p[2]  — loop variable name
    # p[6]  — start expression (inclusive)
    # p[8]  — end expression (exclusive)
    # p[11] — body expression evaluated for each i
    # Returns:
    #   ("for_expr_range", var, start_expr, end_expr, body_expr)
    p[0] = ("for_expr_range", p[2], p[6], p[8], p[11])


def p_func_factor_for_expr(p):
    """func_factor : FOR ID COLON TYPE LPAREN func_expr RPAREN ARROW func_expr"""  # noqa: E501
    # Explicit size for-expression inside a function body.
    # Example:
    #   for i : ℕ(n)
    # Parameters:
    # p[2] — loop variable name
    # p[6] — size expression (n)
    # p[9] — body expression
    # Returns:
    #  ("for_expr", var, size_expr, body_expr)
    p[0] = ("for_expr", p[2], p[6], p[9])


def p_func_factor_for_expr_auto(p):
    """func_factor : FOR ID ARROW func_expr"""
    # Infer size for-expression inside a function body.
    # Example:
    #   for i -> arr[i] * 2.0
    loop_var = p[2]
    body_expr = p[4]
    indexed = find_indexed_arrays(body_expr, loop_var)
    if not indexed:
        raise SyntaxError(
            f"'for {loop_var} ->' body has no '{loop_var}'-indexed array to infer size from"  # noqa: E501
        )
    p[0] = ("for_expr", loop_var, ("call", "len", [("var", indexed[0])]),
            body_expr)


def p_func_elements_single(p):
    """func_elements : func_expr"""
    p[0] = [p[1]]


def p_func_elements_multi(p):
    """func_elements : func_expr COMMA func_elements"""
    p[0] = [p[1]] + p[3]


def p_func_args_empty(p):
    """func_args : """
    p[0] = []


def p_func_args_single(p):
    """func_args : func_expr"""
    p[0] = [p[1]]


def p_func_args_multi(p):
    """func_args : func_expr COMMA func_args"""
    p[0] = [p[1]] + p[3]


def p_expr_plus(p):
    """expr : expr PLUS term"""
    # Build AST node instead of immediate evaluation
    p[0] = ("add", p[1], p[3])


def p_expr_minus(p):
    """expr : expr MINUS term"""
    p[0] = ("sub", p[1], p[3])


def p_expr_term(p):
    """expr : term"""
    p[0] = p[1]


def p_term_binop(p):
    """term : term TIMES factor
            | term DIVIDE factor
            | term MATMUL factor
            | term POWER factor"""
    if p[2] == "*":
        p[0] = ("mul", p[1], p[3])
    elif p[2] == "/":
        p[0] = ("div", p[1], p[3])
    elif p[2] == "**":
        p[0] = ("pow", p[1], p[3])
    else:  # @
        p[0] = ("matmul", p[1], p[3])


# Factors
def p_term_factor(p):
    """term : factor"""
    p[0] = p[1]


def p_factor_call(p):
    """factor : ID LPAREN args RPAREN"""
    # Return AST node for function call
    func_name = p[1]
    args = p[3]
    p[0] = ("call", func_name, args)


def p_factor_number(p):
    """factor : NUMBER"""
    # Return AST node for number literal
    p[0] = ("num", p[1])


def p_factor_neg(p):
    """factor : MINUS factor"""
    # Return AST node for unary minus
    p[0] = ("neg", p[2])


def p_factor_id(p):
    """factor : ID"""
    # Return AST node for variable reference
    p[0] = ("var", p[1])


def p_factor_group(p):
    """factor : LPAREN expr RPAREN"""
    p[0] = p[2]


def p_factor_array(p):
    """factor : LBRACKET elements RBRACKET"""
    # Return AST node for array literal
    p[0] = ("array", p[2])


def p_factor_string(p):
    """factor : STRING"""
    # String literal (for equations and symbolic): 'x0 = a + b'
    p[0] = ("equation_string", p[1])


def p_factor_for_expr_range(p):
    """factor : FOR ID COLON TYPE LPAREN func_expr COMMA func_expr RPAREN ARROW func_expr"""  # noqa: E501
    # Ranged for-expression at program level with explicit start and end.
    # Iterates the loop variable over range(start, end) with exclusive end.
    # Example:
    #   cos_wave : ℝ[5] = for i : ℕ(1, 6) -> cos(i * 0.5)
    # Parameters:
    # p[2]  — loop variable name
    # p[6]  — start expression (inclusive)
    # p[8]  — end expression (exclusive)
    # p[11] — body expression evaluated for each i
    # Returns:
    #   ("for_expr_range", var, start_expr, end_expr, body_expr)
    p[0] = ("for_expr_range", p[2], p[6], p[8], p[11])


def p_factor_for_expr(p):
    """factor : FOR ID COLON TYPE LPAREN func_expr RPAREN ARROW func_expr"""
    # Explicit size for-expression at program level.
    # Example:
    #   for i : ℕ(n)
    # Parameters:
    # p[2] — loop variable name
    # p[6] — size expression (n)
    # p[9] — body expression
    # Returns:
    #  ("for_expr", var, size_expr, body_expr)
    p[0] = ("for_expr", p[2], p[6], p[9])


def p_factor_for_expr_auto(p):
    """factor : FOR ID ARROW func_expr"""
    # Infer size for-expression at program level.
    # Example:
    #   for i -> arr[i] * 2.0
    loop_var = p[2]
    body_expr = p[4]
    indexed = find_indexed_arrays(body_expr, loop_var)
    if not indexed:
        raise SyntaxError(
            f"'for {loop_var} ->' body has no '{loop_var}'-indexed array to infer size from"  # noqa: E501
        )
    p[0] = ("for_expr", loop_var, ("call", "len", [("var", indexed[0])]),
            body_expr)


def p_elements_single(p):
    """elements : expr"""
    p[0] = [p[1]]


def p_elements_multi(p):
    """elements : expr COMMA elements"""
    p[0] = [p[1]] + p[3]


def p_elements_newline(p):
    """elements : NEWLINE elements
                | elements NEWLINE"""
    # Allow newlines within array definitions
    if len(p) == 3:
        p[0] = p[2] if isinstance(p[2], list) else p[1]


def p_factor_index(p):
    """factor : ID LBRACKET NUMBER RBRACKET"""
    # Return AST node for array indexing
    p[0] = ("index", p[1], ("num", p[3]))


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


def p_factor_slice(p):
    """factor : ID LBRACKET NUMBER COLON NUMBER RBRACKET"""
    # Return AST node for array slicing
    p[0] = ("slice", p[1], ("num", p[3]), ("num", p[5]))


# Function Arguments
def p_args_empty(p):
    """args : """
    p[0] = []


def p_args_single(p):
    """args : expr"""
    p[0] = [p[1]]


def p_args_multi(p):
    """args : expr COMMA args"""
    p[0] = [p[1]] + p[3]


def p_error(p):
    if p:
        raise SyntaxError(f"Syntax error at '{p.value}'")
    else:
        raise SyntaxError("Syntax error at EOF")


# Symbolic types
def p_statement_symbol_decl(p):
    """statement : ID COLON SYMBOL NEWLINE"""
    # Declares symbolic variable using Sympy's Symbol
    # Example:
    #   x : Symbol
    # Parameters:
    #   p[1] — symbol name
    # Returns:
    #   ("symbol_decl", name)
    p[0] = ('symbol_decl', p[1])


def p_statement_function_decl(p):
    """statement : ID COLON FUNCTION NEWLINE"""
    # Declares symbolic funcion using Sympy's Function
    # Example:
    #   u : Function
    # Parameters:
    #   p[1] — function name
    # Returns:
    #   ("function_decl", name)
    p[0] = ('function_decl', p[1])


def p_id_list(p):
    """id_list : ID
               | id_list COMMA ID"""
    # Builds a list of identifiers separated by commas.
    # Used for multi-symbol and multi-function declarations.
    # Parameters:
    #   p[1] — single ID
    #   p[3] — next ID (multiple declarations)
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_statement_symbol_multi_decl(p):
    """statement : id_list COLON SYMBOL NEWLINE"""
    # Declares multiple symbolic variables using Sympy's Symbol
    # Example:
    #   x, y : Symbol
    # Parameters:
    #   p[1] — list of symbol names from id_list
    # Returns:
    #   ("symbol_decl_multi", [name, ...])
    p[0] = ("symbol_decl_multi", p[1])


def p_statement_function_multi_decl(p):
    # Declares multiple symbolic functions using Sympy's Function
    # Example:
    #   f, u : Function
    # Parameters:
    #   p[1] — list of function names from id_list
    # Returns:
    #   (function_decl_multi", [name, ...])
    """statement : id_list COLON FUNCTION NEWLINE"""
    p[0] = ("function_decl_multi", p[1])


def p_statemet_equation_decl(p):
    """statement : ID COLON EQUATION WALRUS \
                 func_expr EQUALS func_expr NEWLINE"""
    # Declares equation using symbolic variables
    # Example:
    #   eq: Equation := 2.0*x + 3.0 = 7.0
    # Parameters:
    #   p[1] — equation name
    #   p[5] — expression on LHS
    #   p[7] — expression on RHS
    p[0] = ("equation_decl", p[1], p[5], p[7])


parser = yacc.yacc()
