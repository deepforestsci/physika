import sys
import re
import ply.lex as lex
import ply.yacc as yacc

from utils import to_torch, from_torch, is_torch_tensor, is_scalar, is_vector, is_matrix, get_shape, is_function, infer_type, format_tensor_type, flatten, reshape

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pyvista as pv # For 3D animations
    import numpy as np
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import matplotlib.pyplot as plt # For 3D/2D animations if pyvista not available
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ======================================================
# GLOBAL STATE
# ======================================================

symbol_table = {}
print_separator = False
parsing_function_body = False



# ======================================================
# LEXER
# ======================================================

tokens = (
    "ID", "NUMBER", "TYPE", "STRING",
    "PLUS", "MINUS", "TIMES", "DIVIDE", "MATMUL", "POWER",
    "EQUALS", "PLUSEQ", "COLON", "COMMA", "ARROW",
    "LPAREN", "RPAREN",
    "LBRACKET", "RBRACKET",
    "NEWLINE",
    "DEF", "RETURN", "FOR", "IN", "RANGE", "END",
    "CLASS", "LAMBDA",
    "TANGENT",
    "IMAGINARY",
)

reserved = {
    "def": "DEF",
    "return": "RETURN",
    "for": "FOR",
    "in": "IN",
    "range": "RANGE",
    "end": "END",
    "class": "CLASS",
}

t_POWER    = r"\*\*"
t_PLUSEQ   = r"\+="
t_ARROW    = r"→"
t_PLUS     = r"\+"
t_MINUS    = r"-"
t_TIMES    = r"\*"
t_DIVIDE   = r"/"
t_MATMUL   = r"@"
t_EQUALS   = r"="
t_COLON    = r":"
t_COMMA    = r","

def t_LAMBDA(t):
    r"λ"
    return t
t_LPAREN   = r"\("
t_RPAREN   = r"\)"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_TANGENT  = r"T"

t_ignore = " \t"

def t_COMMENT(t):
    r"\#[^\n]*"
    pass  # Ignore comments

def t_STRING(t):
    r"'[^']*'|\"[^\"]*\""
    # Remove quotes from string
    t.value = t.value[1:-1]
    return t

def t_IMAGINARY(t):
    r"(?<![a-zA-Z0-9_])i(?![a-zA-Z0-9_])"
    return t

def t_TYPE(t):
    r"(ℝ|\\mathbb\{R\}|\\R|ℤ|ℕ|R(?![a-zA-Z0-9_])|Z(?![a-zA-Z0-9_])|N(?![a-zA-Z0-9_]))"
    if t.value in ("ℤ", "Z"):
        t.value = "ℤ"
    elif t.value in ("ℕ", "N"):
        t.value = "ℕ"
    else:
        t.value = "ℝ"
    return t

def t_NUMBER(t):
    r"\d+(\.\d+)?"
    t.value = float(t.value)
    return t

def t_NEWLINE(t):
    r"\n+"
    t.lexer.lineno += len(t.value)
    t.value = len(t.value)
    return t

def t_ID(t):
    r"[a-zA-Z_][a-zA-Z0-9_]*"
    t.type = reserved.get(t.value, "ID")
    return t

def t_error(t):
    raise SyntaxError(f"Illegal character '{t.value[0]}'")

lexer = lex.lex()

# ======================================================
# PARSER
# ======================================================

precedence = (
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "DIVIDE"),
    ("left", "MATMUL"),
    ("right", "POWER"),
    ("right", "EQUALS"),
)

# ----------------------
# Program
# ----------------------

def p_program(p):
    """program : statements"""
    p[0] = p[1]

def p_statements_multi(p):
    """statements : statements statement"""
    p[0] = p[1] + ([] if p[2] is None else [p[2]])

def p_statements_single(p):
    """statements : statement"""
    p[0] = [] if p[1] is None else [p[1]]

# ----------------------
# Types
# ----------------------

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

# ----------------------
# Statements
# ----------------------

def p_statement_function(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE RETURN func_expr"""
    name, params, return_type, body = p[2], p[4], p[7], p[11]

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
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE func_body_stmts RETURN func_expr"""
    # def funcname(params): return_type:
    #     stmt1
    #     stmt2
    #     return expr
    name = p[2]
    params = p[4]
    return_type = p[7]
    body_stmts = p[10]
    final_expr = p[12]

    func_def = {
        "params": params,
        "return_type": return_type,
        "body": final_expr,
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

def p_statement_function_with_loop(p):
    """statement : DEF ID LPAREN params RPAREN COLON type_spec COLON NEWLINE func_init FOR ID IN RANGE LPAREN ID RPAREN COLON NEWLINE func_loop_body END NEWLINE RETURN func_expr"""
    # def funcname(params): return_type:
    #     init_stmts
    #     for i in range(n):
    #         loop_body
    #     end
    #     return expr
    name = p[2]
    params = p[4]
    return_type = p[7]
    init_stmts = p[10]
    loop_var = p[12]
    loop_count_var = p[16]
    loop_body = p[20]
    final_expr = p[24]

    func_def = {
        "params": params,
        "return_type": return_type,
        "has_loop": True,
        "init_stmts": init_stmts,
        "loop_var": loop_var,
        "loop_count_var": loop_count_var,
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
    p[0] = ("loop_assign", p[1], p[3])

def p_func_loop_stmt_pluseq(p):
    """func_loop_stmt : ID PLUSEQ func_expr NEWLINE"""
    p[0] = ("loop_pluseq", p[1], p[3])

def p_func_loop_stmt_empty(p):
    """func_loop_stmt : NEWLINE"""
    p[0] = None

def p_statement_class(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE RETURN func_expr"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         return expr
    class_name = p[2]
    class_params = p[4]  # Parameters for the class (like weights)
    lambda_params = p[11]  # Parameters for the lambda (like input x)
    return_type = p[14]
    body = p[18]

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
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE RETURN func_expr NEWLINE DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE RETURN func_expr"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         return forward_expr
    #     def loss(params) → R:
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[11]
    return_type = p[14]
    body = p[18]
    loss_name = p[21]  # Should be "loss"
    loss_params = p[23]
    loss_return_type = p[26]
    loss_body = p[30]

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

def p_statement_class_with_loop(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE FOR ID IN RANGE LPAREN ID RPAREN COLON NEWLINE lambda_loop_body END NEWLINE RETURN func_expr"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         for i in range(n):
    #             x = expr
    #         return final_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[11]
    return_type = p[14]
    loop_var = p[18]
    loop_count_var = p[22]  # Variable name for iteration count (e.g., 'n')
    loop_body = p[26]
    final_expr = p[30]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "has_loop": True,
        "has_loss": False,
        "loop_var": loop_var,
        "loop_count_var": loop_count_var,
        "loop_body": loop_body,
        "body": final_expr
    }

    symbol_table[class_name] = {"type": "class", "value": class_def}
    p[0] = ("class_def", class_name)

def p_statement_class_with_loop_and_loss(p):
    """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE FOR ID IN RANGE LPAREN ID RPAREN COLON NEWLINE lambda_loop_body END NEWLINE RETURN func_expr NEWLINE DEF ID LPAREN params RPAREN ARROW type_spec COLON NEWLINE RETURN func_expr"""
    # class ClassName(params):
    #     def λ(x: type) → return_type:
    #         for i in range(n):
    #             x = expr
    #         return final_expr
    #     def loss(params) → R:
    #         return loss_expr
    class_name = p[2]
    class_params = p[4]
    lambda_params = p[11]
    return_type = p[14]
    loop_var = p[18]
    loop_count_var = p[22]
    loop_body = p[26]
    final_expr = p[30]
    loss_name = p[33]
    loss_params = p[35]
    loss_return_type = p[38]
    loss_body = p[42]

    class_def = {
        "class_params": class_params,
        "lambda_params": lambda_params,
        "return_type": return_type,
        "has_loop": True,
        "has_loss": True,
        "loop_var": loop_var,
        "loop_count_var": loop_count_var,
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
    """statement : ID COLON type_spec EQUALS expr"""
    name, declared, value = p[1], p[3], p[5]

    # Handle class instance
    if isinstance(value, tuple) and value[0] == "instance":
        instance = value[1]
        symbol_table[name] = {"type": "instance", "value": instance}
        p[0] = ("decl", name)
        return

    # Handle Sequential layer construction
    if isinstance(value, tuple) and value[0] == "sequential":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        layers = [build_layer(spec) for spec in value[1]]
        value = nn.Sequential(*layers)
    
    # Handle other layer types
    elif isinstance(value, tuple) and value[0] in ["linear", "tanh", "relu", "sigmoid"]:
        value = build_layer(value)

    # Convert to torch tensor if not already (and not a nn.Module)
    elif HAS_TORCH and not is_torch_tensor(value) and not isinstance(value, nn.Module):
        value = to_torch(value)

    if isinstance(declared, tuple) and declared[0] == "tensor":
        # Extract just the dimensions (ignore variance for shape checking)
        expected_dims = [dim for dim, _ in declared[1]]
        expected_shape = tuple(expected_dims)
        actual_shape = get_shape(value)
        if actual_shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {actual_shape}")

    symbol_table[name] = {"type": declared, "value": value}
    p[0] = ("decl", name)

def p_statement_assign(p):
    """statement : ID EQUALS expr"""
    name, value = p[1], p[3]

    # Handle class instance (wrapped in tuple)
    if isinstance(value, tuple) and value[0] == "instance":
        instance = value[1]
        symbol_table[name] = {"type": "instance", "value": instance}
        p[0] = ("assign", name)
        return

    # Handle class instance (direct dict from symbol table lookup)
    if isinstance(value, dict) and "bound_params" in value:
        symbol_table[name] = {"type": "instance", "value": value}
        p[0] = ("assign", name)
        return

    # Convert to torch tensor if not already (and not a nn.Module)
    if HAS_TORCH and not is_torch_tensor(value) and not isinstance(value, nn.Module):
        value = to_torch(value)

    # Infer type from value
    inferred_type = infer_type(value)

    symbol_table[name] = {"type": inferred_type, "value": value}
    p[0] = ("assign", name)

def p_statement_expr(p):
    """statement : expr"""
    global print_separator
    if print_separator:
        print()
        print_separator = False
    
    # Convert torch tensor to display format
    display_value = from_torch(p[1]) if is_torch_tensor(p[1]) else p[1]
    type_str = infer_type(p[1])
    
    print(f"{display_value} ∈ {type_str}")
    p[0] = ("expr", p[1])

def p_statement_empty(p):
    """statement : NEWLINE"""
    global print_separator
    if p[1] >= 2:
        print_separator = True
    p[0] = None

def p_statement_for(p):
    """statement : FOR ID IN RANGE LPAREN NUMBER RPAREN COLON NEWLINE for_body END"""
    loop_var = p[2]
    iterations = int(p[6])
    body_statements = p[10]

    # Execute the loop
    for i in range(iterations):
        # Set loop variable
        symbol_table[loop_var] = {"type": "ℝ", "value": float(i)}

        # Execute body statements
        for stmt in body_statements:
            execute_for_statement(stmt)

    p[0] = ("for_loop", loop_var, iterations)

def p_for_body_empty(p):
    """for_body : """
    p[0] = []

def p_for_body_multi(p):
    """for_body : for_body for_statement"""
    p[0] = p[1] + ([p[2]] if p[2] is not None else [])

def p_for_statement_assign(p):
    """for_statement : ID EQUALS func_expr NEWLINE"""
    # Store as AST to be evaluated later
    p[0] = ("for_assign", p[1], p[3])

def p_for_statement_call(p):
    """for_statement : ID LPAREN func_args RPAREN NEWLINE"""
    # Store function call as AST
    p[0] = ("for_call", p[1], p[3])

def p_for_statement_empty(p):
    """for_statement : NEWLINE"""
    p[0] = None

def execute_for_statement(stmt):
    """Execute a statement inside a for loop"""
    if stmt is None:
        return

    op = stmt[0]

    if op == "for_assign":
        _, name, ast_expr = stmt
        # Evaluate the AST expression with current symbol table
        value = evaluate_ast(ast_expr, {})
        if name in symbol_table:
            symbol_table[name]["value"] = value
        else:
            symbol_table[name] = {"type": "ℝ", "value": value}

    elif op == "for_call":
        _, func_name, arg_asts = stmt
        # Evaluate arguments
        args = [evaluate_ast(arg, {}) for arg in arg_asts]
        execute_for_call(func_name, args)

def execute_for_call(func_name, args):
    """Execute a function call inside a for loop"""
    if func_name == "print":
        if len(args) == 1:
            value = args[0]
            display_value = from_torch(value) if is_torch_tensor(value) else value
            print(f"  {display_value}")

# ----------------------
# Parameters
# ----------------------

def p_params_empty(p):
    """params : """
    p[0] = []

def p_params_single(p):
    """params : ID COLON type_spec"""
    p[0] = [(p[1], p[3])]

def p_params_multi(p):
    """params : ID COLON type_spec COMMA params"""
    p[0] = [(p[1], p[3])] + p[5]

# ----------------------
# Function Expression (AST nodes)
# ----------------------

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
    """func_term : func_term TIMES func_factor
                 | func_term DIVIDE func_factor
                 | func_term MATMUL func_factor
                 | func_term POWER func_factor"""
    if p[2] == "*":
        p[0] = ("mul", p[1], p[3])
    elif p[2] == "/":
        p[0] = ("div", p[1], p[3])
    elif p[2] == "**":
        p[0] = ("pow", p[1], p[3])
    else:  # @
        p[0] = ("matmul", p[1], p[3])

def p_func_term_factor(p):
    """func_term : func_factor"""
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

def p_func_factor_call_index(p):
    """func_factor : ID LPAREN func_args RPAREN LBRACKET func_expr RBRACKET"""
    # Indexing a function call result: grad(H, x)[0]
    p[0] = ("call_index", p[1], p[3], p[6])

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
    p[0] = ("imaginary",)

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

# ----------------------
# Expressions (evaluated immediately)
# ----------------------

def execute_statement(stmt):
    """Execute a statement tuple from for loop body"""
    if stmt[0] == "assign":
        _, name, value = stmt
        if name in symbol_table:
            symbol_table[name]["value"] = value
        else:
            symbol_table[name] = {"type": "ℝ", "value": value}
    elif stmt[0] == "decl":
        _, name, type_spec, value = stmt
        symbol_table[name] = {"type": type_spec, "value": value}

def create_pytorch_model(class_def):
    """Create a PyTorch nn.Module from class definition"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")
    
    class DynamicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            
            # Build layers from class body
            for stmt in class_def["body"]:
                if stmt[0] == "layer":
                    layer_name, layer_spec = stmt[1], stmt[2]
                    self.layers[layer_name] = build_layer(layer_spec)
        
        def forward(self, x):
            # Execute forward pass through sequential layers
            for layer in self.layers.values():
                x = layer(x)
            return x
    
    return DynamicModel()

def build_layer(layer_spec):
    """Build a PyTorch layer from specification"""
    layer_type = layer_spec[0]
    
    if layer_type == "linear":
        in_features, out_features = layer_spec[1], layer_spec[2]
        return nn.Linear(in_features, out_features)
    
    elif layer_type == "sequential":
        layers = [build_layer(spec) for spec in layer_spec[1]]
        return nn.Sequential(*layers)
    
    elif layer_type == "tanh":
        return nn.Tanh()
    
    elif layer_type == "relu":
        return nn.ReLU()
    
    elif layer_type == "sigmoid":
        return nn.Sigmoid()
    
    raise ValueError(f"Unknown layer type: {layer_type}")

def tensor_contract(a, b, a_type, b_type):
    """
    Tensor contraction using Einstein summation convention.
    Contracts over matching covariant-contravariant pairs.
    """
    # If we don't have type info, fall back to regular matmul
    if not (isinstance(a_type, tuple) and a_type[0] == "tensor" and
            isinstance(b_type, tuple) and b_type[0] == "tensor"):
        return matmul(a, b)
    
    a_dims = a_type[1]  # List of (size, variance) tuples
    b_dims = b_type[1]
    
    # Find contraction pairs: covariant in one, contravariant in other
    # For now, implement simple matrix-like contraction:
    # Last index of a (if contravariant) with first index of b (if covariant)
    
    # Simple case: matrix-like contraction
    if len(a_dims) == 2 and len(b_dims) == 2:
        a_rows, a_var1 = a_dims[0]
        a_cols, a_var2 = a_dims[1]
        b_rows, b_var1 = b_dims[0]
        b_cols, b_var2 = b_dims[1]
        
        # Check if we can contract: a's last must be contravariant, b's first must be covariant
        # or vice versa
        can_contract = ((a_var2 == "contravariant" and b_var1 == "covariant") or
                       (a_var2 == "covariant" and b_var1 == "contravariant"))
        
        if can_contract and a_cols == b_rows:
            return matmul(a, b)
    
    raise ValueError("Cannot contract these tensors - variance mismatch")

def matmul(a, b, a_type=None, b_type=None):
    """Matrix multiplication for vectors and matrices"""
    # Handle torch tensors
    if is_torch_tensor(a) or is_torch_tensor(b):
        if not is_torch_tensor(a):
            a = to_torch(a)
        if not is_torch_tensor(b):
            b = to_torch(b)
        return torch.matmul(a, b)
    
    # Vector @ Vector (dot product)
    if is_vector(a) and is_vector(b):
        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
        return sum(a[i] * b[i] for i in range(len(a)))
    
    # Matrix @ Vector
    if is_matrix(a) and is_vector(b):
        shape_a = get_shape(a)
        if len(shape_a) != 2:
            raise ValueError("Matrix must be 2D for matrix-vector multiplication")
        rows, cols = shape_a
        if cols != len(b):
            raise ValueError(f"Dimension mismatch: matrix has {cols} columns, vector has {len(b)} elements")
        
        result = []
        for row in a:
            result.append(sum(row[i] * b[i] for i in range(len(b))))
        return result
    
    # Vector @ Matrix
    if is_vector(a) and is_matrix(b):
        shape_b = get_shape(b)
        if len(shape_b) != 2:
            raise ValueError("Matrix must be 2D for vector-matrix multiplication")
        rows, cols = shape_b
        if len(a) != rows:
            raise ValueError(f"Dimension mismatch: vector has {len(a)} elements, matrix has {rows} rows")
        
        result = []
        for j in range(cols):
            result.append(sum(a[i] * b[i][j] for i in range(rows)))
        return result
    
    # Matrix @ Matrix
    if is_matrix(a) and is_matrix(b):
        shape_a = get_shape(a)
        shape_b = get_shape(b)
        
        if len(shape_a) != 2 or len(shape_b) != 2:
            raise ValueError("Both operands must be 2D matrices")
        
        rows_a, cols_a = shape_a
        rows_b, cols_b = shape_b
        
        if cols_a != rows_b:
            raise ValueError(f"Dimension mismatch: ({rows_a}x{cols_a}) @ ({rows_b}x{cols_b})")
        
        result = []
        for i in range(rows_a):
            row = []
            for j in range(cols_b):
                value = sum(a[i][k] * b[k][j] for k in range(cols_a))
                row.append(value)
            result.append(row)
        return result
    
    raise TypeError("Invalid operands for matrix multiplication")

def element_wise_op(a, b, op):
    """Perform element-wise operation with broadcasting"""
    # Handle torch tensors
    if is_torch_tensor(a) or is_torch_tensor(b):
        if not is_torch_tensor(a):
            a = to_torch(a)
        if not is_torch_tensor(b):
            b = to_torch(b)
        return op(a, b)
    
    # Scalar op Scalar
    if is_scalar(a) and is_scalar(b):
        return op(a, b)
    
    # Scalar op Tensor (broadcasting)
    if is_scalar(a) and isinstance(b, list):
        return [element_wise_op(a, item, op) for item in b]
    
    # Tensor op Scalar (broadcasting)
    if isinstance(a, list) and is_scalar(b):
        return [element_wise_op(item, b, op) for item in a]
    
    # Tensor op Tensor (element-wise)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Tensor size mismatch")
        return [element_wise_op(a[i], b[i], op) for i in range(len(a))]
    
    raise TypeError("Invalid operands")

def p_expr_plus(p):
    """expr : expr PLUS term"""
    p[0] = element_wise_op(p[1], p[3], lambda x, y: x + y)

def p_expr_minus(p):
    """expr : expr MINUS term"""
    p[0] = element_wise_op(p[1], p[3], lambda x, y: x - y)

def p_expr_term(p):
    """expr : term"""
    p[0] = p[1]

def p_term_binop(p):
    """term : term TIMES factor
            | term DIVIDE factor
            | term MATMUL factor
            | term POWER factor"""
    if p[2] == "*":
        p[0] = element_wise_op(p[1], p[3], lambda x, y: x * y)
    elif p[2] == "/":
        p[0] = element_wise_op(p[1], p[3], lambda x, y: x / y)
    elif p[2] == "**":
        p[0] = element_wise_op(p[1], p[3], lambda x, y: x ** y)
    else:  # @
        p[0] = matmul(p[1], p[3])



# ======================================================
# AST EVALUATOR
# ======================================================

# ======================================================
# SYMBOLIC EQUATION PARSER FOR solve()
# ======================================================



    HAS_TORCH = False

def parse_equation(eq_str, local_scope):
    """
    Parse an equation string like 'x0 = a + b' or 'v0 = i*omega*a - i*omega*b'
    Returns: (lhs_var, coefficients_dict) where coefficients_dict maps unknown -> coefficient
    """
    # Split by '='
    if '=' not in eq_str:
        raise ValueError(f"Equation must contain '=': {eq_str}")

    lhs, rhs = eq_str.split('=', 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    return (lhs, rhs)

def tokenize_expr(expr):
    """Tokenize a mathematical expression into tokens"""
    # Handle special unicode minus
    expr = expr.replace('−', '-')

    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
        elif c in '+-*/()':
            tokens.append(c)
            i += 1
        elif c.isalpha() or c == '_':
            # Identifier
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif c.isdigit() or c == '.':
            # Number
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(float(expr[i:j]))
            i = j
        else:
            raise ValueError(f"Unknown character in expression: {c}")
    return tokens

def extract_linear_coefficients(rhs_str, unknowns, local_scope):
    """
    Extract coefficients for each unknown from a linear expression.
    Example: 'a + b' with unknowns ['a', 'b'] -> {a: 1, b: 1}
    Example: 'i*omega*a - i*omega*b' -> {a: i*omega, b: -i*omega}
    """
    # Replace unicode minus
    rhs_str = rhs_str.replace('−', '-')

    coeffs = {u: 0.0 for u in unknowns}

    # Split into terms by + and - (keeping the sign)
    # First, normalize: add + at start if doesn't start with -
    rhs_str = rhs_str.strip()
    if not rhs_str.startswith('-'):
        rhs_str = '+' + rhs_str

    # Split by + and - keeping delimiters
    terms = re.split(r'(?=[+-])', rhs_str)
    terms = [t.strip() for t in terms if t.strip()]

    for term in terms:
        # Determine sign
        sign = 1.0
        if term.startswith('-'):
            sign = -1.0
            term = term[1:].strip()
        elif term.startswith('+'):
            term = term[1:].strip()

        # Find which unknown this term contains
        found_unknown = None
        for u in unknowns:
            # Check if unknown appears as a separate token
            pattern = r'(?<![a-zA-Z0-9_])' + u + r'(?![a-zA-Z0-9_])'
            if re.search(pattern, term):
                found_unknown = u
                break

        if found_unknown is None:
            continue  # Constant term, ignore for now

        # Extract coefficient (everything except the unknown)
        coeff_str = re.sub(r'(?<![a-zA-Z0-9_])' + found_unknown + r'(?![a-zA-Z0-9_])', '', term)
        coeff_str = coeff_str.replace('*', ' ').strip()

        # Evaluate the coefficient
        if coeff_str == '' or coeff_str == '*':
            coeff = 1.0
        else:
            # Parse coefficient which may contain i, omega, etc.
            coeff = evaluate_coefficient(coeff_str, local_scope)

        coeffs[found_unknown] = coeffs[found_unknown] + sign * coeff

    return coeffs

def evaluate_coefficient(coeff_str, local_scope):
    """Evaluate a coefficient expression like 'i*omega' or '2.0'"""
    coeff_str = coeff_str.strip()
    if not coeff_str:
        return 1.0

    # Tokenize and evaluate
    tokens = coeff_str.split()
    result = 1.0

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token == 'i':
            # Imaginary unit
            if HAS_TORCH:
                result = result * torch.complex(torch.tensor(0.0), torch.tensor(1.0))
            else:
                result = result * complex(0, 1)
        elif token.replace('.', '').replace('-', '').isdigit():
            result = result * float(token)
        elif token in local_scope:
            val = local_scope[token]
            if is_torch_tensor(val):
                result = result * val.item() if val.numel() == 1 else result * val
            else:
                result = result * val
        elif token in symbol_table:
            val = symbol_table[token]["value"]
            if is_torch_tensor(val):
                result = result * val.item() if val.numel() == 1 else result * val
            else:
                result = result * val

    return result

def solve_equations(equations, local_scope):
    """
    Solve a system of linear equations.
    equations: list of equation strings like ['x0 = a + b', 'v0 = i*omega*a - i*omega*b']
    Returns: dict mapping unknowns to their solved values
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for solve")

    # Parse equations to get LHS and RHS
    parsed = []
    for eq in equations:
        if isinstance(eq, tuple) and eq[0] == "equation_string":
            eq = eq[1]
        lhs, rhs = parse_equation(eq, local_scope)
        parsed.append((lhs, rhs))

    # Collect all identifiers from RHS to find unknowns
    # Unknowns are variables that appear in RHS but are not in local_scope or symbol_table
    all_rhs_vars = set()
    for lhs, rhs in parsed:
        # Find all identifiers in RHS
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rhs)
        all_rhs_vars.update(tokens)

    # Filter out known variables and special tokens
    special = {'i', 'exp', 'sin', 'cos', 'sqrt'}
    unknowns = []
    for var in all_rhs_vars:
        if var in special:
            continue
        if var in local_scope:
            continue
        if var in symbol_table:
            continue
        unknowns.append(var)

    # Sort unknowns for consistent ordering
    unknowns = sorted(unknowns)

    if len(unknowns) != len(parsed):
        raise ValueError(f"Number of unknowns ({len(unknowns)}: {unknowns}) must match number of equations ({len(parsed)})")

    # Build coefficient matrix and RHS vector
    n = len(unknowns)
    use_complex = False

    # First pass: check if we need complex numbers
    for lhs, rhs in parsed:
        if 'i' in rhs:
            use_complex = True
            break

    # Build the system
    if use_complex:
        A = torch.zeros((n, n), dtype=torch.complex64)
        b = torch.zeros(n, dtype=torch.complex64)
    else:
        A = torch.zeros((n, n), dtype=torch.float32)
        b = torch.zeros(n, dtype=torch.float32)

    for i, (lhs, rhs) in enumerate(parsed):
        # Get LHS value (known)
        if lhs in local_scope:
            lhs_val = local_scope[lhs]
        elif lhs in symbol_table:
            lhs_val = symbol_table[lhs]["value"]
        else:
            raise ValueError(f"Unknown variable on LHS: {lhs}")

        if is_torch_tensor(lhs_val):
            lhs_val = lhs_val.item() if lhs_val.numel() == 1 else lhs_val

        b[i] = lhs_val

        # Extract coefficients for each unknown
        coeffs = extract_linear_coefficients(rhs, unknowns, local_scope)

        for j, u in enumerate(unknowns):
            coeff = coeffs[u]
            if is_torch_tensor(coeff):
                coeff = coeff.item() if coeff.numel() == 1 else coeff
            A[i, j] = coeff

    # Solve the system
    solution = torch.linalg.solve(A, b)

    # Return as dictionary
    result = {}
    for i, u in enumerate(unknowns):
        result[u] = solution[i]

    return result, unknowns

def call_builtin(func_name, args):
    """Call a built-in function. Returns None if not a built-in."""
    import math

    # exp
    if func_name == "exp":
        if len(args) == 1:
            x = args[0]
            if is_torch_tensor(x):
                return torch.exp(x)
            elif is_scalar(x):
                return math.exp(x)
            elif isinstance(x, list):
                # Element-wise
                def apply_exp(v):
                    if isinstance(v, list):
                        return [apply_exp(e) for e in v]
                    return math.exp(v)
                return apply_exp(x)
        return None

    # sin
    if func_name == "sin":
        if len(args) == 1:
            x = args[0]
            if is_torch_tensor(x):
                return torch.sin(x)
            return math.sin(x)
        return None

    # real (extract real part)
    if func_name == "real":
        if len(args) == 1:
            x = args[0]
            if is_torch_tensor(x):
                if x.is_complex():
                    return x.real
                return x
            elif isinstance(x, complex):
                return x.real
            return x
        return None

    # imag (extract imaginary part)
    if func_name == "imag":
        if len(args) == 1:
            x = args[0]
            if is_torch_tensor(x):
                if x.is_complex():
                    return x.imag
                return torch.tensor(0.0)
            elif isinstance(x, complex):
                return x.imag
            return 0.0
        return None

    # cos
    if func_name == "cos":
        if len(args) == 1:
            x = args[0]
            if is_torch_tensor(x):
                return torch.cos(x)
            return math.cos(x)
        return None

    # pi constant
    if func_name == "pi":
        if len(args) == 0:
            if HAS_TORCH:
                return torch.tensor(math.pi, requires_grad=True)
            return math.pi
        return None

    # factorial(n)
    if func_name == "factorial":
        if len(args) == 1:
            n = args[0]
            if is_torch_tensor(n):
                n = int(n.item())
            else:
                n = int(n)
            result = math.factorial(n)
            if HAS_TORCH:
                return torch.tensor(float(result), requires_grad=True)
            return float(result)
        return None

    # hermite(n, x) - Physicist's Hermite polynomials
    if func_name == "hermite":
        if len(args) == 2:
            n = args[0]
            x = args[1]
            if is_torch_tensor(n):
                n = int(n.item())
            else:
                n = int(n)

            # Compute H_n(x) using recurrence: H_n = 2x*H_{n-1} - 2(n-1)*H_{n-2}
            if n == 0:
                if is_torch_tensor(x):
                    return torch.ones_like(x)
                return 1.0
            elif n == 1:
                return 2.0 * x
            else:
                if not is_torch_tensor(x) and HAS_TORCH:
                    x = to_torch(x)
                H_prev2 = torch.ones_like(x) if is_torch_tensor(x) else 1.0  # H_0
                H_prev1 = 2.0 * x  # H_1
                for k in range(2, n + 1):
                    H_curr = 2.0 * x * H_prev1 - 2.0 * (k - 1) * H_prev2
                    H_prev2 = H_prev1
                    H_prev1 = H_curr
                return H_prev1
        return None

    # PDESolve - Solve partial differential equations symbolically
    # Currently supports: Schrödinger equation for harmonic oscillator
    if func_name == "PDESolve":
        if len(args) >= 1 and HAS_TORCH:
            eqn = args[0]
            if isinstance(eqn, tuple) and eqn[0] == "equation_string":
                eqn = eqn[1]

            # Check if it's the quantum harmonic oscillator Schrödinger equation
            eqn_lower = eqn.lower().replace(" ", "")
            is_schrodinger_ho = ("hbar" in eqn_lower or "ℏ" in eqn_lower) and \
                                ("d2" in eqn_lower or "∂2" in eqn_lower or "d^2" in eqn_lower) and \
                                ("omega" in eqn_lower or "ω" in eqn_lower)

            if is_schrodinger_ho:
                # Return a solver object that can generate eigenstates
                return ("qho_solutions",)
            else:
                raise ValueError(f"PDESolve: Unrecognized equation type: {eqn}")
        return None

    # # qho_eigenstate(n, x, hbar, m, omega) - Quantum harmonic oscillator eigenstate
    # if func_name == "qho_eigenstate":
    #     if len(args) == 5 and HAS_TORCH:
    #         n, x, hbar, m, omega = args
    #         if is_torch_tensor(n):
    #             n = int(n.item())
    #         else:
    #             n = int(n)

    #         # ψ_n(x) = (1/√(2^n n!)) * (mω/πℏ)^(1/4) * exp(-mωx²/2ℏ) * H_n(√(mω/ℏ) * x)
    #         if not is_torch_tensor(x):
    #             x = to_torch(x)
    #         if not is_torch_tensor(hbar):
    #             hbar = to_torch(hbar)
    #         if not is_torch_tensor(m):
    #             m = to_torch(m)
    #         if not is_torch_tensor(omega):
    #             omega = to_torch(omega)

    #         # Compute xi = sqrt(m*omega/hbar) * x
    #         xi = torch.sqrt(m * omega / hbar) * x

    #         # Normalization: (1/sqrt(2^n * n!)) * (m*omega/(pi*hbar))^(1/4)
    #         norm = (1.0 / torch.sqrt(torch.tensor(2.0**n * math.factorial(n)))) * \
    #                (m * omega / (math.pi * hbar)) ** 0.25

    #         # Gaussian: exp(-xi^2 / 2)
    #         gaussian = torch.exp(-xi**2 / 2.0)

    #         # Hermite polynomial H_n(xi)
    #         H_n = call_builtin("hermite", [n, xi])

    #         return norm * gaussian * H_n
    #     return None

    # # qho_energy(n, hbar, omega) - Quantum harmonic oscillator energy eigenvalue
    # if func_name == "qho_energy":
    #     if len(args) == 3 and HAS_TORCH:
    #         n, hbar, omega = args
    #         if is_torch_tensor(n):
    #             n = int(n.item())
    #         else:
    #             n = int(n)
    #         if not is_torch_tensor(hbar):
    #             hbar = to_torch(hbar)
    #         if not is_torch_tensor(omega):
    #             omega = to_torch(omega)

    #         # E_n = ℏω(n + 1/2)
    #         return hbar * omega * (n + 0.5)
    #     return None

    # mean
    if func_name == "mean":
        if len(args) == 1 and HAS_TORCH:
            tensor = args[0]
            if not is_torch_tensor(tensor):
                tensor = to_torch(tensor)
            return tensor.mean()
        return None

    # print
    if func_name == "print":
        if len(args) == 1:
            value = args[0]
            display_value = from_torch(value) if is_torch_tensor(value) else value
            print(f"  {display_value}")
            return 0.0  # Return dummy value
        return None

    # grad
    if func_name == "grad":
        if len(args) == 2 and HAS_TORCH:
            output, input_var = args[0], args[1]
            if is_torch_tensor(output) and is_torch_tensor(input_var):
                grad_value = torch.autograd.grad(output, input_var,
                    create_graph=True, retain_graph=True)[0]
                return grad_value
        return None

    # solve - Solve linear system
    # Can be called as:
    #   solve(A, b) - matrix form: Ax = b
    #   solve(eq1, eq2, ...) - equation strings: 'x0 = a + b', 'v0 = i*omega*a - i*omega*b'
    if func_name == "solve":
        if not HAS_TORCH:
            return None

        # Check if first arg is an equation string
        if len(args) >= 1:
            first_arg = args[0]
            is_equation = (isinstance(first_arg, tuple) and first_arg[0] == "equation_string") or \
                          (isinstance(first_arg, str) and '=' in first_arg)

            if is_equation:
                # Symbolic equation solving
                equations = []
                for arg in args:
                    if isinstance(arg, tuple) and arg[0] == "equation_string":
                        equations.append(arg[1])
                    elif isinstance(arg, str):
                        equations.append(arg)
                    else:
                        raise ValueError(f"Invalid equation: {arg}")

                # Need to pass local_scope - get it from the call context
                # For now, use symbol_table as scope
                result_dict, unknowns = solve_equations(equations, {})
                # Return as tuple of values in order of unknowns
                return ("solved_values", result_dict, unknowns)

        # Matrix form: solve(A, b)
        if len(args) == 2:
            A = args[0]
            b = args[1]
            if not is_torch_tensor(A):
                A = to_torch(A)
            if not is_torch_tensor(b):
                b = to_torch(b)
            # Ensure A is 2D and b is 1D
            if A.dim() == 2 and b.dim() == 1:
                result = torch.linalg.solve(A.float(), b.float())
                return result
            raise ValueError("solve(A, b) requires 2D matrix A and 1D vector b")
        return None

    # sqrt
    if func_name == "sqrt":
        if len(args) == 1:
            x = args[0]
            if is_torch_tensor(x):
                return torch.sqrt(x)
            return x ** 0.5
        return None

    # train(net, X, y, epochs, lr) - Train network using PyTorch backend
    # Uses custom loss if defined in class, otherwise MSE
    if func_name == "train":
        if len(args) == 5 and HAS_TORCH:
            instance, X, y, epochs, lr = args

            # Handle instance tuple from class instantiation
            if isinstance(instance, tuple) and instance[0] == "instance":
                instance = instance[1]

            if not isinstance(instance, dict) or "bound_params" not in instance:
                raise TypeError("train() requires a network instance")

            # Get parameters from instance
            bound_params = instance["bound_params"]

            # Convert data to torch tensors
            if not is_torch_tensor(X):
                X = to_torch(X)
            if not is_torch_tensor(y):
                y = to_torch(y)

            # Create parameter tensors with gradients
            param_tensors = {}
            for name, value in bound_params.items():
                if isinstance(value, dict):  # Skip function references
                    param_tensors[name] = value
                else:
                    if not is_torch_tensor(value):
                        value = to_torch(value)
                    param_tensors[name] = value.clone().detach().requires_grad_(True)

            epochs = int(epochs)
            lr = float(lr)
            has_custom_loss = instance.get("has_loss", False)

            # Training loop
            if has_custom_loss:
                print(f"  Training with custom loss for {epochs} epochs with lr={lr}")
            else:
                print(f"  Training with MSE loss for {epochs} epochs with lr={lr}")

            for epoch in range(epochs):
                total_loss = torch.tensor(0.0, requires_grad=True)

                # Forward pass for each sample
                for i in range(X.shape[0]):
                    x_i = X[i]
                    y_i = y[i]

                    # For custom loss (like HNN), input needs requires_grad for gradient computation
                    if has_custom_loss:
                        x_i = x_i.clone().detach().requires_grad_(True)

                    # Evaluate network with current parameters
                    local_scope = dict(param_tensors)

                    # Handle lambda params
                    lambda_params = instance["lambda_params"]
                    for (param_name, _), arg_val in zip(lambda_params, [x_i]):
                        local_scope[param_name] = arg_val

                    # Handle loop if present
                    if instance.get("has_loop"):
                        loop_var = instance["loop_var"]
                        loop_count_var = instance["loop_count_var"]
                        loop_body = instance["loop_body"]

                        n = local_scope.get(loop_count_var, 0)
                        if is_torch_tensor(n):
                            n = int(n.item())
                        else:
                            n = int(n)

                        for j in range(n):
                            local_scope[loop_var] = float(j)
                            for stmt in loop_body:
                                if stmt and stmt[0] == "loop_assign":
                                    var_name = stmt[1]
                                    expr_ast = stmt[2]
                                    local_scope[var_name] = evaluate_ast(expr_ast, local_scope)

                    # Evaluate forward pass
                    pred = evaluate_ast(instance["body"], local_scope)
                    if not is_torch_tensor(pred):
                        pred = to_torch(pred)

                    # Compute loss
                    if has_custom_loss:
                        # Use custom loss function defined in class
                        loss_scope = dict(local_scope)
                        loss_params = instance["loss_params"]
                        # Bind loss parameters: typically (pred, target) or custom
                        loss_args = [pred, y_i]  # Default binding
                        for (param_name, _), arg_val in zip(loss_params, loss_args):
                            loss_scope[param_name] = arg_val
                        loss_i = evaluate_ast(instance["loss_body"], loss_scope)
                    else:
                        # Default MSE loss
                        loss_i = (pred - y_i) ** 2

                    if not is_torch_tensor(loss_i):
                        loss_i = to_torch(loss_i)
                    total_loss = total_loss + loss_i

                # Backward pass
                total_loss.backward()

                # Update parameters
                with torch.no_grad():
                    for name, tensor in param_tensors.items():
                        if isinstance(tensor, torch.Tensor) and tensor.grad is not None:
                            tensor -= lr * tensor.grad
                            tensor.grad.zero_()

                # Print progress
                if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                    print(f"    Epoch {epoch}: Loss = {total_loss.item():.6f}")

            # Create new instance with trained parameters
            new_bound_params = {}
            for name, value in param_tensors.items():
                if isinstance(value, torch.Tensor):
                    new_bound_params[name] = value.detach().clone().requires_grad_(True)
                else:
                    # Keep non-tensor values (like function references) as-is
                    new_bound_params[name] = value

            trained_instance = {
                "class_name": instance["class_name"],
                "bound_params": new_bound_params,
                "lambda_params": instance["lambda_params"],
                "return_type": instance["return_type"],
                "body": instance["body"],
                "has_loop": instance.get("has_loop", False),
                "loop_var": instance.get("loop_var"),
                "loop_count_var": instance.get("loop_count_var"),
                "loop_body": instance.get("loop_body"),
                "has_loss": instance.get("has_loss", False),
                "loss_params": instance.get("loss_params"),
                "loss_body": instance.get("loss_body")
            }

            return ("instance", trained_instance)
        return None

    # evaluate(net, X, y) - Compute loss on data using custom loss if defined
    if func_name == "evaluate":
        if len(args) == 3 and HAS_TORCH:
            instance, X, y = args

            # Handle instance tuple
            if isinstance(instance, tuple) and instance[0] == "instance":
                instance = instance[1]

            if not isinstance(instance, dict) or "bound_params" not in instance:
                raise TypeError("evaluate() requires a network instance")

            # Convert data to torch
            if not is_torch_tensor(X):
                X = to_torch(X)
            if not is_torch_tensor(y):
                y = to_torch(y)

            total_loss = 0.0
            n_samples = X.shape[0]
            has_custom_loss = instance.get("has_loss", False)

            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                # For custom loss (like HNN), input needs requires_grad for gradient computation
                if has_custom_loss:
                    x_i = x_i.clone().detach().requires_grad_(True)

                # Build scope
                local_scope = dict(instance["bound_params"])
                lambda_params = instance["lambda_params"]
                for (param_name, _), arg_val in zip(lambda_params, [x_i]):
                    local_scope[param_name] = arg_val

                # Handle loop
                if instance.get("has_loop"):
                    loop_var = instance["loop_var"]
                    loop_count_var = instance["loop_count_var"]
                    loop_body = instance["loop_body"]

                    n = local_scope.get(loop_count_var, 0)
                    if is_torch_tensor(n):
                        n = int(n.item())
                    else:
                        n = int(n)

                    for j in range(n):
                        local_scope[loop_var] = float(j)
                        for stmt in loop_body:
                            if stmt and stmt[0] == "loop_assign":
                                var_name = stmt[1]
                                expr_ast = stmt[2]
                                local_scope[var_name] = evaluate_ast(expr_ast, local_scope)

                pred = evaluate_ast(instance["body"], local_scope)

                # Compute loss
                if has_custom_loss:
                    # Use custom loss function
                    loss_scope = dict(local_scope)
                    loss_params = instance["loss_params"]
                    loss_args = [pred, y_i]
                    for (param_name, _), arg_val in zip(loss_params, loss_args):
                        loss_scope[param_name] = arg_val
                    loss_i = evaluate_ast(instance["loss_body"], loss_scope)
                    if is_torch_tensor(loss_i):
                        loss_i = loss_i.item()
                else:
                    # Default MSE loss
                    if is_torch_tensor(pred):
                        pred = pred.item()
                    if is_torch_tensor(y_i):
                        y_i = y_i.item()
                    loss_i = (pred - y_i) ** 2

                total_loss += loss_i

            return total_loss / n_samples  # Return mean loss
        return None

    # ======================================================
    # VISUALIZATION FUNCTIONS
    # ======================================================
    import numpy as np

    # plot(x, y) or plot(y) - 2D line plot
    if func_name == "plot":
        if len(args) >= 1:
            if len(args) == 1:
                y = args[0]
                if is_torch_tensor(y):
                    y = y.detach().tolist()
                if isinstance(y, list):
                    y = np.array(y)
                x = np.arange(len(y))
            else:
                x, y = args[0], args[1]
                if is_torch_tensor(x):
                    x = x.detach().tolist()
                if isinstance(x, list):
                    x = np.array(x)
                if is_torch_tensor(y):
                    y = y.detach().tolist()
                if isinstance(y, list):
                    y = np.array(y)

            title = args[2] if len(args) > 2 else "Physika Plot"

            if HAS_PYVISTA:
                # Use PyVista for 2D plot
                points = np.column_stack([x, y, np.zeros_like(x)])
                plotter = pv.Plotter()
                plotter.add_lines(points, color='blue', width=3)
                plotter.add_title(str(title))
                plotter.camera_position = 'xy'
                plotter.show()
            elif HAS_MATPLOTLIB:
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, 'b-', linewidth=2)
                plt.title(str(title))
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid(True)
                plt.show()
            else:
                print("  [plot] No visualization backend available (install pyvista or matplotlib)")
            return 0.0
        return None

    # plot3d(x, y, z) - 3D line plot
    if func_name == "plot3d":
        if len(args) >= 3:
            x, y, z = args[0], args[1], args[2]
            if is_torch_tensor(x):
                x = x.detach().tolist()
            if is_torch_tensor(y):
                y = y.detach().tolist()
            if is_torch_tensor(z):
                z = z.detach().tolist()
            if isinstance(x, list):
                x = np.array(x)
            if isinstance(y, list):
                y = np.array(y)
            if isinstance(z, list):
                z = np.array(z)

            title = args[3] if len(args) > 3 else "Physika 3D Plot"

            if HAS_PYVISTA:
                points = np.column_stack([x, y, z])
                plotter = pv.Plotter()
                plotter.add_lines(points, color='blue', width=3)
                plotter.add_title(str(title))
                plotter.show()
            else:
                print("  [plot3d] PyVista not available")
            return 0.0
        return None

    # phase_plot(x, v) - Phase space plot
    # if func_name == "phase_plot":
    #     if len(args) >= 2:
    #         x, v = args[0], args[1]
    #         if is_torch_tensor(x):
    #             x = x.detach().tolist()
    #         if is_torch_tensor(v):
    #             v = v.detach().tolist()
    #         if isinstance(x, list):
    #             x = np.array(x)
    #         if isinstance(v, list):
    #             v = np.array(v)

    #         title = args[2] if len(args) > 2 else "Phase Space"

    #         if HAS_PYVISTA:
    #             points = np.column_stack([x, v, np.zeros_like(x)])
    #             plotter = pv.Plotter()
    #             plotter.add_lines(points, color='red', width=3)
    #             plotter.add_title(str(title))
    #             plotter.add_axes_at_origin(labels_off=False)
    #             plotter.camera_position = 'xy'
    #             plotter.show()
    #         elif HAS_MATPLOTLIB:
    #             plt.figure(figsize=(8, 8))
    #             plt.plot(x, v, 'r-', linewidth=2)
    #             plt.title(str(title))
    #             plt.xlabel('Position x')
    #             plt.ylabel('Velocity v')
    #             plt.grid(True)
    #             plt.axis('equal')
    #             plt.show()
    #         else:
    #             print("  [phase_plot] No visualization backend available")
    #         return 0.0
        # return None


    # linspace for generating ranges
    if func_name == "linspace":
        if len(args) == 3:
            start, end, n = args
            if is_torch_tensor(start):
                start = start.item()
            if is_torch_tensor(end):
                end = end.item()
            if is_torch_tensor(n):
                n = int(n.item())
            else:
                n = int(n)
            if HAS_TORCH:
                return torch.linspace(float(start), float(end), n)
            else:
                return list(np.linspace(float(start), float(end), n))
        return None

    # visualize(func, fixed_args..., time_min, time_max, n_points) - Visualize a Physika function
    # The function's THIRD parameter (index 2) is varied over [time_min, time_max]
    # For U(k, m, t, x0, v0): visualize(U, k, m, x0, v0, time_min, time_max, n_points)
    if func_name == "visualize":
        if len(args) >= 4:
            # First arg is the function (as dict or tuple reference)
            func_ref = args[0]

            # Get the function definition
            if isinstance(func_ref, dict) and "params" in func_ref:
                func_def = func_ref
            elif isinstance(func_ref, tuple) and func_ref[0] == "func_ref":
                func_def = func_ref[2]
            else:
                print(f"  [visualize] Invalid function reference: {type(func_ref)}")
                return 0.0

            # Last 3 args are time_min, time_max, n_points
            time_min = args[-3]
            time_max = args[-2]
            n_points = args[-1]

            if is_torch_tensor(time_min):
                time_min = time_min.item()
            if is_torch_tensor(time_max):
                time_max = time_max.item()
            if is_torch_tensor(n_points):
                n_points = int(n_points.item())
            else:
                n_points = int(n_points)

            # Fixed args are everything between func and the last 3
            fixed_args = list(args[1:-3])

            # Generate time values
            time_vals = np.linspace(float(time_min), float(time_max), n_points)
            y_values = []

            # Evaluate function at each time point
            # Function signature is (k, m, t, x0, v0) - t is at index 2
            for t in time_vals:
                # Build args: [k, m, t, x0, v0]
                # fixed_args = [k, m, x0, v0], insert t at position 2
                call_args = fixed_args[:2] + [float(t)] + fixed_args[2:]

                # Convert to torch tensors for evaluation
                torch_args = []
                for a in call_args:
                    if is_torch_tensor(a):
                        torch_args.append(a)
                    else:
                        torch_args.append(torch.tensor(float(a), requires_grad=True) if HAS_TORCH else a)

                result = evaluate_function(func_def, torch_args)

                # Take real part if complex
                if is_torch_tensor(result):
                    if result.is_complex():
                        result = result.real
                    y_values.append(result.item())
                elif isinstance(result, complex):
                    y_values.append(result.real)
                else:
                    y_values.append(float(result))

            y_values = np.array(y_values)

            # Plot
            if HAS_PYVISTA:
                points = np.column_stack([time_vals, y_values, np.zeros_like(time_vals)])
                plotter = pv.Plotter()
                plotter.add_lines(points, color='blue', width=3)
                plotter.add_title("Physika Function Visualization")
                plotter.camera_position = 'xy'
                plotter.show()
            elif HAS_MATPLOTLIB:
                plt.figure(figsize=(10, 6))
                plt.plot(time_vals, y_values, 'b-', linewidth=2)
                plt.title("Physika Function Visualization")
                plt.xlabel('t')
                plt.ylabel('f(t)')
                plt.grid(True)
                plt.show()
            else:
                print("  [visualize] No visualization backend available")

            return 0.0
        return None

    # animate(func, fixed_args..., time_min, time_max, [n_points]) - Animate a Physika function
    # n_points is optional, defaults to 200
    if func_name == "animate":
        if len(args) >= 3:
            func_ref = args[0]

            if isinstance(func_ref, dict) and "params" in func_ref:
                func_def = func_ref
            elif isinstance(func_ref, tuple) and func_ref[0] == "func_ref":
                func_def = func_ref[2]
            else:
                print(f"  [animate] Invalid function reference: {type(func_ref)}")
                return 0.0

            # Detect if n_points was provided
            # n_points is typically a large integer (>= 10), time values are typically small floats
            last_arg = args[-1]
            if is_torch_tensor(last_arg):
                last_arg_val = last_arg.item()
            else:
                last_arg_val = last_arg

            # Check if last arg is an integer (no decimal part) and >= 10 (typical n_points range)
            def is_integer_like(val):
                if isinstance(val, int):
                    return True
                if isinstance(val, float):
                    return val == int(val)
                return False

            is_n_points_provided = is_integer_like(last_arg_val) and last_arg_val >= 10

            if is_n_points_provided:
                time_min = args[-3]
                time_max = args[-2]
                n_points = int(last_arg_val)
                fixed_args = list(args[1:-3])
            else:
                # n_points not provided, use default
                time_min = args[-2]
                time_max = args[-1]
                n_points = 200
                fixed_args = list(args[1:-2])

            if is_torch_tensor(time_min):
                time_min = time_min.item()
            if is_torch_tensor(time_max):
                time_max = time_max.item()
            time_vals = np.linspace(float(time_min), float(time_max), n_points)
            x_values = []

            for t in time_vals:
                call_args = fixed_args[:2] + [float(t)] + fixed_args[2:]
                torch_args = []
                for a in call_args:
                    if is_torch_tensor(a):
                        torch_args.append(a)
                    else:
                        torch_args.append(torch.tensor(float(a), requires_grad=True) if HAS_TORCH else a)

                result = evaluate_function(func_def, torch_args)

                if is_torch_tensor(result):
                    if result.is_complex():
                        result = result.real
                    x_values.append(result.item())
                elif isinstance(result, complex):
                    x_values.append(result.real)
                else:
                    x_values.append(float(result))

            x_values = np.array(x_values)

            # Compute velocity via numerical differentiation: v = dx/dt
            dt = (float(time_max) - float(time_min)) / (n_points - 1)
            v_values = np.gradient(x_values, dt)

            # Animate
            if HAS_PYVISTA:
                plotter = pv.Plotter()
                plotter.add_title(
                    "Physika \n\nHarmonic Oscillator Animation",
                    # position="upper_edge",
                    font_size=24,
                    font="times",
                    shadow=True
                )

                # plotter.add_text(
                #     "Harmonic Oscillator Animation",
                #     position=(0.3, 0.90),  # normalized coordinates
                #     viewport=True,
                #     font_size=14,
                #     font="times"
                # )
                sphere = pv.Sphere(radius=0.1, center=(x_values[0], 0, 0))
                plotter.add_mesh(sphere, color='blue')
                plotter.add_mesh(pv.Line((-2, 0, 0), (2, 0, 0)), color='black', line_width=3)

                # Red dot indicating initial position x0
                x0_marker = pv.Sphere(radius=0.03, center=(x_values[0], 0, 0))
                plotter.add_mesh(x0_marker, color='red')
               
                plotter.camera_position = [(0, 5, 0), (0, 0, 0), (0, 0, 1)]

                # State for pause and loop
                anim_state = {"paused": False, "running": True}

                def on_key_press(key):
                    if key == "space":
                        anim_state["paused"] = not anim_state["paused"]
                    elif key == "q" or key == "Escape":
                        anim_state["running"] = False

                plotter.add_key_event("space", lambda: on_key_press("space"))
                plotter.add_key_event("q", lambda: on_key_press("q"))

                # Add text actor for dynamic values
                # Using position=(10, 10) for absolute positioning returns a vtkTextActor
                # which supports SetInput() for flicker-free updates
                text_actor = plotter.add_text(
                    f"t = {time_vals[0]:.3f}\nx = {x_values[0]:.4f}\nv = {v_values[0]:.4f}\n[SPACE: pause | Q: quit]",
                    position=(10, 10), font_size=15, font="times"
                )

                plotter.show(auto_close=False, interactive_update=True)

                import time as time_module

                # Animation loop with reset capability
                while anim_state["running"]:
                    for i, x in enumerate(x_values):
                        if not anim_state["running"]:
                            break

                        # Handle pause
                        while anim_state["paused"] and anim_state["running"]:
                            plotter.update()
                            time_module.sleep(0.05)

                        if not anim_state["running"]:
                            break

                        sphere.points = pv.Sphere(radius=0.1, center=(x, 0, 0)).points

                        # Update text in place (no blinking) using VTK's SetInput
                        pause_status = "[PAUSED]" if anim_state["paused"] else "[SPACE: pause | Q: quit]"
                        text_actor.SetInput(
                            f"t = {time_vals[i]:.3f}\nx = {x_values[i]:.4f}\nv = {v_values[i]:.4f}\n{pause_status}"
                        )

                        plotter.update()
                        time_module.sleep(0.03)

                    # Loop resets automatically to beginning

                plotter.close()
            elif HAS_MATPLOTLIB:
                from matplotlib.animation import FuncAnimation

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.set_xlim(-2, 2)
                ax.set_ylim(-0.5, 0.5)
                ax.set_aspect('equal')
                ax.axhline(y=0, color='black', linewidth=2)
                ax.set_title("Harmonic Oscillator Animation [SPACE: pause/resume | R: reset]")

                mass, = ax.plot([], [], 'bo', markersize=20)
                spring, = ax.plot([], [], 'gray', linewidth=2)

                # Red dot indicating initial position x0
                ax.plot([x_values[0]], [0], 'ro', markersize=8, label=f'x₀ = {x_values[0]:.2f}')

                # Text elements for dynamic values
                info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                    verticalalignment='top', fontfamily='monospace',
                                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Animation state
                anim_state = {"paused": False, "frame": 0}
                ani_ref = [None]  # Use list to allow modification in nested function

                def init():
                    mass.set_data([], [])
                    spring.set_data([], [])
                    info_text.set_text('')
                    return mass, spring, info_text

                def anim(i):
                    if anim_state["paused"]:
                        i = anim_state["frame"]
                    else:
                        anim_state["frame"] = i

                    mass.set_data([x_values[i]], [0])
                    spring.set_data([0, x_values[i]], [0, 0])

                    # Update info text with position, velocity, time
                    pause_str = " [PAUSED]" if anim_state["paused"] else ""
                    info_text.set_text(f't = {time_vals[i]:.3f}{pause_str}\nx = {x_values[i]:.4f}\nv = {v_values[i]:.4f}')

                    return mass, spring, info_text

                def on_key(event):
                    if event.key == ' ':
                        anim_state["paused"] = not anim_state["paused"]
                        if not anim_state["paused"] and ani_ref[0] is not None:
                            # Resume from current frame
                            ani_ref[0].frame_seq = ani_ref[0].new_frame_seq()
                    elif event.key == 'r':
                        # Reset to beginning
                        anim_state["frame"] = 0
                        anim_state["paused"] = False
                        if ani_ref[0] is not None:
                            ani_ref[0].frame_seq = ani_ref[0].new_frame_seq()

                fig.canvas.mpl_connect('key_press_event', on_key)

                # Use repeat=True for looping animation
                ani = FuncAnimation(fig, anim, init_func=init, frames=len(x_values),
                                    interval=30, blit=True, repeat=True)
                ani_ref[0] = ani
                plt.show()
            else:
                print("  [animate] No visualization backend available")

            return 0.0
        return None

    # Not a built-in
    return None

def evaluate_function(func_def, args):
    """Evaluate a function with given arguments."""
    params = func_def["params"]

    # Check for element-wise application: scalar function on vector/tensor input
    if len(params) == 1 and not func_def.get("has_loop"):
        param_name, param_type = params[0]
        arg_value = args[0]

        # If function expects scalar but got vector/tensor, apply element-wise
        if param_type == "ℝ" and (is_vector(arg_value) or is_matrix(arg_value)):
            if is_torch_tensor(arg_value):
                # Apply to each element of torch tensor
                flat = arg_value.flatten()
                results = []
                for i in range(flat.numel()):
                    elem = flat[i]
                    local_scope = {param_name: elem}
                    results.append(evaluate_ast(func_def["body"], local_scope))
                return torch.stack(results).reshape(arg_value.shape)
            else:
                # Apply to nested list recursively
                def apply_elementwise(v):
                    if isinstance(v, list):
                        return [apply_elementwise(e) for e in v]
                    local_scope = {param_name: v}
                    return evaluate_ast(func_def["body"], local_scope)
                return apply_elementwise(arg_value)

    # Build local scope with parameters
    local_scope = {}
    for (param_name, param_type), arg_value in zip(params, args):
        # Handle callable parameters (instances)
        if isinstance(arg_value, dict) and "bound_params" in arg_value:
            local_scope[param_name] = arg_value  # Store instance directly
        elif isinstance(arg_value, tuple) and arg_value[0] == "func_ref":
            local_scope[param_name] = arg_value[2]  # Store function def
        else:
            local_scope[param_name] = arg_value

    # Execute body statements (assignments, declarations, tuple unpacking)
    for stmt in func_def.get("statements", []):
        if stmt is None:
            continue
        stmt_type = stmt[0]

        if stmt_type == "body_assign":
            # Simple assignment: x = expr
            var_name = stmt[1]
            expr_ast = stmt[2]
            local_scope[var_name] = evaluate_ast(expr_ast, local_scope)

        elif stmt_type == "body_decl":
            # Typed declaration: x : R = expr
            var_name = stmt[1]
            # type_spec = stmt[2]  # We can use this for type checking later
            expr_ast = stmt[3]
            local_scope[var_name] = evaluate_ast(expr_ast, local_scope)

        elif stmt_type == "body_tuple_unpack":
            # Tuple unpacking: a, b = expr
            var_names = stmt[1]
            expr_ast = stmt[2]
            result = evaluate_ast(expr_ast, local_scope)

            # Handle solved_values from solve()
            if isinstance(result, tuple) and result[0] == "solved_values":
                result_dict = result[1]
                unknowns = result[2]
                # Assign values in order
                for i, var_name in enumerate(var_names):
                    if i < len(unknowns):
                        local_scope[var_name] = result_dict[unknowns[i]]
            elif isinstance(result, (list, tuple)):
                # Regular tuple/list unpacking
                for i, var_name in enumerate(var_names):
                    if i < len(result):
                        local_scope[var_name] = result[i]
            elif is_torch_tensor(result):
                # Tensor unpacking
                for i, var_name in enumerate(var_names):
                    if i < result.numel():
                        local_scope[var_name] = result[i]

    # Handle functions with for loops
    if func_def.get("has_loop"):
        # Execute init statements
        for stmt in func_def.get("init_stmts", []):
            if stmt and stmt[0] == "init_assign":
                var_name = stmt[1]
                expr_ast = stmt[2]
                local_scope[var_name] = evaluate_ast(expr_ast, local_scope)

        # Get loop parameters
        loop_var = func_def["loop_var"]
        loop_count_var = func_def["loop_count_var"]
        loop_body = func_def["loop_body"]

        # Get loop count
        n = local_scope.get(loop_count_var, 0)
        if is_torch_tensor(n):
            n = int(n.item())
        else:
            n = int(n)

        # Execute loop
        for i in range(n):
            local_scope[loop_var] = float(i)
            for stmt in loop_body:
                if stmt and stmt[0] == "loop_assign":
                    var_name = stmt[1]
                    expr_ast = stmt[2]
                    local_scope[var_name] = evaluate_ast(expr_ast, local_scope)
                elif stmt and stmt[0] == "loop_pluseq":
                    var_name = stmt[1]
                    expr_ast = stmt[2]
                    current = local_scope.get(var_name, 0.0)
                    addition = evaluate_ast(expr_ast, local_scope)
                    local_scope[var_name] = element_wise_op(current, addition, lambda x, y: x + y)

    # Evaluate final expression
    result = evaluate_ast(func_def["body"], local_scope)
    return result

def evaluate_ast(node, local_scope):
    """Recursively evaluate an AST node."""
    if isinstance(node, tuple):
        op = node[0]
        
        if op == "num":
            return node[1]
        
        elif op == "var":
            var_name = node[1]
            # Check local scope first, then global
            if var_name in local_scope:
                return local_scope[var_name]
            elif var_name in symbol_table:
                return symbol_table[var_name]["value"]
            else:
                raise NameError(f"Undefined variable '{var_name}'")
        
        elif op == "add":
            left = evaluate_ast(node[1], local_scope)
            right = evaluate_ast(node[2], local_scope)
            return element_wise_op(left, right, lambda x, y: x + y)
        
        elif op == "sub":
            left = evaluate_ast(node[1], local_scope)
            right = evaluate_ast(node[2], local_scope)
            return element_wise_op(left, right, lambda x, y: x - y)
        
        elif op == "mul":
            left = evaluate_ast(node[1], local_scope)
            right = evaluate_ast(node[2], local_scope)
            return element_wise_op(left, right, lambda x, y: x * y)
        
        elif op == "div":
            left = evaluate_ast(node[1], local_scope)
            right = evaluate_ast(node[2], local_scope)
            return element_wise_op(left, right, lambda x, y: x / y)
        
        elif op == "matmul":
            left = evaluate_ast(node[1], local_scope)
            right = evaluate_ast(node[2], local_scope)
            return matmul(left, right)

        elif op == "pow":
            left = evaluate_ast(node[1], local_scope)
            right = evaluate_ast(node[2], local_scope)
            return element_wise_op(left, right, lambda x, y: x ** y)

        elif op == "array":
            # Array literal in function body
            elements = node[1]
            evaluated = [evaluate_ast(elem, local_scope) for elem in elements]
            return evaluated

        elif op == "string":
            # String literal - return as-is for equation parsing
            return ("equation_string", node[1])

        elif op == "imaginary":
            # Imaginary unit i
            if HAS_TORCH:
                return torch.complex(torch.tensor(0.0), torch.tensor(1.0))
            else:
                return complex(0, 1)

        elif op == "index":
            # Tensor indexing: W[i]
            var_name = node[1]
            index_ast = node[2]

            # Get the tensor
            if var_name in local_scope:
                tensor = local_scope[var_name]
            elif var_name in symbol_table:
                tensor = symbol_table[var_name]["value"]
            else:
                raise NameError(f"Undefined variable '{var_name}'")

            # Evaluate the index
            index = evaluate_ast(index_ast, local_scope)
            if is_torch_tensor(index):
                index = int(index.item())
            else:
                index = int(index)

            # Index the tensor
            if is_torch_tensor(tensor):
                return tensor[index]
            else:
                return tensor[index]

        elif op == "call_index":
            # Indexing a function call result: grad(H, x)[0]
            func_name = node[1]
            arg_asts = node[2]
            index_ast = node[3]

            # Evaluate the function call first
            args = [evaluate_ast(arg, local_scope) for arg in arg_asts]
            result = call_builtin(func_name, args)
            if result is None:
                # Try user-defined function
                if func_name in symbol_table:
                    func_entry = symbol_table[func_name]
                    if func_entry["type"] == "function":
                        result = evaluate_function(func_entry["value"], args)

            if result is None:
                raise NameError(f"Undefined function '{func_name}'")

            # Evaluate the index
            index = evaluate_ast(index_ast, local_scope)
            if is_torch_tensor(index):
                index = int(index.item())
            else:
                index = int(index)

            # Index the result
            return result[index]

        elif op == "call":
            func_name = node[1]
            arg_asts = node[2]

            # Evaluate arguments first
            args = [evaluate_ast(arg, local_scope) for arg in arg_asts]

            # Special handling for solve with equation strings
            if func_name == "solve" and len(args) >= 1:
                first_arg = args[0]
                is_equation = (isinstance(first_arg, tuple) and first_arg[0] == "equation_string") or \
                              (isinstance(first_arg, str) and '=' in first_arg)
                if is_equation:
                    equations = []
                    for arg in args:
                        if isinstance(arg, tuple) and arg[0] == "equation_string":
                            equations.append(arg[1])
                        elif isinstance(arg, str):
                            equations.append(arg)
                        else:
                            raise ValueError(f"Invalid equation: {arg}")
                    result_dict, unknowns = solve_equations(equations, local_scope)
                    return ("solved_values", result_dict, unknowns)

            # Check for built-in functions
            result = call_builtin(func_name, args)
            if result is not None:
                return result

            # Check if it's a callable in local scope (function or instance)
            if func_name in local_scope:
                callable_ref = local_scope[func_name]

                # If it's a function definition dict
                if isinstance(callable_ref, dict) and "params" in callable_ref:
                    return evaluate_function(callable_ref, args)

                # If it's an instance (network), call its lambda
                if isinstance(callable_ref, dict) and "bound_params" in callable_ref:
                    instance = callable_ref
                    lambda_params = instance["lambda_params"]

                    # Build scope with bound params and lambda params
                    instance_scope = dict(instance["bound_params"])
                    for (param_name, param_type), arg_value in zip(lambda_params, args):
                        instance_scope[param_name] = arg_value

                    # Handle instances with for loops
                    if instance.get("has_loop"):
                        loop_var = instance["loop_var"]
                        loop_count_var = instance["loop_count_var"]
                        loop_body = instance["loop_body"]

                        n = instance_scope.get(loop_count_var, 0)
                        if is_torch_tensor(n):
                            n = int(n.item())
                        else:
                            n = int(n)

                        for i in range(n):
                            instance_scope[loop_var] = float(i)
                            for stmt in loop_body:
                                if stmt and stmt[0] == "loop_assign":
                                    var_name = stmt[1]
                                    expr_ast = stmt[2]
                                    instance_scope[var_name] = evaluate_ast(expr_ast, instance_scope)

                    return evaluate_ast(instance["body"], instance_scope)

            # Check user-defined functions in symbol table
            if func_name not in symbol_table:
                raise NameError(f"Undefined function '{func_name}'")

            func_entry = symbol_table[func_name]
            if func_entry["type"] != "function":
                raise TypeError(f"'{func_name}' is not a function")

            # Call user-defined function
            return evaluate_function(func_entry["value"], args)

    return node

# ------
# Factors
# -------
def p_term_factor(p):
    """term : factor"""
    p[0] = p[1]

def p_factor_call(p):
    """factor : ID LPAREN args RPAREN"""
    func_name = p[1]
    args = p[3]
    
    # Handle built-in functions first (before symbol table lookup)
    
    # linspace - returns flat 1D array for visualization
    if func_name == "linspace" and len(args) == 3:
        start = float(args[0]) if is_scalar(args[0]) else args[0]
        end = float(args[1]) if is_scalar(args[1]) else args[1]
        num = int(args[2]) if is_scalar(args[2]) else int(args[2])

        if HAS_TORCH:
            result = torch.linspace(start, end, num, requires_grad=False)
            p[0] = result  # Flat 1D tensor
        else:
            import numpy as np
            p[0] = np.linspace(start, end, num).tolist()
        return

    # exp (exponential function)
    if func_name == "exp" and len(args) == 1:
        x = args[0]
        if is_torch_tensor(x):
            p[0] = torch.exp(x)
        elif is_scalar(x):
            import math
            p[0] = math.exp(x)
        else:
            # Element-wise for lists
            import math
            def apply_exp(v):
                if isinstance(v, list):
                    return [apply_exp(e) for e in v]
                return math.exp(v)
            p[0] = apply_exp(x)
        return

    # mod (modulo function)
    if func_name == "mod" and len(args) == 2:
        a, b = args[0], args[1]
        if is_torch_tensor(a):
            a = a.item() if a.numel() == 1 else a
        if is_torch_tensor(b):
            b = b.item() if b.numel() == 1 else b
        p[0] = float(int(a) % int(b))
        return

    # sin (sine function)
    if func_name == "sin" and len(args) == 1:
        x = args[0]
        if is_torch_tensor(x):
            p[0] = torch.sin(x)
        else:
            import math
            p[0] = math.sin(x)
        return

    # cos (cosine function)
    if func_name == "cos" and len(args) == 1:
        x = args[0]
        if is_torch_tensor(x):
            p[0] = torch.cos(x)
        else:
            import math
            p[0] = math.cos(x)
        return

    # real (extract real part of complex number)
    if func_name == "real" and len(args) == 1:
        x = args[0]
        if is_torch_tensor(x):
            if x.is_complex():
                p[0] = x.real
            else:
                p[0] = x
        elif isinstance(x, complex):
            p[0] = x.real
        else:
            p[0] = x
        return

    # imag (extract imaginary part of complex number)
    if func_name == "imag" and len(args) == 1:
        x = args[0]
        if is_torch_tensor(x):
            if x.is_complex():
                p[0] = x.imag
            else:
                p[0] = torch.tensor(0.0)
        elif isinstance(x, complex):
            p[0] = x.imag
        else:
            p[0] = 0.0
        return

    # pi constant
    if func_name == "pi" and len(args) == 0:
        import math
        if HAS_TORCH:
            p[0] = torch.tensor(math.pi, requires_grad=True)
        else:
            p[0] = math.pi
        return

    # factorial(n)
    if func_name == "factorial" and len(args) == 1:
        n = args[0]
        if is_torch_tensor(n):
            n = int(n.item())
        else:
            n = int(n)
        import math
        result = math.factorial(n)
        if HAS_TORCH:
            p[0] = torch.tensor(float(result), requires_grad=True)
        else:
            p[0] = float(result)
        return

    # hermite(n, x) - Physicist's Hermite polynomials
    if func_name == "hermite" and len(args) == 2:
        n = args[0]
        x = args[1]
        if is_torch_tensor(n):
            n = int(n.item())
        else:
            n = int(n)

        # Compute H_n(x) using recurrence: H_n = 2x*H_{n-1} - 2(n-1)*H_{n-2}
        if n == 0:
            if is_torch_tensor(x):
                p[0] = torch.ones_like(x)
            else:
                p[0] = 1.0
        elif n == 1:
            if is_torch_tensor(x):
                p[0] = 2.0 * x
            else:
                p[0] = 2.0 * x
        else:
            if not is_torch_tensor(x):
                x = to_torch(x) if HAS_TORCH else x
            H_prev2 = torch.ones_like(x) if is_torch_tensor(x) else 1.0  # H_0
            H_prev1 = 2.0 * x  # H_1
            for k in range(2, n + 1):
                H_curr = 2.0 * x * H_prev1 - 2.0 * (k - 1) * H_prev2
                H_prev2 = H_prev1
                H_prev1 = H_curr
            p[0] = H_prev1
        return

    # mean
    if func_name == "mean" and len(args) == 1:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for mean")
        tensor = args[0]
        if not is_torch_tensor(tensor):
            tensor = to_torch(tensor)
        p[0] = tensor.mean()
        return

    # pow
    if func_name == "pow" and len(args) == 2:
        base = args[0]
        exp = args[1]
        if is_torch_tensor(base) or is_torch_tensor(exp):
            if not is_torch_tensor(base):
                base = to_torch(base)
            if not is_torch_tensor(exp):
                exp = to_torch(exp)
            p[0] = torch.pow(base, exp)
        else:
            p[0] = base ** exp
        return
    
    # print
    if func_name == "print" and len(args) == 1:
        value = args[0]
        display_value = from_torch(value) if is_torch_tensor(value) else value
        print(f"  {display_value}")
        p[0] = None
        return
    
    # detach
    if func_name == "detach" and len(args) == 1:
        tensor = args[0]
        if is_torch_tensor(tensor):
            p[0] = tensor.detach()
        else:
            p[0] = tensor
        return
    
    # item
    if func_name == "item" and len(args) == 1:
        tensor = args[0]
        if is_torch_tensor(tensor):
            p[0] = tensor.item()
        else:
            p[0] = tensor
        return
    
    # Handle layer constructors - they return tuples (specs), not values
    if func_name == "Linear" and len(args) == 2:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for Linear layer")
        in_f = args[0]
        out_f = args[1]
        if is_scalar(in_f) and is_scalar(out_f):
            p[0] = ("linear", int(in_f), int(out_f))
            return
        # If args are already layer specs (tuples), this is nested - shouldn't happen
        raise ValueError(f"Linear expects two numbers, got {in_f}, {out_f}")
    
    if func_name in ["Tanh", "ReLU", "Sigmoid"] and len(args) == 0:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for activation functions")
        p[0] = (func_name.lower(),)
        return
    
    if func_name == "Sequential":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for Sequential")
        # args should be a list of layer specs (tuples)
        # Each element is already evaluated by p_args, so they're layer specs
        p[0] = ("sequential", args)
        return
    
    # Handle grad function
    if func_name == "grad" and len(args) == 2:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for gradient computation")
        
        output = args[0]
        input_var = args[1]
        
        if not is_torch_tensor(output) or not is_torch_tensor(input_var):
            raise TypeError("Gradient computation requires torch tensors")
        
        # Compute gradient
        grad_value = torch.autograd.grad(output, input_var, create_graph=True, retain_graph=True)[0]
        p[0] = grad_value
        return
    
    # Handle backward function (for loss.backward())
    if func_name == "backward" and len(args) == 1:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for backward")
        
        loss = args[0]
        if not is_torch_tensor(loss):
            raise TypeError("backward() requires a torch tensor")
        
        loss.backward()
        p[0] = None  # backward returns nothing
        return
    
    # Handle optimizer step
    if func_name == "step" and len(args) == 1:
        # This will be for optimizer.step()
        optimizer = args[0]
        if hasattr(optimizer, 'step'):
            optimizer.step()
        p[0] = None
        return
    
    # Handle zero_grad
    if func_name == "zero_grad" and len(args) == 1:
        # This will be for optimizer.zero_grad()
        optimizer = args[0]
        if hasattr(optimizer, 'zero_grad'):
            optimizer.zero_grad()
        p[0] = None
        return
    
    # Handle Adam optimizer creation
    if func_name == "Adam" and len(args) == 2:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for Adam optimizer")
        
        model_entry = args[0]
        lr = args[1]
        
        # Get the actual model from symbol table if it's a string reference
        if isinstance(model_entry, str) and model_entry in symbol_table:
            model = symbol_table[model_entry]["value"]
        else:
            model = model_entry
        
        if not isinstance(model, nn.Module):
            raise TypeError("Adam expects a neural network model")
        
        if not is_scalar(lr):
            raise TypeError("Learning rate must be a scalar")
        
        optimizer = optim.Adam(model.parameters(), lr=float(lr))
        p[0] = optimizer
        return
    
    # Handle print function
    if func_name == "print" and len(args) == 1:
        value = args[0]
        display_value = from_torch(value) if is_torch_tensor(value) else value
        print(f"  {display_value}")
        p[0] = None
        return

    # Handle concat(a, b) - concatenate tensors
    if func_name == "concat" and len(args) >= 2:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for concat")
        tensors = []
        for arg in args:
            if not is_torch_tensor(arg):
                arg = to_torch(arg)
            if arg.dim() == 0:
                arg = arg.unsqueeze(0)
            tensors.append(arg)
        p[0] = torch.cat(tensors)
        return

    # Handle zeros(n) - create zero vector
    if func_name == "zeros" and len(args) == 1:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for zeros")
        n = int(args[0]) if is_scalar(args[0]) else int(args[0].item())
        p[0] = torch.zeros(n, requires_grad=True)
        return

    # Handle solve(A, b) - Solve linear system Ax = b using torch.linalg.solve
    if func_name == "solve" and len(args) == 2:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for solve")
        A = args[0]
        b = args[1]
        if not is_torch_tensor(A):
            A = to_torch(A)
        if not is_torch_tensor(b):
            b = to_torch(b)
        # Ensure A is 2D and b is 1D
        if A.dim() == 2 and b.dim() == 1:
            result = torch.linalg.solve(A.float(), b.float())
            p[0] = result
            return
        raise ValueError("solve(A, b) requires 2D matrix A and 1D vector b")

    # Handle sqrt(x) - square root
    if func_name == "sqrt" and len(args) == 1:
        x = args[0]
        if is_torch_tensor(x):
            p[0] = torch.sqrt(x)
        else:
            p[0] = x ** 0.5
        return

    # # Handle qho_energy(n, hbar, omega) - Quantum harmonic oscillator energy
    # if func_name == "qho_energy" and len(args) == 3:
    #     result = call_builtin("qho_energy", args)
    #     if result is not None:
    #         p[0] = result
    #         return
    #     raise RuntimeError("qho_energy() failed")

    # # Handle qho_eigenstate(n, x, hbar, m, omega) - Quantum harmonic oscillator wavefunction
    # if func_name == "qho_eigenstate" and len(args) == 5:
    #     result = call_builtin("qho_eigenstate", args)
    #     if result is not None:
    #         p[0] = result
    #         return
    #     raise RuntimeError("qho_eigenstate() failed")

    # Handle hermite(n, x) - Hermite polynomials
    if func_name == "hermite" and len(args) == 2:
        result = call_builtin("hermite", args)
        if result is not None:
            p[0] = result
            return
        raise RuntimeError("hermite() failed")

    # Handle PDESolve(eqn) - Solve PDEs symbolically
    if func_name == "PDESolve" and len(args) >= 1:
        result = call_builtin("PDESolve", args)
        if result is not None:
            p[0] = result
            return
        raise RuntimeError("PDESolve() failed")

    # Handle train(net, X, y, epochs, lr)
    if func_name == "train" and len(args) == 5:
        result = call_builtin("train", args)
        if result is not None:
            p[0] = result
            return
        raise RuntimeError("train() failed")

    # Handle evaluate(net, X, y)
    if func_name == "evaluate" and len(args) == 3:
        result = call_builtin("evaluate", args)
        if result is not None:
            p[0] = result
            return
        raise RuntimeError("evaluate() failed")


    # Handle linspace (create evenly spaced points)
    if func_name == "linspace" and len(args) == 3:
        start = float(args[0]) if is_scalar(args[0]) else args[0]
        end = float(args[1]) if is_scalar(args[1]) else args[1]
        num = int(args[2]) if is_scalar(args[2]) else int(args[2])

        if HAS_TORCH:
            result = torch.linspace(start, end, num, requires_grad=False)
            p[0] = result  # Flat 1D tensor for visualization
        else:
            import numpy as np
            result = np.linspace(start, end, num).tolist()
            p[0] = result
        return
    
    # Handle detach (stop gradient tracking)
    if func_name == "detach" and len(args) == 1:
        tensor = args[0]
        if is_torch_tensor(tensor):
            p[0] = tensor.detach()
        else:
            p[0] = tensor
        return
    
    # Handle item() to extract scalar from tensor
    if func_name == "item" and len(args) == 1:
        tensor = args[0]
        if is_torch_tensor(tensor):
            p[0] = tensor.item()
        else:
            p[0] = tensor
        return

    # ======================================================
    # VISUALIZATION FUNCTIONS IN PARSER
    # ======================================================

    # plot(x, y) or plot(y) - 2D line plot
    if func_name == "plot":
        result = call_builtin("plot", args)
        if result is not None:
            p[0] = result
            return
        p[0] = 0.0
        return

    # plot3d(x, y, z) - 3D line plot
    if func_name == "plot3d":
        result = call_builtin("plot3d", args)
        if result is not None:
            p[0] = result
            return
        p[0] = 0.0
        return

    # visualize(func, args..., time_min, time_max, n_points) - Visualize a Physika function
    if func_name == "visualize":
        result = call_builtin("visualize", args)
        if result is not None:
            p[0] = result
            return
        p[0] = 0.0
        return

    # animate(func, args..., time_min, time_max, n_points) - Animate a Physika function
    if func_name == "animate":
        result = call_builtin("animate", args)
        if result is not None:
            p[0] = result
            return
        p[0] = 0.0
        return

    # Regular function call
    if func_name not in symbol_table:
        raise NameError(f"Undefined function '{func_name}'")

    func_entry = symbol_table[func_name]

    # Handle class instantiation: ClassName(params...) -> instance
    if func_entry["type"] == "class":
        class_def = func_entry["value"]
        class_params = class_def["class_params"]

        if len(args) != len(class_params):
            raise ValueError(f"Class '{func_name}' expects {len(class_params)} parameters, got {len(args)}")

        # Create instance with bound parameters
        bound_params = {}
        for (param_name, param_type), arg_value in zip(class_params, args):
            # Handle function references (functions passed as parameters)
            if isinstance(arg_value, tuple) and arg_value[0] == "func_ref":
                bound_params[param_name] = arg_value[2]  # Store the function definition
            else:
                bound_params[param_name] = arg_value

        instance = {
            "class_name": func_name,
            "bound_params": bound_params,
            "lambda_params": class_def["lambda_params"],
            "return_type": class_def["return_type"],
            "body": class_def["body"],
            "has_loop": class_def.get("has_loop", False),
            "loop_var": class_def.get("loop_var"),
            "loop_count_var": class_def.get("loop_count_var"),
            "loop_body": class_def.get("loop_body"),
            "has_loss": class_def.get("has_loss", False),
            "loss_params": class_def.get("loss_params"),
            "loss_body": class_def.get("loss_body")
        }
        p[0] = ("instance", instance)
        return

    # Handle instance call: instance(x) -> evaluate lambda
    if func_entry["type"] == "instance":
        instance = func_entry["value"]
        lambda_params = instance["lambda_params"]

        if len(args) != len(lambda_params):
            raise ValueError(f"Instance expects {len(lambda_params)} arguments, got {len(args)}")

        # Build local scope with bound class params and lambda params
        local_scope = dict(instance["bound_params"])
        for (param_name, param_type), arg_value in zip(lambda_params, args):
            local_scope[param_name] = arg_value

        # Handle classes with for loops
        if instance.get("has_loop"):
            loop_var = instance["loop_var"]
            loop_count_var = instance["loop_count_var"]
            loop_body = instance["loop_body"]

            # Get the loop count from bound params
            n = local_scope.get(loop_count_var, 0)
            if is_torch_tensor(n):
                n = int(n.item())
            else:
                n = int(n)

            # Execute the loop
            for i in range(n):
                local_scope[loop_var] = float(i)
                # Execute loop body statements
                for stmt in loop_body:
                    if stmt and stmt[0] == "loop_assign":
                        var_name = stmt[1]
                        expr_ast = stmt[2]
                        local_scope[var_name] = evaluate_ast(expr_ast, local_scope)

        # Evaluate the final expression
        p[0] = evaluate_ast(instance["body"], local_scope)
        return

    # Handle neural network forward pass
    if isinstance(func_entry.get("value"), nn.Module):
        model = func_entry["value"]
        
        if len(args) != 1:
            raise ValueError("Neural network expects single input")
        
        input_data = args[0]
        
        # Convert to torch tensor with correct shape
        if not is_torch_tensor(input_data):
            input_tensor = to_torch(input_data)
        else:
            input_tensor = input_data
        
        # Ensure correct shape: if 1D vector, add batch dimension
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # Shape: (1, N)
        
        # Forward pass
        output = model(input_tensor)
        
        # Remove batch dimension if added
        if output.shape[0] == 1:
            output = output.squeeze(0)
        
        p[0] = output
        return
    
    if func_entry["type"] != "function":
        raise TypeError(f"'{func_name}' is not a function")
    
    func_def = func_entry["value"]
    
    if len(args) != len(func_def["params"]):
        raise ValueError(f"Function '{func_name}' expects {len(func_def['params'])} arguments, got {len(args)}")
    
    # Evaluate function with arguments
    p[0] = evaluate_function(func_def, args)

def p_factor_number(p):
    """factor : NUMBER"""
    p[0] = p[1]

def p_factor_neg(p):
    """factor : MINUS factor"""
    # Unary minus: -x
    val = p[2]
    if is_torch_tensor(val):
        p[0] = -val
    elif isinstance(val, list):
        # Element-wise negation
        def negate(v):
            if isinstance(v, list):
                return [negate(e) for e in v]
            return -v
        p[0] = negate(val)
    else:
        p[0] = -val

def p_factor_id(p):
    """factor : ID"""
    if p[1] not in symbol_table:
        raise NameError(f"Undefined variable '{p[1]}'")
    entry = symbol_table[p[1]]
    if entry["type"] == "function":
        # Return function reference (for passing functions as parameters)
        p[0] = ("func_ref", p[1], entry["value"])
        return
    if entry["type"] == "class":
        raise TypeError(f"'{p[1]}' is a class, not a variable")
    p[0] = entry["value"]

def p_factor_group(p):
    """factor : LPAREN expr RPAREN"""
    p[0] = p[2]

def p_factor_array(p):
    """factor : LBRACKET elements RBRACKET"""
    p[0] = p[2]

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
    arr = symbol_table[p[1]]["value"]
    p[0] = arr[int(p[3])]

def p_factor_slice(p):
    """factor : ID LBRACKET NUMBER COLON NUMBER RBRACKET"""
    arr = symbol_table[p[1]]["value"]
    p[0] = arr[int(p[3]):int(p[5]) + 1]

# ----------------------
# Function Arguments
# ----------------------

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


parser = yacc.yacc()

# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    # cd to pkysika and run: python -m execute examples/example_arrays.phyk
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        parser.parse(f.read(), lexer=lexer)
