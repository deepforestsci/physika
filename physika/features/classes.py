import re
from typing import Optional, Callable


def is_learnable(type_spec: str) -> bool:
    """
    Helper functions that returns True for "ℝ" and "ℝ[n]" types that should
    become ``nn.Parameter``.

    Parameters
    ----------
    type_spec: str
        Physika types converted to strings.

    Returns
    -------
    bool
        Returns True for ℝ and tensor ("ℝ[n]") types, which
        are learnable

    Examples
    --------
    >>> from physika.features.classes import is_learnable
    >>> is_learnable("ℝ")
    True
    >>> is_learnable(("tensor", [2]))
    True
    >>> is_learnable("int")
    False
    """
    if type_spec in ("ℝ", "R"):
        return True
    if isinstance(type_spec, tuple) and type_spec[0] == "tensor":
        return True
    return False


def replace_class_params(code: str, all_params: list) -> str:
    """
    Add `self.` prefix to class params to match Python code.

    Parameters
    ----------
    code: str
        Converted Python code using `emit_body_stmts` that contains params
        as bare param names and needs `self.` prefix to run in Python.
    all_params: list

    Returns
    -------
    code: str
        Same Pythonic code with prefix added for class params.

    Examples
    --------
    >>> from physika.features.classes import replace_class_params
    >>> replace_class_params("0.5 * mass", [("mass", "ℝ")])
    '0.5 * self.mass'
    >>> replace_class_params("self.mass * 2", [("mass", "ℝ")])
    'self.mass * 2'
    """
    for cp_name, _ in all_params:
        code = re.sub(rf'(?<!\.)\b{cp_name}\b', f'self.{cp_name}', code)
    return code


def unwrap_return(ret: Optional[tuple]) -> Optional[tuple]:
    """
    Helper function to build a return expression that will be used
    by the parser.

    Parameters
    ----------
    ret: tuple[str]
        Return expressions from class methods

    Returns
    -------
    Optional[tuple]
        Depending on the case, None return expression, single or tuple
        return expressions

    Examples
    --------
    >>> from physika.features.classes import unwrap_return
    >>> # equivalent of: 0.5 * this.mass
    >>> expr = ("mul",
    ... ("num", 0.5), ("field_access", ("var", "this"), "mass"))
    >>> unwrap_return(("return_single", expr)) == expr
    True
    >>> unwrap_return(("return_tuple", ("var", "a"), ("var", "b")))
    ('tuple_return', ('var', 'a'), ('var', 'b'))
    >>> unwrap_return(None) is None
    True
    """

    if ret is None:
        return None
    if ret[0] == "return_single":
        return ret[1]
    if ret[0] == "return_tuple":
        return ("tuple_return", ret[1], ret[2])
    return None


def build_class(constructor_params: Optional[list], body_items: list) -> dict:
    """
    Build a dict of the class from parsed body items.

    Parameters
    ----------
    constructor_params: Optional[list]
        - None for empty physika classes params:
            For example:
            class Particle:
                .
                .
                .
       - List of params and types for non-empty Physika classes:
            For example:
            class HamiltonianNet(W1: ℝ[M,N], b1: ℝ[M], w2: ℝ[M], b2: ℝ):
                .
                .
                .
    body_items: list
        Body expressions and statemetns that could be declarations, assignments,
        methods, fields, for-loops, if-else, etc.

    Returns
    -------
    dict
        Dictionary with constructor params, fields and methods needed
        for building a Python class.

    Examples
    --------
        class Particle:
            mass : ℝ
            def ke() : ℝ: ...

    >>> from physika.features.classes import build_class
    >>> ke = {"name": "ke", "params": [], "return_type": "ℝ", "statements": [], "body": None}
    >>> body_items = [("field_decl", "mass", "ℝ"), ("method_def", ke)]
    >>> result = build_class(None, body_items)
    >>> result["constructor_params"]
    [('mass', 'ℝ')]
    >>> result["fields"]
    []
    >>> result
    {'constructor_params': [('mass', 'ℝ')], 'fields': [], 'methods': [{'name': 'ke', 'params': [], 'return_type': 'ℝ', 'statements': [], 'body': None}]}  # noqa :E501
    """
    fields = [(item[1], item[2]) for item in body_items
              if item[0] == "field_decl"]
    methods = [item[1] for item in body_items if item[0] == "method_def"]

    if constructor_params is None:
        # no parameters defined, fields becomes constructor params
        return {"constructor_params": fields, "fields": [], "methods": methods}
    return {
        "constructor_params": list(constructor_params),
        "fields": fields,
        "methods": methods
    }


def emit_method(method: dict, all_params: list, to_expr: Callable,
                scalar_only: bool) -> list[str]:
    """
    Emit code for a class method as an ``nn.Module`` class.

    Parameters
    ----------
    method: dict
        Dictionary that contains method's class definition information in order "name",
        "params" with declared types, "return_type", body statements and expressions.  # noqa :E501
    all_params: list
        List of method's parameters with declared types.
    to_expr: Callable
        ``ast_to_torch_expr`` for getting the associated torch code.
    scalar_only: bool
        Boolean that indicate how to define if-else blocks when emiting body stmts.  # noqa :E501

    Returns
    -------
    method_lines: list[str]
        Pytorch code lines for a given Physika class method.

    Examples
    --------
    >>> from physika.features.classes import emit_method
    >>> body = ("mul", ("num", 0.5), ("field_access", ("var", "this"), "mass"))
    >>> ke = {"name": "ke", "params": [], "return_type": "ℝ", "statements": [], "body": body}
    >>> emit_method(ke, [("mass", "ℝ")], lambda _: "0.5 * this.mass", True)
    ['', '    def ke(self):', '        this = self', '        return 0.5 * self.mass']  # noqa :E501
    """
    from physika.utils.ast_utils import emit_body_stmts

    method_name = method["name"]
    if method_name == "λ":
        py_name = "forward"
    else:
        py_name = method_name

    params = method.get("params", [])
    statements = method.get("statements", [])
    body = method.get("body")

    param_names = [p[0] for p in params]
    all_args = ["self"] + param_names
    method_lines = [
        "",
        f"    def {py_name}({', '.join(all_args)}):",
        "        this = self",  # runtime alias to access ``self``
    ]

    for pname, ptype in params:
        if is_learnable(ptype):
            method_lines.append(
                f"        {pname} = torch.as_tensor({pname}).float()")

    if statements:
        stmt_method_lines: list[str] = []
        emit_body_stmts(statements, 2, stmt_method_lines, list(param_names),
                        set(), to_expr, scalar_only)
        for line in stmt_method_lines:
            line_sub = re.sub(r'\bthis\b', 'self', line)
            method_lines.append(replace_class_params(line_sub, all_params))

    if body is not None:
        this_re = r'\bthis\b'
        if isinstance(body, tuple) and body[0] == "tuple_return":
            _, e1, e2 = body
            e1_sub = re.sub(this_re, 'self', to_expr(e1))
            e2_sub = re.sub(this_re, 'self', to_expr(e2))

            r1 = replace_class_params(e1_sub, all_params)
            r2 = replace_class_params(e2_sub, all_params)
            method_lines.append(f"        return ({r1}, {r2})")
        else:
            body_sub = re.sub(this_re, 'self', to_expr(body))
            ret = replace_class_params(body_sub, all_params)
            method_lines.append(f"        return {ret}")

    return method_lines


def generate_class(name: str, class_def: dict) -> str:
    """
    Emit a ``nn.Module`` subclass from a Physika class definition.


    Parameters
    ----------
    name: str
        Name of a defined Physika class.
    class_def: dict
        Dictionary that contains all the information for the defined Physika class.
        In order, "constructor_params" and types, methods, statements and body.

    Returns
    -------
    str
        PyTorch source code for the nn.Module subclass.
 
    Examples
    --------
    Physika class:

        class Particle:
            mass : ℝ
            def ke() : ℝ:
                return 0.5 * this.mass

    >>> from physika.features.classes import generate_class, build_class
    >>> ke = {"name": "ke", "params": [], "return_type": "ℝ", "statements": [], "body": ("mul", ("num", 0.5), ("field_access", ("var", "this"), "mass"))}
    >>> class_def = build_class(None, [("field_decl", "mass", "ℝ"), ("method_def", ke)])  # noqa :E501
    >>> print(generate_class("Particle", class_def))
    class Particle(nn.Module):
        def __init__(self, mass):
            super().__init__()
            self.mass = torch.as_tensor(mass).float()
    <BLANKLINE>
        def ke(self):
            this = self
            return (0.5 * self.mass)
    <BLANKLINE>
        @property
        def params(self):
            return list(self.parameters())
    <BLANKLINE>
        def update(self, lr, grads):
            with torch.no_grad():
                for p, g in zip(self.parameters(), grads):
                    if g is not None:
                        p -= lr * g
    """
    from physika.utils.ast_utils import ast_to_torch_expr, collect_grad_targets
    constructor_params = class_def["constructor_params"]
    fields = class_def.get("fields", [])
    methods = class_def["methods"]

    all_params = list(constructor_params) + list(fields)

    forward = next((m for m in methods if m["name"] == "λ"), None)

    # class header
    class_lines = [f"class {name}(nn.Module):"]

    # initiailizer
    init_names = [p[0] for p in constructor_params]
    class_lines.append(f"    def __init__(self, {', '.join(init_names)}):")
    class_lines.append("        super().__init__()")

    has_forward = forward is not None

    # Constructor params
    for pname, ptype in constructor_params:

        # Checks pname is an instance of a Physika class
        if isinstance(ptype, tuple) and ptype[0] == "struct_type":
            class_lines.append(f"        self.add_module('{pname}', {pname})")
        elif is_learnable(ptype):
            if has_forward:
                # adds nn.Parameter
                class_lines.append(
                    f"        self.{pname} = nn.Parameter(torch.as_tensor({pname}).float())"  # noqa :E501
                )
            else:
                # add tensor so grad flows through
                class_lines.append(
                    f"        self.{pname} = torch.as_tensor({pname}).float()")
        else:
            class_lines.append(
                f"        self.{pname} = torch.as_tensor({pname}).float() "
                f"if isinstance({pname}, (int, float, torch.Tensor)) else {pname}"  # noqa :E501
            )
    # Fields are initialized to zero as nn.Parameter
    for fname, ftype in fields:
        if isinstance(ftype, tuple) and ftype[0] == "tensor":
            dims = ", ".join(str(d) for d in ftype[1])
            class_lines.append(
                f"        self.{fname} = nn.Parameter(torch.zeros({dims}))")
        elif is_learnable(ftype) and ftype[0] in ["R", "ℝ"]:
            class_lines.append(
                f"        self.{fname} = nn.Parameter(torch.tensor(0.0))")
        else:
            class_lines.append(f"        self.{fname} = None")

    # Methods
    for method in methods:
        method_params = list(method.get("params", []))
        local_names = {p[0] for p in method_params}

        # Collect grad differentiation variables
        # used in this method
        grad_targets: set[str] = set()
        for s in method.get("statements", []):
            collect_grad_targets(s, grad_targets)
        collect_grad_targets(method.get("body"), grad_targets)

        wrt_diff_vars = []
        if forward:
            fwd_params = forward.get("params", [])
        else:
            fwd_params = []

        for p_name, p_type in fwd_params:
            if p_name in grad_targets and p_name not in local_names:
                # params that this method differentiates wrt,
                # but are not in the method's param list
                wrt_diff_vars.append((p_name, p_type))
        if wrt_diff_vars:
            method_params = method_params + wrt_diff_vars

        scalar_only = all(pt == "ℝ" for _, pt in method.get("params", []))
        class_lines.extend(
            emit_method({
                **method, "params": method_params
            }, all_params, ast_to_torch_expr, scalar_only))

    # params property and gradient descent update helper
    class_lines += [
        "",
        "    @property",
        "    def params(self):",
        "        return list(self.parameters())",
        "",
        "    def update(self, lr, grads):",
        "        with torch.no_grad():",
        "            for p, g in zip(self.parameters(), grads):",
        "                if g is not None:",
        "                    p -= lr * g",
    ]

    return "\n".join(class_lines)


def make_parser_rules():
    """
    PLY grammar functions for Physika class syntax.
    """

    def p_statement_class_no_params(p):
        """statement : CLASS ID COLON NEWLINE INDENT class_items DEDENT"""
        # Class with fields declared in the body (no constructor params).
        # Example:
        #   class Particle:
        #       mass : ℝ
        #       def kinetic_energy() : ℝ:
        #           .
        #           .
        #           .
        # Parameters:
        #   p[2] - class name
        #   p[6] - list of class_item nodes with field_decl and method_def
        from physika import parser as parser_mod
        name = p[2]
        class_def = build_class(None, p[6])
        parser_mod.symbol_table[name] = {"type": "class", "value": class_def}
        p[0] = ("class_def", name)

    def p_statement_class_with_params(p):
        """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT class_items DEDENT"""  # noqa :E501
        # class with explicit constructor params in the header, usually to
        # define DL models and layers.
        # Example:
        #   class HNN(W1: ℝ[M,N], b1: ℝ[M]):
        #       def λ(x: ℝ[N]) → ℝ:
        # Parameters:
        #   p[2] - class name
        #   p[4] - constructor param list with types
        #   p[9] - list of class_item nodes with field_decl and method_def
        from physika import parser as parser_mod
        name = p[2]
        class_def = build_class(p[4], p[9])
        parser_mod.symbol_table[name] = {"type": "class", "value": class_def}
        p[0] = ("class_def", name)

    def p_class_items_multi(p):
        """class_items : class_items class_item"""
        # Accumulates multiple field/method items into a list.
        # p[1] - existing item list
        # p[2]  (next item appended)
        p[0] = p[1] + [p[2]]

    def p_class_items_single(p):
        """class_items : class_item"""
        # Base case:
        # A single field or method item starts the list.
        p[0] = [p[1]]

    def p_class_item_field(p):
        """class_item : ID COLON type_spec NEWLINE"""
        # Field declaration inside a class body.
        # Example:
        # clas Particle:
        #   mass : ℝ
        # Parameters:
        #    p[1] - field name
        #    p[3] - type spec
        p[0] = ("field_decl", p[1], p[3])

    def p_class_item_method(p):
        """class_item : class_method"""
        # Wraps a parsed class_method dict as a method_def item.
        p[0] = ("method_def", p[1])

    def p_class_method_params_body(p):
        """class_method : DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT
                        | DEF ID    LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT  # noqa :E501
                        | DEF ID    LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""  # noqa :E501
        # Method with params and statements.
        #  methods; arrow → or colon : return-type separator.
        # Example:
        #   def loss(H: ℝ, target: ℝ[N]) → ℝ:
        #       dH : ℝ = grad(H, x)
        #       return dH
        # Parameters:
        #   p[2] - method name
        #   p[4] - params
        #   p[7] - return type
        #   p[11] - body statements
        #   p[12] - return node
        p[0] = {
            "name": p[2],
            "params": p[4],
            "return_type": p[7],
            "statements": p[11],
            "body": unwrap_return(p[12])
        }

    def p_class_method_params_simple(p):
        """class_method : DEF LAMBDA LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT class_method_return DEDENT
                        | DEF ID    LPAREN params RPAREN ARROW type_spec COLON NEWLINE INDENT class_method_return DEDENT
                        | DEF ID    LPAREN params RPAREN COLON type_spec COLON NEWLINE INDENT class_method_return DEDENT"""  # noqa :E501
        # Method with params and a single return expression (no statements between return and method definition).  # noqa :E501
        # Example:
        #   def dot(other: Vec) → ℝ:
        #       return this.x * other.x + this.y * other.y
        # Parameters:
        #   p[2] - method name
        #   p[4] - params
        #   p[7] - return type
        #   p[11] - return node
        p[0] = {
            "name": p[2],
            "params": p[4],
            "return_type": p[7],
            "statements": [],
            "body": unwrap_return(p[11])
        }

    def p_class_method_no_params_body(p):
        """class_method : DEF ID LPAREN RPAREN ARROW type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT
                        | DEF ID LPAREN RPAREN COLON type_spec COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""  # noqa :E501
        # Method with no params and intermediate statements before the return.
        # Example:
        #   def ke() : ℝ:
        #       v2 : ℝ = sum(this.vel * this.vel)
        #       return 0.5 * this.mass * v2
        # Parameters:
        #   p[2] - method name
        #   p[6] - return type
        #   p[10] - body statements
        #   p[11] - return node
        p[0] = {
            "name": p[2],
            "params": [],
            "return_type": p[6],
            "statements": p[10],
            "body": unwrap_return(p[11])
        }

    def p_class_method_no_params_simple(p):
        """class_method : DEF ID LPAREN RPAREN ARROW type_spec COLON NEWLINE INDENT class_method_return DEDENT
                        | DEF ID LPAREN RPAREN COLON type_spec COLON NEWLINE INDENT class_method_return DEDENT"""  # noqa :E501
        # Method with no params and a single return expression.
        # Example:
        #   def norm_sq() : ℝ:
        #       return this.x * this.x + this.y * this.y
        # Parameters:
        #   p[2] - method name
        #   p[6] - return type
        #   p[10] - return node
        p[0] = {
            "name": p[2],
            "params": [],
            "return_type": p[6],
            "statements": [],
            "body": unwrap_return(p[10])
        }

    def p_class_method_return_single(p):
        """class_method_return : RETURN func_expr NEWLINE"""
        # Single value return at the end of a class method.
        # Example:
        #   return this.x * this.x + this.y * this.y
        # Parameters:
        #   p[2] - return expression
        p[0] = ("return_single", p[2])

    def p_class_method_return_tuple(p):
        """class_method_return : RETURN func_expr COMMA func_expr NEWLINE"""
        # Two value tuple return at the end of a class method.
        # Example:
        #   return new_pos, new_vel
        # Parameters:
        #   p[2] - first expression
        #   p[4] - second expression
        p[0] = ("return_tuple", p[2], p[4])

    def p_field_access(p):
        """factor      : factor DOT ID
           func_factor : func_factor DOT ID"""
        # Read a field from a class instance.
        # Example:
        #   vec.x
        # Parameters:
        #   p[1] - class instance
        #   p[3] - field name
        p[0] = ("field_access", p[1], p[3])

    def p_method_call(p):
        """factor      : factor DOT ID LPAREN args RPAREN
           func_factor : func_factor DOT ID LPAREN func_args RPAREN"""
        # Call a method on a class instance
        # Example:
        #   a.dot(b)
        # Parameters:
        #   p[1] - class instance
        #   p[3] - method name
        #   p[5] - argument list
        p[0] = ("method_call", p[1], p[3], p[5] or [])

    def p_type_class(p):
        """type_spec : ID"""
        # User defined class type.
        # Example:
        #   pos : Particle
        # Parameters:
        #   p[1] - class name used as a type annotation
        p[0] = ("struct_type", p[1])

    def p_func_body_stmt_method_call(p):
        """func_body_stmt : func_factor DOT ID LPAREN func_args RPAREN NEWLINE"""  # noqa :E501
        # Method call used as a statement.
        # Example:
        #   inside another method
        #   p.step(force, dt)
        # Parameters:
        #   p[1] - class instance
        #   p[3] - method name
        #   p[5] - argument list
        p[0] = ("body_expr", ("method_call", p[1], p[3], p[5] or []))

    return [
        p_statement_class_no_params,
        p_statement_class_with_params,
        p_class_items_multi,
        p_class_items_single,
        p_class_item_field,
        p_class_item_method,
        p_class_method_params_body,
        p_class_method_params_simple,
        p_class_method_no_params_body,
        p_class_method_no_params_simple,
        p_class_method_return_single,
        p_class_method_return_tuple,
        p_field_access,
        p_method_call,
        p_type_class,
        p_func_body_stmt_method_call,
    ]
