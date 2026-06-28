# flake8: noqa: E501
import re
from typing import Any, Optional, Callable
from physika.elf import ELF


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
        # to support n return values
        return ("tuple_return", *ret[1:])
    return None


def build_class(constructor_params: Optional[list], body_items: list) -> dict:
    """
    Build a dict of the class from parsed body items.

    Parameters
    ----------
    constructor_params: Optional[list]
        ``None`` when fields are declared in the class body (e.g.
        ``class Particle: mass : ℝ``).
        A list of ``(name, type)`` pairs when params appear in the header
        (e.g. ``class HamiltonianNet(W1: ℝ[M,N], b1: ℝ[M], ...):``).
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
    ::

        class Particle:
            mass : ℝ
            def ke() : ℝ: ...

    >>> from physika.features.classes import build_class
    >>> ke = {"name": "ke", "params": [], "return_type": "ℝ", "statements": [], "body": None}
    >>> body_items = [("field_decl", "mass", "ℝ"), ("method_def", ke)]
    >>> result = build_class(None, body_items)
    >>> result["class_params"]
    [('mass', 'ℝ')]
    >>> result["fields"]
    []
    >>> result
    {'class_params': [('mass', 'ℝ')], 'fields': [], 'methods': [{'name': 'ke', 'params': [], 'return_type': 'ℝ', 'statements': [], 'body': None}]}
    """
    fields = [(item[1], item[2]) for item in body_items
              if item[0] == "field_decl"]
    methods = [item[1] for item in body_items if item[0] == "method_def"]

    if constructor_params is None:
        # no parameters defined, fields becomes constructor params
        return {"class_params": fields, "fields": [], "methods": methods}
    return {
        "class_params": list(constructor_params),
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
    ['', '    def ke(self):', '        this = self', '        return 0.5 * self.mass']
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
            parts = []
            for e in body[1:]:
                e_sub = re.sub(this_re, 'self', to_expr(e))
                parts.append(replace_class_params(e_sub, all_params))
            method_lines.append(f"        return ({', '.join(parts)})")
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
        In order, "class_params" and types, methods, statements and body.

    Returns
    -------
    str
        PyTorch source code for the nn.Module subclass.
 
    Examples
    --------
    Physika class::

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
    constructor_params = class_def["class_params"]
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
    # Fields should not be learnable (register_buffer)
    for fname, ftype in fields:
        if isinstance(ftype, tuple) and ftype[0] == "tensor":
            # Only use torch.zeros if all dims are concrete integers.
            raw_dims = ftype[1]
            int_dims = [d for d in raw_dims if isinstance(d, int)]
            # case all dims are integers
            if len(int_dims) == len(raw_dims):
                dims = ", ".join(str(d) for d in int_dims)
                class_lines.append(
                    f"        self.register_buffer('{fname}', torch.zeros({dims}))"
                )
            else:
                # case symbolic dims (ℝ[n])
                class_lines.append(f"        self.{fname} = None")
        # case scalar field, initialize to 0.0
        elif isinstance(ftype, str) and ftype in ("ℝ", "R"):
            class_lines.append(
                f"        self.register_buffer('{fname}', torch.tensor(0.0))")
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
        """statement : CLASS ID LPAREN params RPAREN COLON NEWLINE INDENT class_items DEDENT"""
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
        """class_method : DEF LAMBDA LPAREN params RPAREN ARROW return_type COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT
                        | DEF ID    LPAREN params RPAREN ARROW return_type COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT
                        | DEF ID    LPAREN params RPAREN COLON return_type COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""
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
        """class_method : DEF LAMBDA LPAREN params RPAREN ARROW return_type COLON NEWLINE INDENT class_method_return DEDENT
                        | DEF ID    LPAREN params RPAREN ARROW return_type COLON NEWLINE INDENT class_method_return DEDENT
                        | DEF ID    LPAREN params RPAREN COLON return_type COLON NEWLINE INDENT class_method_return DEDENT"""
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
        """class_method : DEF ID LPAREN RPAREN ARROW return_type COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT
                        | DEF ID LPAREN RPAREN COLON return_type COLON NEWLINE INDENT func_body_stmts class_method_return DEDENT"""
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
        """class_method : DEF ID LPAREN RPAREN ARROW return_type COLON NEWLINE INDENT class_method_return DEDENT
                        | DEF ID LPAREN RPAREN COLON return_type COLON NEWLINE INDENT class_method_return DEDENT"""
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

    def p_class_method_void(p):
        """class_method : DEF ID LPAREN params RPAREN COLON NEWLINE INDENT func_body_stmts DEDENT
                        | DEF ID LPAREN RPAREN COLON NEWLINE INDENT func_body_stmts DEDENT"""
        # Void method
        # no return type and no return statement.
        # Example:
        #   def train(J: ℝ, h: ℝ, n: ℝ, n_steps: ℕ, lr: ℝ):
        #       for step : ℕ(n_steps):
        #           .
        #           .
        #           .
        # Paremeters:
        #   p[2] - method name
        #   p[4] - params
        #   p[9] - body statements

        # Contains class params:
        # DEF ID ( params ) : NEWLINE INDENT stmts DEDENT
        if len(p) == 11:
            p[0] = {
                "name": p[2],
                "params": p[4],
                "return_type": None,
                "statements": p[9],
                "body": None,
            }
        # No params:
        # DEF ID (        ) : NEWLINE INDENT stmts DEDENT
        else:
            p[0] = {
                "name": p[2],
                "params": [],
                "return_type": None,
                "statements": p[8],
                "body": None,
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
        """class_method_return : RETURN return_expr_list NEWLINE"""
        # N-value tuple return at the end of a class method.
        # Example:
        #   return new_pos, new_vel, new_acc
        # Parameters:
        #   p[2] - list of expressions from return_expr_list
        p[0] = ("return_tuple", *p[2])

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
        """func_body_stmt : func_factor DOT ID LPAREN func_args RPAREN NEWLINE"""
        # Method call used as a statement.
        # Example:
        #   inside another method
        #   p.step(force, dt)
        # Parameters:
        #   p[1] - class instance
        #   p[3] - method name
        #   p[5] - argument list
        p[0] = ("body_expr", ("method_call", p[1], p[3], p[5] or []))

    def p_func_body_stmt_field_assign(p):
        """func_body_stmt : func_factor DOT ID EQUALS func_expr NEWLINE"""
        # Field assignment on an instance inside a method.
        # Example:
        #   this.b = b
        # Parameters:
        #   p[1] - object expression ("var", "this")
        #   p[3] - field name
        #   p[5] - value expression
        p[0] = ("body_field_assign", p[1], p[3], p[5])

    def p_member_expr_base(p):
        """member_expr : ID"""
        # Base case for member expression.
        # Parameters:
        #   p[1] - variable name
        # Returns:
        #   ("var", name)
        p[0] = ("var", p[1])

    def p_member_expr_field(p):
        """member_expr : member_expr DOT ID"""
        # Recursive field access on a member expression.
        # Example:
        #   this.x
        #   this.model.bias
        # Parameters:
        #   p[1] - object expression
        #   p[3] - field name
        # Returns:
        #   ("field_access", object, field_name)
        p[0] = ("field_access", p[1], p[3])

    def p_member_expr_method(p):
        """member_expr : member_expr DOT ID LPAREN func_args RPAREN"""
        # Recursive method call on a member expression.
        # Example:
        #   this.model.forward(x)
        # Parameters:
        #   p[1] - object expression
        #   p[3] - method name
        #   p[5] - argument list
        # Returns:
        #   ("method_call", object, method_name, args)
        p[0] = ("method_call", p[1], p[3], p[5] or [])

    def p_func_loop_stmt_field_assign(p):
        """func_loop_stmt : member_expr DOT ID EQUALS func_expr NEWLINE"""
        # Field assignment on an instance inside a for loop.
        # Example:
        #   this.b = b
        # Parameters:
        #   p[1] - object expression ("var", "this")
        #   p[3] - field name
        #   p[5] - value expression
        p[0] = ("body_field_assign", p[1], p[3], p[5])

    def p_func_loop_stmt_method_call(p):
        """func_loop_stmt : member_expr DOT ID LPAREN func_args RPAREN NEWLINE"""
        # Method call used as a statement inside a for loop of a class method.
        # Example:
        # class PhysikaClass:
        #   def loss(preds: ℝ[n], target: ℝ[n]) → ℝ:
        #       ...
        #   def train(target: ℝ[n], n_steps: ℕ):
        #       for step : ℕ(n_steps):
        #           loss: ℝ = this.loss(this(target), target)
        # Parameters:
        #   p[1] - class instance expression ("var", "this")
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
        p_func_body_stmt_field_assign,
        p_class_method_void,
        p_member_expr_base,
        p_member_expr_field,
        p_member_expr_method,
        p_func_loop_stmt_field_assign,
        p_func_loop_stmt_method_call,
    ]


class ClassFeature(ELF):
    """
    Physika classes implemented as an ELF subclass.

    ``ClassFeature`` injects rules via ``REGISTRY`` at lexer, parser, type
    checker, and code generator.

    **Lexer rules**
    Adds two new tokens, ``CLASS`` reserved keyword (``"class"``) and ``DOT``
    token (``"."``) for field and method access

    **Parser rules**
    Sixteen PLY grammar functions (see ``make_parser_rules``) handle class
    declarations with and without constructor parameters, field
    declarations, method definitions, and single or two tuple valued returns.

    **Type rules**
    Registers ``class_env`` entries so the type checker can
    resolve field types, method calls and constructor calls.

    **Forward rules**
    Three code-generation handlers were defined. ``class_def`` emits a complete
    ``nn.Module``. ``field_access` emits ``obj.field``. ``method_call``emits
    ``obj.method(args)``.

    Physika classes are fully differentiable using Pytroch as backend. Scalar
    and tensor constructor parameters are converted to ``torch.as_tensor``
    objects. Parameters used inside a forward method are wrapped in
    ``nn.Parameter``.

    Physika syntax example (see ``examples/physika_class.phyk``)::

        class Vec:
            x : ℝ
            y : ℝ
            def dot(other : Vec) : ℝ:
                return this.x * other.x + this.y * other.y
            def norm_sq() : ℝ:
                return this.x * this.x + this.y * this.y

        a = Vec(3.0, 4.0)
        a.norm_sq()

    Examples
    --------
    >>> from physika.lexer import lexer
    >>> from physika.parser import parser, symbol_table
    >>> from physika.utils.ast_utils import build_unified_ast
    >>> from physika.codegen import from_ast_to_torch
    >>> def run_phyk(src):
    ...     symbol_table.clear()
    ...     lexer.lexer.lineno = 1
    ...     ast = build_unified_ast(parser.parse(src, lexer=lexer), symbol_table)
    ...     exec(from_ast_to_torch(ast, print_code=False), {})

    >>> # Physika class example
    >>> src = '''
    ... class Vec:
    ...     x : ℝ
    ...     y : ℝ
    ...     def norm_sq() : ℝ:
    ...         return this.x * this.x + this.y * this.y
    ... a = Vec(3.0, 4.0)
    ... a.x
    ... a.y
    ... a.norm_sq()
    ... '''

    >>> # Execute code and verify outputs
    >>> run_phyk(src)
    3.0 ∈ ℝ
    4.0 ∈ ℝ
    25.0 ∈ ℝ
    """
    name = "physika-class"

    def lexer_rules(self) -> dict:
        """
        Adds two new tokens, ``CLASS`` reserved keyword (``"class"``) and
        ``DOT`` token (``"."``) for field and method access.

        Returns
        -------
        dict
            Dictionary with reserved keywords, tokens and tokens functions

        Examples
        --------
        >>> from physika.features import ClassFeature
        >>> rules = ClassFeature().lexer_rules()
        >>> rules["reserved"]
        {'class': 'CLASS'}
        >>> rules["tokens"]
        ['CLASS', 'DOT']

        """

        def t_DOT(t: Any) -> Any:
            # regex matches a dot (".") for field and method access
            r"\."
            return t

        return {
            "reserved": {
                "class": "CLASS"
            },
            "tokens": ["CLASS", "DOT"],
            "token_funcs": [t_DOT],
        }

    def parser_rules(self) -> list:
        """
        Override ``parser_rules`` handler for new grammar rules.

        Sixteen PLY grammar functions (see ``make_parser_rules``) handle class
        declarations with and without constructor parameters, field declarations
        , method definitions, and single or two tuple valued returns.

        Returns
        -------
        list
            List of PLY grammar functions to be injected into ``physika.parser``.

        Examples
        --------
        >>> from physika.features import ClassFeature
        >>> rules = ClassFeature().parser_rules()
        >>> len(rules)
        20
        >>> rules[0].__name__
        'p_statement_class_no_params'
        """
        return make_parser_rules()

    def type_rules(self) -> dict:
        """
        Registers two type-checking handlers that validate field access and
        method calls on class instances. ``field_access``infers ``obj.field``
        by looking up the field name in the class_env and returns its declared
        type. Raises an error if the field does not exist or if a class
        constructor is not instance. ``method_call`` infers 
        ``obj.method(args)`` by checking the number of arguments and types
        against the method's declared parameters and returning its declared
        return type. Raises an error if the method does not exist
        or if argument types do not match.

        Returns
        -------
        dict
            Dispatch table mapping ``"field_access"`` and ``"method_call"`` AST
            tags to their type inference handlers.

        Examples
        --------
        >>> from physika.features import ClassFeature
        >>> rules = ClassFeature().type_rules()
        >>> sorted(rules.keys())
        ['field_access', 'method_call']
        """
        from physika.utils.types import TInstance, Substitution
        from physika.utils.type_checker_utils import from_typespec, type_to_str
        from typing import Callable, Any

        def check_not_constructor(
            expr: tuple,
            class_env: dict,
            add_error: Callable[[str], None],
            expr_name: str,
        ) -> bool:
            """
            Physika classes must be initialized before accesing fields
            or methods. This function checks this behavior by looking at
            ASTNodes for classes fields and methods and comparing if these
            are also defined in the class enviroment.

            Parameters
            ----------
            expr: tuple
                ASTNode for the defined class that contains the parsed
                information for `field_access` or `method_call` expressions.
            class_env: dict
                Dictionary that contains details about fields, methods, and
                types used inside a class.
            add_error: Callable[[str], None]
                Append function to register an error.
            what: str
                Expression that is being called on a non-initialized class
                (``field_access`` or ``method_call``).

            Returns
            -------
            bool
                True if trying to accesing a field without class instance, False
                otherwise.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> rules = ClassFeature().type_rules()
            >>> check_field = rules["field_access"]
            >>> class_env = {"Particle": {"class_params": [("x", "ℝ")], "fields": [], "methods": {}}}
            >>> # Wrong Particle.x, needs to be an instance first
            >>> node = ("field_access", ("var", "Particle"), "x")
            >>> def infer(expr, env, s):
            ...     return None, s
            >>> errors = []
            >>> _ = check_field(node, {}, {}, {}, class_env, errors.append, infer)
            >>> errors[0]
            "'Particle' is a class constructor, not an instance; use an instance to access field 'x'"
            """

            # ('field_access', ('var', 'Vect'), 'x')
            # is equivalent of:
            # Vect.x
            # where Vect is a Physika class
            if (isinstance(expr, tuple) and expr[0] == "var"
                    and expr[1] in class_env):
                add_error(
                    f"'{expr[1]}' is a class constructor, not an instance; "
                    f"use an instance to access {expr_name}")
                return True
            return False

        def check_field_access(
            node: tuple,
            env: dict,
            s: Substitution,
            func_env: dict,
            class_env: dict,
            add_error: Callable[[str], None],
            infer_expr: Callable[..., tuple],
        ) -> tuple[Any, Substitution]:
            """
            Type rules for ``field_access`` AST node.

            Infers the type of class instance, then looks ``field`` in the class
            definition to return its declared type. Registers an error if the
            field does not exist or if a class constructor is used
            directly instead of an instance.

            Parameters
            ----------
            node : tuple
                AST node of the form ``("field_access", obj_expr, field_name)``.
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
            tuple[Any, Substitution]
                The inferred field type and the updated substitution, or
                ``(None, s)`` if the type cannot be resolved.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> from physika.utils.types import TInstance, Substitution
            >>> rules = ClassFeature().type_rules()
            >>> check_field = rules["field_access"]
            >>> s = Substitution()
            >>> class_env = {"Vec": {"class_params": [("x", "ℝ"), ("y", "ℝ")], "fields": [], "methods": {}}}
            >>> def infer(expr, env, s):
            ...     return TInstance("Vec"), s
            >>> node = ("field_access", ("var", "v"), "x")
            >>> t, _ = check_field(node, {}, s, {}, class_env, print, infer)
            >>> t
            ('scalar',)
            """
            _, obj_expr, field_name = node

            obj_type, s = infer_expr(obj_expr, env, s, func_env, class_env,
                                     add_error)

            if isinstance(obj_type, TInstance):
                # get class info (fields, methods, returm types)
                info = class_env.get(obj_type.class_name)
                if info:
                    all_fields = dict(
                        info.get("class_params", []) + info.get("fields", []))
                    if field_name in all_fields:
                        return from_typespec(all_fields[field_name]), s
                    # params and update are defined nn.Module methods
                    if field_name in ("params", "update"):
                        return None, s
                    add_error(
                        f"Class '{obj_type.class_name}' has no field '{field_name}'"
                    )
            elif obj_type is None:
                # Case ClassName.field where ClassName is `var` not TInstance (error)
                check_not_constructor(obj_expr, class_env, add_error,
                                      f"field '{field_name}'")
            return None, s

        def check_method_call(
            node: tuple,
            env: dict,
            s: Substitution,
            func_env: dict,
            class_env: dict,
            add_error: Callable[[str], None],
            infer_expr: Callable[..., tuple],
        ) -> tuple[Any, Substitution]:
            """
            Type inference rules fpr ``method_call`` AST nodes.

            Based on a ``method`` definition in a class, validates arguments
            and types against the method's declared parameters. Returns its
            declared return type.
            
            Registers an error if the method does not exist, argument count mismatches, or
            argument types do not match.

            Parameters
            ----------
            node : tuple
                AST node of the form
                ``("method_call", obj_expr, method_name, args)``.
            env : dict
                Current type environment mapping variable names to their types.
            s : Substitution
                Substitution dict containing bindings accumulated so far.
            func_env : dict
                Function definitions available in scope.
            class_env : dict
                Class definitions mapping class names to their definition dicts.
            add_error : Callable[[str], None]
                Callback to register a type error message.
            infer_expr : Callable
                Recursive type inference function for sub-expressions.

            Returns
            -------
            tuple[Any, Substitution]
                The inferred return type of the method and the updated
                substitution, or ``(None, s)`` if the type cannot be resolved.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> from physika.utils.types import TInstance, Substitution
            >>> rules = ClassFeature().type_rules()
            >>> check_method = rules["method_call"]
            >>> s = Substitution()
            >>> ke = {"params": [], "return_type": "R"}
            >>> class_env = {"Particle": {"class_params": [("mass", "R")], "fields": [], "methods": {"ke": ke}}}
            >>> def infer(expr, env, s):
            ...     return TInstance("Particle"), s
            >>> node = ("method_call", ("var", "p"), "ke", [])
            >>> t, _ = check_method(node, {}, s, {}, class_env, print, infer)
            >>> t
            ('scalar',)
            """
            _, obj_expr, method_name, args = node

            obj_type, s = infer_expr(obj_expr, env, s, func_env, class_env,
                                     add_error)
            # check proper method call ClassName.method() (classes must be first initialized)
            # this expression "ClassName.method()" would have the form of "('var', ClassName)"
            # which will infer to None
            if obj_type is None:

                check_not_constructor(obj_expr, class_env, add_error,
                                      f"method '{method_name}'")
                return None, s

            if isinstance(obj_type, TInstance):
                info = class_env.get(obj_type.class_name)
                if info:
                    methods = info.get("methods", {})
                    if method_name in methods:
                        method_info = methods[method_name]
                        expected_params = method_info.get("params", [])

                        # check args matches
                        if len(args) != len(expected_params):
                            add_error(
                                f"Method '{obj_type.class_name}.{method_name}' expects "
                                f"{len(expected_params)} argument(s), got {len(args)}"
                            )
                        else:
                            for arg, (pname, ptype_spec) in zip(
                                    args, expected_params):
                                # Type check args
                                arg_type, s = infer_expr(
                                    arg, env, s, func_env, class_env,
                                    add_error)
                                expected_type = from_typespec(ptype_spec)

                                # skip if inferred type is unknown
                                if expected_type is None:
                                    continue
                                if arg_type != expected_type:
                                    add_error(
                                        f"Method '{obj_type.class_name}.{method_name}' "
                                        f"parameter '{pname}': expected "
                                        f"'{type_to_str(expected_type)}', "
                                        f"got '{type_to_str(arg_type)}'")

                        return from_typespec(method_info.get("return_type")), s

                    add_error(
                        f"Class '{obj_type.class_name}' has no method '{method_name}'"
                    )
            return None, s

        return {
            "field_access": check_field_access,
            "method_call": check_method_call,
        }

    def forward_rules(self) -> dict:
        """
        Three code-generation handlers were defined. ``class_def`` emits a complete
        ``nn.Module``. ``field_access` emits ``obj.field``. ``method_call``emits
        ``obj.method(args)``.

        Physika classes are fully differentiable using Pytroch as backend. Scalar
        and tensor constructor parameters are converted to ``torch.as_tensor``
        objects. Parameters used inside a forward method are wrapped in
        ``nn.Parameter``.

        Returns
        -------
        dict
            Dictionary containg code generation handlers.

        Examples
        --------
        >>> from physika.features import ClassFeature
        >>> from physika.features.classes import build_class
        >>> rules = ClassFeature().forward_rules()
        >>> class_def = build_class([("mass", "ℝ")], [])
        >>> code = rules["class_def"](("class_def", "Particle", class_def), **{})
        
        >>> # Physika code:
        >>> # class Particle:
        >>> #   mass: ℝ

        >>> nl = chr(10) # unicode for \n
        >>> expected = nl.join([
        ...     "class Particle(nn.Module):",
        ...     "    def __init__(self, mass):",
        ...     "        super().__init__()",
        ...     "        self.mass = torch.as_tensor(mass).float()",
        ...     "",
        ...     "    @property",
        ...     "    def params(self):",
        ...     "        return list(self.parameters())",
        ...     "",
        ...     "    def update(self, lr, grads):",
        ...     "        with torch.no_grad():",
        ...     "            for p, g in zip(self.parameters(), grads):",
        ...     "                if g is not None:",
        ...     "                    p -= lr * g",
        ... ])
        >>> code == expected
        True
        """

        def emit_field_access(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit a Python string attribute acces given a ``field_access`` ASTNode.

            Parameters
            ----------
            node : tuple
                AST node of the form ``("field_access", obj_expr, field_name)``.
            to_expr : Callable
                Recursive codegen function that converts an AST node to a
                Python code string.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism;
                not used directly.

            Returns
            -------
            str
                Python attribute access string, e.g. ``"p.mass"``.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> rules = ClassFeature().forward_rules()
            >>> emit = rules["field_access"]
            >>> node = ("field_access", ("var", "p"), "mass")
            >>> emit(node, lambda n: n[1])
            'p.mass'
            """
            _, obj_expr, field_name = node
            return f"{to_expr(obj_expr)}.{field_name}"

        def emit_method_call(node: tuple, to_expr: Callable, **ctx) -> str:
            """
            Generates a ``method_call`` AST node as a Python method call.

            Parameters
            ----------
            node : tuple
                AST node of the form
                ``("method_call", obj_expr, method_name, args)``.
            to_expr : Callable
                Recursive codegen function that converts an AST node to a
                Python expression string.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism;
                not used directly.

            Returns
            -------
            str
                Python method call string, e.g. ``"p.ke()"``.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> rules = ClassFeature().forward_rules()
            >>> emit = rules["method_call"]
            >>> node = ("method_call", ("var", "p"), "ke", [])
            >>> emit(node, lambda n: n[1])
            'p.ke()'
            """
            _, obj_expr, method_name, args = node
            args_str = ", ".join(to_expr(a) for a in args)
            return f"{to_expr(obj_expr)}.{method_name}({args_str})"

        def emit_class_def(node: tuple, **ctx) -> str:
            """
            Emit a ``class_def`` AST node as a full PyTorch ``nn.Module`` class.

            Converts ``("class_def", name, class_def)`` into a Python source
            string by calling to ``generate_class``, which produces the
            ``__init__``, method defs, and ``params``/``update`` helpers.

            Parameters
            ----------
            node : tuple
                AST node of the form ``("class_def", name, class_def)`` where
                ``class_def`` is the dict returned by ``build_class``.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism;
                not used directly.

            Returns
            -------
            str
                Full Python source string for an ``nn.Module`` subclass.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> from physika.features.classes import build_class
            >>> rules = ClassFeature().forward_rules()
            >>> emit = rules["class_def"]
            >>> class_def = build_class([("mass", "ℝ")], [])
            >>> code = emit(("class_def", "Particle", class_def))
            >>> "class Particle(nn.Module):" in code
            True
            """
            _, name, class_def = node
            return generate_class(name, class_def)

        def emit_body_expr(node: tuple,
                           to_expr: Callable,
                           current_loop_var=None,
                           **ctx) -> str:
            """
            Emit a ``body_expr`` AST node when a method call is used inside
            a method body or for loop.

            Parameters
            ----------
            node : tuple
                AST node of the form ``("body_expr", method_call)`` where
                ``method_call`` is an AST node representing a method call.
            to_expr : Callable
                Code generation function that converts an  AST node to a
                Pytorch code (generally from_ast_to_torch util function).
            current_loop_var : str, optional
                Name of the current loop variable if inside a for loop,
                by default None.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism.

            Returns
            -------
            str
                Python source string for a method call.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = ClassFeature().forward_rules()
            >>> emit = rules["body_expr"]
            >>> method_call = ("method_call", ("var", "p"), "ke", [])
            >>> code = emit(("body_expr", method_call), ast_to_torch_expr, current_loop_var=None)
            >>> "p.ke()" in code
            True
            """
            _, inner_expr = node
            return to_expr(inner_expr, current_loop_var=current_loop_var)

        def emit_body_field_assign(node: tuple,
                                   to_expr: Callable,
                                   current_loop_var=None,
                                   **ctx) -> str:
            """
            Emits a ``body_field_assign`` AST node when a field assignment statement
            is used inside a method body or for loop.

            
            Parameters
            ----------
            node : tuple
                AST node of the form ``('body_field_assign', obj_expr, field_name, expr)`` where
                ``obj_expr`` refers to `this` var name, ``field_name`` the name of the field being
                assigned and ``expr`` the expression to be done.
            to_expr : Callable
                Code generation function that converts an  AST node to a
                Pytorch code (generally ``from_ast_to_torch`` util function).
            current_loop_var : str, optional
                Name of the current loop variable if inside a for loop,
                by default None.
            **ctx
                Extra keyword arguments forwarded by the dispatch mechanism.

            Returns
            -------
            str
                Python source string for a method call.

            Examples
            --------
            >>> from physika.features import ClassFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = ClassFeature().forward_rules()
            >>> emit = rules["body_field_assign"]
            >>> field_assign = ("body_field_assign", ("var", "this"), "b", (add, 1, ("var", "b")))
            >>> code = emit(field_assign, ast_to_torch_expr, current_loop_var=None)
            >>> "self.b = (1 + b)" in code
            True
            """
            _, obj_expr, field_name, expr = node
            raw = to_expr(obj_expr, current_loop_var=current_loop_var)
            if raw == "this":
                obj_code = "self"
            else:
                obj_code = raw
            val_code = to_expr(expr, current_loop_var=current_loop_var)
            return f"{obj_code}.{field_name} = {val_code}"

        return {
            "field_access": emit_field_access,
            "method_call": emit_method_call,
            "class_def": emit_class_def,
            "body_expr": emit_body_expr,
            "body_field_assign": emit_body_field_assign,
        }
