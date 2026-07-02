from typing import Callable, Tuple
from physika.elf import ELF
from physika.utils.types import Substitution


class TupleUnpackFeature(ELF):
    """
    ELF that adds tuple unpacking support for Physika programs at top-level
    program, inside for loops, functions and classes.

    ``TupleUnpackFeature`` allows for declaring multiple variables in one line
    separated with commas. Also, this ELFs allows to assign variable names to
    a multiple variable return function or class method.

    Parser rules are added for three levels (top-level programs, for-loops,
    functions), which also supports typed tuple unpack statements. Type rules
    where also added to check a tuple unpack is well typed. Type annotation for
    tuple unpack is optional, but inferred values are used to verify a program is
    well defined. Finally, code generation rules produces a python equivalent code
    string for tuple unpack.

    Examples
    --------
    >>> from physika.features.tuple_unpack import TupleUnpackFeature
    >>> f = TupleUnpackFeature()
    >>> sorted(f.forward_rules().keys())
    ['expr_list', 'loop_tuple_unpack', 'stmt_tuple_unpack']
    >>> sorted(f.type_rules().keys())
    ['body_tuple_unpack', 'expr_list', 'loop_tuple_unpack', 'stmt_tuple_unpack', 'tuple_return']
    >>> len(f.parser_rules())
    9
    >>> [r.__name__ for r in f.parser_rules()]
    ['p_return_type_single', 'p_return_type_tuple', 'p_typed_id_list', 'p_return_expr_list', 'p_top_level_expr_list', 'p_func_body_stmt_tuple_unpack', 'p_func_loop_stmt_tuple_unpack', 'p_statement_tuple_unpack', 'p_for_statement_tuple_unpack']  # noqa: E501
    """

    name = "tuple_unpack"

    def parser_rules(self) -> list:
        """
        Nine PLY grammar functions where added for tuple unpacking in Physika.

        A class method's type that returns a tuple (multiple values) must be
        composed of the same number of return variables and matching types.
        For example, a method's type returning ``a`` and ``b``, where both
        are of type ``ℝ``, must be ``ℝ → ℝ``. ``return_type`` parser rules
        where added to support this.

        Also, comma separated values parser rules where added at every Physika
        level. Tuple unpack are used for two functionalities when writing
        Physika programs. Parser rules where added to support these
        expressions. First, typed variable assignment:
        ``a: ℝ, b: ℝ, c: ℝ, d: ℝ = 1.0, 2.0, 3.0, 4.0``

        Second, for splitting a multiple valued tuple produced from a function
        or class.

        Returns
        -------
        list
            Nine PLY grammar functions.

        Examples
        --------
        >>> from physika.features.tuple_unpack import TupleUnpackFeature
        >>> rules = TupleUnpackFeature().parser_rules()
        >>> rules[4].__doc__.strip().splitlines()[0]
        'top_level_expr_list : expr COMMA expr'
        >>> rules[7].__doc__.strip().splitlines()[0]
        'statement : id_list EQUALS expr NEWLINE'
        >>> rules[8].__doc__.strip().splitlines()[0]
        'for_statement : id_list EQUALS func_expr NEWLINE'
        """

        def p_return_type_single(p):
            """return_type : type_spec"""
            p[0] = p[1]

        def p_return_type_tuple(p):
            """return_type : type_spec COMMA type_spec
                           | return_type COMMA type_spec"""
            if isinstance(p[1], tuple) and p[1][0] == "tuple_type":
                p[0] = ("tuple_type", p[1][1] + [p[3]])
            else:
                p[0] = ("tuple_type", [p[1], p[3]])

        def p_typed_id_list(p):
            """typed_id_list : ID COLON type_spec COMMA ID COLON type_spec
                             | typed_id_list COMMA ID COLON type_spec"""
            # Returns [(name, type_spec), ...] pairs and the type checker
            # verifies each declared type against the actual element type.
            if isinstance(p[1], str):
                p[0] = [(p[1], p[3]), (p[5], p[7])]
            else:
                p[0] = p[1] + [(p[3], p[5])]

        def p_return_expr_list(p):
            """return_expr_list : func_expr COMMA func_expr
                                | return_expr_list COMMA func_expr"""
            if isinstance(p[1], list):
                p[0] = p[1] + [p[3]]
            else:
                p[0] = [p[1], p[3]]

        def p_func_body_stmt_tuple_unpack(p):
            """func_body_stmt : id_list EQUALS func_expr NEWLINE
                              | typed_id_list EQUALS func_expr NEWLINE
                              | id_list EQUALS return_expr_list NEWLINE
                              | typed_id_list EQUALS return_expr_list NEWLINE"""  # noqa: E501
            rhs = ("expr_list", p[3]) if isinstance(p[3], list) else p[3]
            p[0] = ("body_tuple_unpack", p[1], rhs)

        def p_func_loop_stmt_tuple_unpack(p):
            """func_loop_stmt : id_list EQUALS func_expr NEWLINE
                              | typed_id_list EQUALS func_expr NEWLINE
                              | id_list EQUALS return_expr_list NEWLINE
                              | typed_id_list EQUALS return_expr_list NEWLINE"""  # noqa: E501
            rhs = ("expr_list", p[3]) if isinstance(p[3], list) else p[3]
            p[0] = ("loop_tuple_unpack", p[1], rhs)

        def p_top_level_expr_list(p):
            """top_level_expr_list : expr COMMA expr
                                   | top_level_expr_list COMMA expr"""
            if isinstance(p[1], list):
                p[0] = p[1] + [p[3]]
            else:
                p[0] = [p[1], p[3]]

        def p_statement_tuple_unpack(p):
            """statement : id_list EQUALS expr NEWLINE
                         | typed_id_list EQUALS expr NEWLINE
                         | id_list EQUALS top_level_expr_list NEWLINE
                         | typed_id_list EQUALS top_level_expr_list NEWLINE"""
            rhs = ("expr_list", p[3]) if isinstance(p[3], list) else p[3]
            p[0] = ("stmt_tuple_unpack", p[1], rhs)

        def p_for_statement_tuple_unpack(p):
            """for_statement : id_list EQUALS func_expr NEWLINE
                             | typed_id_list EQUALS func_expr NEWLINE
                             | id_list EQUALS top_level_expr_list NEWLINE
                             | typed_id_list EQUALS top_level_expr_list NEWLINE"""  # noqa: E501
            rhs = ("expr_list", p[3]) if isinstance(p[3], list) else p[3]
            p[0] = ("stmt_tuple_unpack", p[1], rhs)

        return [
            p_return_type_single,
            p_return_type_tuple,
            p_typed_id_list,
            p_return_expr_list,
            p_top_level_expr_list,
            p_func_body_stmt_tuple_unpack,
            p_func_loop_stmt_tuple_unpack,
            p_statement_tuple_unpack,
            p_for_statement_tuple_unpack,
        ]

    def forward_rules(self) -> dict:
        """
        Three tags are handled:

        - ``"expr_list"``: comma-separated RHS literal values, e.g.
          ``a, b = 1, 2``.  Each element is emitted independently and
          joined with ``", "``.
        - ``"loop_tuple_unpack"``: produced inside ``for k:`` bodies of
          functions and class methods.  Forwards ``current_loop_var`` so
          index expressions (``arr[k]``) resolve correctly.
        - ``"stmt_tuple_unpack"``: produced at program level statements and
          top level for-loop bodies.  The caller passes ``current_loop_var``
          into the ``to_expr`` lambda before dispatching, so no extra
          forwarding is needed here.

        The two unpack handlers extract plain name strings from the LHS,
        which may be a plain ``id_list`` (``["a", "b"]``) or a
        ``typed_id_list`` (``[("a", type), ("b", type)]``).

        Returns
        -------
        dict
            ``{"expr_list": handler, "loop_tuple_unpack": handler,
            "stmt_tuple_unpack": handler}``.

        Examples
        --------
        >>> from physika.features.tuple_unpack import TupleUnpackFeature
        >>> from physika.utils.ast_utils import ast_to_torch_expr
        >>> rules = TupleUnpackFeature().forward_rules()
        >>> node = ("loop_tuple_unpack", ["spins", "lp"],
        ...         ("call", "f", [("var", "n")]))
        >>> rules["loop_tuple_unpack"](node, ast_to_torch_expr)
        'spins, lp = f(n)'
        >>> node2 = ("stmt_tuple_unpack", ["a", "b"],
        ...          ("call", "g", [("var", "x")]))
        >>> rules["stmt_tuple_unpack"](node2, ast_to_torch_expr)
        'a, b = g(x)'
        """

        def emit_expr_list(node: Tuple, to_expr: Callable, **ctx) -> str:
            # Comma-separated RHS of a tuple unpack: `x, y = a, b`.
            # Each element is emitted independently and joined with ", ".
            _, exprs = node
            return ", ".join(to_expr(e) for e in exprs)

        def emit_loop_tuple_unpack(node: Tuple,
                                   to_expr: Callable,
                                   current_loop_var=None,
                                   **ctx) -> str:
            _, var_names, expr = node
            names = [n if isinstance(n, str) else n[0] for n in var_names]
            expr_code = to_expr(expr, current_loop_var=current_loop_var)
            return f"{', '.join(names)} = {expr_code}"

        def emit_stmt_tuple_unpack(node: Tuple, to_expr: Callable,
                                   **ctx) -> str:
            _, var_names, expr = node
            names = [n if isinstance(n, str) else n[0] for n in var_names]
            return f"{', '.join(names)} = {to_expr(expr)}"

        return {
            "expr_list": emit_expr_list,
            "loop_tuple_unpack": emit_loop_tuple_unpack,
            "stmt_tuple_unpack": emit_stmt_tuple_unpack,
        }

    def type_rules(self) -> dict:
        """
        Type rules for tuple unpack AST nodes.

        ``infer_expr``checks type rules when a class
        method body is ``("tuple_return", e1, e2)``.  Both expressions
        ``e1`` and ``e2``  are type checked independently, so errors are
        catched and reported if types not properly defined.

        Returns
        -------
        dict
            Five handlers keyed by AST node tag.

        Examples
        --------
        >>> from physika.features.tuple_unpack import TupleUnpackFeature
        >>> from physika.utils.types import Substitution
        >>> rules = TupleUnpackFeature().type_rules()
        >>> sorted(rules.keys())
        ['body_tuple_unpack', 'expr_list', 'loop_tuple_unpack', 'stmt_tuple_unpack', 'tuple_return']  # noqa: E501
        >>> errors = []
        >>> env = {"a": None, "b": None}
        >>> from physika.utils.infer_expr import infer_expr
        >>> node = ("tuple_return", ("var", "a"), ("var", "b"))
        >>> t, _ = rules["tuple_return"](node, env, Substitution(), {}, {},
        ...                              errors.append, infer_expr)
        >>> t is None
        True
        >>> errors  # no errors for known vars
        []
        >>> errors2 = []
        >>> node2 = ("tuple_return", ("var", "missing"), ("var", "b"))
        >>> rules["tuple_return"](node2, {}, Substitution(), {}, {},
        ...                       errors2.append, infer_expr)
        (None, {})
        >>> len(errors2) > 0
        False
        >>> errors3 = []
        >>> env3 = {}
        >>> node3 = ("body_tuple_unpack", ["x", "y"], ("num", 1.0))
        >>> rules["body_tuple_unpack"](node3, env3, Substitution(), {}, {},
        ...                            errors3.append, infer_expr)
        (None, {})
        >>> "x" in env3 and "y" in env3
        True
        """
        from physika.utils.types import TScalar

        def check_tuple_return(node: tuple, env: dict, s: Substitution,
                               func_env: dict, class_env: dict,
                               add_error: Callable, infer_expr: Callable):
            for expr in node[1:]:
                _, s = infer_expr(expr, env, s, func_env, class_env, add_error)
            return None, s

        def check_expr_list(node: tuple, env: dict, s: Substitution,
                            func_env: dict, class_env: dict,
                            add_error: Callable, infer_expr: Callable):
            # ("expr_list", [e1, e2, ...]) comma RHS of a tuple unpack.
            for sub in node[1]:
                # type check each sub expression
                _, s = infer_expr(sub, env, s, func_env, class_env, add_error)
            return None, s

        def check_tuple_unpack(node: tuple, env: dict, s: Substitution,
                               func_env: dict, class_env: dict,
                               add_error: Callable, infer_expr: Callable):
            from physika.utils.type_checker_utils import from_typespec, type_to_str  # noqa: E501
            _, names, expr = node

            if isinstance(expr, tuple) and expr[0] == "expr_list":
                # RHS is a literal comma list: `a, b = 1.0, 2.0`
                # Infer the type of each RHS element independently
                rhs_exprs = expr[1]
                rhs_types = []
                for e in rhs_exprs:
                    t, s = infer_expr(e, env, s, func_env, class_env,
                                      add_error)
                    rhs_types.append(t)

                for i, entry in enumerate(names):
                    # Pair each LHS name with its corresponding RHS type
                    # Falls back to None if the lists are unequal length

                    actual = rhs_types[i] if i < len(rhs_types) else None

                    if isinstance(entry, tuple):
                        # check declared type matches inferred.
                        name, type_spec = entry
                        declared = from_typespec(type_spec)
                        if actual is not None and declared != actual:
                            add_error(
                                f"Type mismatch in tuple unpack: '{name}' declared "  # noqa: E501
                                f"as {type_to_str(declared)} but got "
                                f"{type_to_str(actual)}")
                            # Register the actual type so later statements
                            env[name] = actual
                        else:
                            # Use declared type when inferred is None,
                            # otherwise trust the inferred type.
                            env[name] = declared if actual is None else actual
                    else:
                        # Untyped LHS register inferred type, default ℝ.
                        env[entry] = actual if actual is not None else TScalar(
                            "ℝ")
            else:
                # when rhs is a single expression: `a, b = f()` (function or
                # method call).
                _, s = infer_expr(expr, env, s, func_env, class_env, add_error)
                element_type = TScalar("ℝ")
                for entry in names:
                    if isinstance(entry, tuple):
                        # flag a mismatch if declared type isn't ℝ.
                        name, type_spec = entry
                        declared = from_typespec(type_spec)
                        if declared != element_type:
                            add_error(
                                f"Type mismatch in tuple unpack: '{name}' declared "  # noqa: E501
                                f"as {type_to_str(declared)} but element type is "  # noqa: E501
                                f"{type_to_str(element_type)}")
                            env[name] = element_type
                        else:
                            env[name] = declared
                    else:
                        # Untyped LHS register as ℝ.
                        env[entry] = element_type
            return None, s

        return {
            "tuple_return": check_tuple_return,
            "expr_list": check_expr_list,
            "body_tuple_unpack": check_tuple_unpack,
            "loop_tuple_unpack": check_tuple_unpack,
            "stmt_tuple_unpack": check_tuple_unpack,
        }
