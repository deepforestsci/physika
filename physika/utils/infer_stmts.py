from typing import Any, Callable, Optional, Tuple
from physika.utils.types import Substitution, Type, TVar, TDim, T_NAT, new_var, new_dim


class StmtContext:
    """
    Data class that contains the context of a Physika statement that is being
    inferred.

    ``StmtContext`` is passed to every ``stmt_*`` inference statement handler.

    Attributes
    ----------
    env : dict
        Maps variable names to their current types
    s : Substitution
        Dicitionary accumulated with valid substitutions. Each statement
        handler aims to resolve any unknown types using previous bindings
        or adding new ones.
    func_env : dict
        Maps user defined function names to ``(param_types, return_type)``.
    class_env : dict
        Maps class names to their definition dicts (``class_params``,
        ``return_type``, ...).
    add_error : Callable
        Error callback.
    func_name : str
        User defined function name for checking check body statements. This
        field is used when calling ``check_function`` from main type
        checking algorithm.
    return_type : Optional[Type]
        Used especifically in ``body_if_return`` and ``body_if_else_return``
        to unify the return expression type against it. Propagated into nested
        function body scopes so that return statements inside can still
        be checked.

    Examples
    --------
    >>> from physika.utils.infer_stmts import StmtContext
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(
    ...     env={}, s=Substitution(), func_env={}, class_env={},
    ...     add_error=errors.append, func_name="f", return_type=T_REAL,
    ... )
    >>> t = ctx.infer_type(("num", 1.0))
    >>> t
    ℝ
    >>> errors
    []
    """

    def __init__(self,
                 env: dict,
                 s: Substitution,
                 func_env: dict,
                 class_env: dict,
                 add_error: Callable,
                 func_name: str = "?",
                 return_type: Optional[Type] = None) -> None:
        self.env: dict = env
        self.s: Substitution = s
        self.func_env: dict = func_env
        self.class_env: dict = class_env
        self.add_error: Callable = add_error
        self.func_name: str = func_name
        self.return_type: Optional[Type] = return_type

    def infer_type(self, expr: Any) -> Optional[Type]:
        """Infer the type of a Physika expression.

        Calls ``infer_expr`` using the current context environments and
        updates ``self.s`` in place with new bindings if any.

        Parameters
        ----------
        expr : Any
            A Physika AST expression node  such as
            ``("add", left, right)``, ``("call", name, args)``,
            ``("index", arr, idx)``, ``("for_expr", var, size, body)``,
            or a numeric literal (``int`` / ``float``).

        Returns
        -------
        Optional[Type]
            The inferred ``Type`` (e.g. ``T_REAL``),
            or ``None`` if the expression is not resolved (unknown variable)


        Examples
        --------
        >>> from physika.utils.infer_stmts import StmtContext
        >>> from physika.utils.types import Substitution, T_REAL, TTensor
        >>> errors = []
        >>> ctx = StmtContext(
        ...     env={"v": TTensor(((3, "invariant"),))}, s=Substitution(), func_env={}, class_env={},  # noqa: E501
        ...     add_error=errors.append, func_name="f", return_type=T_REAL,
        ... )
        >>> ctx.infer_type(("num", 2.0))
        ℝ
        >>> ctx.infer_type(("var", "v"))
        ℝ[3]
        >>> errors
        []
        """
        from physika.utils.infer_expr import infer_expr

        t, self.s = infer_expr(expr, self.env, self.s, self.func_env,
                               self.class_env, self.add_error)
        return t


def stmt_body_decl(stmt: Tuple, ctx: StmtContext) -> None:
    """
    Infer type of declaration statements inside functions.

    The inferred type is obtained by calling ``ctx.infer_type`` on the expression ``expr`` found
    in ``body_decl`` statement. Then, the inferred type is unified with declared type including the updated bindings
    at ``s: Substitution`` . If there is a mismatch, an error is reported.

    Parameters
    ----------
    stmt: tuple
        AST node of the form ``("body_decl", 'var', 'var_type', expr)`` where *expr* can be any
        for of supported expressions at ``infer_expr``.
    ctx: StmtContext
        Current inference context

    Returns
    -------
    None
        If there is any type mismatch between inferred and declared, report an
        error and add to ``env`` the inferred type if not None. Else, add a
        new unknown var type. If there is no mismatch, add declared type at ``env``.
        If declared type is None, add inferred type if known, else add unknown var type.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_decl, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> stmt = ('body_decl', 'x', 'ℝ', ('add', ('num', 3), ('var', 'x')))
    >>> s = Substitution()
    >>> errors = []
    >>> ctx = StmtContext(
    ...     env={'x': T_REAL},
    ...     s=s,
    ...     func_env={'f': (['ℝ'], 'ℝ')},
    ...     class_env={},
    ...     add_error=errors.append,
    ...     func_name='f',
    ...     return_type=T_REAL)
    >>> stmt_body_decl(stmt, ctx)
    >>> print(errors)
    []
    >>> mismatch_stmt = ('body_decl', 'v', ('tensor', [(3, 'invariant')]), ('num', 2.0))  # noqa: E501
    >>> mismatch_errors = []
    >>> ctx2 = StmtContext(
    ...     env={}, s=Substitution(), func_env={}, class_env={},
    ...     add_error=mismatch_errors.append, func_name='f', return_type=T_REAL,
    ... )
    >>> stmt_body_decl(mismatch_stmt, ctx2)
    >>> mismatch_errors
    ["In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"]
    """
    from physika.utils.type_checker_utils import from_typespec, unify, type_to_str  # noqa: E501

    # example stmt node: ('body_decl', var, var_type, expr)
    _, var_name, var_type_spec, expr = stmt
    inferred = ctx.infer_type(expr)
    declared = from_typespec(var_type_spec)
    mismatch = False
    if declared is not None and inferred is not None:
        try:
            ctx.s = unify(declared, inferred, ctx.s)
        except TypeError as e:
            mismatch = True
            ctx.add_error(
                f"In '{ctx.func_name}': '{var_name}' declared {type_to_str(declared)}, "  # noqa: E501
                f"inferred {type_to_str(ctx.s.apply(inferred))}: {e}")

    # Update env dictionary
    if mismatch:
        if inferred is not None:
            ctx.env[var_name] = inferred
        else:
            ctx.env[var_name] = new_var()
    else:
        # No mismatch and declared exists
        if declared is not None:
            ctx.env[var_name] = declared
        else:
            # Add inferred value at env, or create a new variable
            # if none exists
            ctx.env[var_name] = inferred or new_var()


def stmt_body_assign(stmt: Any, ctx: StmtContext) -> None:
    """
    Inferred type of assingment statements inside functions.

    ``stmt_body_assign`` is similar to ``stmt_body_decl``, but in
    ``body_assign`` statements ``type_var`` is not declared. However,
    it is inferred and used for following unification until function
    return.

    If inference returns ``None`` (unknown expression), a new ``TVar``
    type variable is stored instead.

    Parameters
    ----------
    stmt: tuple
        AST node of the form ``("body_assign", var_name, expr)``.
    ctx: StmtContext
        Current inference context.

    Returns
    -------
    None
        Updates ``ctx.env[var_name]`` in place.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_assign, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL}, s=Substitution(), func_env={},
    ...                   class_env={}, add_error=errors.append)
    >>> stmt_body_assign(('body_assign', 'y', ('add', ('var', 'x'), ('num', 1.0))), ctx)  # noqa: E501
    >>> ctx.env['y']
    ℝ
    >>> errors
    []
    >>> ctx2 = StmtContext(env={}, s=Substitution(), func_env={}, class_env={},
    ...                    add_error=errors.append, return_type=T_REAL)
    >>> stmt_body_assign(('body_assign', 'v', ('array', [('num', 1.0), ('num', 2.0)])), ctx2)  # noqa: E501
    >>> ctx2.env['v']
    ℝ[2]
    >>> errors
    []
    """
    # assing statement example node: ("body_assign", var_name, expr)
    _, var_name, expr = stmt
    inferred = ctx.infer_type(expr)
    ctx.env[var_name] = inferred if inferred is not None else new_var()


def stmt_body_if_return(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer and check the return expression of an ``if`` return statement.

    The general form of an body if-return node looks like:
    - ``("body_if_return", cond_expr, ret_expr)``

    Correspoinding to a phyiska source code of the form::
    def func(x: ℝ) -> ℝ:
        if cond:
            return expr

    If ``ctx.return_type`` is set, the
    inferred type is unified against it and a type-mismatch error is reported
    on failure.

    Parameters
    ----------
    stmt : tuple
        AST node of the form ``("body_if_return", cond_expr, ret_expr)``.
    ctx : StmtContext
        Current inference context.  ``ctx.return_type`` must be set (by
        ``check_function``) for the return-type check to fire.

    Returns
    -------
    None
        Updates ``ctx.s`` with any new unification bindings.
        Calls ``ctx.add_error`` if the inferred return type does not match
        ``ctx.return_type``.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_if_return, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL},
    ...                        s=Substitution(),
    ...                        func_env={},
    ...                        class_env={},
    ...                        func_name='f',
    ...                        return_type=T_REAL,
    ...                        add_error=errors.append)
    >>> cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
    >>> stmt_body_if_return(('body_if_return', cond, ('var', 'x')), ctx)
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import unify, type_to_str

    _, cond, ret_expr = stmt
    ctx.infer_type(cond)
    ret_t = ctx.infer_type(ret_expr)
    if ctx.return_type is not None and ret_t is not None:
        try:
            ctx.s = unify(ctx.return_type, ctx.s.apply(ret_t), ctx.s)
        except TypeError as e:
            ctx.add_error(f"if-return type mismatch: "
                          f"declared {type_to_str(ctx.return_type)}, "
                          f"got {type_to_str(ctx.s.apply(ret_t))}: {e}")


def stmt_body_if_else_return(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer and check both branches of an ``if/else`` return statement.

    Handles ``("body_if_else_return", cond_expr, then_expr, else_expr)`` nodes.

    Physika source code would look like:

        if cond:
            return then_expr
        else:
            return else_expr

    Type inference checks for ``then_expr`` and ``else_expr`` types, which are
    unified against each other. A mismatch here means the two branches disagree
    on what the function returns.

    The unified branch type is unified against
    ``ctx.return_type`` (the declared return type of the function).
    A mismatch here means the ``if`` and ``else`` branches match types,
    but do not match the declaration.

    Parameters
    ----------
    stmt : tuple
        AST node of the form
        ``("body_if_else_return", cond_expr, then_expr, else_expr)``.
    ctx : StmtContext
        Current inference context.  ``ctx.return_type`` must be set for the
        return-type check to fire.

    Returns
    -------
    None
        Updates ``ctx.s`` with any new unification bindings.
        Calls ``ctx.add_error`` for each failed unification.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_if_else_return, StmtContext  # noqa: E501
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL},
    ...                        s=Substitution(),
    ...                        func_name='f',
    ...                        return_type=T_REAL,
    ...                        add_error=errors.append,
    ...                        func_env={},
    ...                        class_env={})
    >>> cond = ("cond_gt", ("var", "x"), ("num", 0.0))
    >>> stmt = ("body_if_else_return", cond, ("var", "x"),
    ...         ("num", 0.0))
    >>> stmt_body_if_else_return(stmt, ctx)
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import unify, type_to_str

    _, cond, then_expr, else_expr = stmt
    ctx.infer_type(cond)
    then_t = ctx.infer_type(then_expr)
    else_t = ctx.infer_type(else_expr)

    # Phase 1: branch consistency — then and else must agree
    if then_t is not None and else_t is not None:
        try:
            ctx.s = unify(ctx.s.apply(then_t), ctx.s.apply(else_t), ctx.s)
        except TypeError as e:
            ctx.add_error(f"if/else branch type mismatch: "
                          f"then={type_to_str(ctx.s.apply(then_t))}, "
                          f"else={type_to_str(ctx.s.apply(else_t))}: {e}")

    # Phase 2: unified branch type must match declared return type
    unified_branch = ctx.s.apply(then_t) if then_t is not None else (
        ctx.s.apply(else_t) if else_t is not None else None)
    if ctx.return_type is not None and unified_branch is not None:
        try:
            ctx.s = unify(ctx.return_type, unified_branch, ctx.s)
        except TypeError as e:
            ctx.add_error(f"if/else return type mismatch: "
                          f"declared {type_to_str(ctx.return_type)}, "
                          f"got {type_to_str(unified_branch)}: {e}")


def stmt_body_if_else(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer and check both branches of an ``if/else`` statement, without return.

    Handles ``("body_if_else", cond_expr, then_stmts, else_stmts)`` and
    ``("body_if", cond_expr, then_stmts)`` nodes.

    Runs type inference on each branch body to catch expression errors inside
    them.


    Parameters
    ----------
    stmt : tuple
        AST node of the form
        ``("body_if_else", cond_expr, then_stmts, else_stmts)`` or
        ``("body_if", cond_expr, then_stmts)``.
    ctx : StmtContext
        Current inference context.

    Returns
    -------
    None
        Calls ``ctx.add_error`` for each type error found inside a branch.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_if_else, StmtContext  # noqa: E501
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL},
    ...                        s=Substitution(),
    ...                        func_name='f',
    ...                        return_type=T_REAL,
    ...                        add_error=errors.append,
    ...                        func_env={},
    ...                        class_env={})
    >>> cond = ("cond_gt", ("var", "x"), ("num", 0.0))
    >>> then_stmts = [("body_assign", "y", ("var", "x"))]
    >>> else_stmts = [("body_assign", "y", ("num", 0.0))]
    >>> stmt = ("body_if_else", cond, then_stmts, else_stmts)
    >>> stmt_body_if_else(stmt, ctx)
    >>> errors
    []
    """
    op = stmt[0]
    _, _, then_stmts = stmt[:3]
    else_stmts = stmt[3] if op == "body_if_else" else []

    infer_stmts(then_stmts, ctx.env, ctx.s,
                ctx.func_env, ctx.class_env, ctx.add_error,
                ctx.func_name, ctx.return_type)
    infer_stmts(else_stmts, ctx.env, ctx.s,
                ctx.func_env, ctx.class_env, ctx.add_error,
                ctx.func_name, ctx.return_type)


def stmt_body_for(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer the types of ``body_for`` node statements which calls ``infer_stmts``
    since there are statements inside body for loops. A general physika program
    that will trigger this inference handler would look like:

    def f(x : ℝ[n]): ℝ:
        for i:
            total = op

    ``stmt_body_for`` will infer and compare types of any operations defined
    inside for-loop body.

    Parameters
    ----------
    stmt : tuple
        AST node of the form
        ``("body_for", loop_var, body_for)``.
    ctx : StmtContext
        Current inference context.  ``ctx.return_type``.
    
    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_for, StmtContext  # noqa: E501
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL},
    ...                        s=Substitution(),
    ...                        func_name='f',
    ...                        return_type=T_REAL,
    ...                        add_error=errors.append,
    ...                        func_env={},
    ...                        class_env={})
    >>> stmt = ('body_for', 'i', [('loop_pluseq', 'total', ('index', 'arr', ('imaginary',)))], ['arr']) # noqa: E501
    >>> stmt_body_for(stmt, ctx)
    >>> errors
    []
    """
    _, loop_var, loop_body, _ = stmt
    ctx.env[loop_var] = T_NAT
    ctx.env, ctx.s = infer_stmts(loop_body, ctx.env, ctx.s,
                                         ctx.func_env, ctx.class_env, ctx.add_error,
                                         ctx.func_name, ctx.return_type)

def stmt_body_for_range(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer the types of ``body_for_range`` node statements inside a ranged for loop.

    Handles ``("body_for_range", loop_var, start, stop, body)`` nodes where the
    loop variable iterates over an integer range.  A Physika program
    that triggers this handler looks like:

    def f(x : ℝ[n]): ℝ:
        for i: ℕ(n):
            total = op

    The loop variable is registered as ``T_NAT`` so that index expressions
    inside the body can be checked against array dimensions.

    Parameters
    ----------
    stmt : tuple
        AST node of the form
        ``("body_for_range", loop_var, start, stop, body)``.
    ctx : StmtContext
        Current inference context.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_for_range, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL}, s=Substitution(), func_name='f',
    ...                   return_type=T_REAL, add_error=errors.append,
    ...                   func_env={}, class_env={})
    >>> stmt = ('body_for_range', 'i', ('num', 0), ('num', 4),
    ...         [('loop_assign', 'y', ('var', 'x'))])
    >>> stmt_body_for_range(stmt, ctx)
    >>> errors
    []
    """
    _, loop_var, _, _, loop_body = stmt
    ctx.env[loop_var] = T_NAT
    ctx.env, ctx.s = infer_stmts(loop_body, ctx.env, ctx.s,
                                         ctx.func_env, ctx.class_env, ctx.add_error,
                                         ctx.func_name, ctx.return_type)

def stmt_body_zeros_decl(stmt: Any, ctx: StmtContext) -> None:
    """Register a zero initialised array declaration in the type environment.

    Handles ``("body_zeros_decl", name, type_spec)`` nodes. The declared type
    is added to ``ctx.env`` so that subsequent node statements can
    look up the array's shape.

    Parameters
    ----------
    stmt : tuple
        AST node of the form ``("body_zeros_decl", name, type_spec)``.
    ctx : StmtContext
        Current inference context.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_zeros_decl, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(env={}, s=Substitution(), func_name='f',
    ...                   return_type=T_REAL, add_error=errors.append,
    ...                   func_env={}, class_env={})
    >>> stmt_body_zeros_decl(('body_zeros_decl', 'C',
    ...                       ('tensor', [(3, 'invariant'), (3, 'invariant')])), ctx)
    >>> ctx.env['C']
    ℝ[3,3]
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import from_typespec

    _, var_name, type_spec = stmt
    declared = from_typespec(type_spec)
    if declared is not None: 
        ctx.env[var_name] = declared
    else:
        ctx.env[var_name] = new_var()


def stmt_body_for_accum(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer the types of ``body_for_accum`` node statements inside an
    multi-iter variable for loop.

    Handles ``("body_for_accum", loop_vars, body)`` nodes where multiple loop
    variables are used to make an in place accumulation ( ``+=``) #TODO allow
    general operations at ast_utils.py.
    
    A Physika program that triggers this handler looks like:

    def physika_matmul(A : ℝ[n, m], B : ℝ[m, o]): ℝ[n, o]:
        C : ℝ[n, o]
        for i j k:
            C[i, j] += A[i, k] * B[k, j]

    Each loop variable is registered as a new dimension type variable for
    later unification.

    Parameters
    ----------
    stmt : tuple
        AST node of the form ``("body_for_accum", loop_vars, body)`` where
        ``loop_vars`` is a list of loop variable name strings.
    ctx : StmtContext
        Current inference context.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_body_for_accum, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL}, s=Substitution(), func_name='f',
    ...                   return_type=T_REAL, add_error=errors.append,
    ...                   func_env={}, class_env={})
    >>> stmt = ('body_for_accum', ['i', 'j'],
    ...         [('loop_assign', 'y', ('var', 'x'))])
    >>> stmt_body_for_accum(stmt, ctx)
    >>> errors
    []
    """
    _, loop_vars, loop_body = stmt
    for lv in loop_vars:
        ctx.env[lv] = new_dim()
    ctx.env, ctx.s = infer_stmts(loop_body, ctx.env, ctx.s,
                                         ctx.func_env, ctx.class_env, ctx.add_error,
                                         ctx.func_name, ctx.return_type)

def stmt_for_assign(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer the type of an assignment statement inside a for loop body.

    Handles ``("loop_assign", var_name, rhs)`` nodes. Infer the type of rhs
    of assignment and its type is stored in the environment under ``var_name``.
    A Physika program that triggers this handler looks like:

    def f(x : ℝ[n]): ℝ:
        for i:
            y = x[i]

    If the rhs type cannot be inferred, a fresh type variable is used so that
    subsequent statements inside the same loop body can still be checked/unified.

    Parameters
    ----------
    stmt : tuple
        AST node of the form ``("loop_assign", var_name, rhs_expr)``.
    ctx : StmtContext
        Current inference context.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_for_assign, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL}, s=Substitution(), func_name='f',
    ...                   return_type=T_REAL, add_error=errors.append,
    ...                   func_env={}, class_env={})
    >>> stmt_for_assign(('loop_assign', 'y', ('var', 'x')), ctx)
    >>> ctx.env['y']
    ℝ
    >>> errors
    []
    """
    from physika.utils.infer_expr import infer_expr
    _, var_name, rhs = stmt
    rhs_t, ctx.s = infer_expr(rhs, ctx.env, ctx.s,
                                  ctx.func_env, ctx.class_env, ctx.add_error)
    if rhs_t is not None:
        ctx.env[var_name] = rhs_t
    else:
        new_var()

def stmt_for_pluseq(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer types for ``+=`` accumulation statements inside a for loop body.

    Handles two ASTNodes dispatched to the same handler:

    - ``("for_pluseq", arr_name, idx_exprs, rhs)`
    - ``("loop_index_pluseq", arr_name, idx_exprs, rhs)``, where each index
    expression is unified against the known array dimension

    A Physika program that triggers this handler looks like:

    def f(A : ℝ[n, m], B : ℝ[m, o]): ℝ[n, o]:
        C : ℝ[n, o]
        for i j k:
            C[i, j] += A[i, k] * B[k, j] # stmt_for_pluseq node

    Parameters
    ----------
    stmt : tuple
        AST node of the form ``("for_pluseq", arr_name, idx_exprs, rhs)``
        or ``("loop_index_pluseq", arr_name, idx_exprs, rhs)``.
    ctx : StmtContext
        Current inference context.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_for_pluseq, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL, TTensor
    >>> errors = []
    >>> ctx = StmtContext(
    ...     env={'C': TTensor(((3, 'invariant'), (3, 'invariant')))},
    ...     s=Substitution(), func_name='f', return_type=T_REAL,
    ...     add_error=errors.append, func_env={}, class_env={})
    >>> stmt_for_pluseq(('for_pluseq', 'C', [], ('num', 1.0)), ctx)
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import get_tensor_shape, unify_dim
    from physika.utils.infer_expr import infer_expr
    op = stmt[0]
    # case basic accumulator (no indexing)
    _, ctx.s = infer_expr(stmt[-1], ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, ctx.add_error)
    # case indexed acummulator
    if op == "loop_index_pluseq":
        _, arr_name, idx_exprs, _ = stmt
        arr_t = ctx.s.apply(ctx.env[arr_name]) if arr_name in ctx.env else None
        shape = get_tensor_shape(arr_t)
        if shape is not None:
            for idx_expr, dim in zip(idx_exprs, shape):
                idx_t, ctx.s = infer_expr(idx_expr, ctx.env, ctx.s,
                                              ctx.func_env, ctx.class_env, ctx.add_error)
                if isinstance(idx_t, (TVar, TDim, str, int)):
                    try:
                        ctx.s = unify_dim(idx_t, dim, ctx.s)
                    except TypeError as e:
                        ctx.add_error(f"Index mismatch for '{arr_name}': {e}")

def stmt_loop_if(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer types for an ``if` statement inside a for loop body.

    Handles ``("loop_if", cond_expr, then_body)`` nodes. The condition
    and statements in the then branch are type checked. A Physika program
    that triggers this handler looks like:

    def f(x : ℝ[n]): ℝ:
        for i:
            if x[i] > 0.0: # if only branch
                total = total + x[i]

    Parameters
    ----------
    stmt : tuple
        AST node of the form ``("loop_if", cond_expr, then_body)`` where
        ``then_body`` is a list of loop body statement nodes.
    ctx : StmtContext
        Current inference context.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_loop_if, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL}, s=Substitution(), func_name='f',
    ...                   return_type=T_REAL, add_error=errors.append,
    ...                   func_env={}, class_env={})
    >>> cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
    >>> stmt_loop_if(('loop_if', cond, [('loop_assign', 'y', ('var', 'x'))]), ctx)
    >>> errors
    []
    """
    from physika.utils.infer_expr import infer_expr

    _, cond, then_body = stmt
    infer_expr(cond, ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error)
    ctx.env, ctx.s = infer_stmts(then_body, ctx.env, ctx.s,
                                 ctx.func_env, ctx.class_env, ctx.add_error,
                                 ctx.func_name, ctx.return_type)

def stmt_loop_if_else(stmt: Any, ctx: StmtContext) -> None:
    """
    Infer types for ``if/else`` statement inside a for loop body.

    Handles ``("loop_if_else", cond_expr, then_body, else_body)`` nodes.
    The condition and both branch bodies are type-checked. A Physika program
    that triggers this handler looks like:

    def f(x : ℝ[n]): ℝ:
        for i:
            if x[i] > 0.0:
                pos = pos + x[i]
            else:
                neg = neg + x[i]

    Parameters
    ----------
    stmt : tuple
        AST node of the form
        ``("loop_if_else", cond_expr, then_body, else_body)`` where each
        body is a list of loop body statement nodes.
    ctx : StmtContext
        Current inference context.

    Examples
    --------
    >>> from physika.utils.infer_stmts import stmt_loop_if_else, StmtContext
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(env={'x': T_REAL}, s=Substitution(), func_name='f',
    ...                   return_type=T_REAL, add_error=errors.append,
    ...                   func_env={}, class_env={})
    >>> cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
    >>> stmt_loop_if_else(('loop_if_else', cond,
    ...                    [('loop_assign', 'y', ('var', 'x'))],
    ...                    [('loop_assign', 'y', ('num', 0.0))]), ctx)
    >>> errors
    []
    """
    from physika.utils.infer_expr import infer_expr

    _, cond, then_body, else_body = stmt
    infer_expr(cond, ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error)
    ctx.env, ctx.s = infer_stmts(then_body, ctx.env, ctx.s,
                                 ctx.func_env, ctx.class_env, ctx.add_error,
                                 ctx.func_name, ctx.return_type)
    ctx.env, ctx.s = infer_stmts(else_body, ctx.env, ctx.s,
                                 ctx.func_env, ctx.class_env, ctx.add_error,
                                 ctx.func_name, ctx.return_type)



STMT_DISPATCH: dict = {
    "body_decl":           stmt_body_decl,
    "body_assign":         stmt_body_assign,
    "body_if_return":      stmt_body_if_return,
    "body_if_else_return": stmt_body_if_else_return,
    "body_if_else":        stmt_body_if_else,
    "body_if":             stmt_body_if_else,
    "body_for":            stmt_body_for,
    "body_for_range":      stmt_body_for_range,
    "body_zeros_decl":     stmt_body_zeros_decl,
    "body_for_accum":      stmt_body_for_accum,
    "for_assign":          stmt_for_assign,
    "loop_assign":         stmt_for_assign,
    "for_pluseq":          stmt_for_pluseq,
    "loop_index_pluseq":   stmt_for_pluseq,
    "loop_if":             stmt_loop_if,
    "loop_if_else":        stmt_loop_if_else,
}

def infer_stmts(
    stmts: list,
    env: dict,
    s: Substitution,
    func_env: dict,
    class_env: dict,
    add_error: Callable,
    func_name: str = "?",
    return_type: Optional[Type] = None,
) -> Tuple[dict, Substitution]:
    """
    Infer types for a given statement ASTNode.

    ``infer_stmts`` is called from main type algorithm at inference time
    when checking types for functions (body) and top-level programs.

    Returns a tuple of updated enviroment, which is a dictionary containing
    variable names and type annotations, and substitution (``Substitution``)
    object with resolved types).

    Parameters
    ----------
    stmts: list
        List of statements ASTNodes to check types at body and top-level
        program. Every `statement` contain operations with `expressions`,
        so ``infer_expr`` is called.
    env: dict
        Dictionary object that represents the current enviroment with
        declared functions, varibles and types.
    s: Substitution
        Substitution object with resolved and unresolved types.
    func_env: dict
        Dictionary object representing the enviroment of a user defined
        function in a physika program. Contains variables names and types
        , arguments and return types.
    class_env: dict
        Dictionary object representing the enviroment of a user defined
        class in a physika program. Contains variables names and types
        , arguments and return types.
    add_error: Callable
        Callable append function to store collected type checker errors.
    func_name: str
        User defined function name for checking body statements. This
        field is used when calling ``check_function`` from main type
        checking algorithm and passed to ``s:Substitution`` dict.
    return_type: Optional[Type]
        Passed to ``s:Substitution`` dict for checking if-else-body statements.
    
    Example
    -------
    >>> from physika.utils.infer_stmts import infer_stmts
    >>> from physika.utils.types import Substitution, T_REAL
    >>> errors = []
    >>> ctx = StmtContext(
    ...     env={}, s=Substitution(), func_env={}, class_env={},
    ...     add_error=errors.append, func_name="f", return_type=T_REAL,
    ... )
    >>> t = ctx.infer_type(("num", 1.0))
    >>> t
    ℝ
    >>> errors
    []
    >>> stmts = [("num", 1.0), ('body_assign', 'y', ('add', ('var', 'x'), ('num', 1.0)))]
    >>> infer_stmts(stmts, env={}, s=Substitution(), func_env={},
    ...                   class_env={}, add_error=errors.append)
    ({'y': ℝ}, {})
    """
    ctx = StmtContext(
        env=dict(env),
        s=s,
        func_env=func_env,
        class_env=class_env,
        add_error=add_error,
        func_name=func_name,
        return_type=return_type,
    )
    for stmt in stmts:
        if stmt is None:
            continue
        handler = STMT_DISPATCH.get(stmt[0])
        if handler is not None:
            handler(stmt, ctx)
    return ctx.env, ctx.s
