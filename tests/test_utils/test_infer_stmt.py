from physika.utils.types import (
    TTensor,
    T_REAL,
    T_NAT,
    T_COMPLEX,
    TVar,
    TDim,
    Substitution,
    new_dim,
)
from physika.utils.infer_stmts import (
    StmtContext,
    stmt_body_decl,
    stmt_body_assign,
    stmt_body_if_return,
    stmt_body_if_else_return,
    stmt_if as stmt_body_if_else,
    stmt_for as stmt_body_for,
    stmt_for as stmt_body_for_range,
    stmt_body_zeros_decl,
    stmt_body_for_accum,
    stmt_for_assign,
    stmt_for_pluseq,
    stmt_if as stmt_loop_if,
    stmt_if as stmt_loop_if_else,
    stmt_decl,
    stmt_assign,
    stmt_expr,
    infer_stmts,
)


class TestStmtContext:
    """
    Tests for ``StmtContext``
    """

    def test_fields(self):
        """
        Checks that constructor arguments are stored as attributes.
        """
        env = {"x": T_REAL}
        s = Substitution({"α0": T_NAT})
        func_env = {"f": ([T_REAL], T_REAL)}
        class_env = {"Net": {"class_params": []}}
        errors = []
        cb = errors.append
        func_name = 'f'
        return_type = T_REAL
        ctx = StmtContext(env, s, func_env, class_env, cb, func_name,
                          return_type)
        assert ctx.env is env
        assert ctx.s is s
        assert ctx.func_env is func_env
        assert ctx.class_env is class_env
        assert ctx.add_error is cb

    def test_empty_dicts(self):
        """
        All dict arguments may be empty.
        """
        ctx = StmtContext(env={},
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=[].append)
        assert ctx.env == {}
        assert ctx.func_env == {}
        assert ctx.class_env == {}

    def test_add_error(self):
        """
        ``add_error`` stores error messages.
        """
        errors = []
        ctx = StmtContext(env={},
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=errors.append)

        ctx.add_error("error #1")
        assert errors == ["error #1"]

        # Multiple errors accumulate in order.
        ctx.add_error("error #2")
        assert errors == ["error #1", "error #2"]

    def test_s_substitution(self):
        """
        ``s`` is stored as the exact Substitution passed in.
        """
        s = Substitution({"α1": T_COMPLEX})
        ctx = StmtContext(env={},
                          s=s,
                          func_env={},
                          class_env={},
                          add_error=[].append)
        assert isinstance(ctx.s, Substitution)
        assert ctx.s["α1"] == T_COMPLEX

    def test_env_mutation(self):
        """
        Mutating the env dict after construction changes ctx.env
        """
        env = {}
        ctx = StmtContext(env=env,
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=[].append)
        env["y"] = T_NAT
        assert ctx.env["y"] == T_NAT


# helper function to create a StmtContext object
def make_stmt_ctx(env=None,
                  s=None,
                  func_env=None,
                  class_env=None,
                  errors=None,
                  func_name=None,
                  return_type=None):
    """
    Build an StmtContext with sensible defaults for unit tests.
    """
    if env is None:
        env = {}
    if s is None:
        s = Substitution()
    if func_env is None:
        func_env = {}
    if class_env is None:
        class_env = {}
    if errors is None:
        errors = []
    if func_name is None:
        func_name = ''

    return StmtContext(
        env=env,
        s=s,
        func_env=func_env,
        class_env=class_env,
        add_error=errors.append,
        func_name=func_name,
        return_type=return_type,
    )


class TestInferTypeMethod:
    """Test StmtContext.infer_type infers correct types for expressions."""

    def test_numeric(self):
        """Numeric expressions infers to ℝ."""
        ctx = make_stmt_ctx()
        assert ctx.infer_type(("num", 3.14)) == T_REAL

        ctx = make_stmt_ctx()
        assert ctx.infer_type(("num", 0)) == T_REAL

        # lookup from env
        ctx = make_stmt_ctx(env={"x": T_REAL})
        assert ctx.infer_type(("var", "x")) == T_REAL

    def test_tensor_var(self):
        """Tensor variable returns its tensor type."""
        vec = TTensor(((3, "invariant"), ))
        ctx = make_stmt_ctx(env={"v": vec})
        assert ctx.infer_type(("var", "v")) == vec

    def test_add_expression(self):
        """Addition of two scalars infers to ℝ."""
        ctx = make_stmt_ctx(env={"x": T_REAL})
        assert ctx.infer_type(("add", ("var", "x"), ("num", 1.0))) == T_REAL

    def test_substitution_updated(self):
        """self.s is the same object after inference."""
        ctx = make_stmt_ctx()
        s_before = ctx.s
        ctx.infer_type(("num", 1.0))
        assert ctx.s is s_before

    def test_unknown_variable_returns_none(self):
        """Variable not in env returns None without raising."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        t = ctx.infer_type(("var", "unknown"))
        assert t is None


class TestStmtBodyDecl:
    """Test function body declaration statements (stmt_body_decl)."""

    def test_add_expr(self):
        """Inferred ℝ matches declared ℝ"""
        stmt = ('body_decl', 'x', 'ℝ', ('add', ('num', 3), ('var', 'x')))
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_env={'f': (['ℝ'], 'ℝ')},
                            errors=errors)
        stmt_body_decl(stmt, ctx)
        assert ctx.env['x'] == T_REAL
        assert errors == []

    def test_numeric_literal(self):
        """Declaring a variable with a numeric literal"""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'y', 'ℝ', ('num', 42.0)), ctx)
        assert ctx.env['y'] == T_REAL
        assert errors == []

    def test_tensor_declared_and_inferred(self):
        """Inferred array ℝ[3] matches declared ℝ[3]."""
        errors = []
        a_type = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(errors=errors)
        stmt = ('body_decl', 'v', ('tensor', [(3, 'invariant')]),
                ('array', [('num', 1.0), ('num', 2.0), ('num', 3.0)]))
        stmt_body_decl(stmt, ctx)
        assert ctx.env['v'] == a_type
        assert errors == []

    def test_no_declared_type(self):
        """No type annotation, but env dict gets the inferred type."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'z', None, ('num', 7.0)), ctx)
        assert ctx.env['z'] == T_REAL
        assert errors == []

    def test_type_mismatch_reports_error(self):
        """Declared ℝ[3] but infers ℝ."""
        errors = []
        ctx = make_stmt_ctx(func_name='f', errors=errors)
        stmt = ('body_decl', 'v', ('tensor', [(3, 'invariant')]), ('num', 2.0))
        stmt_body_decl(stmt, ctx)
        assert len(errors) == 1
        assert errors == [
            "In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3]"
            " with scalar ℝ"
        ]
        # if mismatch, env gets the inferred type
        assert ctx.env['v'] == T_REAL

    def test_type_mismatch(self):
        """Declared ℝ but infers ℝ[2]"""
        errors = []
        ctx = make_stmt_ctx(func_name='f', errors=errors)
        stmt = ('body_decl', 'x', 'ℝ', ('array', [('num', 1.0), ('num', 2.0)]))
        stmt_body_decl(stmt, ctx)
        assert len(errors) == 1
        assert errors == [
            "In 'f': 'x' declared ℝ, inferred ℝ[2]: Cannot unify scalar ℝ with"
            " tensor ℝ[2]"
        ]

    def test_env_updated(self):
        """Variable is accessible in env for subsequent expressions."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'a', 'ℝ', ('num', 1.0)), ctx)
        assert ctx.env['a'] == T_REAL
        stmt_body_decl(
            ('body_decl', 'b', 'ℝ', ('add', ('var', 'a'), ('num', 2.0))), ctx)
        assert ctx.env['b'] == T_REAL
        assert errors == []

    def test_index(self):
        """
        Indexing a ℝ[3,4] matrix infers to ℝ[4] — declared ℝ[4] matches.
        """
        errors = []
        mat = TTensor(((3, 'invariant'), (4, 'invariant')))
        ctx = make_stmt_ctx(env={'A': mat}, errors=errors)
        stmt_body_decl(('body_decl', 'r', ('tensor', [(4, 'invariant')]),
                        ('index', 'A', ('num', 0.0))), ctx)
        assert ctx.env['r'] == TTensor(((4, 'invariant'), ))
        assert errors == []

    def test_slice(self):
        """Slicing ℝ[6] with literal bounds infers ℝ[3] type"""
        errors = []
        vec = TTensor(((6, 'invariant'), ))
        sliced = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec}, errors=errors)
        stmt_body_decl(('body_decl', 's', ('tensor', [(3, 'invariant')]),
                        ('slice', 'v', ('num', 1.0), ('num', 4.0))), ctx)
        assert ctx.env['s'] == sliced
        assert errors == []

    def test_for_expr(self):
        """for i : ℕ(3) infers ℝ[3] type"""
        errors = []
        vec3 = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_decl(('body_decl', 'v', ('tensor', [(3, 'invariant')]),
                        ('for_expr', 'i', ('num', 3.0), ('num', 1.0))), ctx)
        assert ctx.env['v'] == vec3
        assert errors == []

    def test_index_mismatch(self):
        """Declared ℝ[4] but index of ℝ[4] yields ℝ — mismatch reported."""
        errors = []
        vec = TTensor(((4, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec}, func_name='f', errors=errors)
        stmt_body_decl(('body_decl', 'x', ('tensor', [(4, 'invariant')]),
                        ('index', 'v', ('num', 0.0))), ctx)
        assert len(errors) == 1
        assert errors == [
            "In 'f': 'x' declared ℝ[4], inferred ℝ: Cannot unify tensor ℝ[4] with scalar ℝ"  # noqa: E501
        ]


class TestStmtBodyAssign:
    """Test function body assignment statements (stmt_body_assign)."""

    def test_scalar_assignment(self):
        """Assigning a numeric literal registers ℝ in env."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'x', ('num', 3.0)), ctx)
        assert ctx.env['x'] == T_REAL
        assert errors == []

    def test_add_expression(self):
        """Assigning a scalar addition registers ℝ in env."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        stmt_body_assign(
            ('body_assign', 'y', ('add', ('var', 'x'), ('num', 1.0))), ctx)
        assert ctx.env['y'] == T_REAL
        assert errors == []

    def test_array_assignment(self):
        """Assigning an array literal registers tensor type in env."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'v', ('array', [('num', 1.0),
                                                         ('num', 2.0),
                                                         ('num', 3.0)])), ctx)
        assert ctx.env['v'] == TTensor(((3, 'invariant'), ))
        assert errors == []

    def test_no_type_error(self):
        """stmt_body_assign dont reports errors."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, func_name='f', errors=errors)
        # Assigning a tensor to x even though x was ℝ
        stmt_body_assign(('body_assign', 'x', ('array', [('num', 1.0),
                                                         ('num', 2.0)])), ctx)
        assert errors == []
        assert ctx.env['x'] == TTensor(((2, 'invariant'), ))

    def test_env_updated(self):
        """Assigned variable is used in following assignments."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'a', ('num', 5.0)), ctx)
        stmt_body_assign(
            ('body_assign', 'b', ('add', ('var', 'a'), ('num', 1.0))), ctx)
        assert ctx.env['a'] == T_REAL
        assert ctx.env['b'] == T_REAL
        assert errors == []

    def test_slice(self):
        """
        Assigning a slice of ℝ[6] with literal bounds registers ℝ[3]
        in env.
        """
        errors = []
        vec = TTensor(((6, 'invariant'), ))
        sliced = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec}, errors=errors)
        stmt_body_assign(
            ('body_assign', 's', ('slice', 'v', ('num', 1.0), ('num', 4.0))),
            ctx)
        assert ctx.env['s'] == sliced
        assert errors == []

    def test_for_expr_scalar_body(self):
        """Assigning for i : ℕ(3) → num registers ℝ[3] in env."""
        errors = []
        vec3 = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'v', ('for_expr', 'i', ('num', 3.0),
                                               ('num', 1.0))), ctx)
        assert ctx.env['v'] == vec3
        assert errors == []

    def test_fresh_var(self):
        """If type cannot be inferred, a new TVar is stored """
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_assign(('body_assign', 'z', ('var', 'unknown')), ctx)
        assert isinstance(ctx.env['z'], TVar)


class TestStmtBodyIfReturn:
    """Test if return statement type checking."""

    def test_basic_if_return_cond_infer(self):
        """Return type ℝ matches declared ℝ and conditions infers to ℝ"""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
        stmt_body_if_return(('body_if_return', cond, ('var', 'x')), ctx)
        assert errors == []

    def test_return_type_mismatch(self):
        """Return type ℝ[3] does not match declared ℝ."""
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('num', 1.0), ('num', 0.0))
        stmt_body_if_return(('body_if_return', cond, ('var', 'v')), ctx)
        vec_type = type(vec)
        ret_type = ctx.return_type
        assert vec_type == TTensor
        assert ret_type == T_REAL
        assert vec_type != ret_type
        assert len(errors) == 1
        assert errors == [
            'if-return type mismatch: declared ℝ, got ℝ[3]: Cannot unify scalar ℝ with tensor ℝ[3]'  # noqa: E501
        ]

    def test_return_cond_type_mismatch(self):
        """
        Declared and return type mismatch.
        Condition type also mismatches:
          TTensor!=T_REAL
        """
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('var', 'v'), ('num', 0.0))
        stmt_body_if_return(('body_if_return', cond, ('var', 'v')), ctx)
        vec_type = type(vec)
        ret_type = ctx.return_type
        assert vec_type == TTensor
        assert ret_type == T_REAL
        assert vec_type != ret_type
        assert len(errors) == 2
        assert errors[
            0] == "ℝ[3] is not comparable with ℝ at 'cond_gt' expression"
        assert errors[
            1] == 'if-return type mismatch: declared ℝ, got ℝ[3]: Cannot unify scalar ℝ with tensor ℝ[3]'  # noqa: E501


class TestStmtBodyIfElseReturn:
    """Test if-else-return statement type checking."""

    def test_both_branches_match_declared(self):
        """Both branches return ℝ matching declared ℝ"""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
        stmt = ('body_if_else_return', cond, ('var', 'x'), ('num', 0.0))
        stmt_body_if_else_return(stmt, ctx)
        assert errors == []

    def test_branchs_types_mismatch(self):
        """If branch returns ℝ[3], else returns ℝ"""
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('num', 1.0), ('num', 0.0))
        stmt = ('body_if_else_return', cond, ('var', 'v'), ('num', 0.0))
        stmt_body_if_else_return(stmt, ctx)

        assert len(errors) == 2
        assert errors[
            0] == "if/else branch type mismatch: then=ℝ[3], else=ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501
        assert errors[
            1] == 'if/else return type mismatch: declared ℝ, got ℝ[3]: Cannot unify scalar ℝ with tensor ℝ[3]'  # noqa: E501

    def test_return_type_mismatch(self):
        """Both branches return ℝ but declared return type is ℝ[3]"""
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=vec,
                            errors=errors)
        cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
        stmt = ('body_if_else_return', cond, ('var', 'x'), ('num', 0.0))
        stmt_body_if_else_return(stmt, ctx)
        assert len(errors) == 1
        assert errors[
            0] == 'if/else return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ'  # noqa: E501

    def test_branchs_types_cond_mismatch(self):
        """
        If branch returns ℝ[3], else returns ℝ.
        Condition also has type mismatch:
          TTensor!=T_REAL
        """
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('var', 'v'), ('num', 0.0))
        stmt = ('body_if_else_return', cond, ('var', 'v'), ('num', 0.0))
        stmt_body_if_else_return(stmt, ctx)

        assert len(errors) == 3
        assert errors[
            0] == "ℝ[3] is not comparable with ℝ at 'cond_gt' expression"
        assert errors[
            1] == "if/else branch type mismatch: then=ℝ[3], else=ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501
        assert errors[
            2] == 'if/else return type mismatch: declared ℝ, got ℝ[3]: Cannot unify scalar ℝ with tensor ℝ[3]'  # noqa: E501


class TestStmtBodyIfElse:
    """
    Test body-if-else statement type checking.
    """

    def test_basic_branches(self):
        """
        Correct typed then/else branches does not produce errors.
        """
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ("cond_gt", ("var", "x"), ("num", 0.0))
        then_stmts = [("body_assign", "y", ("var", "x"))]
        else_stmts = [("body_assign", "y", ("num", 1.0))]
        stmt_body_if_else(("body_if_else", cond, then_stmts, else_stmts), ctx)
        assert errors == []

    def test_different_branch_type(self):
        """
        Branches assigning different types produce no error here.

        Then and else branches are inferred independently.
        """
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ("cond_gt", ("num", 1.0), ("num", 0.0))

        then_stmts = [("body_assign", "y", ("var", "v"))]  # ℝ[3]
        else_stmts = [("body_assign", "y", ("num", 0.0))]  # ℝ
        stmt_body_if_else(("body_if_else", cond, then_stmts, else_stmts), ctx)
        assert errors == []

    def test_return_type_mismatch(self):
        """return_type mismatch with branch body types is not a type fail
        since these are not the return values. variable assingments can be
        used to perform more operations and then return."""
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=vec,
                            errors=errors)
        cond = ("cond_gt", ("var", "x"), ("num", 0.0))
        then_stmts = [("body_assign", "y", ("var", "x"))]
        else_stmts = [("body_assign", "y", ("num", 0.0))]
        stmt_body_if_else(("body_if_else", cond, then_stmts, else_stmts), ctx)

        assert errors == []

    def test_error_in_branches(self):
        """
        A body_decl type mismatch inside a branch is caught.
        """
        # Case then branch error
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ("cond_gt", ("var", "x"), ("num", 0.0))

        # Declaring wrong type v : ℝ[3] = 2.0 (type error)
        bad_decl = (
            "body_decl",
            "v",
            ("tensor", [(3, 'invariant')]),  # ℝ
            ("num", 2.0))  # ℝ
        stmt_body_if_else(("body_if_else", cond, [bad_decl], []), ctx)
        assert len(errors) == 1
        print(errors)
        assert errors[
            0] == "In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # Case else branch error
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ("cond_gt", ("var", "x"), ("num", 0.0))

        stmt_body_if_else(("body_if_else", cond, [], [bad_decl]), ctx)
        assert len(errors) == 1
        assert errors[
            0] == "In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

    def test_body_if_only(self):
        """
        body_if runs only the then branch without error.
        """
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ("cond_gt", ("var", "x"), ("num", 0.0))
        then_stmts = [("body_assign", "y", ("var", "x"))]
        # body_if has no else_stmts argument
        stmt_body_if_else(("body_if", cond, then_stmts), ctx)
        assert errors == []


class TestStmtBodyFor:
    """
    Test body_for statement type inference.
    """

    def test_body_for_assigns(self):
        errors = []
        arr_type = TTensor(((4, 'invariant'), ))
        ctx = make_stmt_ctx(env={'arr': arr_type},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        # equivalent of doing:
        # for i:
        #   total = arr[i]
        body = [('loop_assign', 'total', ('index', 'arr', ('var', 'i')))]
        stmt_body_for(('body_for', 'i', body, ['arr']), ctx)
        assert ctx.env.get('total') == T_REAL  # arr[i] : ℝ
        assert ctx.env['i'] == T_NAT
        assert errors == []

    def test_multiple_body_stmts(self):
        """
        All body statements are type-checked and hoisted to env.
        """
        errors = []
        arr_type = TTensor(((4, 'invariant'), ))
        ctx = make_stmt_ctx(env={'arr': arr_type}, errors=errors)
        # for i:
        #   a = arr[i] # ℝ
        #   b = 2
        body = [
            ('loop_assign', 'a', ('index', 'arr', ('var',
                                                   'i'))),  # a = arr[i] : ℝ
            ('loop_assign', 'b', ('num', 2.0)),
        ]
        stmt_body_for(('body_for', 'i', body, ['arr']), ctx)
        assert ctx.env.get('a') == T_REAL
        assert ctx.env.get('b') == T_REAL
        assert errors == []

    def test_error_inside_for_body(self):
        """A type mismatch inside the body is reported via add_error."""
        errors = []
        ctx = make_stmt_ctx(func_name='f', errors=errors)
        # v : ℝ[3] = 1.0
        # inferred ℝ, declared ℝ[3]
        bad = ('body_decl', 'v', ('tensor', [(3, 'invariant')]), ('num', 1.0))
        stmt_body_for(('body_for', 'i', [bad], []), ctx)
        assert len(errors) == 1
        assert "In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ" in errors[  # noqa: E501
            0]  # noqa: E501


class TestStmtBodyForRange:
    """
    Test body_for_range statement type inference.
    """

    def test_body_stmts(self):
        """Body assignment updates env and loop var becomes T_NAT."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        stmt_body_for_range(
            ('body_for_range', 'i', ('num', 0),
             ('num', 10), [('loop_assign', 'acc', ('var', 'x'))]), ctx)
        assert ctx.env.get('acc') == T_REAL
        assert ctx.env['i'] == T_NAT
        assert errors == []

    def test_multiple_body_stmts(self):
        """body statements adds bindings to env."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        body = [
            ('loop_assign', 'p', ('var', 'x')),
            ('loop_assign', 'q', ('num', 0.0)),
        ]
        stmt_body_for_range(
            ('body_for_range', 'i', ('num', 0), ('num', 4), body), ctx)
        assert ctx.env['i'] == T_NAT
        assert ctx.env.get('p') == T_REAL
        assert ctx.env.get('q') == T_REAL
        assert errors == []

    def test_loop_var_in_body(self):
        """Body statements can reference the loop variable as T_NAT."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        body = [('loop_assign', 'step', ('var', 'i'))]
        stmt_body_for_range(
            ('body_for_range', 'i', ('num', 0), ('num', 10), body), ctx)
        assert ctx.env.get('step') == T_NAT
        assert errors == []

    def test_error_inside_body_is_propagated(self):
        """A type mismatch in the body is reported via add_error."""
        errors = []
        ctx = make_stmt_ctx(func_name='f', errors=errors)
        # v : ℝ[3] = 1.0 — inferred ℝ ≠ declared ℝ[3]
        bad = ('body_decl', 'v', ('tensor', [(3, 'invariant')]), ('num', 1.0))
        stmt_body_for_range(
            ('body_for_range', 'i', ('num', 0), ('num', 10), [bad]), ctx)
        assert len(errors) == 1
        assert "In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ" in errors[  # noqa: E501
            0]  # noqa: E501


class TestStmtBodyZerosDecl:
    """Test ``stmt_body_zeros_decl`` statement inference handler."""

    def test_zero_decl(self):
        """Declaring a ℝ[n] variable registers correct type in env."""
        # Case ℝ
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_zeros_decl(('body_zeros_decl', 'x', 'ℝ'), ctx)
        assert ctx.env['x'] == T_REAL
        assert errors == []

        # Case ℝ[4]
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_zeros_decl(
            ('body_zeros_decl', 'v', ('tensor', [(4, 'invariant')])), ctx)
        assert ctx.env['v'] == TTensor(((4, 'invariant'), ))

        # Case ℝ[3,3]
        """Declaring ℝ[3,3] registers a TTensor with two dimensions."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_zeros_decl(
            ('body_zeros_decl', 'C', ('tensor', [(3, 'invariant'),
                                                 (3, 'invariant')])), ctx)
        assert ctx.env['C'] == TTensor(((3, 'invariant'), (3, 'invariant')))

    def test_none_type_adds_tvar(self):
        """None type_spec stores a new TVar."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_zeros_decl(('body_zeros_decl', 'z', None), ctx)
        assert isinstance(ctx.env['z'], TVar)


class TestStmtBodyForAccum:
    """Test body_for_accum statement type inference."""

    def test_loop_vars_types(self):
        """loop variables are added to env as TDim types."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_body_for_accum(('body_for_accum', ['i', 'j', 'k'], []), ctx)
        # Each loop var should be a TDim
        for lv in ('i', 'j', 'k'):
            assert isinstance(ctx.env[lv], TDim)

    def test_body_stmts_for_accum(self):
        """Statements in the accumulation body are inferred."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        body = [('loop_assign', 'acc', ('var', 'x'))]
        stmt_body_for_accum(('body_for_accum', ['i'], body), ctx)
        assert ctx.env.get('acc') == T_REAL
        assert errors == []


class TestStmtForAssign:
    """Test loop_assign statement type inference."""

    def test_inferred_type_registered_in_env(self):
        """inferred type is stored under the assigned name."""
        # case ℝ
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        stmt_for_assign(('loop_assign', 'y', ('var', 'x')), ctx)
        assert ctx.env['y'] == T_REAL
        assert errors == []

        # case ℝ[3]
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec}, errors=errors)
        stmt_for_assign(('loop_assign', 'w', ('var', 'v')), ctx)
        assert ctx.env['w'] == vec

        # case infer numeric
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_for_assign(('loop_assign', 'z', ('num', 3.14)), ctx)
        assert ctx.env['z'] == T_REAL


class TestStmtForPluseq:
    """Test += accumulation statement type inference."""

    def test_basic_pluseq(self):
        """Basic for_pluseq infers rhs type."""
        errors = []
        ctx = make_stmt_ctx(env={'total': T_REAL, 'x': T_REAL}, errors=errors)
        stmt_for_pluseq(('for_pluseq', 'total', [], ('var', 'x')), ctx)
        assert errors == []

    def test_indexed_pluseq(self):
        """
        loop_index_pluseq unifies each loop var's TDim to the array dimension.
        """
        errors = []
        mat = TTensor(((3, 'invariant'), (4, 'invariant')))
        ctx = make_stmt_ctx(env={'C': mat}, errors=errors)
        # Simulate i, j registered by stmt_body_for_accum before this stmt
        i_dim = new_dim()
        j_dim = new_dim()
        ctx.env['i'] = i_dim
        ctx.env['j'] = j_dim
        stmt_for_pluseq(
            ('loop_index_pluseq', 'C', [('var', 'i'),
                                        ('var', 'j')], ('num', 1.0)), ctx)
        assert errors == []
        # env keeps the raw TDim objects unchanged
        assert ctx.env['i'] is i_dim
        assert ctx.env['j'] is j_dim
        # substitution resolves them to concrete dimensions
        assert ctx.s.apply(i_dim) == 3
        assert ctx.s.apply(j_dim) == 4


class TestStmtLoopIf:
    """Test loop_if (if-only inside a for body) type inference."""

    def test_valid_condition_and_body(self):
        """Valid condition and body produce no errors."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
        body = [('loop_assign', 'y', ('var', 'x'))]
        stmt_loop_if(('loop_if', cond, body), ctx)
        assert errors == []

    def test_body_env_updated(self):
        """
        Variable assigned inside the if body is visible in env.
        """
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
        body = [('loop_assign', 'result', ('var', 'x'))]
        stmt_loop_if(('loop_if', cond, body), ctx)
        assert ctx.env.get('result') == T_REAL

    def test_tensor_condition_reports_error(self):
        """Using a ℝ[3] tensor as a comparison operand reports an error."""
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        # Comparing a ℝ[3] tensor with a scalar
        # should report an error
        cond = ('cond_gt', ('var', 'v'), ('num', 0.0))
        stmt_loop_if(('loop_if', cond, []), ctx)
        assert len(errors) == 1
        assert "ℝ[3] is not comparable with ℝ at 'cond_gt' expression" == errors[  # noqa: E501
            0]


class TestStmtLoopIfElse:
    """Test loop_if_else type inference."""

    def test_valid_branches(self):
        """Valid condition and both branches produce no errors."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
        then_body = [('loop_assign', 'positive', ('var', 'x'))]
        else_body = [('loop_assign', 'zero', ('num', 0.0))]
        stmt_loop_if_else(('loop_if_else', cond, then_body, else_body), ctx)
        assert errors == []

        # test env updates
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        cond = ('cond_gt', ('var', 'x'), ('num', 0.0))
        then_body = [('loop_assign', 'a', ('var', 'x'))]
        else_body = [('loop_assign', 'b', ('num', 0.0))]
        stmt_loop_if_else(('loop_if_else', cond, then_body, else_body), ctx)
        assert ctx.env.get('a') == T_REAL
        assert ctx.env.get('b') == T_REAL

    def test_tensor_condition_reports_error(self):
        """ℝ[3] tensor as condition operand reports an error."""
        errors = []
        vec = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'v': vec},
                            func_name='f',
                            return_type=T_REAL,
                            errors=errors)
        cond = ('cond_gt', ('var', 'v'), ('num', 0.0))
        stmt_loop_if_else(('loop_if_else', cond, [], []), ctx)
        assert len(errors) == 1
        assert "ℝ[3] is not comparable with ℝ at 'cond_gt' expression" == errors[  # noqa: E501
            0]


class TestInferStmts:
    """Test the main dispatcher ``infer_stmts``."""

    def test_dispatches_body_decl(self):
        """infer_stmts dispatches body_decl and updates env."""
        errors = []
        env, _ = infer_stmts([('body_decl', 'x', 'ℝ', ('num', 1.0))], {},
                             Substitution(), {}, {}, errors.append)
        assert env['x'] == T_REAL
        assert errors == []

    def test_dispatches_body_assign(self):
        """infer_stmts dispatches body_assign and updates env."""
        errors = []
        env, _ = infer_stmts([('body_assign', 'x', ('num', 2.0))], {},
                             Substitution(), {}, {}, errors.append)
        assert env['x'] == T_REAL

    def test_dispatches_loop_assign(self):
        """infer_stmts dispatches loop_assign."""
        errors = []
        env, _ = infer_stmts([('loop_assign', 'y', ('num', 5.0))], {},
                             Substitution(), {}, {}, errors.append)
        assert env['y'] == T_REAL

    def test_dispatches_body_for(self):
        """infer_stmts dispatches body_for"""
        errors = []
        env, _ = infer_stmts([('body_for', 'k', [], [])], {}, Substitution(),
                             {}, {}, errors.append)
        assert env['k'] == T_NAT

    def test_dispatches_body_zeros_decl(self):
        """infer_stmts dispatches body_zeros_decl."""
        errors = []
        env, _ = infer_stmts([('body_zeros_decl', 'C',
                               ('tensor', [(2, 'invariant'),
                                           (2, 'invariant')]))], {},
                             Substitution(), {}, {}, errors.append)
        assert env['C'] == TTensor(((2, 'invariant'), (2, 'invariant')))

    def test_multiple_stmts_body_assign(self):
        """Statements types are inferred and added to env."""
        errors = []
        env, _ = infer_stmts([('body_assign', 'a', ('num', 1.0)),
                              ('body_assign', 'b', ('num', 2.0)),
                              ('body_assign', 'c', ('add', ('var', 'a'),
                                                    ('var', 'b')))], {},
                             Substitution(), {}, {}, errors.append)
        assert env['a'] == T_REAL
        assert env['b'] == T_REAL
        assert env['c'] == T_REAL
        assert errors == []


class TestStmtDecl:
    """Tests for ``stmt_decl`` infer types"""

    def test_scalar(self):
        """Declared ℝ matched by a numeric literal"""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_decl(('decl', 'x', 'ℝ', ('num', 1.0)), ctx)
        assert ctx.env['x'] == T_REAL
        assert errors == []

    def test_tensor(self):
        """Declared ℝ[3] matched infer tensor types"""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_decl(
            ('decl', 'v', ('tensor', [(3, 'invariant')]),
             ('array', [('num', 1.0), ('num', 2.0), ('num', 3.0)])),
            ctx,
        )
        assert ctx.env['v'] == TTensor(((3, 'invariant'), ))
        assert errors == []

    def test_type_mismatch(self):
        """Declared ℝ[3] but inferred ℝ report error"""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_decl(
            ('decl', 't', ('tensor', [(3, 'invariant')]), ('num', 0.0)),
            ctx,
        )
        print(errors)
        assert len(errors) == 1
        assert "Type mismatch for 't': declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ" == errors[  # noqa: E501
            0]

        # on a type mismatch the variable is still added to env
        assert 't' in ctx.env


class TestStmtAssign:
    """Tests for ``stmt_assign``"""

    def test_scala(self):
        """Assigning a numeric stores T_REAL in env."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_assign(('assign', 'x', ('num', 3.0)), ctx)
        assert ctx.env['x'] == T_REAL
        assert errors == []

    def test_tensor_rhs(self):
        """Assigning an array stores the tensor type in env."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_assign(
            ('assign', 'v', ('array', [('num', 1.0), ('num', 2.0),
                                       ('num', 3.0)])),
            ctx,
        )
        assert ctx.env['v'] == TTensor(((3, 'invariant'), ))
        assert errors == []

    def test_expression_rhs(self):
        """The result type of an expression is inferred from operands."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        stmt_assign(
            ('assign', 'y', ('add', ('var', 'x'), ('num', 1.0))),
            ctx,
        )
        assert ctx.env['y'] == T_REAL
        assert errors == []

    def test_unknown_tvar(self):
        """An unknown variable stores a fresh TVar in env."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_assign(('assign', 'z', ('var', 'unknown')), ctx)
        assert 'z' in ctx.env
        assert isinstance(ctx.env['z'], TVar)


class TestStmtExpr:
    """Tests for ``stmt_expr``"""

    def test_valid_expression(self):
        """Basic correct expr node."""
        errors = []
        ctx = make_stmt_ctx(errors=errors)
        stmt_expr(('expr', ('add', ('num', 1.0), ('num', 2.0))), ctx)
        assert errors == []

    def test_env_not_modified(self):
        """expressions should never bind a to a variable."""
        errors = []
        ctx = make_stmt_ctx(env={'x': T_REAL}, errors=errors)
        stmt_expr(('expr', ('var', 'x')), ctx)
        assert list(ctx.env.keys()) == ['x']

    def test_shape_mismatch_caught(self):
        """A shape error inside the expression is reported as an error."""
        errors = []
        v2 = TTensor(((2, 'invariant'), ))
        v3 = TTensor(((3, 'invariant'), ))
        ctx = make_stmt_ctx(env={'a': v2, 'b': v3}, errors=errors)
        stmt_expr(
            ('expr', ('add', ('var', 'a'), ('var', 'b'))),
            ctx,
        )

        assert len(errors) == 1
        assert errors[0] == 'Shape mismatch in add: ℝ[2] vs ℝ[3]'

    def test_function_call_expression(self):
        """A function call used as a statement is type checked."""
        errors = []
        ctx = make_stmt_ctx(
            env={'x': T_REAL},
            func_env={'f': ([T_REAL], T_REAL)},
            errors=errors,
        )
        stmt_expr(('expr', ('call', 'f', [('var', 'x')])), ctx)
        assert errors == []
