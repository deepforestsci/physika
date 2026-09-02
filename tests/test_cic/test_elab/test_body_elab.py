from physika.core.elab.elab import Elab
from physika.core.environment import Environment
from physika.core.expr import App, Const, FVar, Lit, MVar, Proj
from physika.core.elab.dim_typespec import _REAL_CONST
from physika.utils.cic_utils.expr_utils import get_app_fn_args
from physika.core.elab.body_elab import (
    error_type_str,
    elaborate_expr,
    elaborate_stmts_with_return,
    elab_body_decl,
    elab_body_assign,
    elab_body_for,
    elab_body_if,
    elab_body_if_return,
    elab_body_if_else_return,
    elab_expr_var,
    elab_expr_num,
    elab_expr_imaginary,
    elab_expr_for,
    elab_expr_array,
    elab_expr_index,
    elab_expr_indexn,
    elab_expr_call,
    elab_expr_field_access,
    elab_expr_method_call,
    elab_expr_binop,
    elab_expr_matmul,
    elab_expr_pow,
    elab_expr_neg,
    elab_expr_if_else_return,
)


class TestErrorTypeStr:
    """
    Tests for ``error_type_str``.
    """

    def test_real_and_vec_types(self):
        """
        Real prints as ``ℝ``, ``Vec Real n`` prints as ``ℝ[n]``.
        """
        elab = Elab(Environment())
        assert error_type_str(_REAL_CONST, elab) == "ℝ"
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        assert error_type_str(vec_ty, elab) == "ℝ[3]"


class TestElaborateExpr:
    """
    Tests for ``elaborate_expr``.
    """

    def test_dispatches_by_tag(self):
        """
        Dispatches the handler registered for a node's own tag.
        """
        elab = Elab(Environment())
        assert elaborate_expr(("num", 1.0), {}, elab, []) == Lit(1.0)
        result = elaborate_expr(("add", ("num", 1.0), ("num", 2.0)), {}, elab,
                                [])
        assert result == App(App(Const("Real.add", ()), Lit(1.0)), Lit(2.0))

    def test_unknown_tag_degrades_to_mvar(self):
        """
        A tag with no registered handler degrades to a new MVar.
        """
        elab = Elab(Environment())
        assert isinstance(elaborate_expr(("weird", ), {}, elab, []), MVar)

    def test_dispatches_var_and_imaginary(self):
        """
        ``var``/``imaginary`` tags resolve to their bound local, or a
        new MVar when unbound.
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        assert elaborate_expr(("var", "x"), {"x": x_fv}, elab, []) is x_fv
        assert isinstance(elaborate_expr(("imaginary", ), {}, elab, []), MVar)

    def test_dispatches_arithmetic_tags(self):
        """
        ``sub``/``mul``/``div`` route through ``elab_expr_binop`` to their
        own ``Real.*`` axiom, same as ``add`` already covered above.
        """
        elab = Elab(Environment())
        ops = {"sub": "Real.sub", "mul": "Real.mul", "div": "Real.div"}
        for tag, const_name in ops.items():
            node = (tag, ("num", 1.0), ("num", 2.0))
            result = elaborate_expr(node, {}, elab, [])
            assert result == App(App(Const(const_name, ()), Lit(1.0)),
                                 Lit(2.0))

    def test_dispatches_pow_and_neg(self):
        """
        ``pow``/``neg`` route to ``Real.pow``/``Real.neg``.
        """
        elab = Elab(Environment())
        node = ("pow", ("num", 2.0), ("num", 3))
        head, _ = get_app_fn_args(elaborate_expr(node, {}, elab, []))
        assert head == Const("Real.pow", ())
        result = elaborate_expr(("neg", ("num", 1.0)), {}, elab, [])
        assert result == App(Const("Real.neg", ()), Lit(1.0))

    def test_dispatches_for_expr_and_array(self):
        """
        ``for_expr``/``array`` route to their own ``Vec.tabulate``/
        ``Vec.cons`` construction.
        """
        elab = Elab(Environment())
        node = ("for_expr", "j", ("num", 2), ("num", 1.0))
        head, _ = get_app_fn_args(elaborate_expr(node, {}, elab, []))
        assert head == Const("Vec.tabulate", ())
        node = ("array", [("num", 1.0), ("num", 2.0)])
        head, _ = get_app_fn_args(elaborate_expr(node, {}, elab, []))
        assert head == Const("Vec.cons", ())

    def test_dispatches_index_and_indexn(self):
        """
        ``index``/``indexN`` both route to ``Vec.get`` construction.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        elab, i_fv = elab.with_local("i", App(Const("Fin", ()), Lit(3)))
        env = {"arr": arr_fv, "i": i_fv}
        node = ("index", "arr", ("var", "i"))
        head, _ = get_app_fn_args(elaborate_expr(node, env, elab, []))
        assert head == Const("Vec.get", ())
        idx = [("index_item", ("var", "i"))]
        node = ("indexN", "arr", idx)
        head, _ = get_app_fn_args(elaborate_expr(node, env, elab, []))
        assert head == Const("Vec.get", ())

    def test_dispatches_call_field_access_and_method_call(self):
        """
        ``call`` resolves a builtin (``len``). An unresolvable
        ``field_access``/``method_call`` degrades to a new MVar.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        node = ("call", "len", [("var", "arr")])
        assert elaborate_expr(node, {"arr": arr_fv}, elab, []) == Lit(3)
        node = ("field_access", ("var", "obj"), "x")
        assert isinstance(elaborate_expr(node, {}, elab, []), MVar)
        node = ("method_call", ("var", "obj"), "m", [])
        assert isinstance(elaborate_expr(node, {}, elab, []), MVar)

    def test_dispatches_matmul_and_if_else_return(self):
        """
        ``matmul`` builds ``Mat.matmul`` for a well-shaped pair.
        ``body_if_else_return`` builds a ``Real.ite`` conditional value.
        """
        elab = Elab(Environment())
        row_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        mat_ty = App(App(Const("Vec", ()), row_ty), Lit(2))
        elab, m_fv = elab.with_local("m", mat_ty)
        node = ("matmul", ("var", "m"), ("var", "m"))
        assert isinstance(elaborate_expr(node, {"m": m_fv}, elab, []),
                          MVar) is False
        cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
        node = ("body_if_else_return", cond, ("num", 1.0), ("num", 2.0))
        head, _ = get_app_fn_args(elaborate_expr(node, {}, elab, []))
        assert head == Const("Real.ite", ())


class TestElaborateStmtsWithReturn:
    """
    Tests for ``elaborate_stmts_with_return``.
    """

    def test_declares_local_then_elaborates_return(self):
        """
        A declard local is bound and referenced from the return node.
        """
        elab = Elab(Environment())
        stmts = [("body_decl", "y", "ℝ", ("num", 1.0))]
        body_cic, new_elab, cur_env, local_decls, fvar_names = (
            elaborate_stmts_with_return(stmts, ("var", "y"), {}, elab, [],
                                        "In function 'f'"))
        assert isinstance(body_cic, FVar)
        assert local_decls == [("y", Lit(1.0))]
        assert list(cur_env) == ["y"]


class TestElabBodyDecl:
    """
    Tests for ``elab_body_decl``.
    """

    def test_declares_local(self):
        """
        A ``x: ℝ = 5.0`` decl binds ``x``.
        """
        elab = Elab(Environment())
        stmt = ("body_decl", "x", "ℝ", ("num", 5.0))
        env, elab, _, _, ctrl = elab_body_decl(stmt, 0, [stmt], None, None, {},
                                               elab, [], {}, [], "f")
        assert ctrl is None
        assert list(env) == ["x"]


class TestElabBodyAssign:
    """
    Tests for ``elab_body_assign``.
    """

    def test_body_assign_rebinds_to_a_new_fvar(self):
        """
        Reassigning a local rebinds it to a new FVar (same value, new id).
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        stmt = ("body_assign", "x", ("num", 2.0))
        env, elab, _, _, ctrl = elab_body_assign(stmt, 0, [stmt], None, None,
                                                 {"x": x_fv}, elab, [], {}, [],
                                                 "f")
        assert env["x"] is not x_fv
        assert ctrl is None


class TestElabBodyFor:
    """
    Tests for ``elab_body_for``.
    """

    def test_fold_loop_rebinds_accumulator(self):
        """
        A single reassignment loop body is elaborated as a fold, rebinding
        the accumulator local to the fold result.
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        stmt = ("body_for", "k", [("loop_assign", "x", ("num", 1.0))], ["arr"])
        env, elab, _, _, ctrl = elab_body_for(stmt, 0, [stmt], None, None, {
            "x": x_fv,
            "arr": arr_fv
        }, elab, [], {}, [], "f")
        assert env["x"] is not x_fv
        assert ctrl is None


class TestElabBodyIf:
    """
    Tests for ``elab_body_if``.
    """

    def test_if_rebinds_reassigned_local(self):
        """
        A local reassigned only in the "then" branch is rebound to an
        ``ite``-merged value.
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
        stmt = ("body_if", cond, [("body_assign", "x", ("num", 9.0))])
        env, elab, _, _, ctrl = elab_body_if(stmt, 0, [stmt], None, None,
                                             {"x": x_fv}, elab, [], {}, [],
                                             "f")
        assert env["x"] is not x_fv
        assert ctrl is None


class TestElabBodyIfReturn:
    """
    Tests for ``elab_body_if_return``.
    """

    def test_returns_ite_merged(self):
        """
        ``if cond: return then`` merges with return via a
        ``("return", ...)``.
        """
        elab = Elab(Environment())
        cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
        stmt = ("body_if_return", cond, ("num", 1.0))
        _, elab, _, _, ctrl = elab_body_if_return(stmt, 0, [stmt],
                                                  ("num", 2.0), None, {}, elab,
                                                  [], {}, [], "f")
        assert ctrl[0] == "return"
        merged_cic = ctrl[1][0]
        head, _ = get_app_fn_args(merged_cic)
        assert head == Const("Real.ite", ())


class TestElabBodyIfElseReturn:
    """
    Tests for ``elab_body_if_else_return``.
    """

    def test_breaks_when_no_terminal_node_yet(self):
        """
        With no terminal node seen yet, signals a ``break`` so the caller
        treats this statement as the function's terminal expression.
        """
        elab = Elab(Environment())
        cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
        stmt = ("body_if_else_return", cond, ("num", 1.0), ("num", 2.0))
        result = elab_body_if_else_return(stmt, 0, [stmt], None, None, {},
                                          elab, [], {}, [], "f")
        assert result[4][0] == "break"

    def test_no_op_when_terminal_node_already_present(self):
        """
        A terminal node already found upstream makes this a no-op.
        """
        elab = Elab(Environment())
        stmt = ("body_if_else_return", ("cond", ), ("num", 1.0), ("num", 2.0))
        result = elab_body_if_else_return(stmt, 0, [stmt], None, ("num", 0.0),
                                          {}, elab, [], {}, [], "f")
        assert result[4] is None


class TestElabExprVar:
    """
    Tests for ``elab_expr_var``.
    """

    def test_resolves_local_by_name(self):
        """
        Resolves a ``var`` node to its bound value in ``fvar_env``.
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        assert elab_expr_var(("var", "x"), {"x": x_fv}, elab, [], None) is x_fv


class TestElabExprNum:
    """
    Tests for ``elab_expr_num``.
    """

    def test_float_literal_returns__lit(self):
        """
        A float literal returns ``Lit``.
        """
        elab = Elab(Environment())
        assert elab_expr_num(("num", 3.0), {}, elab, [], None) == Lit(3.0)

    def test_int_literal_defaults_to_nat_ofnat(self):
        """
        An int literal with no expected Real type uses ``instOfNatNat``.
        """
        elab = Elab(Environment())
        result = elab_expr_num(("num", 3), {}, elab, [], None)
        assert result == Proj("OfNat", 0, App(Const("instOfNatNat", ()),
                                              Lit(3)))


class TestElabExprImaginary:
    """
    Tests for ``elab_expr_imaginary``.
    """

    def test_resolves_to_bound_i(self):
        """
        Resolves to ``i`` in scope, else degrades to a new MVar.
        """
        elab = Elab(Environment())
        elab, i_fv = elab.with_local("i", _REAL_CONST)
        result = elab_expr_imaginary(("imaginary", ), {"i": i_fv}, elab, [],
                                     None)
        assert result is i_fv
        assert isinstance(
            elab_expr_imaginary(("imaginary", ), {}, elab, [], None), MVar)


class TestElabExprFor:
    """
    Tests for ``elab_expr_for``.
    """

    def test_builds_vec_tabulate(self):
        """
        ``for i : ℕ(n) -> body`` elaborates to ``Vec.tabulate``.
        """
        elab = Elab(Environment())
        node = ("for_expr", "j", ("num", 2), ("num", 1.0))
        result = elab_expr_for(node, {}, elab, [], None)
        head, _ = get_app_fn_args(result)
        assert head == Const("Vec.tabulate", ())


class TestElabExprArray:
    """
    Tests for ``elab_expr_array``.
    """

    def test_builds_vec_cons_chain(self):
        """
        Non-empty elements build a concrete ``Vec.cons`` chain, not a MVar.
        """
        elab = Elab(Environment())
        node = ("array", [("num", 1.0), ("num", 2.0)])
        result = elab_expr_array(node, {}, elab, [], None)
        head, _ = get_app_fn_args(result)
        assert head == Const("Vec.cons", ())

    def test_empty_array_degrades_to_mvar(self):
        """
        An empty ``array`` node has no element type to infer, so it
        degrades to a new MVar.
        """
        elab = Elab(Environment())
        assert isinstance(elab_expr_array(("array", []), {}, elab, [], None),
                          MVar)


class TestElabExprIndex:
    """
    Tests for ``elab_expr_index``.
    """

    def test_builds_vec_get(self):
        """
        ``arr[i]`` builds ``Vec.get`` when both shape and index resolve.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        elab, i_fv = elab.with_local("i", App(Const("Fin", ()), Lit(3)))
        env = {"arr": arr_fv, "i": i_fv}
        node = ("index", "arr", ("var", "i"))
        head, _ = get_app_fn_args(elab_expr_index(node, env, elab, [], None))
        assert head == Const("Vec.get", ())


class TestElabExprIndexn:
    """
    Tests for ``elab_expr_indexn``.
    """

    def test_degrades_to_mvar_without_vec_get_registered(self):
        """
        An Environment (no ``Vec.get`` registered) can't infer the
        first ``Vec.get`` application's own type to build the second one,
        so a 2D index degrades to a MVar.
        """
        elab = Elab(Environment())
        row_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        mat_ty = App(App(Const("Vec", ()), row_ty), Lit(2))
        elab, m_fv = elab.with_local("m", mat_ty)
        elab, i_fv = elab.with_local("i", App(Const("Fin", ()), Lit(2)))
        elab, j_fv = elab.with_local("j", App(Const("Fin", ()), Lit(3)))
        env = {"m": m_fv, "i": i_fv, "j": j_fv}
        idx = [("index_item", ("var", "i")), ("index_item", ("var", "j"))]
        result = elab_expr_indexn(("indexN", "m", idx), env, elab, [], None)
        assert isinstance(result, MVar)


class TestElabExprCall:
    """
    Tests for ``elab_expr_call``.
    """

    def test_len_vec_length(self):
        """
        ``len(arr)`` reads the array's own registered ``Vec`` length,
        with no constant registration needed.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        node = ("call", "len", [("var", "arr")])
        assert elab_expr_call(node, {"arr": arr_fv}, elab, [], None) == Lit(3)


class TestElabExprFieldAccess:
    """
    Tests for ``elab_expr_field_access``.
    """

    def test_invalid_object_degrades_to_mvar(self):
        """
        A field access on an unresolvable object degrades to a new MVar.
        """
        elab = Elab(Environment())
        node = ("field_access", ("var", "obj"), "x")
        assert isinstance(elab_expr_field_access(node, {}, elab, [], None),
                          MVar)


class TestElabExprMethodCall:
    """
    Tests for ``elab_expr_method_call``.
    """

    def test_unregistered_method_degrades_to_mvar(self):
        """
        A method call whose class/method isn't registered degrades to a
        new MVar.
        """
        elab = Elab(Environment())
        node = ("method_call", ("var", "obj"), "m", [])
        assert isinstance(elab_expr_method_call(node, {}, elab, [], None),
                          MVar)


class TestElabExprBinop:
    """
    Tests for ``elab_expr_binop``.
    """

    def test_add_two_reals(self):
        """
        Elaborates both operands then dispatches to ``elaborate_binop``.
        """
        elab = Elab(Environment())
        node = ("add", ("num", 1.0), ("num", 2.0))
        result = elab_expr_binop(node, {}, elab, [], None)
        assert result == App(App(Const("Real.add", ()), Lit(1.0)), Lit(2.0))


class TestElabExprMatmul:
    """
    Tests for ``elab_expr_matmul``.
    """

    def test_builds_mat_matmul_for_square_matrix(self):
        """
        A square matrix times itself builds ``Mat.matmul``.
        """
        elab = Elab(Environment())
        row_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        mat_ty = App(App(Const("Vec", ()), row_ty), Lit(2))
        elab, m_fv = elab.with_local("m", mat_ty)
        env = {"m": m_fv}
        node = ("matmul", ("var", "m"), ("var", "m"))
        result = elab_expr_matmul(node, env, elab, [], None)
        assert isinstance(result, MVar) is False


class TestElabExprPow:
    """
    Tests for ``elab_expr_pow``.
    """

    def test_builds_real_pow(self):
        """
        Both base and exponent are coerced to Real before ``Real.pow``.
        """
        elab = Elab(Environment())
        node = ("pow", ("num", 2.0), ("num", 3))
        result = elab_expr_pow(node, {}, elab, [], None)
        head, _ = get_app_fn_args(result)
        assert head == Const("Real.pow", ())


class TestElabExprNeg:
    """
    Tests for ``elab_expr_neg``.
    """

    def test_builds_real_neg(self):
        """
        Negative valus use ``Real.neg``.
        """
        elab = Elab(Environment())
        result = elab_expr_neg(("neg", ("num", 1.0)), {}, elab, [], None)
        assert result == App(Const("Real.neg", ()), Lit(1.0))


class TestElabExprIfElseReturn:
    """
    Tests for ``elab_expr_if_else_return``.
    """

    def test_both_branches_return_builds_ite(self):
        """
        Both branches returning builds a ``Real.ite`` conditional value.
        """
        elab = Elab(Environment())
        cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
        node = ("body_if_else_return", cond, ("num", 1.0), ("num", 2.0))
        result = elab_expr_if_else_return(node, {}, elab, [], None)
        head, _ = get_app_fn_args(result)
        assert head == Const("Real.ite", ())
