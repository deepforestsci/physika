from physika.core.expr import (App, BVar, BinderInfo, Const, ForallE, FVar,
                               Lit, MVar, Proj)
from physika.core.elab.elab import Elab
from physika.core.environment import Environment, ConstantInfo
from physika.core.level import LSucc, LZero
from physika.utils.cic_utils.expr_utils import get_app_fn_args
from physika.utils.cic_utils.elab_utils import (
    arg_decreases,
    body_stmts_assigned_names,
    collect_calls_to,
    return_type_contains_more_elements,
    flatten_loop_assigns,
    loop_body_assigned_names,
    safe_infer_type,
    bind_local,
    bind_declared_local,
    invalidate_local,
    lit_to_real,
    vec_len,
    vec_elem_type,
    build_ite_term,
    mk_prod_chain,
    try_elaborate_dependent_fold_loop,
    coerce_to_fin,
    elaborate_binop,
    elaborate_condition_bool,
    elaborate_branch_values,
    merge_branch_values,
    elaborate_grad_call,
    elaborate_call,
    find_accum_loop_index,
    infer_reduction_bound,
    try_elaborate_fold_loop,
    try_elaborate_index_write_loop,
    invalidate_loop_locals,
    try_elaborate_sum_accumulator,
    try_elaborate_accumulator_body,
    safe_whnf,
    resolve_name,
    expected_elem_type_of,
    head_const_of,
    vec_shape_of,
)
from physika.core.elab.dim_typespec import (_NAT_CONST, _REAL_CONST,
                                            _VEC_CONST, _NAT_ADD)


def append_shape() -> ForallE:
    """
    Pi-type for append function:
        ``∀{m:Nat}, Vec Real m → Real → Vec Real (m+1)``
    """
    return ForallE(
        "m",
        _NAT_CONST,
        ForallE(
            "x",
            App(App(_VEC_CONST, _REAL_CONST), BVar(0)),
            ForallE(
                "v",
                _REAL_CONST,
                App(App(_VEC_CONST, _REAL_CONST),
                    App(App(_NAT_ADD, BVar(2)),
                        Lit(1))),  # type: ignore[arg-type]
                BinderInfo.DEFAULT,
            ),
            BinderInfo.DEFAULT,
        ),
        BinderInfo.IMPLICIT,
    )


class TestContainsMoreElements:
    """
    Tests for ``return_type_contains_more_elements``.
    """

    def test_return_type_contains_more_elements_than_initial_params(self):
        """
        Tests a Pi-type contains the right structure when doing an append
        operation, which mean the return type contains more elements than its
        params.
        """
        assert return_type_contains_more_elements(append_shape()) is True

        # non Pi type should be rejected
        assert return_type_contains_more_elements(_REAL_CONST) is False

        # shpould catch outer binder wrong type
        shape = ForallE("m", _NAT_CONST, _REAL_CONST, BinderInfo.DEFAULT)
        assert return_type_contains_more_elements(shape) is False

        # returns same shape in of params is wrong
        shape = ForallE(
            "m",
            _NAT_CONST,
            ForallE(
                "x",
                App(App(_VEC_CONST, _REAL_CONST), BVar(0)),
                ForallE(
                    "v",
                    _REAL_CONST,
                    App(App(_VEC_CONST, _REAL_CONST), BVar(2)),
                    BinderInfo.DEFAULT,
                ),
                BinderInfo.DEFAULT,
            ),
            BinderInfo.IMPLICIT,
        )
        assert return_type_contains_more_elements(shape) is False


class TestFlattenLoopandBodyAssignments:
    """
    Tests for ``flatten_loop_assigns`` and ``loop_body_assigned_names``.
    """

    def test_flatten_loop_assingments(self):
        """
        Checks a loop assingment is properly flattened
        """
        # single loop assignemt
        assert flatten_loop_assigns([("loop_assign", "x", ("num", 1.0))
                                     ]) == [[("x", ("num", 1.0))]]

        result = flatten_loop_assigns([
            ("loop_tuple_unpack", ["a", "b"], ("expr_list", [("var", "x"),
                                                             ("var", "y")])),
        ])
        assert result == [[("a", ("var", "x")), ("b", ("var", "y"))]]

        # checks loop features, like tuple unpack are flattened properly
        result = flatten_loop_assigns([
            ("loop_assign", "x", ("num", 1.0)),
            ("loop_tuple_unpack", ["a", "b"], ("expr_list", [("var", "x"),
                                                             ("var", "y")])),
        ])
        assert result == [
            [("x", ("num", 1.0))],
            [("a", ("var", "x")), ("b", ("var", "y"))],
        ]

    def test_body_loop_assigned_names(self):
        """
        Checks body loop assignemnt for variable name is handlerd properly
        """
        assert loop_body_assigned_names([("loop_assign", "total", ("num", 1.0))
                                         ]) == {"total"}

        # body var names should be catvhed
        names = loop_body_assigned_names([
            ("loop_pluseq", "total", ("num", 1.0)),
            ("loop_index_assign_nd", "arr", [], ("num", 2.0)),
            ("loop_index_pluseq", "acc", [], ("num", 3.0)),
        ])
        assert names == {"total", "arr", "acc"}

        # var anmes in if-else branches
        names = loop_body_assigned_names([
            ("loop_if_else", ("cond", ), [("loop_assign", "a", ("num", 1.0))],
             [("loop_pluseq", "b", ("num", 2.0))]),
            ("loop_for_range", "k", 0, 10, [("loop_index_assign_nd", "c", [],
                                             ("num", 3.0))]),
        ])
        assert names == {"a", "b", "c"}

    def test_empty_body_returns_empty_set(self):
        assert loop_body_assigned_names([]) == set()


class TestBodyStmtsAssignedNames:
    """
    Tests for ``body_stmts_assigned_names``.
    """

    def test_body_assign(self):
        """
        Test body assingments variable names
        """
        assert body_stmts_assigned_names([("body_assign", "x", ("num", 1.0))
                                          ]) == {"x"}

        # support body assignment calls
        # body_for_accum/body_for_map's 2nd field is the list of loop
        # *iteration* variables (e.g. ["i"]), not the assigned name — the
        # assigned name lives inside the loop body's loop_index_pluseq/
        # loop_index_assign_nd statements (see parser.py's
        # p_func_loop_body_stmt_for).
        stmts = [
            ("body_decl", "a", "ℝ", ("num", 0.0)),
            ("body_zeros_decl", "b", ("tensor", [])),
            ("body_index_assign", "c", [], ("num", 0.0)),
            ("body_index_assign_nd", "d", [], ("num", 0.0)),
            ("body_for_accum", ["i"], [("loop_index_pluseq", "e",
                                        [("var", "i")], ("num", 1.0))]),
            ("body_for_map", ["i"], [("loop_index_assign_nd", "f",
                                      [("var", "i")], ("num", 2.0))]),
        ]
        assert body_stmts_assigned_names(stmts) == {
            "a", "b", "c", "d", "e", "f"
        }

        # if else recursion
        names = body_stmts_assigned_names([
            ("body_if", ("cond", ), [("body_assign", "x", ("num", 1.0))]),
            ("body_if_else", ("cond", ), [("body_assign", "y", ("num", 1.0))],
             [("body_assign", "z", ("num", 2.0))]),
        ])
        assert names == {"x", "y", "z"}

        names = body_stmts_assigned_names([
            ("body_for", "i", [("loop_pluseq", "total", ("num", 1.0))], []),
            ("body_for_range", "j", 0, 10, [("loop_assign", "other", ("num",
                                                                      2.0))]),
        ])
        assert names == {"total", "other"}


class TestCollectCallsTo:
    """
    Tests for ``collect_calls_to``.
    """

    def test_call_in_body(self):
        """
        Chcks collect funciton calls in a AST node
        """
        # should return an empty list
        assert collect_calls_to({"f"}, ("num", 0.0), []) == []

        assert collect_calls_to({"f"}, ("call", "f", [("var", "x")]),
                                []) == [("f", [("var", "x")])]

        # nested call
        found = collect_calls_to({"f"}, ("add", ("call", "f", [("var", "x")]),
                                         ("num", 1.0)), [])
        assert found == [("f", [("var", "x")])]

        # call in statements not just in return
        found = collect_calls_to({"g"}, ("num", 0.0),
                                 [("call", "g", [("var", "y")])])
        assert found == [("g", [("var", "y")])]

        # multipple target names
        found = collect_calls_to(
            {"f", "g"},
            ("add", ("call", "f", [("var", "x")]), ("call", "h", [])),
            [("call", "g", [("var", "y")])],
        )
        assert found == [("f", [("var", "x")]), ("g", [("var", "y")])]


class TestArgDecreases:
    """
    Tests for ``arg_decreases``.
    """

    def test_param_decreases_rec_call(self):
        arg = ("sub", ("var", "n"), ("num", 1.0))
        assert arg_decreases(arg, "n", "ℕ") is True

        # should be decremented by 1 if Nats
        arg = ("sub", ("var", "n"), ("num", 2.0))
        assert arg_decreases(arg, "n", "ℕ") is False

        # Real type decrement should be true
        arg = ("sub", ("var", "n"), ("num", 5.0))
        assert arg_decreases(arg, "n", "ℝ") is True

        # decrementts using negative values should return Flase
        arg = ("sub", ("var", "n"), ("num", -1.0))
        assert arg_decreases(arg, "n", "ℕ") is False


class TestSafeInferType:
    """
    Tests for ``safe_infer_type``.
    """

    def test_infer_litera_type(self):
        """
        Infers Real for a float literal, and returns None on failure.
        """
        elab = Elab(Environment())
        assert safe_infer_type(elab, Lit(1.0)) == _REAL_CONST
        assert safe_infer_type(elab, Const("undefined", ())) is None


class TestBindLocalAndDeclaredLocal:
    """
    Tests for ``bind_local`` and ``bind_declared_local``.
    """

    def test_binds_fvar(self):
        """
        Binds var name to a new FVar, mutating local_decls/fvar_names.
        """
        elab = Elab(Environment())
        decls, names = [], {}
        env, elab = bind_local("x", Lit(1.0), _REAL_CONST, {}, elab, decls,
                               names)
        assert decls == [("x", Lit(1.0))]
        assert isinstance(env["x"], FVar)
        assert names[env["x"].id] == "x"

    def test_no_mismatch_reports_no_error(self):
        """
        No mismatch reported when rhs's inferred type unifies with decl_type.
        """
        elab = Elab(Environment())
        decls, names, errors = [], {}, []
        env, elab = bind_declared_local("x", _REAL_CONST, Lit(1.0), {}, elab,
                                        decls, names, errors, "f")
        assert errors == []
        assert list(env) == ["x"]

    def test_mismatch_reports_error(self):
        """
        A mismatch between decl_type and rhs's inferred type is reported.
        """
        elab = Elab(Environment())
        decls, names, errors = [], {}, []
        bind_declared_local("x", _NAT_CONST, Lit(1.0), {}, elab, decls, names,
                            errors, "f")
        assert len(errors) == 1
        assert errors[
            0] == "f: variable 'x' declared ℕ but assigned value of type ℝ"


class TestInvalidateLocal:
    """
    Tests for ``invalidate_local``.
    """

    def test_fvar_becomes_mvar(self):
        """
        A FVar is replaced by a MVar of the same type. A name
        not bound to a FVar is left unchanged.
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        assert isinstance(invalidate_local("x", {"x": x_fv}, elab)["x"], MVar)
        assert invalidate_local("y", {}, elab) == {}


class TestLitToReal:
    """
    Tests for ``lit_to_real``.
    """

    def test_int_literal_to_real(self):
        """
        Converts a Lit to Real only when expected_type is Real.
        """
        assert lit_to_real(Lit(0), _REAL_CONST) == Lit(0.0)
        assert lit_to_real(Lit(0), _NAT_CONST) == Lit(0)

    def test_ofnat_projection(self):
        """
        For Nat type converts ``OfNat``/``instOfNatNat`` projection's own
        instance.
        """
        ofnat_zero = Proj("OfNat", 0, App(Const("instOfNatNat", ()), Lit(0)))
        coerced = lit_to_real(ofnat_zero, _REAL_CONST)
        assert coerced == Proj("OfNat", 0,
                               App(Const("instOfNatReal", ()), Lit(0)))
        assert lit_to_real(ofnat_zero, _NAT_CONST) is ofnat_zero


class TestVecLenAndElemType:
    """
    Tests for ``vec_len`` and ``vec_elem_type``.
    """

    def test_reads_shape_off_vec_type(self):
        """
        Extracts length and element type of a ``Vec α n`` type, else None.
        """
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        assert vec_len(vec_ty) == Lit(3)
        assert vec_elem_type(vec_ty) == _REAL_CONST
        assert vec_len(_REAL_CONST) is None
        assert vec_elem_type(_REAL_CONST) is None


class TestVecShapeOf:
    """
    Tests for ``vec_shape_of``.
    """

    def test_returns_len_and_elem_type_pair(self):
        """
        ``vec_shape_of`` is ``(vec_len(t), vec_elem_type(t))``.
        """
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        assert vec_shape_of(vec_ty) == (Lit(3), _REAL_CONST)


class TestBuildIteTerm:
    """
    Tests for ``build_ite_term``.
    """

    def test_ite_variant(self):
        """
        Uses ``Real.ite``/``Nat.ite``/``Vec.ite`` depending ``then_type``.
        """
        cond = Const("true", ())
        result = build_ite_term(cond, Lit(1.0), Lit(0.0), _REAL_CONST)
        assert result == App(App(App(Const("Real.ite", ()), cond), Lit(1.0)),
                             Lit(0.0))

        result = build_ite_term(cond, Lit(1), Lit(0), _NAT_CONST)
        head, _ = get_app_fn_args(result)
        assert head == Const("Nat.ite", ())

        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        result = build_ite_term(cond, Lit(1.0), Lit(0.0), vec_ty)
        head, args = get_app_fn_args(result)
        assert head == Const("Vec.ite", ())
        assert args[0] == Lit(3)


class TestMkProdChain:
    """
    Tests for ``mk_prod_chain``.
    """

    def test_nests_prod_mk_over_values(self):
        """
        Builds a right associative ``Prod.mk`` nesting, None on empty input.
        """
        elab = Elab(Environment())
        result = mk_prod_chain([Lit(1.0), Lit(2.0)], elab)
        assert result == App(
            App(App(App(Const("Prod.mk", ()), _REAL_CONST), _REAL_CONST),
                Lit(1.0)), Lit(2.0))
        assert mk_prod_chain([], elab) is None


class TestTryElaborateDependentFoldLoop:
    """
    Tests for ``try_elaborate_dependent_fold_loop``.
    """

    def test_declines_when_loop_body_not_flattenable(self):
        """
        Returns None when the loop body isn't a group of reassignments.
        """
        elab = Elab(Environment())
        assert try_elaborate_dependent_fold_loop("k", [], Lit(3), {}, elab,
                                                 []) is None

    def test_builds_nat_rec_over_carried_local(self):
        """
        A local var reassigned each iteration builds a
        ``Nat.rec``-based fold, updating that local to the fold result.
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        loop_body = [("loop_assign", "x", ("var", "x"))]
        updates = try_elaborate_dependent_fold_loop("k", loop_body, Lit(3),
                                                    {"x": x_fv}, elab, [])
        assert list(updates) == ["x"]
        head, args = get_app_fn_args(updates["x"])
        assert head == Const("Nat.rec", (LSucc(LZero()), ))
        assert args[-1] == Lit(3)


class TestCoerceToFin:
    """
    Tests for ``coerce_to_fin``.
    """

    def test_non_fin_index_in_fin_ofnat(self):
        """
        A literal index with a bound should become a
        ``Fin.succ``/``Fin.zero`` chain.
        """

        elab = Elab(Environment())
        assert coerce_to_fin(Lit(0), Lit(3),
                             elab) == App(Const("Fin.zero", ()), Lit(2))
        assert coerce_to_fin(Lit(2), Lit(3), elab) == App(
            App(Const("Fin.succ", ()), Lit(2)),
            App(App(Const("Fin.succ", ()), Lit(1)),
                App(Const("Fin.zero", ()), Lit(0))))

        elab, i_fv = elab.with_local("i", App(Const("Fin", ()), Lit(3)))

        assert isinstance(coerce_to_fin(i_fv, Lit(3), elab), FVar)
        assert isinstance(i_fv, FVar)
        assert coerce_to_fin(i_fv, Lit(3), elab) is i_fv

    def test_out_of_range_index_error(self):
        """
        A literal index out of range shpuld produce an erro and
        degrades to an MVar.
        """
        elab = Elab(Environment())
        errors = []
        result = coerce_to_fin(Lit(5), Lit(3), elab, errors)
        assert isinstance(result, MVar)
        assert errors and "out of range" in errors[0]


class TestElaborateBinop:
    """
    Tests for ``elaborate_binop``.
    """

    def test_scalar_add_real_add(self):
        """
        Two Real operands uses ``Real.add``.
        """
        elab = Elab(Environment())
        result = elaborate_binop("add", Lit(1.0), Lit(2.0), elab)
        assert result == App(App(Const("Real.add", ()), Lit(1.0)), Lit(2.0))

    def test_vecs_use_vadd(self):
        """
        Same-length Vec add uses ``Vec.vadd``. Mismatch lengths use concat.
        """
        elab = Elab(Environment())
        vec2 = App(App(Const("Vec", ()), _REAL_CONST), Lit(2))
        vec3 = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, a_fv = elab.with_local("a", vec2)
        elab, b_fv = elab.with_local("b", vec2)
        elab, c_fv = elab.with_local("c", vec3)

        head, _ = get_app_fn_args(elaborate_binop("add", a_fv, b_fv, elab))
        assert head == Const("Vec.vadd", ())

        head, _ = get_app_fn_args(elaborate_binop("add", a_fv, c_fv, elab))
        assert head == Const("Vec.concat", ())

    def test_mvar_operand_degrades_to_mvar(self):
        """
        Any MVar operand makes the whole binop degrade to a new MVar.
        """
        elab = Elab(Environment())
        mv = elab.new_mvar("_x", _REAL_CONST)
        assert isinstance(elaborate_binop("add", mv, Lit(1.0), elab), MVar)


class TestElaborateConditionBool:
    """
    Tests for ``elaborate_condition_bool``.
    """

    def test_condition_prefix(self):
        """
        Uses ``Real`` or ``Nat`` comparison axiom depending on operand type.
        """
        elab = Elab(Environment())
        cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
        result = elaborate_condition_bool(cond, {}, elab, [])
        assert result == App(App(Const("Real.ltb", ()), Lit(1.0)), Lit(2.0))

    def test_unsupported_shape_degrades_to_mvar(self):
        """
        A node that isn't one of the six comparison tags degrades to MVar.
        """
        elab = Elab(Environment())
        assert isinstance(elaborate_condition_bool(("weird", ), {}, elab, []),
                          MVar)


class TestElaborateBranchValues:
    """
    Tests for ``elaborate_branch_values``.
    """

    def test_assign_overrides_base_env_value(self):
        """
        A ``body_assign`` overrides the name's value from ``base_env``.
        """
        elab = Elab(Environment())
        stmts = [("body_assign", "y", ("num", 5.0))]
        result = elaborate_branch_values(stmts, {"y": Lit(1.0)}, elab, [])
        assert result == {"y": Lit(5.0)}


class TestMergeBranchValues:
    """
    Tests for ``merge_branch_values``.
    """

    def test_if_else_builds_ite_term(self):
        """
        An if-else assigning both branches merges into a Real.ite term.
        """
        elab = Elab(Environment())
        cond = ("cond_lt", ("num", 1.0), ("num", 2.0))
        then_stmts = [("body_assign", "y", ("num", 1.0))]
        else_stmts = [("body_assign", "y", ("num", 2.0))]
        merged = merge_branch_values(cond, then_stmts, else_stmts,
                                     {"y": Lit(0.0)}, elab, [])
        assert list(merged) == ["y"]
        head, _ = get_app_fn_args(merged["y"])
        assert head == Const("Real.ite", ())


class TestElaborateGradCall:
    """
    Tests for ``elaborate_grad_call``.
    """

    def test_vec_args_uses_to_vec_grad(self):
        """
        Vec arguments should use ``Vec.grad``.
        """
        elab = Elab(Environment())
        vec_n = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        vec_m = App(App(Const("Vec", ()), _REAL_CONST), Lit(2))
        elab, x_fv = elab.with_local("x", vec_n)
        elab, y_fv = elab.with_local("y", vec_m)
        head, args = get_app_fn_args(
            elaborate_grad_call([x_fv, y_fv], elab, []))
        assert head == Const("Vec.grad", ())
        assert args[:2] == [Lit(3), Lit(2)]

    def test_wrong_param_numbers_use_elaborate_call(self):
        """
        Not exactly 2 args fallback to ``elaborate_call``.
        """
        elab = Elab(Environment())
        assert isinstance(elaborate_grad_call([Lit(1.0)], elab, []), MVar)


class TestElaborateCall:
    """
    Tests for ``elaborate_call``.
    """

    def test_explicit_implicit_args(self):
        """
        Explicit args are consumed in order. Unregistered call degrades to
        a MVar.
        """
        env = Environment()
        env.add_constant(
            ConstantInfo("double", (), ForallE("x", _REAL_CONST, _REAL_CONST),
                         None))
        elab = Elab(env)
        result = elaborate_call("double", [Lit(1.0)], elab, [])
        assert result == App(Const("double", ()), Lit(1.0))

        elab = Elab(Environment())
        assert isinstance(elaborate_call("missing", [], elab, []), MVar)


class TestFindAccumLoopIndex:
    """
    Tests for ``find_accum_loop_index``.
    """

    def test_for_loop_after_declarations(self):
        """
        Test fir finding the accumulating loop.
        """
        stmts = [
            ("body_decl", "total", "ℝ", ("num", 0.0)),
            ("body_decl", "temp", "ℝ", ("num", 0.0)),
            ("body_for", "i", [("loop_pluseq", "total", ("var", "i"))],
             ["arr"]),
        ]
        assert find_accum_loop_index(stmts, 1, "total") == 2
        assert find_accum_loop_index([("body_assign", "x", ("num", 1.0))], 0,
                                     "total") is None


class TestInferReductionBound:
    """
    Tests for ``infer_reduction_bound``.
    """

    def test_finds_bound_from_indexed_array(self):
        """
        Reads reduction bound off an array indexed by the reduction var.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, a_fv = elab.with_local("A", vec_ty)
        node = ("index", "A", ("var", "k"))
        assert infer_reduction_bound(node, "k", {"A": a_fv}, elab) == Lit(3)
        assert infer_reduction_bound(("num", 1.0), "k", {}, elab) is None


class TestTryElaborateFoldLoop:
    """
    Tests for ``try_elaborate_fold_loop``.
    """

    def test_builds_vec_foldl_over_indexed_array_length(self):
        """
        A single ``loop_assign`` builds ``Vec.foldl`` over an indexed
        array's own length, seeded at the accumulator's value after
        entering the loop.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(2))
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        elab, arr_fv = elab.with_local("arr", vec_ty)
        loop_body = [("loop_assign", "x", ("num", 1.0))]
        cur_env = {"x": x_fv, "arr": arr_fv}
        result = try_elaborate_fold_loop("k", loop_body, ["arr"], cur_env,
                                         elab, [])
        head, args = get_app_fn_args(result)
        assert head == Const("Vec.foldl", ())
        assert args[-1] is x_fv


class TestTryElaborateIndexWriteLoop:
    """
    Tests for ``try_elaborate_index_write_loop``.
    """

    def test_builds_vec_tabulate_for_index_write(self):
        """
        A single ``for i: arr[i] = expr`` builds using ``Vec.tabulate``.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        elab, dst_fv = elab.with_local("dst", vec_ty)
        loop_body = [("loop_index_assign_nd", "dst",
                      [("index_item", ("imaginary", ))], ("num", 2.0))]
        name, result = try_elaborate_index_write_loop("i", loop_body, {
            "arr": arr_fv,
            "dst": dst_fv
        }, elab, [])
        assert name == "dst"
        head, _ = get_app_fn_args(result)
        assert head == Const("Vec.tabulate", ())

    def test_rhs_declines(self):
        """
        When RHS reads the array being written should return None.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        loop_body = [("loop_index_assign_nd", "arr", [
            ("index_item", ("imaginary", ))
        ], ("index", "arr", ("imaginary", )))]
        assert try_elaborate_index_write_loop("i", loop_body, {"arr": arr_fv},
                                              elab, []) is None


class TestInvalidateLoopLocals:
    """
    Tests for ``invalidate_loop_locals``.
    """

    def test_reassigned_locals_become_mvars(self):
        """
        Locals reassigned by ``loop_body`` become fresh MVars.
        """
        elab = Elab(Environment())
        elab, total_fv = elab.with_local("total", _REAL_CONST)
        loop_body = [("loop_assign", "total", ("num", 1.0))]
        new_env = invalidate_loop_locals(loop_body, {"total": total_fv}, elab)
        assert isinstance(new_env["total"], MVar)

        new_env = invalidate_loop_locals(loop_body, {"total": total_fv},
                                         elab,
                                         exclude={"total"})
        assert new_env["total"] is total_fv


class TestTryElaborateSumAccumulator:
    """
    Tests for ``try_elaborate_sum_accumulator``.
    """

    def test_builds_vec_sum_of_tabulate(self):
        """
        ``result: ℝ`` followed by ``for i: result += expr`` builds
        ``Vec.sum n (Vec.tabulate n (fun i => expr))``.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(2))
        elab, arr_fv = elab.with_local("arr", vec_ty)
        next_stmt = ("body_for", "i", [("loop_pluseq", "total", ("num", 1.0))],
                     ["arr"])
        result = try_elaborate_sum_accumulator("total", "ℝ", next_stmt,
                                               {"arr": arr_fv}, elab, [])
        head, args = get_app_fn_args(result)
        assert head == Const("Vec.sum", ())
        assert args[0] == Lit(2)

    def test_non_real_type_spec_declines(self):
        """
        Only a ``ℝ`` type spec is recognized as a sum accumulator.
        """
        elab = Elab(Environment())
        assert try_elaborate_sum_accumulator("total", "ℕ", None, {}, elab,
                                             []) is None


class TestTryElaborateAccumulatorBody:
    """
    Tests for ``try_elaborate_accumulator_body``.
    """

    def test_pluseq_rhs_is_the_summand(self):
        """
        A ``result += expr`` summand is ``expr`` itself.
        """
        elab = Elab(Environment())
        loop_body = [("loop_pluseq", "total", ("num", 1.0))]
        result = try_elaborate_accumulator_body(loop_body, "total", {}, elab,
                                                [])
        assert result == Lit(1.0)

    def test_if_pluseq_summand_is_real_ite(self):
        """
        ``if cond: result += expr`` summand is ``Real.ite cond expr 0.0``.
        """
        elab = Elab(Environment())
        loop_body = [
            ("loop_if", ("cond_lt", ("num", 1.0), ("num", 2.0)),
             [("loop_pluseq", "total", ("num", 3.0))]),
        ]
        result = try_elaborate_accumulator_body(loop_body, "total", {}, elab,
                                                [])
        head, _ = get_app_fn_args(result)
        assert head == Const("Real.ite", ())


class TestSafeWhnf:
    """
    Tests for ``safe_whnf``.
    """

    def test_reduces_whnf(self):
        """
        Reduces a type to whnf, returns None on failure.
        """
        elab = Elab(Environment())
        assert safe_whnf(elab, _REAL_CONST) == _REAL_CONST

        # recursion exceeded returns None
        deep = Const("f", ())
        for _ in range(3000):
            deep = App(deep, Lit(1))
        assert safe_whnf(elab, deep) is None


class TestResolveName:
    """
    Tests for ``resolve_name``.
    """

    def test_local_constant_mvar_order(self):
        """
        A local in ``fvar_env`` is prioritized over a registered constant,
        which wins over the fresh-MVar fallback.
        """
        env = Environment()
        env.add_constant(ConstantInfo("pi", (), _REAL_CONST, None))
        elab = Elab(env)
        elab, x_fv = elab.with_local("x", _REAL_CONST)
        assert resolve_name("x", {"x": x_fv}, elab) is x_fv
        assert resolve_name("pi", {}, elab) == Const("pi", ())
        assert isinstance(resolve_name("undefined", {}, elab), MVar)


class TestExpectedElemTypeOf:
    """
    Tests for ``expected_elem_type_of``.
    """

    def test_peels_one_vec_layer(self):
        """
        Peels one Vec layer off a known expected type, None if unknown.
        """
        elab = Elab(Environment())
        vec_ty = App(App(Const("Vec", ()), _REAL_CONST), Lit(3))
        assert expected_elem_type_of(elab, vec_ty) == _REAL_CONST
        assert expected_elem_type_of(elab, None) is None


class TestHeadConstOf:
    """
    Tests for ``head_const_of``.
    """

    def test_returns_head_of_inferred_type(self):
        """
        Infers ``cic``'s type and returns its applied head Const.
        """
        elab = Elab(Environment())
        assert head_const_of(elab, Lit(1.0)) == _REAL_CONST
