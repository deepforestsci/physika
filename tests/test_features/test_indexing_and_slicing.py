from physika.features.indexing_and_slicing import IndexingandSlicing
from physika.utils.types import TTensor, T_REAL, TDim, Substitution, new_dim
from tests.test_utils.test_infer_stmt import make_stmt_ctx
from tests.test_utils.test_infer_expr import make_ctx
from physika.utils.infer_expr import infer_expr

rules = IndexingandSlicing().type_rules()
stmt_for_eq = rules["loop_index_assign_nd"]
stmt_for_pluseq = rules["for_pluseq"]
expr_index = rules["index"]
expr_indexN = rules["indexN"]


class TestIndexingandSlicing:
    """
    Checks parser is emitting the correct AST nodes,
    that the forward rules have the correct keys and type
    rules catches proper errors.
    """

    def test_elf_name(self):
        """
        ELF is registered under the name 'Indexing_and_Slicing'.
        """
        assert IndexingandSlicing.name == "Indexing_and_Slicing"

    def test_parser_rules(self):
        """
        parser_rules should return nineteen handlers.
        """
        rules = IndexingandSlicing().parser_rules()
        assert len(rules) == 22
        names = [r.__name__ for r in rules]
        assert "p_factor_index" in names
        assert "p_factor_index_var" in names
        assert "p_factor_indexN" in names
        assert "p_func_factor_index" in names
        assert "p_multi_index_item_index" in names
        assert "p_multi_index_item_slice" in names
        assert "p_multi_index_list_single" in names
        assert "p_multi_index_list_base" in names
        assert "p_multi_index_list_extend" in names
        assert "p_func_factor_indexN" in names
        assert "p_for_statement_index_assign_nd" in names
        assert "p_statement_index_assign" in names
        assert "p_statement_index_assign_nd" in names
        assert "p_loop_index_item_index" in names
        assert "p_loop_index_item_slice" in names
        assert "p_loop_index_list_single" in names
        assert "p_loop_index_list_base" in names
        assert "p_loop_index_list_extend" in names
        assert "p_func_loop_stmt_index_pluseq" in names
        assert "p_func_loop_stmt_index_assign_nd" in names
        assert "p_func_body_stmt_index_assign" in names
        assert "p_func_body_stmt_index_assign_nd" in names

    def test_forward_rules_keys(self):
        """
        forward rules should return nine handlers.
        """
        rules = IndexingandSlicing().forward_rules()
        assert len(rules) == 9
        assert set(rules.keys()) == {
            "index", "indexN", "index_assign", "index_assign_nd",
            "for_index_assign_nd", "loop_index_pluseq", "loop_index_assign_nd",
            "body_index_assign", "body_index_assign_nd"
        }
        assert all(callable(v) for v in rules.values())

    def test_type_rules_keys(self):
        """
        type rules should return six handlers.
        """
        rules = IndexingandSlicing().type_rules()
        assert len(rules) == 6
        assert set(rules.keys()) == {
            "index", "indexN", "loop_index_assign_nd", "body_for_map",
            "for_pluseq", "loop_index_pluseq"
        }
        assert all(callable(v) for v in rules.values())


class TestExprIndex:
    """
    Tests for ``expr_index``
    """

    def test_index(self):
        """Indexing a 1D vector produces a scalar ℝ."""
        ctx = make_ctx(env={"v": TTensor(((4, "invariant"), ))})
        t, _ = expr_index(("index", "v", ("num", 0)), ctx.env, ctx.s,
                          ctx.func_env, ctx.class_env, ctx.add_error,
                          infer_expr)
        assert t == T_REAL

        # Indexing the first dimension of a matrix produces a row vector
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_index(("index", "A", ("num", 0)), ctx.env, ctx.s,
                          ctx.func_env, ctx.class_env, ctx.add_error,
                          infer_expr)
        assert t == TTensor(((4, "invariant"), ))

        # Indexing a 3D tensor along dim-0 produces a 2D matrix
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_index(("index", "T", ("num", 0)), ctx.env, ctx.s,
                          ctx.func_env, ctx.class_env, ctx.add_error,
                          infer_expr)
        assert t == TTensor(((3, "invariant"), (4, "invariant")))

    def test_unknown_variable(self):
        """
        Unknown array name returns (None, ctx.s) without error, error is raised
        at compile time.
        """
        errors = []
        ctx = make_ctx(errors=errors)
        t, _ = expr_index(("index", "missing", ("num", 0)), ctx.env, ctx.s,
                          ctx.func_env, ctx.class_env, ctx.add_error,
                          infer_expr)
        assert t is None
        assert errors == []

    def test_scalar_indexed_reports_error(self):
        """
        Indexing a scalar variable reports an error.
        """
        errors = []
        ctx = make_ctx(env={"x": T_REAL}, errors=errors)
        expr_index(("index", "x", ("num", 0)), ctx.env, ctx.s, ctx.func_env,
                   ctx.class_env, ctx.add_error, infer_expr)
        assert len(errors) == 1
        assert errors[0] == "Cannot index scalar 'x'"

    def test_expression_index(self):
        """The index expression may itself be a variable (loop counter)."""
        ctx = make_ctx(env={"v": TTensor(((5, "invariant"), )), "i": T_REAL})
        t, _ = expr_index(("index", "v", ("var", "i")), ctx.env, ctx.s,
                          ctx.func_env, ctx.class_env, ctx.add_error,
                          infer_expr)
        assert t == T_REAL

    def test_substitution_returned(self):
        """Substitution is returned unchanged for simple cases."""
        s = Substitution()
        ctx = make_ctx(env={"v": TTensor(((3, "invariant"), ))}, s=s)
        _, s_out = expr_index(("index", "v", ("num", 0)), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, ctx.add_error,
                              infer_expr)
        assert isinstance(s_out, Substitution)

    def test_substitution_gains_binding(self):
        """
        Indexing with a TDim index variable unifies it against the
        leading dimension.

        ``v : ℝ[5]``, ``i : δ0``:

        ``expr_index`` calls ``unify_dim(δ0, 5, s)``
        which writes the binding ``δ0 -> 5`` into the returned substitution.
        """
        s = Substitution()
        ctx = make_ctx(
            env={
                "v": TTensor(((5, "invariant"), )),
                "i": TDim("δ0")
            },
            s=s,
        )
        _, s_out = expr_index(("index", "v", ("var", "i")), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, ctx.add_error,
                              infer_expr)
        assert s_out["δ0"] == 5


class TestExprIndexN:
    """
    Tests for ``expr_indexN``
    """

    def test_2d_index(self):
        """
        Indexing all dims of a matrix gives type ℝ.
        """
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [("index_item", ("num", 0)),
                                            ("index_item", ("num", 1))]),
                           ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                           ctx.add_error, infer_expr)
        assert t == T_REAL

    def test_3d_index(self):
        """
        Indexing the first two dims of a 3-D tensor gives ℝ[k].
        """
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_indexN(("indexN", "T", [("index_item", ("num", 0)),
                                            ("index_item", ("num", 1))]),
                           ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                           ctx.add_error, infer_expr)
        assert t == TTensor(((4, "invariant"), ))

        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_indexN(("indexN", "T", [("index_item", ("num", 0)),
                                            ("index_item", ("num", 1)),
                                            ("index_item", ("num", 2))]),
                           ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                           ctx.add_error, infer_expr)
        assert t == T_REAL

    def test_matrix_index(self):
        """One index on a matrix gives row vector."""
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [("index_item", ("num", 1))]),
                           ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                           ctx.add_error, infer_expr)
        assert t == TTensor(((4, "invariant"), ))

    def test_unknown_variable_returns_none(self):
        """
        Variable not in scope gives (None, s)
        """
        errors = []
        ctx = make_ctx(errors=errors)
        t, _ = expr_indexN(("indexN", "missing", [("num", 0), ("num", 1)]),
                           ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                           ctx.add_error, infer_expr)
        assert t is None
        assert errors == []

    def test_scalar_variable_returns_none(self):
        """
        Indexing a scalar gives None.
        """
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_indexN(("indexN", "x", [("num", 0), ("num", 1)]), ctx.env,
                           ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t is None

    def test_over_indexed(self):
        """More indices than dimensions gives None + error reported."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((3, "invariant"), ))}, errors=errors)
        t, _ = expr_indexN(("indexN", "v", [("index_item", ("num", 0)),
                                            ("index_item", ("num", 1)),
                                            ("index_item", ("num", 2))]),
                           ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                           ctx.add_error, infer_expr)
        assert t is None
        assert len(errors) == 1
        assert errors[0] == "Over-indexed 'v': 3 indices for a rank-1 tensor"
        assert "v" in errors[0]
        assert "3" in errors[0]

    def test_variable_indices(self):
        """
        Loop variables used as indices works properly.
        """
        ctx = make_ctx(
            env={
                "A": TTensor(((3, "invariant"), (4, "invariant"))),
                "i": T_REAL,
                "j": T_REAL
            })
        t, _ = expr_indexN(("indexN", "A", [("index_item", ("var", "i")),
                                            ("index_item", ("var", "j"))]),
                           ctx.env, ctx.s, ctx.func_env, ctx.class_env,
                           ctx.add_error, infer_expr)
        assert t == T_REAL

    def test_2d_full_slice(self):
        """
        Slicing all dimensions of matrix.
        A is shape of [3, 4]
        slicing for [:, :]
        """
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [
            ("slice_item", None, None),
            ("slice_item", None, None),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(((3, "invariant"), (4, "invariant")))

    def test_2d_slice_index(self):
        """
        Slicing one dimension and indexing another which returns vector.
        A is shape of [3, 4]
        slicing for [:, 1]
        """
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [
            ("slice_item", None, None),
            ("index_item", ("num", 1)),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(((3, "invariant"), ))

    def test_2d_index_slice(self):
        """
        Indexing one dimension and slicing another returns a vector.

        A is shape [3, 4]
        indexing for [1, :]
        """
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [
            ("index_item", ("num", 1)),
            ("slice_item", None, None),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(((4, "invariant"), ))

    def test_2d_partial_slice(self):
        """
        Slicing a range along one dimension.

        A is shape [3, 4]
        slicing for [1:3, :]
        """
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [
            ("slice_item", ("num", 1), ("num", 3)),
            ("slice_item", None, None),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(((2, "invariant"), (4, "invariant")))

    def test_2d_partial_slice_both_dims(self):
        """
        Slicing ranges on both dimensions.

        A is shape [3, 4]
        slicing for [1:3, 1:3]
        """
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [
            ("slice_item", ("num", 1), ("num", 3)),
            ("slice_item", ("num", 1), ("num", 3)),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(((2, "invariant"), (2, "invariant")))

    def test_3d_full_slice(self):
        """
        Slicing all dimensions of a 3-D tensor.

        T is shape [2, 3, 4]
        slicing for [:, :, :]
        """
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_indexN(("indexN", "T", [
            ("slice_item", None, None),
            ("slice_item", None, None),
            ("slice_item", None, None),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(
            ((2, "invariant"), (3, "invariant"), (4, "invariant")))

    def test_3d_slice_index_slice(self):
        """
        Slicing, indexing and slicing on a 3-D tensor.

        T is shape [2, 3, 4]
        slicing for [:, 1, :]
        """
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_indexN(("indexN", "T", [
            ("slice_item", None, None),
            ("index_item", ("num", 1)),
            ("slice_item", None, None),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(((2, "invariant"), (4, "invariant")))

    def test_3d_partial_slice(self):
        """
        Partial slicing on a 3-D tensor.

        T is shape [2, 3, 4]
        slicing for [0:2, 1:3, :]
        """
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_indexN(("indexN", "T", [
            ("slice_item", ("num", 0), ("num", 2)),
            ("slice_item", ("num", 1), ("num", 3)),
            ("slice_item", None, None),
        ]), ctx.env, ctx.s, ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == TTensor(
            ((2, "invariant"), (2, "invariant"), (4, "invariant")))


class TestStmtForEq:
    """Test index assignment ``=`` statement type inference."""

    def test_basic_eq(self):
        """Basic loop_index_assign_nd infers rhs type."""
        errors = []
        ctx = make_stmt_ctx(
            env={
                "results": T_REAL,
                "x": T_REAL,
            },
            errors=errors,
        )

        stmt_for_eq = IndexingandSlicing().type_rules()["loop_index_assign_nd"]

        _, ctx.s = stmt_for_eq(
            ("loop_index_assign_nd", "results", [], ("var", "x")),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []

    def test_indexed_eq(self):
        """
        loop_index_assign_nd unifies each loop var's TDim to the array
        dimension.
        """

        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"results": mat}, errors=errors)

        # Simulate i, j registered by stmt_body_for_accum before this stmt
        i_dim = new_dim()
        j_dim = new_dim()
        ctx.env["i"] = i_dim
        ctx.env["j"] = j_dim

        stmt_for_eq = IndexingandSlicing().type_rules()["loop_index_assign_nd"]

        _, ctx.s = stmt_for_eq(
            (
                "loop_index_assign_nd",
                "results",
                [
                    ("index_item", ("var", "i")),
                    ("index_item", ("var", "j")),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []
        # env keeps the raw TDim objects unchanged
        assert ctx.env["i"] is i_dim
        assert ctx.env["j"] is j_dim
        # substitution resolves them to concrete dimensions
        assert ctx.s.apply(i_dim) == 3
        assert ctx.s.apply(j_dim) == 4

    def test_slice_start_end_eq(self):
        """
        loop_index_assign_nd unifies slice start/end variables with the
        corresponding tensor dimension.
        """

        errors = []
        mat = TTensor(((5, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"results": mat}, errors=errors)

        start_dim = new_dim()
        end_dim = new_dim()

        ctx.env["start"] = start_dim
        ctx.env["end"] = end_dim

        stmt_for_eq = IndexingandSlicing().type_rules()["loop_index_assign_nd"]

        _, ctx.s = stmt_for_eq(
            (
                "loop_index_assign_nd",
                "results",
                [
                    ("slice_item", ("var", "start"), ("var", "end")),
                    ("index_item", ("num", 1)),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []
        assert ctx.s.apply(start_dim) == 5
        assert ctx.s.apply(end_dim) == 5

    def test_open_slice_eq(self):
        """
        Open-ended slices do not produce type errors.
        """
        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"results": mat}, errors=errors)

        stmt_for_eq = IndexingandSlicing().type_rules()["loop_index_assign_nd"]

        _, ctx.s = stmt_for_eq(
            (
                "loop_index_assign_nd",
                "results",
                [
                    ("slice_item", None, None),
                    ("index_item", ("num", 2)),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []

    def test_body_for_map_basic(self):
        """
        body_for_map infers rhs type.
        """
        errors = []
        ctx = make_stmt_ctx(
            env={
                "results": T_REAL,
                "x": T_REAL,
            },
            errors=errors,
        )

        stmt_for_eq = IndexingandSlicing().type_rules()["body_for_map"]

        _, ctx.s = stmt_for_eq(
            ("body_for_map", "results", [], ("var", "x")),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []

    def test_body_for_map_indexed(self):
        """
        body_for_map unifies each loop var's TDim to the array dimension.
        """
        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"results": mat}, errors=errors)

        i_dim = new_dim()
        j_dim = new_dim()
        ctx.env["i"] = i_dim
        ctx.env["j"] = j_dim

        stmt_for_eq = IndexingandSlicing().type_rules()["body_for_map"]

        _, ctx.s = stmt_for_eq(
            (
                "body_for_map",
                "results",
                [
                    ("index_item", ("var", "i")),
                    ("index_item", ("var", "j")),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []
        assert ctx.env["i"] is i_dim
        assert ctx.env["j"] is j_dim

    def test_body_for_map_slice(self):
        """
        body_for_map unifies slice start/end variables with the corresponding
        tensor dimension.
        """
        errors = []
        mat = TTensor(((5, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"results": mat}, errors=errors)

        start_dim = new_dim()
        end_dim = new_dim()

        ctx.env["start"] = start_dim
        ctx.env["end"] = end_dim

        stmt_for_eq = IndexingandSlicing().type_rules()["body_for_map"]

        _, ctx.s = stmt_for_eq(
            (
                "body_for_map",
                "results",
                [
                    ("slice_item", ("var", "start"), ("var", "end")),
                    ("index_item", ("num", 1)),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []


class TestStmtForPluseq:
    """Test += accumulation statement type inference."""

    def test_basic_pluseq(self):
        """Basic for_pluseq infers rhs type."""
        errors = []
        ctx = make_stmt_ctx(
            env={
                "total": T_REAL,
                "x": T_REAL
            },
            errors=errors,
        )

        stmt_for_pluseq = IndexingandSlicing().type_rules()["for_pluseq"]

        _, ctx.s = stmt_for_pluseq(
            ("for_pluseq", "total", [], ("var", "x")),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []

    def test_indexed_pluseq(self):
        """
        loop_index_pluseq unifies each loop var's TDim to the array dimension.
        """
        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"C": mat}, errors=errors)

        # Simulate i, j registered by stmt_body_for_accum before this stmt
        i_dim = new_dim()
        j_dim = new_dim()
        ctx.env["i"] = i_dim
        ctx.env["j"] = j_dim

        stmt_for_pluseq = IndexingandSlicing().type_rules(
        )["loop_index_pluseq"]

        _, ctx.s = stmt_for_pluseq(
            (
                "loop_index_pluseq",
                "C",
                [
                    ("index_item", ("var", "i")),
                    ("index_item", ("var", "j")),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []
        # env keeps the raw TDim objects unchanged
        assert ctx.env["i"] is i_dim
        assert ctx.env["j"] is j_dim
        # substitution resolves them to concrete dimensions
        assert ctx.s.apply(i_dim) == 3
        assert ctx.s.apply(j_dim) == 4

    def test_indexed_pluseq_slice(self):
        """
        loop_index_pluseq unifies slice bounds with the corresponding
        array dimension.
        """
        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"C": mat}, errors=errors)

        i_dim = new_dim()
        ctx.env["i"] = i_dim

        stmt_for_pluseq = IndexingandSlicing().type_rules(
        )["loop_index_pluseq"]

        _, ctx.s = stmt_for_pluseq(
            (
                "loop_index_pluseq",
                "C",
                [
                    ("slice_item", ("var", "i"), None),
                    ("index_item", ("num", 1)),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []
        assert ctx.env["i"] is i_dim
        assert ctx.s.apply(i_dim) == 3

    def test_for_pluseq_basic_indexed(self):
        """
        Basic indexed for_pluseq infers rhs type.
        """
        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"C": mat}, errors=errors)

        stmt_for_pluseq = IndexingandSlicing().type_rules()["for_pluseq"]

        _, ctx.s = stmt_for_pluseq(
            (
                "for_pluseq",
                "C",
                [
                    ("index_item", ("num", 0)),
                    ("index_item", ("num", 1)),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []

    def test_for_pluseq_indexed(self):
        """
        for_pluseq unifies each loop var's TDim to the array dimension.
        """
        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"C": mat}, errors=errors)

        i_dim = new_dim()
        j_dim = new_dim()
        ctx.env["i"] = i_dim
        ctx.env["j"] = j_dim

        stmt_for_pluseq = IndexingandSlicing().type_rules()["for_pluseq"]

        _, ctx.s = stmt_for_pluseq(
            (
                "for_pluseq",
                "C",
                [
                    ("index_item", ("var", "i")),
                    ("index_item", ("var", "j")),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []
        assert ctx.env["i"] is i_dim
        assert ctx.env["j"] is j_dim

    def test_for_pluseq_slice(self):
        """
        for_pluseq unifies slice bounds with the corresponding array dimension.
        """
        errors = []
        mat = TTensor(((3, "invariant"), (4, "invariant")))
        ctx = make_stmt_ctx(env={"C": mat}, errors=errors)

        i_dim = new_dim()
        ctx.env["i"] = i_dim

        stmt_for_pluseq = IndexingandSlicing().type_rules()["for_pluseq"]

        _, ctx.s = stmt_for_pluseq(
            (
                "for_pluseq",
                "C",
                [
                    ("slice_item", ("var", "i"), None),
                    ("index_item", ("num", 1)),
                ],
                ("num", 1.0),
            ),
            ctx.env,
            ctx.s,
            ctx.func_env,
            ctx.class_env,
            ctx.add_error,
            infer_expr,
        )

        assert errors == []
        assert ctx.env["i"] is i_dim
