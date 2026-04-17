from physika.utils.types import (
    TVar,
    TDim,
    TTensor,
    TFunc,
    T_REAL,
    T_NAT,
    T_COMPLEX,
    Substitution,
)
from physika.utils.infer_expr import (
    ExprContext,
    expr_num,
    expr_imaginary,
    expr_var,
    expr_array,
    expr_index,
    expr_indexN,
    expr_chain_index,
    expr_slice,
    expr_add_sub,
    expr_mul,
    expr_div,
    expr_matmul,
    expr_pow,
    expr_neg,
    expr_call,
    expr_for_expr,
    expr_for_expr_range,
    infer_expr,
)
from physika.utils.types import new_dim


class TestExprContext:
    """
    Tests for ``ExprContext``
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
        ctx = ExprContext(env, s, func_env, class_env, cb)
        assert ctx.env is env
        assert ctx.s is s
        assert ctx.func_env is func_env
        assert ctx.class_env is class_env
        assert ctx.add_error is cb

    def test_empty_dicts(self):
        """
        All dict arguments may be empty.
        """
        ctx = ExprContext(env={},
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
        ctx = ExprContext(env={},
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
        ctx = ExprContext(env={},
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
        ctx = ExprContext(env=env,
                          s=Substitution(),
                          func_env={},
                          class_env={},
                          add_error=[].append)
        env["y"] = T_NAT
        assert ctx.env["y"] == T_NAT


def make_ctx(env=None, s=None, func_env=None, class_env=None, errors=None):
    """
    Build an ExprContext with sensible defaults for unit tests.
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

    return ExprContext(
        env=env,
        s=s,
        func_env=func_env,
        class_env=class_env,
        add_error=errors.append,
    )


class TestExprNum:
    """
    Tests for ``expr_num``.

    Numeric literal inference.
    """

    def test_float(self):
        """Any float literal must infer to ℝ."""
        ctx = make_ctx()
        t, s = expr_num(("num", 3.14), ctx)
        assert t == T_REAL

    def test_int(self):
        """Integer literal refer to ℝ."""
        ctx = make_ctx()
        t, s = expr_num(("num", 42), ctx)
        assert t == T_REAL

        # zero is ℝ
        ctx = make_ctx()
        t, _ = expr_num(("num", 0), ctx)
        assert t == T_REAL

    def test_substitution_unchanged(self):
        """
        The substitution dict does not contain new bindings.
        """
        existing = Substitution({"α0": T_NAT})
        ctx = make_ctx(s=existing)
        _, s_out = expr_num(("num", 1.0), ctx)
        assert s_out == existing

    def test_env_context(self):
        """expr_num infer type with a non-empty environment."""
        ctx = make_ctx(env={"x": T_REAL, "y": TTensor(((3, "invariant"), ))})
        t, _ = expr_num(("num", 7.0), ctx)
        assert t == T_REAL


class TestExprImaginary:
    """Tests for ``expr_imaginary``."""

    def test_ret_complex(self):
        """
        ``i`` is the imaginary unit ℂ.
        """
        ctx = make_ctx()
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_COMPLEX

    def test_loop_var_over_imaginary(self):
        """
        When ``i`` is a live loop variable it shadows ℂ and resolves to ℝ.
        """
        ctx = make_ctx(env={"i": T_REAL})
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_REAL

        # loop variables named ``j``, ``k``, etc,  must not shadow ``i``.
        ctx = make_ctx(env={"j": T_REAL, "k": T_REAL})
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_COMPLEX

    def test_loop_var_substitution_applied(self):
        """
        When ``i`` maps to a TVar that is bound, apply returns the binding.
        """
        alpha = TVar("α0")
        s = Substitution({"α0": T_REAL})
        ctx = make_ctx(env={"i": alpha}, s=s)
        t, _ = expr_imaginary(("imaginary", ), ctx)
        assert t == T_REAL

    def test_substitution(self):
        """No new bindings are introduced."""
        existing = Substitution({"α1": T_NAT})
        ctx = make_ctx(s=existing)
        _, s_out = expr_imaginary(("imaginary", ), ctx)
        assert s_out == existing


class TestExprVar:
    """
    Tests for ``expr_var``.

    Variable lookup in the current environment.
    """

    def test_bounded_variable(self):
        """A variable bound to ℝ resolves to ℝ."""
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_var(("var", "x"), ctx)
        assert t == T_REAL

        # a variable bound to ℕ resolves to ℕ.
        ctx = make_ctx(env={"n": T_NAT})
        t, _ = expr_var(("var", "n"), ctx)
        assert t == T_NAT

        # a variable bound to ℝ[3] resolves to ℝ[3].
        vec_t = TTensor(((3, "invariant"), ))
        ctx = make_ctx(env={"v": vec_t})
        t, _ = expr_var(("var", "v"), ctx)
        assert t == vec_t

        # a variable bound to ℝ[2,4] resolves to ℝ[2,4].
        mat_t = TTensor(((2, "invariant"), (4, "invariant")))
        ctx = make_ctx(env={"M": mat_t})
        t, _ = expr_var(("var", "M"), ctx)
        assert t == mat_t

    def test_unbound_variable_returns_none(self):
        """Looking up a name not in scope returns None without error."""
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_var(("var", "y"), ctx)
        assert t is None

    def test_empty_env_returns_none(self):
        """Empty environment always returns None for any name."""
        ctx = make_ctx()
        t, _ = expr_var(("var", "anything"), ctx)
        assert t is None

    def test_substitution_applied_to_tvar(self):
        """
        Checks that if the variable maps to a TVar,
        the substitution is applied.
        """
        alpha = TVar("α0")
        s = Substitution({"α0": T_REAL})
        ctx = make_ctx(env={"x": alpha}, s=s)
        t, _ = expr_var(("var", "x"), ctx)
        assert t == T_REAL

        # symbolic dims in tensor types are resolved through s
        tensor_t = TTensor(((TDim("δ0"), "invariant"), ))
        s = Substitution({"δ0": 5})
        ctx = make_ctx(env={"v": tensor_t}, s=s)
        t, _ = expr_var(("var", "v"), ctx)
        expected = TTensor(((5, "invariant"), ))
        assert t == expected

    def test_bound_substitution(self):
        """
        The substitution is returned unchanged when the variable is found.
        """
        existing = Substitution({"α0": T_NAT})
        ctx = make_ctx(env={"x": T_REAL}, s=existing)
        _, s_out = expr_var(("var", "x"), ctx)
        assert s_out == existing

        # substitution is returned unchanged even when lookup returns None.
        existing = Substitution({"α0": T_NAT})
        ctx = make_ctx(s=existing)
        _, s_out = expr_var(("var", "missing"), ctx)
        assert s_out == existing

    def test_no_errors_emitted_for_unbound(self):
        """Missing variable does NOT trigger an error — callers handle None."""
        errors = []
        ctx = make_ctx(errors=errors)
        expr_var(("var", "None"), ctx)
        assert errors == []

    def test_func_type_variable(self):
        """A variable holding a function type resolves correctly."""
        f_t = TFunc((T_REAL, ), T_REAL)
        ctx = make_ctx(env={"f": f_t})
        t, _ = expr_var(("var", "f"), ctx)
        assert t == f_t


class TestExprArray:
    """Tests for ``expr_array``."""

    def test_empty_array(self):
        """Empty literal produces ℝ[0]."""
        ctx = make_ctx()
        t, _ = expr_array(("array", []), ctx)
        assert t == TTensor(((0, "invariant"), ))

    def test_scalar_elements(self):
        """Three scalar literals produce ℝ[3]."""
        ctx = make_ctx()
        elems = [("num", 1.0), ("num", 2.0), ("num", 3.0)]
        t, _ = expr_array(("array", elems), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # Length of the resulting tensor equals the number of elements
        for n in (1, 5, 10):
            elems = [("num", float(i)) for i in range(n)]
            t, _ = expr_array(("array", elems), ctx)
            assert t == TTensor(((n, "invariant"), ))

    def test_nested_arrays(self):
        """
        Test matrices:
        [[1,2],[3,4]] should be type ℝ[2, 2].
        """
        ctx = make_ctx()
        row = ("array", [("num", 1.0), ("num", 2.0)])
        t, _ = expr_array(("array", [row, row]), ctx)
        assert t == TTensor(((2, "invariant"), (2, "invariant")))

        # non-square matrices:
        # [[1,2,3],[4,5,6]] should be ℝ[2, 3]
        ctx = make_ctx()
        row = ("array", [("num", 1.0), ("num", 2.0), ("num", 3.0)])
        t, _ = expr_array(("array", [row, row]), ctx)
        assert t == TTensor(((2, "invariant"), (3, "invariant")))

    def test_3D_nesting(self):
        """Nesting three deep"""
        ctx = make_ctx()
        inner = ("array", [("num", 0.0), ("num", 1.0)])
        middle = ("array", [inner, inner])
        t, _ = expr_array(("array", [middle, middle]), ctx)
        # t should be ℝ[2, 2, 2]
        assert t == TTensor(
            ((2, "invariant"), (2, "invariant"), (2, "invariant")))

    def test_variable_elements(self):
        """
        Array of vector variables is scope should be one rank higher.
        """
        # a : R[4]
        # b: R[4]
        ctx = make_ctx(env={
            "a": TTensor(((4, "invariant"), )),
            "b": TTensor(((4, "invariant"), ))
        })
        # t : R[2, 4] = [a, b] = [R[4], R[4]]
        t, _ = expr_array(("array", [("var", "a"), ("var", "b")]), ctx)
        assert t == TTensor(((2, "invariant"), (4, "invariant")))

    def test_substitution(self):
        """
        Unification of a TVar element against a concrete type produces a
        binding.
        """
        alpha = TVar("α0")
        # x has a fresh type variable as its type
        ctx = make_ctx(env={"x": alpha})
        elems = [("var", "x"), ("num", 1.0)]

        # [x, 1.0] where x : α0
        t, s_out = expr_array(("array", elems), ctx)
        assert t == TTensor(((2, "invariant"), ))
        # α0 must now be resolved to T_REAL
        assert s_out["α0"] == T_REAL

    def test_mixed_type_error(self):
        """
        Incompatible element types are reported via add_error, not raised
        ."""
        errors = []
        ctx = make_ctx(
            env={
                "v": TTensor(((3, "invariant"), )),
                "w": TTensor(((5, "invariant"), ))
            },
            errors=errors,
        )
        # ℝ[3] and ℝ[5] cannot be unified, should add an error
        expr_array(("array", [("var", "v"), ("var", "w")]), ctx)
        assert len(errors) == 1
        assert 'Inconsistent array element types at index 1: Dimension mismatch: 3 ≠ 5' == errors[  # noqa: E501
            0]

        # no error for compatible shapes
        errors = []
        ctx_no_error = make_ctx(env={
            "a": TTensor(((3, "invariant"), )),
            "b": TTensor(((3, "invariant"), ))
        },
                                errors=errors)
        expr_array(("array", [("var", "a"), ("var", "b")]), ctx_no_error)
        assert errors == []


class TestExprIndex:
    """
    Tests for ``expr_index``
    """

    def test_index(self):
        """Indexing a 1D vector produces a scalar ℝ."""
        ctx = make_ctx(env={"v": TTensor(((4, "invariant"), ))})
        t, _ = expr_index(("index", "v", ("num", 0)), ctx)
        assert t == T_REAL

        # Indexing the first dimension of a matrix produces a row vector
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_index(("index", "A", ("num", 0)), ctx)
        assert t == ("tensor", [(4, "invariant")])

        # Indexing a 3D tensor along dim-0 produces a 2D matrix
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_index(("index", "T", ("num", 0)), ctx)
        assert t == ("tensor", [(3, "invariant"), (4, "invariant")])

    def test_unknown_variable(self):
        """
        Unknown array name returns (None, ctx.s) without error, error is raised
        at compile time.
        """
        errors = []
        ctx = make_ctx(errors=errors)
        t, _ = expr_index(("index", "missing", ("num", 0)), ctx)
        assert t is None
        assert errors == []

    def test_scalar_indexed_reports_error(self):
        """
        Indexing a scalar variable reports an error.
        """
        errors = []
        ctx = make_ctx(env={"x": T_REAL}, errors=errors)
        expr_index(("index", "x", ("num", 0)), ctx)
        assert len(errors) == 1
        assert errors[0] == "Cannot index scalar 'x'"

    def test_expression_index(self):
        """The index expression may itself be a variable (loop counter)."""
        ctx = make_ctx(env={"v": TTensor(((5, "invariant"), )), "i": T_REAL})
        t, _ = expr_index(("index", "v", ("var", "i")), ctx)
        assert t == T_REAL

    def test_substitution_returned(self):
        """Substitution is returned unchanged for simple cases."""
        s = Substitution()
        ctx = make_ctx(env={"v": TTensor(((3, "invariant"), ))}, s=s)
        _, s_out = expr_index(("index", "v", ("num", 0)), ctx)
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
            },  # "δ0" is the size to range over, determined by unification
            s=s,
        )
        _, s_out = expr_index(("index", "v", ("var", "i")), ctx)
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
        t, _ = expr_indexN(("indexN", "A", [("num", 0), ("num", 1)]), ctx)
        assert t == T_REAL

    def test_3d_index(self):
        """
        Indexing the first two dims of a 3-D tensor gives ℝ[k].
        """
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_indexN(("indexN", "T", [("num", 0), ("num", 1)]), ctx)
        assert t == TTensor(((4, "invariant"), ))

        # Indexing all three dims gives ℝ
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_indexN(("indexN", "T", [("num", 0), ("num", 1),
                                            ("num", 2)]), ctx)
        assert t == T_REAL

    def test_matrix_index(self):
        """One index on a matrix gives row vector."""
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_indexN(("indexN", "A", [("num", 1)]), ctx)
        assert t == TTensor(((4, "invariant"), ))

    def test_unknown_variable_returns_none(self):
        """
        Variable not in scope gives (None, s)
        """
        errors = []
        ctx = make_ctx(errors=errors)
        t, _ = expr_indexN(("indexN", "missing", [("num", 0), ("num", 1)]),
                           ctx)
        assert t is None
        assert errors == []

    def test_scalar_variable_returns_none(self):
        """
        Indexing a scalar gives None.
        """
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_indexN(("indexN", "x", [("num", 0), ("num", 1)]), ctx)
        assert t is None

    def test_over_indexed(self):
        """More indices than dimensions gives None + error reported."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((3, "invariant"), ))}, errors=errors)
        # v is ℝ[3] (rank 1).
        # supplying [i][j][k] indices is overindexing
        t, _ = expr_indexN(("indexN", "v", [("num", 0), ("num", 1),
                                            ("num", 2)]), ctx)
        assert t is None
        assert len(errors) == 1
        assert errors[0] == "Over-indexed 'v': 3 indices for a rank-1 tensor"
        assert "v" in errors[0]
        assert "3" in errors[0]  # n_idx reported in message

    def test_variable_indices(self):
        """
        Loop variables used as indices works properly.
        """
        # A : ℝ[3, 4][i][j] -> ℝ
        ctx = make_ctx(
            env={
                "A": TTensor(((3, "invariant"), (4, "invariant"))),
                "i": T_REAL,
                "j": T_REAL
            })
        t, _ = expr_indexN(("indexN", "A", [("var", "i"), ("var", "j")]), ctx)
        assert t == T_REAL


class TestExprChainIndex:
    """Tests for ``expr_chain_index``."""

    def test_tensor_chain(self):
        """
        Let A: ℝ[m, n];
        A[i] on a 2D matrix gives ℝ[n]
        """
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        inner = ("index", "A", ("num", 0))  # A[0] -> ℝ[4]
        t, _ = expr_chain_index(("chain_index", inner), ctx)
        assert t == T_REAL

        # Let T : ℝ[m, n, o];
        # T[i][j] on a 3-D tensor gives ℝ[4].
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        inner = ("index", "T", ("num", 0)
                 )  # T[0] -> ('tensor', [(3,'invariant'),(4,'invariant')])
        t, _ = expr_chain_index(("chain_index", inner), ctx)
        assert t == TTensor(((4, "invariant"), ))

    def test_1d_chain_is_overindex(self):
        """
        v[0][k] on a 1-D vector is over-indexing:
        v[0] → ℝ,
        then [k] on scalar gives error.
        """
        errors = []
        ctx = make_ctx(env={"v": TTensor(((2, "invariant"), ))}, errors=errors)
        inner = ("index", "v", ("num", 0))  # v[0] → ℝ
        t, s_out = expr_chain_index(("chain_index", inner), ctx)
        assert t is None
        assert len(errors) == 1
        assert "Chain index applied to a scalar: cannot index a scalar" in errors[  # noqa: E501
            0]
        assert isinstance(s_out, Substitution)

    def test_chain_of_var_indices(self):
        """
        Chain index with variable loop counters works correctly.
        """
        ctx = make_ctx(
            env={
                "A": TTensor(((3, "invariant"), (4, "invariant"))),
                "i": T_REAL,
                "j": T_REAL
            })
        inner = ("index", "A", ("var", "i"))  # A[i] → ℝ[4]
        t, _ = expr_chain_index(("chain_index", inner), ctx)
        assert t == T_REAL

    def test_chain_on_scalar(self):
        """
        Chaining [k] onto a scalar result is over-indexing.
        """
        errors = []
        ctx = make_ctx(env={"x": T_REAL}, errors=errors)
        # ("var", "x") infers T_REAL; chain_index then tries to peel
        #  a dim from scalar
        inner = ("var", "x")
        t, _ = expr_chain_index(("chain_index", inner), ctx)
        assert t is None
        assert len(errors) == 1
        assert "Chain index applied to a scalar: cannot index a scalar" in errors[  # noqa: E501
            0]

    def test_chain_on_failed_inner_propagates_none(self):
        """
        When inner inference returns None (e.g. unknown var),
        no second error is added.
        """
        errors = []
        ctx = make_ctx(errors=errors)
        # "missing" is not in env
        inner = ("index", "missing", ("num", 0))
        t, _ = expr_chain_index(("chain_index", inner), ctx)
        assert t is None
        assert errors == []


class TestExprSlice:
    """
    Tests for ``expr_slice``.
    """

    def test_1d_bounds(self):
        """
        v[0:3] on ℝ[6] gives ℝ[3].
        """
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), ))})
        t, _ = expr_slice(("slice", "v", ("num", 0.0), ("num", 3.0)), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # v[2:7] on ℝ[10] gives ℝ[5].
        ctx = make_ctx(env={"v": TTensor(((10, "invariant"), ))})
        t, _ = expr_slice(("slice", "v", ("num", 2.0), ("num", 7.0)), ctx)
        assert t == TTensor(((5, "invariant"), ))  # 7 - 2 = 5

        # v[0:6] on ℝ[6] → ℝ[6]
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), ))})
        t, _ = expr_slice(("slice", "v", ("num", 0.0), ("num", 6.0)), ctx)
        assert t == TTensor(((6, "invariant"), ))

    def test_2d_slice(self):
        """A[0:2] on ℝ[3, 4] gives ℝ[2, 4]"""
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_slice(("slice", "A", ("num", 0.0), ("num", 2.0)), ctx)
        assert t == TTensor(((2, "invariant"), (4, "invariant")))

    def test_dynamic_start_gives_symbolic_dim(self):
        """
        Dynamic start result is ℝ[δN]
        """
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), )), "i": T_REAL})
        t, _ = expr_slice(("slice", "v", ("var", "i"), ("num", 4.0)), ctx)
        assert isinstance(t, TTensor)
        assert len(t.dims) == 1
        dim, _ = t.dims[0]
        assert isinstance(dim, TDim)  # length unknown, but still a 1-D tensor

    def test_dynamic_end_gives_symbolic_dim(self):
        """Dynamic end result is ℝ[δN]"""
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), )), "j": T_REAL})
        t, _ = expr_slice(("slice", "v", ("num", 0.0), ("var", "j")), ctx)
        assert isinstance(t, TTensor)
        assert len(t.dims) == 1
        dim, _ = t.dims[0]
        assert isinstance(dim, TDim)

    def test_both_dynamic_gives_symbolic_dim(self):
        """
        Both bounds dynamic gives ℝ[δN]
        """
        ctx = make_ctx(env={
            "v": TTensor(((6, "invariant"), )),
            "a": T_REAL,
            "b": T_REAL
        })
        t, _ = expr_slice(("slice", "v", ("var", "a"), ("var", "b")), ctx)
        assert isinstance(t, TTensor)
        assert len(t.dims) == 1
        dim, _ = t.dims[0]
        assert isinstance(dim, TDim)

    def test_dynamic_bounds_2d(self):
        """Dynamic start on ℝ[3,4] gives ℝ[δN,4]"""
        ctx = make_ctx(env={
            "A": TTensor(((3, "invariant"), (4, "invariant"))),
            "i": T_REAL
        })
        t, _ = expr_slice(("slice", "A", ("var", "i"), ("num", 2.0)), ctx)
        assert isinstance(t, TTensor)
        assert len(t.dims) == 2
        leading_dim, _ = t.dims[0]
        assert isinstance(leading_dim, TDim)  # sliced dim is symbolic
        assert t.dims[1] == (4, "invariant")  # trailing dim is exact

    def test_unknown_variable_returns_none(self):
        """Unknown array name gives (None, ctx.s)."""
        ctx = make_ctx()
        t, _ = expr_slice(("slice", "missing", ("num", 0.0), ("num", 3.0)),
                          ctx)
        assert t is None

    def test_substitution_unchanged(self):
        """No new substitution bindings are produced for literal slices."""
        s = Substitution({"α0": T_REAL})
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), ))}, s=s)
        _, s_out = expr_slice(("slice", "v", ("num", 0.0), ("num", 3.0)), ctx)
        assert s_out == s

    def test_negative_start_reports_error(self):
        """Negative start is a semantic error gives (None, s) + error."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), ))}, errors=errors)
        t, _ = expr_slice(("slice", "v", ("num", -1.0), ("num", 3.0)), ctx)
        assert t is None
        assert len(errors) == 1
        assert "Slice start -1 is negative in 'v[-1:3]'" == errors[0]

    def test_negative_end_reports_error(self):
        """Negative end is a static semantic error."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), ))}, errors=errors)
        t, _ = expr_slice(("slice", "v", ("num", 0.0), ("num", -2.0)), ctx)
        assert t is None
        assert len(errors) == 1
        assert "Slice end -2 is negative in 'v[0:-2]'" == errors[0]

    def test_end_less_than_start_reports_error(self):
        """end < start raise error."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), ))}, errors=errors)
        t, _ = expr_slice(("slice", "v", ("num", 4.0), ("num", 2.0)), ctx)
        assert t is None
        assert len(errors) == 1
        assert "Slice end 2 is less than start 4 in 'v[4:2]'" == errors[0]

    def test_empty_slice_reports_error(self):
        """start == end produces an empty slice error."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((6, "invariant"), ))}, errors=errors)
        t, _ = expr_slice(("slice", "v", ("num", 3.0), ("num", 3.0)), ctx)
        assert t is None
        assert len(errors) == 1
        assert "Empty slice 'v[3:3]': start equals end" == errors[0]

    def test_start_out_of_bounds_reports_error(self):
        """start >= leading dimension error."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((4, "invariant"), ))}, errors=errors)
        t, _ = expr_slice(("slice", "v", ("num", 4.0), ("num", 5.0)), ctx)
        assert t is None
        assert len(errors) == 1
        assert errors[
            0] == "Slice start 4 is out of bounds for 'v' with leading dimension 4"  # noqa: E501

    def test_end_out_of_bounds_reports_error(self):
        """end > leading dimension error."""
        errors = []
        ctx = make_ctx(env={"v": TTensor(((4, "invariant"), ))}, errors=errors)
        t, _ = expr_slice(("slice", "v", ("num", 0.0), ("num", 5.0)), ctx)
        assert t is None
        assert len(errors) == 1
        assert errors[
            0] == "Slice end 5 is out of bounds for 'v' with leading dimension 4"  # noqa: E501


class TestExprAddSub:
    """Tests for ``expr_add_sub``."""

    def test_scalar(self):
        """ℝ + ℝ → ℝ"""
        ctx = make_ctx()
        t, _ = expr_add_sub(("add", ("num", 1.0), ("num", 2.0)), ctx)
        assert t == T_REAL

        # ℝ - ℝ → ℝ
        ctx = make_ctx()
        t, _ = expr_add_sub(("sub", ("num", 1), ("num", 3)), ctx)
        assert t == T_REAL

    def test_vector(self):
        """ℝ[3] + ℝ[3] → ℝ[3]"""
        ctx = make_ctx(env={
            "x": TTensor(((3, "invariant"), )),
            "y": TTensor(((3, "invariant"), ))
        })
        t, _ = expr_add_sub(("add", ("var", "x"), ("var", "y")), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # ℝ[4] - ℝ[4] → ℝ[4]
        ctx = make_ctx(env={
            "a": TTensor(((4, "invariant"), )),
            "b": TTensor(((4, "invariant"), ))
        })
        t, _ = expr_add_sub(("sub", ("var", "a"), ("var", "b")), ctx)
        assert t == TTensor(((4, "invariant"), ))

    def test_vector_and_scalar(self):
        """ℝ[3] + ℝ → ℝ[3]  (broadcast)."""
        ctx = make_ctx(env={"v": TTensor(((3, "invariant"), ))})
        t, _ = expr_add_sub(("add", ("var", "v"), ("num", 1)), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # ℝ + ℝ[3] → ℝ[3]  (tensor wins regardless of order)
        ctx = make_ctx(env={"v": TTensor(((3, "invariant"), ))})
        t, _ = expr_add_sub(("add", ("num", 1), ("var", "v")), ctx)
        assert t == TTensor(((3, "invariant"), ))

    def test_matrix_and_matrix(self):
        """ℝ[2,3] + ℝ[2,3] → ℝ[2,3]."""
        ctx = make_ctx(
            env={
                "A": TTensor(((2, "invariant"), (3, "invariant"))),
                "B": TTensor(((2, "invariant"), (3, "invariant")))
            })
        t, _ = expr_add_sub(("add", ("var", "A"), ("var", "B")), ctx)
        assert t == TTensor(((2, "invariant"), (3, "invariant")))

        errors = []
        ctx = make_ctx(env={
            "A": TTensor(((3, "invariant"), (2, "invariant"))),
            "B": TTensor(((2, "invariant"), (3, "invariant")))
        },
                       errors=errors)
        t, _ = expr_add_sub(("add", ("var", "A"), ("var", "B")), ctx)
        assert len(errors) == 1
        assert errors[0] == 'Shape mismatch in add: ℝ[3,2] vs ℝ[2,3]'

        errors = []
        ctx = make_ctx(env={
            "A": TTensor(((3, "invariant"), (2, "invariant"))),
            "B": TTensor(((2, "invariant"), (3, "invariant")))
        },
                       errors=errors)
        t, _ = expr_add_sub(("sub", ("var", "A"), ("var", "B")), ctx)
        assert len(errors) == 1
        assert errors[0] == 'Shape mismatch in sub: ℝ[3,2] vs ℝ[2,3]'

    def test_matrix_minus_scalar_broadcast(self):
        """ℝ[2,3] - ℝ → ℝ[2,3]."""
        ctx = make_ctx(
            env={"A": TTensor(((2, "invariant"), (3, "invariant")))})
        t, _ = expr_add_sub(("sub", ("var", "A"), ("num", 1)), ctx)
        assert t == TTensor(((2, "invariant"), (3, "invariant")))

    def test_shape_mismatch_reports_error(self):
        """Mismatched tensor shapes report an error and do not raise."""
        errors = []
        ctx = make_ctx(env={
            "x": TTensor(((3, "invariant"), )),
            "y": TTensor(((5, "invariant"), ))
        },
                       errors=errors)
        expr_add_sub(("add", ("var", "x"), ("var", "y")), ctx)
        assert len(errors) == 1
        assert errors[0] == "Shape mismatch in add: ℝ[3] vs ℝ[5]"

    def test_no_error_for_compatible_shapes(self):
        """Compatible tensor shapes emit no errors."""
        errors = []
        ctx = make_ctx(env={
            "x": TTensor(((3, "invariant"), )),
            "y": TTensor(((3, "invariant"), ))
        },
                       errors=errors)
        expr_add_sub(("add", ("var", "x"), ("var", "y")), ctx)

        assert errors == []

    def test_nested_expressions(self):
        """Nested add/sub expressions resolve correctly."""
        ctx = make_ctx(env={"x": T_REAL, "y": T_REAL})
        # (x + y) - 1.0
        inner = ("add", ("var", "x"), ("var", "y"))
        t, _ = expr_add_sub(("sub", inner, ("num", 1)), ctx)
        assert t == T_REAL


class TestInferExpr:
    """
    Unit tests for ``infer_expr``.

    ``infer_expr`` is the top-level expression dispatcher.
    """

    def test_none_node_returns_none(self):
        """``None`` input returns ``(None, s)`` with no error."""
        errors = []
        t, s = infer_expr(None, {}, Substitution(), {}, {}, errors.append)
        assert t is None
        assert isinstance(s, Substitution)
        assert errors == []

    def test_bare_int_returns_real(self):
        """Bare ``int`` literal returns ``T_REAL``."""
        t, _ = infer_expr(42, {}, Substitution(), {}, {}, [].append)
        assert t == T_REAL

    def test_bare_float_returns_real(self):
        """Bare ``float`` literal returns ``T_REAL``."""
        t, _ = infer_expr(3.14, {}, Substitution(), {}, {}, [].append)
        assert t == T_REAL

    def test_bare_string_returns_none(self):
        """Non-numeric bare value returns ``None`` without error."""
        errors = []
        t, _ = infer_expr("hello", {}, Substitution(), {}, {}, errors.append)
        assert t is None
        assert errors == []

    def test_num_node(self):
        """``("num", v)`` dispatches to ``expr_num`` returns ``T_REAL``."""
        t, _ = infer_expr(("num", 7.0), {}, Substitution(), {}, {}, [].append)
        assert t == T_REAL

    def test_var_node(self):
        """``("var", name)`` with known env entry returns its known type."""
        t, _ = infer_expr(("var", "x"), {"x": T_REAL}, Substitution(), {}, {},
                          [].append)
        assert t == T_REAL

        # for a vector variable returns TTensor type
        vec3 = TTensor(((3, "invariant"), ))
        t, _ = infer_expr(("var", "v"), {"v": vec3}, Substitution(), {}, {},
                          [].append)
        assert t == vec3

        # Unknown variable returns ``None``"""
        errors = []
        t, _ = infer_expr(("var", "z"), {}, Substitution(), {}, {},
                          errors.append)
        assert t is None
        assert errors == []

    def test_imaginary_node(self):
        """``("imaginary",)`` returns ``T_COMPLEX``."""
        t, _ = infer_expr(("imaginary", ), {}, Substitution(), {}, {},
                          [].append)
        assert t == T_COMPLEX

    def test_array_literal(self):
        """``("array", [num, num])`` returns ``ℝ[2]``."""
        node = ("array", [("num", 1.0), ("num", 2.0)])
        t, _ = infer_expr(node, {}, Substitution(), {}, {}, [].append)
        assert t == TTensor(((2, "invariant"), ))

    def test_index_1d(self):
        """1-D subscript on a known vector returns ``T_REAL``."""
        env = {"a": TTensor(((5, "invariant"), ))}
        node = ("index", "a", ("num", 2.0))
        t, _ = infer_expr(node, env, Substitution(), {}, {}, [].append)
        assert t == T_REAL

    def test_index_into_array_literal(self):
        """Index into an inline array literal resolves to scalar."""
        env = {"a": TTensor(((3, "invariant"), ))}
        node = ("index", "a", ("num", 0.0))
        t, _ = infer_expr(node, env, Substitution(), {}, {}, [].append)
        assert t == T_REAL

    def test_slice_returns_shorter_tensor(self):
        """``a[1:4]`` on ``ℝ[9]`` → ``ℝ[3]`` (end-exclusive)."""
        env = {"a": TTensor(((9, "invariant"), ))}
        node = ("slice", "a", ("num", 1), ("num", 4))
        t, _ = infer_expr(node, env, Substitution(), {}, {}, [].append)
        assert t == TTensor(((3, "invariant"), ))

    def test_slice_oob_reports_error(self):
        """Slice end beyond array length → error."""
        errors = []
        env = {"a": TTensor(((5, "invariant"), ))}
        node = ("slice", "a", ("num", 0), ("num", 10))
        infer_expr(node, env, Substitution(), {}, {}, errors.append)
        assert len(errors) >= 1
        assert errors[
            0] == "Slice end 10 is out of bounds for 'a' with leading dimension 5"  # noqa: E501

    def test_add_two_scalars(self):
        """``x + y`` with both scalars returns ``T_REAL``."""
        env = {"x": T_REAL, "y": T_REAL}
        node = ("add", ("var", "x"), ("var", "y"))
        t, _ = infer_expr(node, env, Substitution(), {}, {}, [].append)
        assert t == T_REAL

    def test_sub_scalar_literal(self):
        """``x - 1.0`` returns ``T_REAL``."""
        env = {"x": T_REAL}
        node = ("sub", ("var", "x"), ("num", 1.0))
        t, _ = infer_expr(node, env, Substitution(), {}, {}, [].append)
        assert t == T_REAL

    def test_unknown_tag_reports_error(self):
        """Unrecognised AST tag reports an error."""
        errors = []
        t, _ = infer_expr(("invalid_tag", 99), {}, Substitution(), {}, {},
                          errors.append)
        assert t is None
        assert errors == ["Unknown expression type: invalid_tag"]

        # second unknown tag produces the correct error message
        infer_expr(("mystery_op", "a", "b"), {}, Substitution(), {}, {},
                   errors.append)
        assert len(errors) == 2
        assert errors[-1] == "Unknown expression type: mystery_op"

    def test_returns_substitution_instance(self):
        """
        ``infer_expr`` always returns a ``Substitution`` as second element.
        """
        _, s = infer_expr(("num", 1.0), {}, Substitution(), {}, {}, [].append)
        assert isinstance(s, Substitution)

    def test_nested_add(self):
        """``(x + y) + z`` with all scalars returns ``T_REAL``."""
        env = {"x": T_REAL, "y": T_REAL, "z": T_REAL}
        inner = ("add", ("var", "x"), ("var", "y"))
        outer = ("add", inner, ("var", "z"))
        t, _ = infer_expr(outer, env, Substitution(), {}, {}, [].append)
        assert t == T_REAL


class TestExprMul:
    """
    Tests for ``expr_mul``.
    """

    def test_mult_op(self):
        """ℝ * ℝ → ℝ."""
        ctx = make_ctx()
        t, _ = expr_mul(("mul", ("num", 2.0), ("num", 3.0)), ctx)
        assert t == T_REAL

        # ℝ[3] * ℝ → ℝ[3]
        ctx = make_ctx(env={"x": TTensor(((3, "invariant"), ))})
        t, _ = expr_mul(("mul", ("var", "x"), ("num", 2.0)), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # ℝ * ℝ[3] → ℝ[3]  (order in operands swapped)
        ctx = make_ctx(env={"x": TTensor(((3, "invariant"), ))})
        t, _ = expr_mul(("mul", ("num", 2.0), ("var", "x")), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # ℝ[4] * ℝ[4] → ℝ[4]  (elementwise)
        ctx = make_ctx(
            env={
                "x": TTensor(((4, "invariant"), )),
                "y": TTensor(((4, "invariant"), )),
            })
        t, _ = expr_mul(("mul", ("var", "x"), ("var", "y")), ctx)
        assert t == TTensor(((4, "invariant"), ))

        # ℝ[2,3] * ℝ[2,3] → ℝ[2,3]
        ctx = make_ctx(
            env={
                "A": TTensor(((2, "invariant"), (3, "invariant"))),
                "B": TTensor(((2, "invariant"), (3, "invariant"))),
            })
        t, _ = expr_mul(("mul", ("var", "A"), ("var", "B")), ctx)
        assert t == TTensor(((2, "invariant"), (3, "invariant")))

    def test_shape_mismatch_reports_error(self):
        """ℝ[3] * ℝ[5] report error"""
        errors = []
        ctx = make_ctx(env={
            "x": TTensor(((3, "invariant"), )),
            "y": TTensor(((5, "invariant"), )),
        },
                       errors=errors)
        expr_mul(("mul", ("var", "x"), ("var", "y")), ctx)
        assert len(errors) == 1
        assert errors[0] == "Shape mismatch in mul: ℝ[3] vs ℝ[5]"


class TestExprDiv:
    """
    Tests for ``expr_div``.
    """

    def test_div_op(self):
        """ℝ / ℝ → ℝ."""
        ctx = make_ctx()
        t, _ = expr_div(("div", ("num", 6.0), ("num", 2.0)), ctx)
        assert t == T_REAL

        # ℝ[3] / ℝ → ℝ[3]
        ctx = make_ctx(env={"x": TTensor(((3, "invariant"), ))})
        t, _ = expr_div(("div", ("var", "x"), ("num", 2.0)), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # ℝ[4] / ℝ[4] → ℝ[4]  (elementwise)
        ctx = make_ctx(
            env={
                "x": TTensor(((4, "invariant"), )),
                "y": TTensor(((4, "invariant"), )),
            })
        t, _ = expr_div(("div", ("var", "x"), ("var", "y")), ctx)
        assert t == TTensor(((4, "invariant"), ))

        # ℝ[2,3] / ℝ[2,3] → ℝ[2,3]
        ctx = make_ctx(
            env={
                "A": TTensor(((2, "invariant"), (3, "invariant"))),
                "B": TTensor(((2, "invariant"), (3, "invariant"))),
            })
        t, _ = expr_div(("div", ("var", "A"), ("var", "B")), ctx)
        assert t == TTensor(((2, "invariant"), (3, "invariant")))

    def test_tensor_div_tensor_different_shape(self):
        """ℝ[3] / ℝ[2] reports error."""
        errors = []
        ctx = make_ctx(env={
            "x": TTensor(((3, "invariant"), )),
            "z": TTensor(((2, "invariant"), )),
        },
                       errors=errors)
        expr_div(("div", ("var", "x"), ("var", "z")), ctx)
        assert len(errors) == 1
        assert errors[0] == "Shape mismatch in div: ℝ[3] vs ℝ[2]"


class TestExprMatmul:
    """
    Tests for ``expr_matmul``.
    """

    def test_matmul_op(self):
        """ℝ[3] @ ℝ[3] → ℝ."""
        ctx = make_ctx(
            env={
                "u": TTensor(((3, "invariant"), )),
                "v": TTensor(((3, "invariant"), )),
            })
        t, _ = expr_matmul(("matmul", ("var", "u"), ("var", "v")), ctx)
        assert t == T_REAL

        # ℝ[2,3] @ ℝ[3,4] → ℝ[2,4]
        ctx = make_ctx(
            env={
                "A": TTensor(((2, "invariant"), (3, "invariant"))),
                "B": TTensor(((3, "invariant"), (4, "invariant"))),
            })
        t, _ = expr_matmul(("matmul", ("var", "A"), ("var", "B")), ctx)
        assert t == TTensor(((2, "invariant"), (4, "invariant")))

        # ℝ[3,3] @ ℝ[3,3] → ℝ[3,3]
        ctx = make_ctx(
            env={
                "A": TTensor(((3, "invariant"), (3, "invariant"))),
                "B": TTensor(((3, "invariant"), (3, "invariant"))),
            })
        t, _ = expr_matmul(("matmul", ("var", "A"), ("var", "B")), ctx)
        assert t == TTensor(((3, "invariant"), (3, "invariant")))

    def test_error_report(self):
        # ℝ[2,3] @ ℝ[3] should raise error rank mismatch
        errors = []
        ctx = make_ctx(env={
            "A": TTensor(((2, "invariant"), (3, "invariant"))),
            "v": TTensor(((3, "invariant"), )),
        },
                       errors=errors)
        t, _ = expr_matmul(("matmul", ("var", "A"), ("var", "v")), ctx)
        assert len(errors) == 1
        assert errors[
            0] == "Matmul rank mismatch: ℝ[2,3] @ ℝ[3]; use ℝ[2,3] @ ℝ[3,1] for an explicit matrix form"  # noqa: E501
        assert t is None


class TestExprPow:
    """
    Tests for ``expr_pow``.
    """

    def test_pow_op(self):
        # ℝ ** ℝ → ℝ
        ctx = make_ctx()
        t, _ = expr_pow(("pow", ("num", 2.0), ("num", 3.0)), ctx)
        assert t == T_REAL

        # ℝ[3] ** ℝ → ℝ[3]
        ctx = make_ctx(env={"x": TTensor(((3, "invariant"), ))})
        t, _ = expr_pow(("pow", ("var", "x"), ("num", 2.0)), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # ℝ[2,3] ** ℝ → ℝ[2,3]
        ctx = make_ctx(
            env={"A": TTensor(((2, "invariant"), (3, "invariant")))})
        t, _ = expr_pow(("pow", ("var", "A"), ("num", 2.0)), ctx)
        assert t == TTensor(((2, "invariant"), (3, "invariant")))

    def test_unknown_base_returns_none(self):
        """Unknown base variable returns None."""
        ctx = make_ctx()
        t, _ = expr_pow(("pow", ("var", "missing"), ("num", 2.0)), ctx)
        assert t is None


class TestExprNeg:
    """
    Tests for ``expr_neg``.
    """

    def test_negation(self):
        # -ℝ → ℝ.
        ctx = make_ctx()
        t, _ = expr_neg(("neg", ("num", 1.0)), ctx)
        assert t == T_REAL

        # -ℝ[3] → ℝ[3]
        ctx = make_ctx(env={"x": TTensor(((3, "invariant"), ))})
        t, _ = expr_neg(("neg", ("var", "x")), ctx)
        assert t == TTensor(((3, "invariant"), ))

        # -ℝ[2,4] → ℝ[2,4]
        ctx = make_ctx(
            env={"A": TTensor(((2, "invariant"), (4, "invariant")))})
        t, _ = expr_neg(("neg", ("var", "A")), ctx)
        assert t == TTensor(((2, "invariant"), (4, "invariant")))

        # --ℝ → ℝ  (double negation)
        ctx = make_ctx()
        inner = ("neg", ("num", 5.0))
        t, _ = expr_neg(("neg", inner), ctx)
        assert t == T_REAL

    def test_unknown_variable_returns_none(self):
        """-missing → None."""
        ctx = make_ctx()
        t, _ = expr_neg(("neg", ("var", "missing")), ctx)
        assert t is None


class TestExprCall:
    """Tests for ``expr_call``."""

    def test_elementwise_ops_preserve_shape(self):
        """Element wise builtins preserve operand's shape."""
        vec = TTensor(((4, "invariant"), ))
        for name in ("exp", "log", "sin", "cos", "sqrt", "abs", "tanh"):
            ctx = make_ctx(env={"x": vec})
            t, _ = expr_call(("call", name, [("var", "x")]), ctx)
            assert t == vec

    def test_reduction_sum(self):
        """sum(ℝ[5]) → ℝ."""
        ctx = make_ctx(env={"v": TTensor(((5, "invariant"), ))})
        t, _ = expr_call(("call", "sum", [("var", "v")]), ctx)
        assert t == T_REAL

    def test_grad_returns_x_shape(self):
        """grad(f, x) returns same shape as x."""
        ctx = make_ctx(env={"x": TTensor(((3, "invariant"), ))})
        t, _ = expr_call(("call", "grad", [("num", 1.0), ("var", "x")]), ctx)
        assert t == TTensor(((3, "invariant"), ))

    def test_grad_scalar_x(self):
        """grad(f, x) with scalar x → ℝ."""
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_call(("call", "grad", [("num", 1.0), ("var", "x")]), ctx)
        assert t == T_REAL

    def test_user_defined_function(self):
        """User-defined function: return type taken from func_env."""
        # f → ℝ[3]
        vec3 = TTensor(((3, "invariant"), ))
        func_env = {"f": ([vec3], vec3)}
        ctx = make_ctx(env={"x": vec3}, func_env=func_env)
        t, _ = expr_call(("call", "f", [("var", "x")]), ctx)
        assert t == vec3

        # g : ℝ → ℝ
        func_env = {"g": ([T_REAL], T_REAL)}
        ctx = make_ctx(env={"x": T_REAL}, func_env=func_env)
        t, _ = expr_call(("call", "g", [("var", "x")]), ctx)
        assert t == T_REAL

    def test_user_function_wrong_parameters_error(self):
        """Wrong number of arguments reports an error."""
        errors = []
        func_env = {"f": ([T_REAL, T_REAL], T_REAL)}
        ctx = make_ctx(func_env=func_env, errors=errors)
        expr_call(("call", "f", [("num", 1.0)]), ctx)  # expects 2, got 1
        assert len(errors) == 1
        assert "expects 2 args, got 1" in errors[0]

    def test_unknown_function_returns_none(self):
        """Unrecognised function returns None without raising."""
        ctx = make_ctx()
        t, _ = expr_call(("call", "unknown_fn", [("num", 1.0)]), ctx)
        assert t is None


class TestExprForExpr:
    """Tests for ``expr_for_expr``."""

    def test_scalar_body_produces_1d(self):
        """for i : ℕ(5) → i  produces ℝ[5]."""
        ctx = make_ctx()
        t, _ = expr_for_expr(("for_expr", "i", ("num", 5.0), ("imaginary")),
                             ctx, new_dim)
        assert t == TTensor(((5, "invariant"), ))

        # test for other cases
        ctx = make_ctx()
        for n in (1, 3, 8):
            t, _ = expr_for_expr(
                ("for_expr", "i", ("num", float(n)), ('imaginary')), ctx,
                new_dim)
            assert t == TTensor(((n, "invariant"), ))

    def test_tensor_body_prepends_dim(self):
        """
        b : ℝ[2] = [1, 3]
        A : ℝ[4, 2] = for i : ℕ(4) → b
        """
        ctx = make_ctx()
        body = ("array", [("num", 1.0), ("num", 2.0)])  # ℝ[2]
        t, _ = expr_for_expr(("for_expr", "i", ("num", 4.0), body), ctx,
                             new_dim)
        assert t == TTensor(((4, "invariant"), (2, "invariant")))

    def test_two_level_nesting(self):
        """for i :ℕ(3) → for j : ℕ(4) → i  produces ℝ[3, 4]."""
        ctx = make_ctx()
        inner = ("for_expr", "j", ("num", 4.0), ('imaginary'))
        t, _ = expr_for_expr(("for_expr", "i", ("num", 3.0), inner), ctx,
                             new_dim)
        assert t == TTensor(((3, "invariant"), (4, "invariant")))

    def test_three_level_nesting(self):
        """for i:ℕ(2) → for j:ℕ(3) → for k:ℕ(4) → i  produces ℝ[2, 3, 4]."""
        ctx = make_ctx()
        innermost = ("for_expr", "k", ("num", 4.0), (('imaginary')))
        middle = ("for_expr", "j", ("num", 3.0), innermost)
        t, _ = expr_for_expr(("for_expr", "i", ("num", 2.0), middle), ctx,
                             new_dim)
        assert t == TTensor(
            ((2, "invariant"), (3, "invariant"), (4, "invariant")))

    def test_dynamic_size_gives_tdim(self):
        """Non-literal size_expr introduces a fresh TDim as the outer dim."""
        ctx = make_ctx(env={"n": T_NAT})
        t, _ = expr_for_expr(("for_expr", "i", ("var", "n"), ('imaginary')),
                             ctx, new_dim)

        # t type should be t : ℝ[δ0]
        assert isinstance(t, TTensor)
        assert len(t.dims) == 1
        outer, _ = t.dims[0]
        assert isinstance(outer, TDim)


class TestExprForExprRange:
    """Tests for ``expr_for_expr_range``."""

    def test_literal_bounds_scalar_body(self):
        """for i :  ℕ(0, 4) → i  produces ℝ[4]."""
        ctx = make_ctx()
        t, _ = expr_for_expr_range(
            ("for_expr_range", "i", ("num", 0.0), ("num", 4.0), ('imaginary')),
            ctx, new_dim)
        assert t == TTensor(((4, "invariant"), ))

    def test_range_size_is_end_minus_start(self):
        """Size = end − start for literal bounds."""
        ctx = make_ctx()
        t, _ = expr_for_expr_range(
            ("for_expr_range", "i", ("num", 2.0), ("num", 7.0), ('imaginary')),
            ctx, new_dim)
        assert t == TTensor(((5, "invariant"), ))

    def test_zero_start(self):
        """ℕ(0, n) range gives ℝ[n]."""
        # n = 1, 5, 10
        ctx = make_ctx()
        for n in (1, 5, 10):
            t, _ = expr_for_expr_range(("for_expr_range", "i", ("num", 0.0),
                                        ("num", float(n)), ('imaginary')), ctx,
                                       new_dim)
            assert t == TTensor(((n, "invariant"), ))

    def test_tensor_body_prepends_dim(self):
        """for i : ℕ(0, n) → ℝ[3] body  produces ℝ[2, 3]."""
        ctx = make_ctx()
        body = ("array", [("num", 1.0), ("num", 2.0), ("num", 3.0)])  # ℝ[3]
        t, _ = expr_for_expr_range(
            ("for_expr_range", "i", ("num", 0.0), ("num", 2.0), body), ctx,
            new_dim)
        assert t == TTensor(((2, "invariant"), (3, "invariant")))

    def test_dynamic_bounds_give_tdim(self):
        """Dynamic end bound introduces a fresh TDim."""
        ctx = make_ctx(env={"n": T_NAT})
        t, _ = expr_for_expr_range(
            ("for_expr_range", "i", ("num", 0.0), ("var", "n"), ('imaginary')),
            ctx, new_dim)
        assert isinstance(t, TTensor)
        assert len(t.dims) == 1
        outer, _ = t.dims[0]
        assert isinstance(outer, TDim)
