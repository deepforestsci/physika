import pytest
from physika.utils.types import (TVar, TDim, TTensor, TFunc, TInstance, T_REAL,
                                 T_NAT, T_COMPLEX, T_STRING, T_Z2)
from physika.utils.type_checker_utils import (
    from_typespec,
    occurs_in,
    get_tensor_shape,
    make_tensor,
    unify,
    unify_dim,
    broadcast_op,
    matmul_op,
)
from physika.utils.types import Substitution


class TestFromTypespec:
    """Tests for from_typespec

    Converts parser annotation type from AST to Type used in type checker
    algorithm.
    """

    def test_from_typespec_cases(self):
        """
        Test that invalid Type return None, and correct types are converte
        accordingly.
        """
        assert from_typespec(None) is None
        assert from_typespec("unknown_type") is None
        assert from_typespec(42) is None
        assert from_typespec(("unsupported", )) is None

        assert from_typespec("ℝ") == T_REAL
        assert from_typespec("R") == T_REAL

        assert from_typespec("ℕ") == T_NAT
        assert from_typespec("N") == T_NAT

        assert from_typespec("ℂ") == T_COMPLEX

        assert from_typespec("string") == T_STRING

        assert from_typespec("ℤ2") == T_Z2
        assert from_typespec("Z2") == T_Z2

        result = from_typespec(("tensor", [(3, "invariant")]))
        assert result == TTensor(((3, "invariant"), ))

        result = from_typespec(("tensor", [(2, "invariant"),
                                           (4, "invariant")]))
        assert result == TTensor(((2, "invariant"), (4, "invariant")))

        # "n" must become TDim("n") for unification step
        result = from_typespec(("tensor", [("n", "invariant")]))
        assert result == TTensor(((TDim("n"), "invariant"), ))

        # ℝ[n, 4], where "n" symbolic, 4 concrete
        result = from_typespec(("tensor", [("n", "invariant"),
                                           (4, "invariant")]))
        assert result == TTensor(((TDim("n"), "invariant"), (4, "invariant")))

        # ℝ[n, m, o]
        result = from_typespec(("tensor", [("n", "invariant"),
                                           ("m", "invariant"),
                                           ("o", "invariant")]))
        assert result == TTensor((
            (TDim("n"), "invariant"),
            (TDim("m"), "invariant"),
            (TDim("o"), "invariant"),
        ))

        result = from_typespec(("func_type", "ℝ", "ℝ"))
        assert result == TFunc((T_REAL, ), T_REAL)

        result = from_typespec(
            ("func_type", "ℝ", ("tensor", [(3, "invariant")])))
        assert result == TFunc((T_REAL, ), TTensor(((3, "invariant"), )))

        result = from_typespec(("struct_type", "Ray"))
        assert result == TInstance("Ray")

        result = from_typespec(("instance", "Net"))
        assert result == TInstance("Net")


class TestOccursIn:
    """
    Tests for occurs_in.
    """

    def test_occurs_in(self):
        """
        Occurs check at unification step to prevent infinite types.
        """
        assert occurs_in(TVar("α0"), TVar("α1")) is False
        assert occurs_in(TVar("α0"), T_REAL) is False

        assert occurs_in(TDim("δ0"), TDim("δ0")) is True

        t = TTensor(((TDim("δ0"), "invariant"), ))
        assert occurs_in(TDim("δ0"), t) is True

        t = TTensor(((TDim("δ1"), "invariant"), ))
        assert occurs_in(TDim("δ0"), t) is False

        f = TFunc((TVar("α0"), ), T_REAL)
        assert occurs_in(TVar("α0"), f) is True

        f = TFunc((T_REAL, ), TVar("α0"))
        assert occurs_in(TVar("α0"), f) is True

        f = TFunc((T_REAL, ), T_NAT)
        assert occurs_in(TVar("α0"), f) is False

        t = TTensor(((3, "invariant"), ))
        assert occurs_in(TDim("δ0"), t) is False


class TestGetShape:
    """Tests for get_tensor_shape helper function."""

    def test_get_tensor_shape(self):
        """

        Test that get_tensor_shape return a list of dimensions for TTensor
        types and None for non-tensor types.
        """
        t = TTensor(((3, "invariant"), (4, "invariant")))
        assert get_tensor_shape(t) == [3, 4]

        t = TTensor(((5, "invariant"), ))
        assert get_tensor_shape(t) == [5]

        assert get_tensor_shape(T_REAL) is None
        assert get_tensor_shape(T_NAT) is None

        assert get_tensor_shape(None) is None

        f = TFunc((T_REAL, ), T_REAL)
        assert get_tensor_shape(f) is None

        t = TTensor(((TDim("n"), "invariant"), (3, "invariant")))
        shape = get_tensor_shape(t)
        assert shape == [TDim("n"), 3]


class TestMakeTensor:
    """Tests for make_tensor function."""

    def test_make_tensor(self):
        """
        Test that make_tensor constructs the correct TTensor type from a list
        of dimensions.
        """
        # 1D tensor
        assert make_tensor([3]) == TTensor(((3, "invariant"), ))

        # 2D tensor
        assert make_tensor([2, 3]) == TTensor(
            ((2, "invariant"), (3, "invariant")))

        # Symbolic dim
        result = make_tensor([TDim("n"), 4])
        assert result == TTensor(((TDim("n"), "invariant"), (4, "invariant")))

        # test that get_tensor_shape and make_tensot interoperability
        dims = [2, 3, 4]
        t = make_tensor(dims)
        assert get_tensor_shape(t) == dims
        dims = [TDim("n"), 3]
        t = make_tensor(dims)
        assert get_tensor_shape(t) == dims


class TestUnify:
    """
    Tests for ``unify`` which represents the unification step
    in in type checker algorithm.

    Unification is the process of determining if two types are compatible and
    finding a substitution that makes them equal.
    """

    def s(self):
        """
        Helper to create a fresh empty substitution for each test.
        """
        return Substitution()

    def test_equal_scalars_returns_equal_substitution(self):
        """
        Verify that unifying two identical scalar types succeeds
        """

        s = self.s()
        # unifying T_REAL with T_REAL should succeed with no new bindings
        assert unify(T_REAL, T_REAL, s) is s
        assert s == {}  # no new bindings

        assert unify(T_NAT, T_NAT, s) is s
        assert s == {}  # no new bindings

        assert unify(T_COMPLEX, T_COMPLEX, s) is s
        assert s == {}  # no new bindings

        assert unify(T_STRING, T_STRING, s) is s
        assert s == {}  # no new bindings

        assert unify(T_Z2, T_Z2, s) is s
        assert s == {}  # no new bindings

    def test_nat_real_compatible(self):
        """
        Tests that set of natural number is compatible with real numbers
        (ℕ ⊂ ℝ).
        """
        result = unify(T_REAL, T_NAT, self.s())
        assert result == {}
        result = unify(T_NAT, T_REAL, self.s())
        assert result == {}

    def test_tvar_binds_to_concrete(self):
        """
        Unifying a type variable with a concrete type.
        """
        s = unify(TVar("α0"), T_REAL, self.s())
        assert s.apply(TVar("α0")) == T_REAL
        assert s == {"α0": T_REAL}

        s = unify(TVar("α1"), T_NAT, s)
        assert s == {"α0": T_REAL, "α1": T_NAT}

    def test_tvar_second_binds(self):
        """
        Verifies unifying a concrete type with a type variable on the right
        also binds the variable.
        """
        s = unify(T_REAL, TVar("α0"), self.s())
        assert s.apply(TVar("α0")) == T_REAL

        s = unify(T_NAT, TVar("α1"), s)
        assert s == {"α0": T_REAL, "α1": T_NAT}

    def test_two_tvars_bind_together(self):
        """
        Test that unifying two type variables binds them together.
        """
        s = unify(TVar("α0"), TVar("α1"), self.s())
        assert s == {"α0": TVar("α1")}  # α0 binds to α1

        # after applying, both resolve to the same thing
        assert s.apply(TVar("α0")) == s.apply(TVar("α1"))

    def test_tvar_with_existing_substitution(self):
        """
        Verifies existing binding is conserved.
        """
        s0 = Substitution({"α0": T_REAL})

        # unifying α0 with ℝ returns the same substitution (already compatible)
        s1 = unify(TVar("α0"), T_REAL, s0)
        assert s1.apply(TVar("α0")) == T_REAL
        assert s1 == s0  # no new bindings

    def test_occurs_check_raises(self):
        """
        Unifying α0 with a type that contains α0 must raise TypeError
        (infinite type).
        """
        with pytest.raises(TypeError, match="Occurs check"):
            unify(TVar("α0"), TFunc((TVar("α0"), ), T_REAL), self.s())

    def test_scalar_mismatch_raises(self):
        """
        Checks that unifying two different scalar types raises a TypeError.
        """
        with pytest.raises(TypeError):
            unify(T_REAL, T_COMPLEX, self.s())
        with pytest.raises(TypeError):
            unify(T_Z2, T_COMPLEX, self.s())

    def test_scalar_tensor_mismatch_raises(self):
        """
        Checks that unifying a scalar type with a tensor type raises a
        TypeError.
        """
        t = TTensor(((3, "invariant"), ))
        with pytest.raises(TypeError):
            unify(T_REAL, t, self.s())
        with pytest.raises(TypeError):
            unify(t, T_REAL, self.s())
        with pytest.raises(TypeError):
            unify(T_Z2, t, self.s())
        with pytest.raises(TypeError):
            unify(t, T_Z2, self.s())

    def test_tensors_same_shape(self):
        """
        Tests that unifying two tensors with the same shape returns an
        empty substitution.
        """
        t = TTensor(((3, "invariant"), (4, "invariant")))
        result = unify(t, t, self.s())
        assert result == {}

    def test_tensors_rank_mismatch_raises(self):
        """
        Tests that unifying two tensors with different ranks raises a
        TypeError.
        """
        t1 = TTensor(((3, "invariant"), ))
        t2 = TTensor(((3, "invariant"), (4, "invariant")))
        with pytest.raises(TypeError, match="Rank mismatch"):
            unify(t1, t2, self.s())

    def test_tensor_with_symbolic_dim_binds(self):
        """
        Verifies that unifying a tensor with a symbolic dimension with a tensor
        with a concrete dimension binds the symbolic dimension to the concrete
        one.
        """
        t1 = TTensor(((TDim("δ0"), "invariant"), ))
        t2 = TTensor(((5, "invariant"), ))
        s = unify(t1, t2, self.s())
        assert s.apply_dim(TDim("δ0")) == 5
        assert s == {"δ0": 5}

    def test_func_unification(self):
        """
        Checks that unifying two function types with compatible argument and
        return types succeeds and returns the correct substitution.
        """
        # params and return types match
        f1 = TFunc((TVar("α0"), ), T_REAL)
        f2 = TFunc((T_NAT, ), T_REAL)
        s = unify(f1, f2, self.s())
        assert s.apply(TVar("α0")) == T_NAT

        # param types dont match
        f3 = TFunc((T_REAL, ), T_REAL)
        f4 = TFunc((T_COMPLEX, ), T_REAL)
        with pytest.raises(TypeError):
            unify(f3, f4, self.s())

        # return types dont match
        f5 = TFunc((T_REAL, ), T_REAL)
        f6 = TFunc((T_REAL, ), T_COMPLEX)
        with pytest.raises(TypeError):
            unify(f5, f6, self.s())

    def test_func_param_numbers_mismatch_raises(self):
        """
        Checks that unifying two function types with different number of
        parameters raises a TypeError.
        """
        f1 = TFunc((T_REAL, ), T_REAL)
        f2 = TFunc((T_REAL, T_REAL), T_REAL)
        with pytest.raises(TypeError, match="parameter count mismatch"):
            unify(f1, f2, self.s())

    def test_instance_same_class(self):
        """
        Tests that unifying two instances of the same class succeeds.
        """
        i = TInstance("OneLayerNet")
        result = unify(i, i, self.s())
        assert result == {}

        i = TInstance("FullyConnectedNet")
        result = unify(i, i, self.s())
        assert result == {}

    def test_instance_different_class_raises(self):
        """
        Tests that unifying two instances of different classes raises a
        TypeError.
        """
        with pytest.raises(TypeError, match="Instance mismatch"):
            unify(TInstance("OneLayerNet"), TInstance("FullyConnectedNet"),
                  self.s())


class TestUnifyDim:
    """
    Tests for ``unify_dim``. A dimension-level unification.
    """

    def s(self):
        """
        Helper to create a fresh empty substitution for each test.
        """
        return Substitution()

    def test_int_dims(self):
        """
        Verifies dimension unification of integers.
        """
        result = unify_dim(3, 3, self.s())
        assert result == {}

        with pytest.raises(TypeError, match="Dimension mismatch"):
            unify_dim(3, 4, self.s())

    def test_equal_strings(self):
        """
        Test unify dimensions for symbolic dimensions represented as strings.
        """
        result = unify_dim("n", "n", self.s())
        assert result == {}

        with pytest.raises(TypeError, match="Dimension mismatch"):
            unify_dim("n", "m", self.s())

    def test_tdim_bind(self):
        """
        Test unify TDim("δ0") with 4 binds the symbolic dimension to the
        concrete one.
        """
        s = unify_dim(TDim("δ0"), 4, self.s())
        assert s.apply_dim(TDim("δ0")) == 4
        assert s == {"δ0": 4}

        # TDim as second argument also works
        s = unify_dim(4, TDim("δ0"), self.s())
        assert s.apply_dim(TDim("δ0")) == 4
        assert s == {"δ0": 4}

        # unifying a string with a TDim should bind the TDim to the string
        s = unify_dim("n", TDim("δ0"), self.s())
        assert s.apply_dim(TDim("δ0")) == "n"
        assert s == {"δ0": "n"}

        # unifying two TDim should bind them together
        s = unify_dim(TDim("δ0"), TDim("δ1"), self.s())
        assert s.apply_dim(TDim("δ0")) == s.apply_dim(TDim("δ1"))
        assert s == {"δ0": TDim("δ1")}


class TestBroadcast:
    """
    Tests for ``broadcast_op``.

    Verifies the result type of element-wise binary operations
    (``+``, ``-``, ``*``, ``/``).
    """

    def test_both_scalars(self):
        """scalar OP scalar should return scalar."""
        assert broadcast_op(T_REAL, T_REAL) == T_REAL
        assert broadcast_op(T_NAT, T_NAT) == T_NAT
        assert broadcast_op(T_Z2, T_Z2) == T_Z2

    def test_tensor_wins_over_scalar_left(self):
        """tensor OP scalar should return a tensor."""
        t = TTensor(((3, "invariant"), ))
        assert broadcast_op(t, T_REAL) == t

    def test_tensor_wins_over_scalar_right(self):
        """scalar OP tensor should return a tensor."""
        t = TTensor(((4, "invariant"), ))
        assert broadcast_op(T_REAL, t) == t

    def test_both_tensors_returns_t1(self):
        """tensor OP tensor should return t1."""
        # Shape verified by unify step.
        t1 = TTensor(((3, "invariant"), ))
        t2 = TTensor(((3, "invariant"), ))
        assert broadcast_op(t1, t2) is t1

    def test_none_returns(self):
        """
        Unknown left operand should return propagate t2.
        Unknown right operand should return t1.
        Both unknown should return None.
        """
        t = TTensor(((5, "invariant"), ))
        assert broadcast_op(None, t) == t
        assert broadcast_op(None, T_REAL) == T_REAL

        t = TTensor(((5, "invariant"), ))
        assert broadcast_op(t, None) == t
        assert broadcast_op(T_REAL, None) == T_REAL

        assert broadcast_op(None, None) is None

    def test_2d_tensor_wins_over_scalar(self):
        """2D tensor OP scalar should return 2D tensor."""
        t = TTensor(((2, "invariant"), (3, "invariant")))
        assert broadcast_op(T_REAL, t) == t
        assert broadcast_op(t, T_REAL) == t

        # for higher rank tensors as well
        t = TTensor(((1, "invariant"), (2, "invariant"), (3, "invariant"),
                     (3, "invariant"), (5, "invariant"), (6, "invariant")))
        assert broadcast_op(T_REAL, t) == t
        assert broadcast_op(t, T_REAL) == t


class TestMatmulResult:
    """
    Tests for ``matmul_op``.

    Verifies the inferred result type of ``@``. Tests for batched ND cases,
    symbolic expressions, and mismatch errors.
    """

    def fresh_errors(self):
        """
        Helper function that return an error-accumulating list
        and its append callback for error messages.
        """
        errors = []
        return errors, errors.append

    def test_none_operand(self):
        """
        Either operand None should return None.
        """
        t = TTensor(((3, "invariant"), ))
        errors, cb = self.fresh_errors()
        assert matmul_op(None, t, cb) is None
        assert matmul_op(t, None, cb) is None
        assert matmul_op(None, None, cb) is None
        assert errors == []  # no errors should be added for None cases

    def test_1d_1d(self):
        """
        ℝ[n] @ ℝ[n] should return ℝ where n is a positive integer.
        """
        t = TTensor(((3, "invariant"), ))
        errors, cb = self.fresh_errors()
        assert matmul_op(t, t, cb) == T_REAL
        assert errors == []

    def test_matrix_vector(self):
        """
        ℝ[m,n] @ ℝ[n] should raise an error (mixed rank)
        """
        m, n = 2, 3
        t1 = TTensor(((m, "invariant"), (n, "invariant")))
        t2 = TTensor(((n, "invariant"), ))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) is None
        assert len(errors) == 1
        assert "rank mismatch" in errors[0]
        err_msg = "; use ℝ[%s,%s] @ ℝ[%s,1]" % (m, n, n)
        assert err_msg in errors[0]

        n, p = 3, 4
        t1 = TTensor(((n, "invariant"), ))
        t2 = TTensor(((n, "invariant"), (p, "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) is None
        assert len(errors) == 1
        assert "rank mismatch" in errors[0]
        err_msg = "; use ℝ[1,%s] @ ℝ[%s,%s]" % (n, n, p)
        assert err_msg in errors[0]

    def test_2d_2d(self):
        """
        ℝ[m, n] @ ℝ[n, p] should return ℝ[m, p] (matrix-matrix).
        """
        m, n, p = 2, 3, 4
        t1 = TTensor(((m, "invariant"), (n, "invariant")))
        t2 = TTensor(((n, "invariant"), (p, "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) == make_tensor([m, p])
        assert errors == []

    def test_batched_3d_3d(self):
        """
        ℝ[b, m, n] @ ℝ[b, n, p] should return ℝ[b, m, p]
        """
        b, m, n, p = 5, 2, 3, 4
        t1 = TTensor(((b, "invariant"), (m, "invariant"), (n, "invariant")))
        t2 = TTensor(((b, "invariant"), (n, "invariant"), (p, "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) == make_tensor([b, m, p])
        assert errors == []

        # batch broadcasting:
        # ℝ[1, m, n] @ ℝ[b, n, p] should return ℝ[b, m, p]
        b, m, n, p = 5, 2, 3, 4
        t1 = TTensor(((1, "invariant"), (m, "invariant"), (n, "invariant")))
        t2 = TTensor(((b, "invariant"), (n, "invariant"), (p, "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) == make_tensor([b, m, p])
        assert errors == []

        # ℝ[n] @ ℝ[b,n,p] should raise an error (mixed rank)
        b, n, p = 5, 3, 4
        t1 = TTensor(((n, "invariant"), ))
        t2 = TTensor(((b, "invariant"), (n, "invariant"), (p, "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) is None
        assert len(errors) == 1
        assert "rank mismatch" in errors[0]
        err_msg = "; use ℝ[1,1,%s] @ ℝ[%s,%s,%s]" % (n, b, n, p)
        assert err_msg in errors[0]

        # ℝ[b,m,n] @ ℝ[n] should raise an error (mixed rank)
        b, m, n = 5, 2, 3
        t1 = TTensor(((b, "invariant"), (m, "invariant"), (n, "invariant")))
        t2 = TTensor(((n, "invariant"), ))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) is None
        assert len(errors) == 1
        assert "rank mismatch" in errors[0]

    def test_symbolic_dims_preserved(self):
        """
        ℝ[m,n] @ ℝ[n,p] should return ℝ[m,p] where m,n,p are symbolic types.
        """
        t1 = TTensor(((TDim("m"), "invariant"), (TDim("n"), "invariant")))
        t2 = TTensor(((TDim("n"), "invariant"), (TDim("p"), "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) == make_tensor([TDim("m"), TDim("p")])
        assert errors == []

    def test_dot_product_mismatch(self):
        """
        ℝ[3] @ ℝ[4] should raise an error.
        """
        t1 = TTensor(((3, "invariant"), ))
        t2 = TTensor(((4, "invariant"), ))
        errors, cb = self.fresh_errors()
        result = matmul_op(t1, t2, cb)
        assert result == T_REAL
        assert len(errors) == 1
        assert "3" in errors[0] and "4" in errors[0]
        assert "different dims 3 ≠ 4" in errors[0]

    def test_matrix_matrix_mismatch(self):
        """ℝ[2,3] @ ℝ[4,5] should raise an error."""
        t1 = TTensor(((2, "invariant"), (3, "invariant")))
        t2 = TTensor(((4, "invariant"), (5, "invariant")))
        errors, cb = self.fresh_errors()
        result = matmul_op(t1, t2, cb)
        assert result == make_tensor([2, 5])
        assert len(errors) == 1
        assert "3" in errors[0] and "4" in errors[0]
        assert "different dims 3 ≠ 4" in errors[0]

    def test_batched_mismatch(self):
        """
        ℝ[3,2,4] @ ℝ[5,4,6] should raise an error"""
        t1 = TTensor(((3, "invariant"), (2, "invariant"), (4, "invariant")))
        t2 = TTensor(((5, "invariant"), (4, "invariant"), (6, "invariant")))
        errors, cb = self.fresh_errors()
        matmul_op(t1, t2, cb)
        assert len(errors) == 1
        assert "3" in errors[0] and "5" in errors[0]
        assert "batch dims 3 and 5 are not broadcast-compatible" in errors[0]

    def test_unequal_batch_rank(self):
        """
        ℝ[6,2,3] @ ℝ[4,6,2,3,4] should raise an error
        """
        t1 = TTensor(((6, "invariant"), (2, "invariant"), (3, "invariant")))
        t2 = TTensor(((4, "invariant"), (6, "invariant"), (2, "invariant"),
                      (3, "invariant"), (4, "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) is None
        assert len(errors) == 1
        assert "batch rank mismatch" in errors[0]
        assert "1 vs 3 batch dims, use explicit 1s to broadcast" in errors[0]

    def test_explicit_1_batch_broadcast_valid(self):
        """
        ℝ[1,1,2,3] @ ℝ[4,6,3,4] should return ℝ[4,6,2,4]
        """
        t1 = TTensor(((1, "invariant"), (1, "invariant"), (2, "invariant"),
                      (3, "invariant")))
        t2 = TTensor(((4, "invariant"), (6, "invariant"), (3, "invariant"),
                      (4, "invariant")))
        errors, cb = self.fresh_errors()
        assert matmul_op(t1, t2, cb) == make_tensor([4, 6, 2, 4])
        assert errors == []
