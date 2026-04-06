import pytest
from physika.utils.types import (
    TVar,
    TDim,
    TTensor,
    TFunc,
    TInstance,
    T_REAL,
    T_NAT,
    T_COMPLEX,
    T_STRING,
)
from physika.utils.type_checker_utils import (
    from_typespec,
    occurs_in,
    get_tensor_shape,
    make_tensor,
    unify,
    unify_dim,
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
