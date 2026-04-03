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
)


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
