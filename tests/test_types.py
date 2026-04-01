import pytest

import dataclasses

from physika.utils.types import (TScalar, TTensor, TVar, TDim, TFunc,
                                 TInstance, T_REAL, T_NAT, T_COMPLEX, T_STRING,
                                 VarCounter, Substitution)


class TestHMTypes:
    """
    Tests for the basic HM types: TVar, TDim, TScalar, TTensor, TFunc, and
    TInstance.
    """

    def test_TVar(self):
        """
        TVars are identified by string names "α0", "α1",etc.
        TVars are immutable and hashable, so they can be used as dict keys.
        """
        assert repr(TVar("α0")) == "α0"
        assert repr(TVar("α99")) == "α99"
        assert TVar("α0") == TVar("α0")
        assert TVar("α0") != TVar("α1")

        # substitution dict
        s = {TVar("α0"), TVar("α1"), TVar("α0")}
        assert len(s) == 2

        s = {TVar("α0"), TVar("α1"), TVar("α0")}
        assert len(s) == 2

        with pytest.raises(dataclasses.FrozenInstanceError):
            TVar("α0").name = "α1"  # type: ignore[misc]

    def test_TReal(self):
        """
        T_REAL is a singleton instance of TScalar representing the real
        numbers.
        """
        assert repr(T_REAL) == "ℝ"
        assert T_REAL == TScalar("ℝ")
        assert T_REAL != T_NAT
        assert T_REAL != T_COMPLEX
        assert T_REAL != T_STRING

    def test_TNat(self):
        """
        T_NAT is a singleton instance of TScalar representing the natural
        numbers.
        """
        assert repr(T_NAT) == "ℕ"
        assert T_NAT == TScalar("ℕ")
        assert T_NAT != T_REAL
        assert T_NAT != T_COMPLEX
        assert T_NAT != T_STRING

    def test_TComplex(self):
        """
        T_COMPLEX is a singleton instance of TScalar representing the complex
          numbers.
        """
        assert repr(T_COMPLEX) == "ℂ"
        assert T_COMPLEX == TScalar("ℂ")
        assert T_COMPLEX != T_REAL
        assert T_COMPLEX != T_NAT
        assert T_COMPLEX != T_STRING

    def test_TString(self):
        """
        T_STRING is a singleton instance of TScalar representing strings.
        """
        assert repr(T_STRING) == "string"
        assert T_STRING == TScalar("string")
        assert T_STRING != T_REAL
        assert T_STRING != T_NAT
        assert T_STRING != T_COMPLEX

    def test_TDim(self):
        """
        TDims are identified by string names "δ0", "δ1",etc.
        TDims are immutable and hashable, so they can be used as dict keys.
        """
        d1 = TDim("δ0")
        d2 = TDim("δ1")
        assert repr(d1) == "δ0"
        assert repr(d2) == "δ1"
        assert d1 == TDim("δ0")
        assert d1 != d2

        s = {TDim("δ0"), TDim("δ1"), TDim("δ0")}
        assert len(s) == 2

        with pytest.raises(dataclasses.FrozenInstanceError):
            TDim("δ0").name = "δ1"

        assert repr(TDim("n")) == "n"

    def test_TTensor(self):
        """
        TTensors are identified by their shape, which is a tuple of
        (size, "invariant") pairs.
        """
        t1 = TTensor(((3, "invariant"), (4, "invariant")))
        t2 = TTensor(((3, "invariant"), (4, "invariant")))
        t3 = TTensor(((3, "invariant"), ))
        assert repr(t1) == "ℝ[3,4]"
        assert repr(t3) == "ℝ[3]"
        assert t1 == t2
        assert t1 != t3

        s = {t1, t2, t3}
        assert len(s) == 2

        with pytest.raises(dataclasses.FrozenInstanceError):
            t1.dims = ()  # type: ignore[misc]

        assert repr(TTensor(((TDim("δ0"), "invariant"), ))) == "ℝ[δ0]"
        assert repr(
            TTensor(((TDim("δ0"), "invariant"), (TDim("δ1"),
                                                 "invariant")))) == "ℝ[δ0,δ1]"
        assert repr(TTensor(
            ((TDim("n"), "invariant"), (3, "invariant")))) == "ℝ[n,3]"
        assert repr(
            TTensor(((TDim("n"), "invariant"), (TDim("m"),
                                                "invariant")))) == "ℝ[n,m]"

        with pytest.raises(TypeError, match="TVar"):
            TTensor(((TVar("α0"), "invariant"), ))

        t = TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        assert repr(t) == "ℝ[2,3,4]"

    def test_TFunc(self):
        """
        TFunc is identified by its argument types and return type.
        Two TFuncs are equal if their arg types and ret type are equal.
        """
        f1 = TFunc((TScalar("ℝ"), ), TScalar("ℝ"))
        f2 = TFunc((TScalar("ℝ"), ), TScalar("ℝ"))
        f3 = TFunc((TScalar("ℝ"), TTensor(((3, "invariant"), ))), TScalar("ℝ"))
        assert repr(f1) == "(ℝ) → ℝ"
        assert repr(f3) == "(ℝ, ℝ[3]) → ℝ"
        assert f1 == f2
        assert f1 != f3

        s = {f1, f2, f3}
        assert len(s) == 2

        with pytest.raises(dataclasses.FrozenInstanceError):
            f1.ret = T_NAT

        f4 = TFunc(
            (TTensor(((TDim("n"), "invariant"), )),
             TTensor(((TDim("m"), "invariant"), ))),
            TScalar("ℝ"),
        )
        assert repr(f4) == "(ℝ[n], ℝ[m]) → ℝ"

    def test_TInstance(self):
        """
        TInstances are identified by their class names.
        """
        i1 = TInstance("FullyConnectedNet")
        i2 = TInstance("FullyConnectedNet")
        i3 = TInstance("Net")
        assert repr(i1) == "instance(FullyConnectedNet)"
        assert repr(i3) == "instance(Net)"
        assert i1 == i2
        assert i1 != i3

        s = {i1, i2, i3}
        assert len(s) == 2

        with pytest.raises(dataclasses.FrozenInstanceError):
            i1.class_name = "Net"


class TestVarCounter:
    """
    Tests for VarCounter

    Shared counter for fresh type/dim variables.
    """

    def setup_method(self):
        self.c = VarCounter()

    def test_new_tvar(self):
        v = self.c.new_var()
        assert isinstance(v, TVar)

    def test_new_tdim(self):
        d = self.c.new_dim()
        assert isinstance(d, TDim)

    def test_var_sequential(self):
        c = VarCounter()
        assert c.new_var() == TVar("α0")
        assert c.new_var() == TVar("α1")
        assert c.new_var() == TVar("α2")

    def test_dim_sequential(self):
        c = VarCounter()
        assert c.new_dim() == TDim("δ0")
        assert c.new_dim() == TDim("δ1")

    def test_var_and_dim_counter(self):
        # var and dim draw from the same counter, so names never collide
        c = VarCounter()
        v0 = c.new_var()  # α0
        d1 = c.new_dim()  # δ1
        v2 = c.new_var()  # α2
        assert v0 == TVar("α0")
        assert d1 == TDim("δ1")
        assert v2 == TVar("α2")

    def test_reset_var_dims(self):
        c = VarCounter()
        c.new_var()
        c.new_var()
        c.new_dim()
        c.reset()
        assert c.new_var() == TVar("α0")

        c.new_dim()
        c.reset()
        assert c.new_dim() == TDim("δ0")

    def test_independent_instance(self):
        c1 = VarCounter()
        c2 = VarCounter()
        c1.new_var()
        c1.new_var()

        # c2 starts fresh regardless of c1
        assert c2.new_var() == TVar("α0")

    def test_produced_dims_and_vars_are_unique(self):
        c = VarCounter()
        tvars = [c.new_var() for _ in range(10)]
        assert [v.name for v in tvars] == [f"α{i}" for i in range(10)]
        assert len(set(tvars)) == 10

        c1 = VarCounter()
        tdims = [c1.new_dim() for _ in range(10)]
        assert [d.name for d in tdims] == [f"δ{i}" for i in range(10)]
        assert len(set(tdims)) == 10


class TestSubstitution:
    """
    Tests for Substitution class.

    Substitution represents a mapping from type variables (TVar) and dimension
    variables (TDim) to their bound types/dims.
    """

    def test_substitution(self):
        """
        An empty substitution should return the original type unchanged when
        applied.
        """
        # applying empty substitution to None returns None
        s = Substitution()
        assert s.apply(None) is None
        s = Substitution()

        # test dict access, len, and keys
        s = Substitution({"α0": T_REAL})
        assert s["α0"] == T_REAL
        s = Substitution({"α0": T_REAL, "α1": T_NAT})
        assert len(s) == 2
        assert set(s.keys()) == {"α0", "α1"}

        # applying empty substitution returns the original type unchanged
        assert Substitution().apply(TVar("α0")) == TVar("α0")
        # applying empty substitution to concrete types returns the same type
        assert s.apply(T_REAL) == T_REAL
        assert s.apply(TTensor(((3, "invariant"), ))) == TTensor(
            ((3, "invariant"), ))
        assert s.apply(TFunc((TScalar("ℝ"), ), TScalar("ℝ"))) == TFunc(
            (TScalar("ℝ"), ), TScalar("ℝ"))

    def test_apply_tvar(self):
        """
        Test that applying a substitution to a TVar resolves it to the bound
        type.
        """
        s = Substitution({"α0": T_REAL})
        # bound variable "α0" should have value T_REAL (resolved)
        assert s.apply(TVar("α0")) == T_REAL
        # unbound variable "α1" should be returned as is (not resolved)
        assert s.apply(TVar("α1")) == TVar("α1")

        # apply method follows_chaining substitutions until a concrete type or
        # unbound variable is reached.
        # α1 -> α0 -> ℝ
        s = Substitution({"α0": T_REAL, "α1": TVar("α0")})
        assert s.apply(TVar("α1")) == T_REAL

    def test_apply_ttensor_resolves_dims(self):
        """
        Test that applying a substitution to a ``TTensor`` resolves any bound
        ``TDim`` variables.

        ``TTensor`` dims use ``TDim`` (not ``TVar``) and ``Substitution``
        resolves bound TDims
        """
        s = Substitution({"δ0": 3})
        t = TTensor(((TDim("δ0"), "invariant"), ))
        result = s.apply(t)
        assert result == TTensor(((3, "invariant"), ))

        # applying substitution to TTensor with concrete dims
        # leaves them unchanged
        s = Substitution({"α0": T_REAL})
        t = TTensor(((3, "invariant"), (4, "invariant")))
        assert s.apply(t) == t

        # applying substitution chain to solve bound dims
        s = Substitution({"δ0": TDim("δ1"), "δ1": 5})
        t = TTensor(((TDim("δ0"), "invariant"), ))
        result = s.apply(t)
        assert result == TTensor(((5, "invariant"), ))

    def test_apply_tfunc(self):
        """
        Test that applying a substitution to a ``TFunc`` resolves any bound
        type variables in the parameter types and return type.
        """
        s = Substitution({"α0": T_REAL, "α1": T_NAT})
        f = TFunc((TVar("α0"), ), TVar("α1"))
        result = s.apply(f)
        assert result == TFunc((T_REAL, ), T_NAT)

    def test_apply_tinstance_class(self):
        """
        Test that applying a substitution to a ``TInstance`` leaves it
        unchanged.
        """
        s = Substitution({"α0": T_REAL})
        inst = TInstance("Net")
        assert s.apply(inst) == inst

    def test_apply_env(self):
        """
        Test that applying a substitution to an environment dict applies it to
        all the types in the environment.
        """
        # test for empty env
        s = Substitution({"α0": T_REAL})
        assert s.apply_env({}) == {}

        # test for env with unbound variable
        s = Substitution({"α0": T_REAL})
        env = {"x": TVar("α0"), "y": T_NAT, "z": None}
        result = s.apply_env(env)
        assert result == {"x": T_REAL, "y": T_NAT, "z": None}

    def test_compose_method(self):
        """
        Test that composing two substitutions applies the first substitution
        to the values of the second substitution before merging.
        """
        s1 = Substitution({"α0": T_REAL})
        s2 = Substitution({"α1": TVar("α0")})
        composed = s1.compose(s2)
        assert composed == {"α1": T_REAL, "α0": T_REAL}
        assert composed.apply(TVar("α1")) == T_REAL

        # compose should not modify the original substitutions
        s1 = Substitution({"α0": T_REAL})
        s2 = Substitution({"α1": T_NAT})
        composed = s1.compose(s2)
        assert composed.apply(TVar("α0")) == T_REAL
        assert composed.apply(TVar("α1")) == T_NAT

        # compose should not override other keys.
        s1 = Substitution({"α0": T_NAT})
        s2 = Substitution({"α0": T_REAL})
        composed = s1.compose(s2)
        # s2's α0 is kept
        assert composed.apply(TVar("α0")) == T_REAL

        # test for ``other`` empty
        s1 = Substitution({"α0": T_REAL})
        composed = s1.compose(Substitution())
        assert composed.apply(TVar("α0")) == T_REAL

        # test for self empty
        s2 = Substitution({"α0": T_REAL})
        composed = Substitution().compose(s2)
        assert composed.apply(TVar("α0")) == T_REAL

    def test_apply_dim(self):
        """
        Test for ``apply_dim`` method.

        Verifies applying a substitution to a dimension resolves for bound and
        unbound
        variable.
        """
        # applying substitution to a concrete integer returns the same integer
        s = Substitution({"δ0": 5})
        assert s.apply_dim(3) == 3

        # applying empty substitution to a TDim returns the same TDim
        s = Substitution()
        assert s.apply_dim(TDim("δ0")) == TDim("δ0")

        # applying substitution to a bound integer value
        s = Substitution({"δ0": 4})
        assert s.apply_dim(TDim("δ0")) == 4

        # apply_dim should follow chains of TDim bindings
        s = Substitution({"δ0": TDim("δ1"), "δ1": 7})
        assert s.apply_dim(TDim("δ0")) == 7

        # applying substitution to an unbound TDim returns the same TDim
        s = Substitution({"δ1": 9})
        assert s.apply_dim(TDim("δ0")) == TDim("δ0")
