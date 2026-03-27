import pytest

import dataclasses

from physika.utils.types import (TScalar, TTensor, TVar, TDim, TFunc,
                                 TInstance, T_REAL, T_NAT, T_COMPLEX, T_STRING,
                                 VarCounter)


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
