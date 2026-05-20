import pytest

import dataclasses

from physika.utils.types import (TScalar, TTensor, TVar, TDim, TFunc,
                                 TInstance, T_REAL, T_NAT, T_COMPLEX, T_STRING,
                                 VarCounter, Substitution, check_function,
                                 check_statement, check_class)


def make_fdef(params=None, stmts=None, body=None, return_type=None):
    """
    Helper function to build a physika function as dict based on its defintion.
    """
    return {
        "params": params or [],
        "statements": stmts or [],
        "body": body,
        "return_type": return_type,
    }


def make_cdef(class_params=None, fields=None, methods=None):
    """Helper to build a Physika class definition dict."""
    return {
        "class_params": class_params or [],
        "fields": fields or [],
        "methods": methods or [],
    }


def make_method(name, params=None, stmts=None, body=None, return_type=None):
    """Helper function to add a method inside a class definition as a dict."""
    return {
        "name": name,
        "params": params or [],
        "statements": stmts or [],
        "body": body,
        "return_type": return_type,
    }


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


class TestCheckFunction:
    """
    Tests for ``check_function``.
    """

    def test_types_function_params_returns(self):
        """
        Verifies type checker does not report errors for a well typed program.
        """
        # scalar case
        errors = []
        fdef = make_fdef(params=[("x", "ℝ")],
                         body=("var", "x"),
                         return_type="ℝ")
        check_function("f", fdef, {}, {}, errors.append)
        # Physika program:
        # def f(x: ℝ): ℝ
        #   return x
        assert errors == []

        # tensor case
        errors = []
        fdef = make_fdef(
            params=[("v", ("tensor", [(3, "invariant")]))],
            body=("var", "v"),
            return_type=("tensor", [(3, "invariant")]),
        )
        check_function("f", fdef, {}, {}, errors.append)
        assert errors == []

        # A function without a declared return type does not report an error,
        # instead type is inferred so the type checker can proceed to check the
        # remaning program.
        errors = []
        fdef = make_fdef(params=[("x", "ℝ")], body=("var", "x"))
        check_function("f", fdef, {}, {}, errors.append)
        assert errors == []

    def test_return_type_mismatch(self):
        """Declared ℝ[3] but body returns ℝ"""
        errors = []
        fdef = make_fdef(
            params=[("x", "ℝ")],
            body=("var", "x"),
            return_type=("tensor", [(3, "invariant")]),
        )

        check_function("f", fdef, {}, {}, errors.append)

        assert len(errors) == 1
        assert errors[
            0] == "In function 'f': return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

    def test_statement_type_mismatch(self):
        """Body statement v : ℝ[3] = 1.0"""
        errors = []
        bad_stmt = ("body_decl", "v", ("tensor", [(3, "invariant")]), ("num",
                                                                       1.0))
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[bad_stmt],
            body=("var", "x"),
            return_type="ℝ",
        )
        check_function("g", fdef, {}, {}, errors.append)
        assert len(errors) == 1
        assert errors[
            0] == "In function 'g': In 'g': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

    def test_multiple_statement(self):
        """Two consecutive declarations well typed does not report errors."""
        errors = []
        stmts = [
            ("body_decl", "a", "ℝ", ("num", 1.0)),
            ("body_decl", "b", "ℝ", ("add", ("var", "a"), ("num", 2.0))),
        ]
        fdef = make_fdef(stmts=stmts, body=("var", "b"), return_type="ℝ")
        check_function("h", fdef, {}, {}, errors.append)
        assert errors == []

    def test_function_calls(self):
        """Function calls another function in func_env"""
        errors = []
        func_env = {"sq": (["ℝ"], "ℝ")}

        fdef = make_fdef(
            params=[("x", "ℝ")],
            body=("call", "sq", [("var", "x")]),
            return_type="ℝ",
        )
        check_function("apply_sq", fdef, func_env, {}, errors.append)
        # Well typed functions
        # sq recieves and returns a scalar (ℝ)
        # apply_sq recieves an scalar (ℝ) and returns a call of `sq`, which is
        # an scalar (ℝ)
        assert errors == []

        # Bad types (wrong number of arguments)
        errors = []
        func_env = {"sq": (["ℝ"], "ℝ")}
        fdef = make_fdef(
            params=[("x", "ℝ"), ("y", "ℝ")],
            body=("call", "sq", [("var", "x"), ("var", "y")]),
            return_type="ℝ",
        )
        check_function("bad", fdef, func_env, {}, errors.append)
        # Bad typed function
        # sq recieves and returns a scalar (ℝ)
        # bad recieves an two scalars (ℝ, ℝ) and returns a call of `sq`,
        # which is expects one scalar as argument but 2 were args (x, y)
        # were passed.
        assert len(errors) == 1
        assert errors[
            0] == "In function 'bad': Function 'sq' expects 1 args, got 2"

        # Bad types (wrong number of arguments)
        errors = []
        func_env = {"sq": (["ℝ"], "ℝ")}
        fdef = make_fdef(
            params=[("x", "ℝ"), ("y", "ℝ")],
            body=("call", "sq", [("var", "x")]),
            return_type=("tensor", [(3, "invariant")]),
        )
        check_function("bad_types", fdef, func_env, {}, errors.append)
        # Bad typed function
        # sq recieves and returns a scalar (ℝ)
        # bad_types recieves two scalars (ℝ, ℝ) and returns a call of `sq`,
        # which is scalar type (ℝ), but declared return type is ℝ[3].
        assert len(errors) == 1
        assert errors[
            0] == "In function 'bad_types': return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

    def test_if_else_func(self):
        """
        Test cases for if-else stmts type checking inside function bodies.
        """
        # well typed body_if with valid condition
        errors = []
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_if", ("cond_gt", ("var", "x"), ("num", 0.0)),
                    [("body_assign", "y", ("num", 1.0))])],
            body=("var", "x"),
            return_type="ℝ",
        )
        check_function("f", fdef, {}, {}, errors.append)
        assert errors == []

        # body_if_else with valid condition
        errors = []
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_if_else", ("cond_gt", ("var", "x"), ("num", 0.0)), [
                ("body_assign", "y", ("num", 1.0))
            ], [("body_assign", "y", ("num", 0.0))])],
            body=("var", "x"),
            return_type="ℝ",
        )
        check_function("f", fdef, {}, {}, errors.append)
        assert errors == []

        # wrong args number call inside then-branch reports error
        errors = []
        func_env = {"sq": (["ℝ"], "ℝ")}
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_if", ("cond_gt", ("var", "x"), ("num", 0.0)),
                    [("body_assign", "z", ("call", "sq", [("var", "x"),
                                                          ("var", "x")]))])],
            body=("var", "x"),
            return_type="ℝ",
        )
        check_function("g", fdef, func_env, {}, errors.append)
        assert len(errors) == 1
        assert errors[
            0] == "In function 'g': Function 'sq' expects 1 args, got 2"

        # both branches return ℝ, declared return ℝ
        errors = []
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_if_else_return", ("cond_gt", ("var", "x"),
                                            ("num", 0.0)), ("num", 1.0),
                    ("num", 0.0))],
            body=None,
            return_type="ℝ",
        )
        check_function("sign", fdef, {}, {}, errors.append)
        assert errors == []

        # then-branch returns ℝ[2] but else and declared return are ℝ
        # reports two errors
        errors = []
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_if_else_return", ("cond_gt", ("var", "x"), ("num",
                                                                      0.0)),
                    ("array", [("num", 1.0), ("num", 2.0)]), ("num", 0.0))],
            body=None,
            return_type="ℝ",
        )
        check_function("bad", fdef, {}, {}, errors.append)
        assert len(errors) == 2
        assert errors[
            0] == "In function 'bad': if/else branch type mismatch: then=ℝ[2], else=ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501
        assert errors[
            1] == "In function 'bad': if/else return type mismatch: declared ℝ, got ℝ[2]: Cannot unify scalar ℝ with tensor ℝ[2]"  # noqa: E501

    def test_for_loops_func(self):
        """
        Test cases for for-loops stmts type checking inside function bodies.
        """
        # well typed for loop insidde funciton
        errors = []
        func_env = {"f": (["ℝ"], "ℝ")}
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_for", "i", [("body_assign", "z",
                                       ("call", "f", [("var", "x")]))], [])],
            body=("var", "x"),
            return_type="ℝ",
        )
        check_function("outer", fdef, func_env, {}, errors.append)
        assert errors == []

        # well typed for loop range
        errors = []
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_for_range", "k", ("num", 0.0), ("num", 5.0),
                    [("body_assign", "y", ("var", "x"))])],
            body=("var", "x"),
            return_type="ℝ",
        )
        check_function("range_loop", fdef, {}, {}, errors.append)
        assert errors == []

        # type mismatch inside body_for body is reported
        errors = []
        func_env = {"f": ([("tensor", [(3, "invariant")])], "ℝ")}
        # f expects ℝ[3] but receives a scalar inside a for loop
        fdef = make_fdef(
            params=[("x", "ℝ")],
            stmts=[("body_for", "i", [("body_assign", "z",
                                       ("call", "f", [("var", "x")]))], [])],
            body=("var", "x"),
            return_type="ℝ",
        )
        check_function("type_err_loop", fdef, func_env, {}, errors.append)
        assert len(errors) == 1
        assert errors[
            0] == "In function 'type_err_loop': Arg 0 of 'f': Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501


class TestCheckClass:
    """Tests for ``check_class``."""

    def test_constructor_in_env(self):
        """
        Verify that ``check_class`` adds the constructor signature to func_env.
        """
        func_env = {}
        class_env = {}
        cdef = make_cdef(class_params=[("x", "ℝ"), ("y", "ℝ")])
        check_class("Vec", cdef, func_env, class_env, [].append)
        param_types, class_inst = func_env["Vec"]

        assert "Vec" in func_env
        assert len(param_types) == 2  # for x and y
        assert class_inst == TInstance("Vec")

    def test_fields_in_class_env(self):
        """
        Verifies that field are registered in class_env.
        """
        func_env = {}
        class_env = {}
        cdef = make_cdef(class_params=[("x", "ℝ"), ("y", "ℝ")])
        check_class("Vec", cdef, func_env, class_env, [].append)
        field_names = [name for name, _ in class_env["Vec"]["fields"]]

        assert "Vec" in class_env
        assert "x" in field_names
        assert "y" in field_names

    def test_method(self):
        """
        Checks a well typed method (params, stmts and return) does not
        produce errors
        """
        errors = []
        method = make_method(
            "norm_sq",
            body=("add", ("mul", ("field_access", ("var", "this"), "x"),
                          ("field_access", ("var", "this"), "x")),
                  ("mul", ("field_access", ("var", "this"), "y"),
                   ("field_access", ("var", "this"), "y"))),
            return_type="ℝ",
        )
        cdef = make_cdef(
            class_params=[("x", "ℝ"), ("y", "ℝ")],
            methods=[method],
        )
        check_class("Vec", cdef, {}, {}, errors.append)
        assert errors == []

    def test_method_return_type_mismatch(self):
        """
        Method declared return ℝ[2] but body yields ℝ reports an error.
        """
        errors = []
        method = make_method(
            "bad",
            body=("field_access", ("var", "this"), "x"),
            return_type=("tensor", [(2, "invariant")]),
        )
        cdef = make_cdef(
            class_params=[("x", "ℝ"), ("y", "ℝ")],
            methods=[method],
        )
        check_class("Vec", cdef, {}, {}, errors.append)
        assert len(errors) == 1
        print(errors)
        assert errors[
            0] == "In class 'Vec', method 'bad': return type mismatch: declared ℝ[2], got ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501

    def test_method_stmt_type_mismatch(self):
        """
        Body_decl inside a method with a declared type that
        does not match the inferred type reports an error.
        """
        errors = []
        # v : ℝ[3] = this.x  — this.x
        # inferred is ℝ, declared is ℝ[3]
        bad_stmt = ("body_decl", "v", ("tensor", [(3, "invariant")]),
                    ("field_access", ("var", "this"), "x"))
        method = make_method(
            "bad_decl",
            stmts=[bad_stmt],
            body=("field_access", ("var", "this"), "x"),
            return_type="ℝ",
        )
        cdef = make_cdef(class_params=[("x", "ℝ")], methods=[method])
        check_class("Pt", cdef, {}, {}, errors.append)
        assert len(errors) == 1
        assert errors[
            0] == "In class 'Pt', method 'bad_decl': In 'Pt': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

    def test_method_sees_other_class_field(self):
        """
        A method parameter typed as another class instance can have to its
        fields accessed.
        """
        errors = []
        # dot(other: Vec): ℝ:
        #  return this.x * other.x + this.y * other.y
        method = make_method(
            "dot",
            params=[("other", ("struct_type", "Vec"))],
            body=("add", ("mul", ("field_access", ("var", "this"), "x"),
                          ("field_access", ("var", "other"), "x")),
                  ("mul", ("field_access", ("var", "this"), "y"),
                   ("field_access", ("var", "other"), "y"))),
            return_type="ℝ",
        )
        cdef = make_cdef(
            class_params=[("x", "ℝ"), ("y", "ℝ")],
            methods=[method],
        )
        func_env = {"Vec": (["ℝ", "ℝ"], TInstance("Vec"))}
        class_env = {
            "Vec": {
                "fields": [("x", "ℝ"), ("y", "ℝ")],
                "methods": {}
            }
        }

        check_class("Vec", cdef, func_env, class_env, errors.append)
        assert errors == []

    def test_multiple_methods_independent_errors(self):
        """
        Two methods with independent errors each report their own error.
        """
        errors = []
        # declared return ℝ[2] but inferred is ℝ
        m1 = make_method(
            "m1",
            body=("field_access", ("var", "this"), "x"),
            return_type=("tensor", [(2, "invariant")]),
        )
        # declared return ℝ[3] but inferred is ℝ
        m2 = make_method(
            "m2",
            body=("field_access", ("var", "this"), "y"),
            return_type=("tensor", [(3, "invariant")]),
        )

        cdef = make_cdef(
            class_params=[("x", "ℝ"), ("y", "ℝ")],
            methods=[m1, m2],
        )
        check_class("Vec", cdef, {}, {}, errors.append)
        assert len(errors) == 2
        assert errors[
            0] == "In class 'Vec', method 'm1': return type mismatch: declared ℝ[2], got ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501
        assert errors[
            1] == "In class 'Vec', method 'm2': return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501


class TestCheckStatement:
    """Tests for ``check_statement``."""

    def test_var_registers(self):
        """statement adds the declared variable to type_env."""
        # decl case
        errors = []
        env = {}
        check_statement(("decl", "x", "ℝ", ("num", 1.0), 1), env, {}, {},
                        errors.append)
        assert env["x"] == T_REAL
        assert errors == []

        # assing case
        errors = []
        env = {}
        check_statement(("assign", "y", ("num", 3.14), 2), env, {}, {},
                        errors.append)
        assert env["y"] == T_REAL
        assert errors == []

        # expr statement calling a function
        errors = []
        env = {"x": T_REAL}
        func_env = {"sq": (["ℝ"], "ℝ")}
        check_statement(
            ("expr", ("call", "sq", [("var", "x")]), 3),
            env,
            func_env,
            {},
            errors.append,
        )
        assert errors == []

    def test_decl_type_mismatch(self):
        """Declared ℝ[3] but inferred ℝ"""
        errors = []
        env = {}
        lineno = 7
        check_statement(
            ("decl", "v", ("tensor", [(3, "invariant")]),
             ("num", 0.0), lineno),
            env,
            {},
            {},
            errors.append,
        )
        assert len(errors) == 1
        assert errors[0].startswith("Line 7:")
        assert errors[
            0] == "Line 7: Type mismatch for 'v': declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # multiple mismatched decls
        errors = []
        env = {}
        stmts = [
            ("decl", "z", ("tensor", [(1, "invariant")]), ("num", 9.0), 1),
            ("decl", "z", ("tensor", [(2, "invariant")]), ("num", 9.0), 2),
            ("decl", "z", ("tensor", [(3, "invariant")]), ("num", 9.0), 3),
            ("decl", "z", ("tensor", [(5, "invariant")]), ("num", 9.0), 5),
            ("decl", "z", ("tensor", [(7, "invariant")]), ("num", 9.0), 7),
        ]
        for stmt in stmts:
            check_statement(stmt, env, {}, {}, errors.append)
        assert len(errors) == 5
        for error, stmt in zip(errors, stmts):
            assert error.startswith(f"Line {stmt[-1]}:")

    def test_expr_reports_error(self):
        """
        expr calling sq() with two args when it expects one reports an error.
        """
        errors = []
        env = {"x": T_REAL}
        func_env = {"sq": (["ℝ"], "ℝ")}
        check_statement(
            ("expr", ("call", "sq", [("var", "x"), ("var", "x")]), 5),
            env,
            func_env,
            {},
            errors.append,
        )
        assert len(errors) == 1
        assert errors[0] == "Line 5: Function 'sq' expects 1 args, got 2"

    def test_class_constructor(self):
        """
        Physika class from examples Vec has 2 fields, so the next program
        is well tytped.
        class Vec:
            x : ℝ
            y : ℝ
        Vec(1.0, 2.0)
        """
        errors = []
        env = {}
        class_env = {
            "Vec": {
                "fields": [("x", "ℝ"), ("y", "ℝ")],
                "methods": {},
            }
        }
        func_env = {"Vec": (["ℝ", "ℝ"], TInstance("Vec"))}
        check_statement(
            ("expr", ("call", "Vec", [("num", 1.0), ("num", 2.0)]), 10),
            env,
            func_env,
            class_env,
            errors.append,
        )
        assert errors == []

        # wrong number of args
        # calling Vec(1.0) when Vec expects 2 args reports an error
        errors = []
        env = {}
        func_env = {"Vec": (["ℝ", "ℝ"], TInstance("Vec"))}
        check_statement(
            ("expr", ("call", "Vec", [("num", 1.0)]), 4),
            env,
            func_env,
            {},
            errors.append,
        )
        assert len(errors) == 1
        assert errors[0] == "Line 4: Function 'Vec' expects 2 args, got 1"

    def test_decl_stmts_update_env(self):
        """check_statement calls accumulate bindings in type_env."""
        errors = []
        env = {}
        check_statement(("decl", "a", "ℝ", ("num", 1.0), 1), env, {}, {},
                        errors.append)
        check_statement(
            ("decl", "b", "ℝ", ("add", ("var", "a"), ("num", 2.0)), 2),
            env,
            {},
            {},
            errors.append,
        )
        assert env["a"] == T_REAL
        assert env["b"] == T_REAL
        assert errors == []

    def test_if_else_stmt(self):
        """
        Test cases for if-else stmts type checking.
        """
        # Case only if with a valid scalar condition
        errors = []
        env = {"x": T_REAL}
        cond = ("cond_gt", ("var", "x"), ("num", 0.0))

        check_statement(
            ("if_only", cond, [("assign", "y", ("num", 1.0))]),
            env,
            {},
            {},
            errors.append,
        )
        assert errors == []

        # case if_else with valid condition
        errors = []
        env = {"x": T_REAL}
        cond = ("cond_gt", ("var", "x"), ("num", 0.0))
        then_branch = [("assign", "y", ("num", 1.0))]
        else_branch = [("assign", "y", ("num", 0.0))]
        check_statement(
            ("if_else", cond, then_branch, else_branch, 8),
            env,
            {},
            {},
            errors.append,
        )
        assert errors == []

        # if condition referencing an undeclared function reports an error
        errors = []
        env = {}
        cond = ("cond_gt", ("call", "not_defined_func", []), ("num", 0.0))
        check_statement(
            ("if_else", cond, [], [], 10),
            env,
            {},
            {},
            errors.append,
        )
        assert len(errors) == 1
        assert errors[
            0] == "Line 10: unknown is not comparable with ℝ at 'cond_gt' expression"  # noqa: E501

        # number args error inside if-else block
        errors = []
        env = {}
        func_env = {"f": (["ℝ", "ℝ"], "ℝ")}
        # f expects 2 args but gets 1
        cond = ("cond_gt", ("call", "f", [("num", 1.0)]), ("num", 0.0))
        check_statement(
            ("if_else", cond, [], [], 15),
            env,
            func_env,
            {},
            errors.append,
        )
        assert len(errors) == 1
        assert errors[0] == "Line 15: Function 'f' expects 2 args, got 1"

    def test_for_loops_stmt(self):
        """
        Test various cases for for-loops stmts type checking.
        """
        # for loop registers the loop variable as ℕ
        errors = []
        env = {"x": T_REAL}
        check_statement(
            ("for_loop", "i", [("assign", "y", ("var", "x"))], 3),
            env,
            {},
            {},
            errors.append,
        )
        assert errors == []
        assert env["i"] == T_NAT

        # case for loop range
        errors = []
        env = {}
        check_statement(
            ("for_loop_range", "k", ("num", 0.0), ("num", 10.0), [], 7),
            env,
            {},
            {},
            errors.append,
        )
        assert errors == []
        assert env["k"] == T_NAT

        # wrong args nujmber inside a for_loop body is reported
        errors = []
        env = {}
        func_env = {"f": (["ℝ"], "ℝ")}
        # body calls f with wrong args nujmber
        bad_call = ("expr", ("call", "f", [("num", 1.0), ("num", 2.0)]))
        check_statement(
            ("for_loop", "i", [bad_call], 9),
            env,
            func_env,
            {},
            errors.append,
        )
        assert len(errors) == 1
        assert errors[0] == "Line 9: Function 'f' expects 1 args, got 2"

        # type error inside a for_loop body is reported
        errors = []
        env = {}
        func_env = {"f": ([("tensor", [(1, "invariant")])], "ℝ")}
        # body calls f with wrong args nujmber
        bad_call = ("expr", ("call", "f", [("num", 1.0)]))
        check_statement(
            ("for_loop", "i", [bad_call], 10),
            env,
            func_env,
            {},
            errors.append,
        )
        assert len(errors) == 1
        assert errors[
            0] == "Line 10: Arg 0 of 'f': Cannot unify tensor ℝ[1] with scalar ℝ"  # noqa: E501

        # vars declared before the loop are visible inside the loop body.
        errors = []
        env = {"z": T_REAL}
        func_env = {"f": (["ℝ"], "ℝ")}
        body = [("expr", ("call", "f", [("var", "z")]))]
        check_statement(
            ("for_loop", "i", body, 2),
            env,
            func_env,
            {},
            errors.append,
        )
        assert errors == []
