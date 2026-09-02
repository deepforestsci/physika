import pytest
from physika.core.elab.dim_typespec import (
    _NAT_CONST,
    _REAL_CONST,
    _VEC_CONST,
    _NAT_ADD,
    flatten_nat_chain,
    bvar_resolver,
    canon_nat_shape,
    collect_dim_vars_ordered,
    dim_bvar,
    dim_leaf_names,
    dim_to_cic_resolved,
    elaborate_func_type,
    elaborate_method_type,
    elaborate_struct_kind_and_ctor,
    nat_lit_int,
    struct_field_names,
    typespec_to_cic,
    typespec_to_cic_resolved,
)
from physika.core.environment import ConstantInfo, Environment, InductiveInfo
from physika.core.expr import (App, BVar, BinderInfo, Const, ForallE, Lit,
                               MVar, MVarId, NatLit, Sort)
from physika.core.inductive import Constructor, InductiveDecl
from physika.core.level import LSucc, LZero


class _StubElab:
    """
    Minimal stand-in for the elaborator, exposing only ``new_mvar``.
    """

    def new_mvar(self, name: str, type_: object) -> MVar:
        return MVar(MVarId(name))


def register_struct(env: Environment, class_name: str) -> None:
    """
    Register a simple inductive type in environment for testing.
    """
    kind, ctor_type, num_params, _ = elaborate_struct_kind_and_ctor(
        class_name, [("n", "ℝ"), ("m", "ℝ")])
    ctor_name = f"{class_name}.mk"
    decl = InductiveDecl(
        name=class_name,
        level_params=(),
        num_params=num_params,
        type=kind,
        constructors=(Constructor(ctor_name, ctor_type), ),
        is_recursive=False,
    )
    ctor_ci = ConstantInfo(ctor_name, (), ctor_type, None)
    rec_ci = ConstantInfo(f"{class_name}.rec", ("u", ), kind, None)
    env.add_inductive(
        InductiveInfo(decl=decl, ctors={ctor_name: ctor_ci}, recursor=rec_ci))


class TestDimLeafNames:
    """
    Tests for ``dim_leaf_names``.
    """

    def test_dim_leafs(self):
        """
        Verifies dim_leaf_names function works properly for different inputs.
        """
        # int literal has no leaves
        assert dim_leaf_names(3) == []

        assert dim_leaf_names("n") == ["n"]
        assert dim_leaf_names(("mul_dim", "n", 2)) == ["n"]

        # arithmetic tags
        assert dim_leaf_names(("add_dim_id", "n", "m")) == ["n", "m"]
        assert dim_leaf_names(
            ("mul_dim", ("add_dim_id", "n", "m"), 2)) == ["n", "m"]

        # unkown should reutn empty list
        assert dim_leaf_names(("unknown_tag", "n", "m")) == []


class TestCollectDimVarsOrdered:
    """
    Tests for ``collect_dim_vars_ordered``.
    """

    def test_shared_dim_var_across_two_params(self):
        """
        Two function parameters that share dim var should detect this "n" var
        """
        assert collect_dim_vars_ordered(
            [("u", ("tensor", [("n", "invariant")])),
             ("v", ("tensor", [("n", "invariant")]))],
            "ℝ",
        ) == ["n"]
        # non dim var shoudl return ewmpt list
        assert collect_dim_vars_ordered([("x", "ℝ")], "ℝ") == []

    def test_appearance_order_params_and_return(self):
        """
        Verifies dim vars are collected in order of appeareance
        """
        assert collect_dim_vars_ordered(
            [("A", ("tensor", [("n", "invariant"), ("m", "invariant")]))],
            ("tensor", [("n", "invariant")]),
        ) == ["n", "m"]


class TestDimBvar:
    """
    Tests for ``dim_bvar``.
    """

    def test_resolves_to_bvar_at_correct_depth(self):
        """
        Checks dim_bvar gets the correct BVar de bruijn index.
        """
        assert dim_bvar("n", ["n", "v"], depth=1) == BVar(0)
        assert dim_bvar("v", ["n", "v"], depth=2) == BVar(0)
        assert dim_bvar("n", ["n", "v"], depth=2) == BVar(1)

        # should raise a ValueError
        with pytest.raises(ValueError):
            # z not a paramater
            dim_bvar("z", ["n", "v"], depth=2)


class TestDimToCiCResolved:
    """
    Tests for ``dim_to_cic_resolved``.
    """

    def test_correct_resolved_CIC_term(self):
        """
        Verifies ``dim_to_cic_resolved`` produces the correcrt CIC terms for
        Lit, BVars, and inductive type recursive rules.
        """
        assert dim_to_cic_resolved(3, lambda name: None) == Lit(3)
        # produces the correct BVar
        assert dim_to_cic_resolved(
            "n", lambda name: {"n": BVar(0)}.get(name)) == BVar(0)

        # not recognized dim var returns None
        assert dim_to_cic_resolved("missing", lambda name: None) is None

        # arithmetic op with dim var
        assert dim_to_cic_resolved(
            ("mul_dim", "n", 2), lambda name: {"n": BVar(0)}.get(name)) == App(
                App(Const("Nat.mul", ()), BVar(0)), Lit(2))

        assert dim_to_cic_resolved(
            ("add_dim", "n", 1), lambda name: {"n": BVar(0)}.get(name)) == App(
                App(_NAT_ADD, BVar(0)), Lit(1))

        # two dim vars
        result = dim_to_cic_resolved(
            ("add_dim_id", "n", "m"),
            lambda name: {
                "n": BVar(1),
                "m": BVar(0)
            }.get(name),
        )
        assert result == App(App(_NAT_ADD, BVar(1)), BVar(0))


class TestNatLitInt:
    """
    Tests for ``nat_lit_int``.
    """

    def test_int_literal(self):
        """A non negative ``Lit(int)`` returns its value."""
        assert nat_lit_int(Lit(0)) == 0
        assert nat_lit_int(Lit(7)) == 7
        assert nat_lit_int(Lit(NatLit(4))) == 4

        # non literal expression shoudl return none
        assert nat_lit_int(BVar(0)) is None
        assert nat_lit_int(App(App(_NAT_ADD, BVar(0)), Lit(1))) is None

    def test_negative_and_bool_rejected(self):
        """Negative ints and ``bool`` are not accepted as Nat literals."""
        assert nat_lit_int(Lit(-1)) is None
        assert nat_lit_int(Lit(True)) is None


class TestFlattenNatChain:
    """
    Tests for ``flatten_nat_chain``.
    """

    def test_flattens_nested_chain(self):
        """``(a + b) + c`` -> ``[a, b, c]``."""
        add = Const("Nat.add", ())
        chain = App(App(add, App(App(add, BVar(0)), BVar(1))), BVar(2))
        acc = []
        flatten_nat_chain(chain, "Nat.add", acc)
        assert acc == [BVar(0), BVar(1), BVar(2)]

    def test_non_matching_op_is_a_single_leaf(self):
        """A chain of a *different* operator."""
        mul = Const("Nat.mul", ())
        expr = App(App(mul, BVar(0)), BVar(1))
        acc = []
        flatten_nat_chain(expr, "Nat.add", acc)
        assert acc == [expr]

    def test_atom_is_a_single_leaf(self):
        """A non application node is appended."""
        acc = []
        flatten_nat_chain(BVar(3), "Nat.add", acc)
        assert acc == [BVar(3)]


class TestTypespecToCicResolved:
    """
    Tests for ``typespec_to_cic_resolved``.
    """

    def test_get_correct_CIC_term_from_typespec(self):
        """
        Evaluate if two typespec resolves to get correct CIC terms
        """
        assert typespec_to_cic_resolved("ℝ", lambda name: None) == _REAL_CONST
        assert typespec_to_cic_resolved("ℕ", lambda name: None) == _NAT_CONST
        assert typespec_to_cic_resolved("ℤ", lambda name: None) == _NAT_CONST

        # Test for tensor type (1D)
        result = typespec_to_cic_resolved(
            ("tensor", [("n", "invariant")]),
            lambda name: {"n": BVar(0)}.get(name),
        )
        assert result == App(App(_VEC_CONST, _REAL_CONST), BVar(0))

        # Test for tensor type (2D)
        result = typespec_to_cic_resolved(
            ("tensor", [("m", "invariant"), ("n", "invariant")]),
            lambda name: {
                "m": BVar(1),
                "n": BVar(0)
            }.get(name),
        )
        inner = App(App(_VEC_CONST, _REAL_CONST), BVar(0))
        assert result == App(App(_VEC_CONST, inner), BVar(1))

        # A 0 dim tensor is a scalar
        assert typespec_to_cic_resolved(("tensor", []),
                                        lambda name: None) == _REAL_CONST

        # for an unassigned dim var an error is catched, but resolves to 0 so
        # inference continues
        result = typespec_to_cic_resolved(
            ("tensor", [("n", "invariant")]),
            lambda name: None,
        )
        assert result == App(App(_VEC_CONST, _REAL_CONST), Lit(0))

        # physika classes
        assert typespec_to_cic_resolved(("struct_type", "Box"),
                                        lambda name: None) == Const("Box", ())

        # multivalue return type
        result = typespec_to_cic_resolved(
            ("tuple_type", ["ℝ", "ℝ"]),
            lambda name: None,
        )
        assert result == App(App(Const("Prod", ()), _REAL_CONST), _REAL_CONST)

    def test_canon_nat_shape_rewrites(self):
        """
        ``canon_nat_shape`` folds literals and drop identities
        so equal shapes are the same ``Expr``.
        """
        add = Const("Nat.add", ())
        mul = Const("Nat.mul", ())
        sub = Const("Nat.sub", ())

        # constant folding
        assert canon_nat_shape(App(App(add, Lit(3)), Lit(1))) == Lit(4)
        assert canon_nat_shape(App(App(mul, Lit(3)), Lit(4))) == Lit(12)
        assert canon_nat_shape(App(App(sub, Lit(3)), Lit(5))) == Lit(0)

        # identities
        assert canon_nat_shape(App(App(add, Lit(0)), BVar(0))) == BVar(0)
        assert canon_nat_shape(App(App(add, BVar(0)), Lit(0))) == BVar(0)
        assert canon_nat_shape(App(App(sub, BVar(0)), Lit(0))) == BVar(0)
        assert canon_nat_shape(App(App(mul, BVar(0)), Lit(1))) == BVar(0)
        assert canon_nat_shape(App(App(mul, BVar(0)), Lit(0))) == Lit(0)

        # m + n == n + m
        m_n = App(App(add, BVar(1)), BVar(0))
        n_m = App(App(add, BVar(0)), BVar(1))
        assert canon_nat_shape(m_n) == canon_nat_shape(n_m)

        # folded literal stays on the right
        assert canon_nat_shape(App(App(add, Lit(1)),
                                   BVar(0))) == App(App(add, BVar(0)), Lit(1))
        assert canon_nat_shape(App(App(add, BVar(0)),
                                   Lit(1))) == App(App(add, BVar(0)), Lit(1))

    def test_canon_nat_shapes(self):
        """Applying ``canon_nat_shape`` twice equals applying it once."""
        add = Const("Nat.add", ())
        exprs = [
            App(App(add, BVar(1)), BVar(0)),
            App(App(add, Lit(0)), BVar(0)),
            App(App(add, App(App(add, Lit(2)), BVar(0))), Lit(1)),
        ]
        for e in exprs:
            once = canon_nat_shape(e)
            assert canon_nat_shape(once) == once

    def test_equal_shapes_produce_same_cic(self):
        """
        ``ℝ[m+n]`` and ``ℝ[n+m]`, and ``ℝ[0+n]`` vs ``ℝ[n]`` should resolve to
        the identical CIC ``Expr`` after canonicalization.
        """
        resolve = lambda name: {  # noqa: E731
            "m": BVar(1),
            "n": BVar(0)
        }.get(name)  # noqa: E731
        mn = typespec_to_cic_resolved(
            ("tensor", [(("add_dim_id", "m", "n"), "invariant")]), resolve)
        nm = typespec_to_cic_resolved(
            ("tensor", [(("add_dim_id", "n", "m"), "invariant")]), resolve)
        assert mn == nm

        zn = typespec_to_cic_resolved(
            ("tensor", [(("add_dim", "n", 0), "invariant")]), resolve)
        just_n = typespec_to_cic_resolved(("tensor", [("n", "invariant")]),
                                          resolve)
        assert zn == just_n


class TestBvarResolver:
    """
    Tests for ``bvar_resolver``.
    """

    def test_bvar_resolves(self):
        """
        Tests BVar resolver returns a correct function that solves the depth
        of correct given dim var.
        """
        # resolves ordinary binder
        resolve = bvar_resolver(["n", "v"], depth=1)
        assert resolve("n") == BVar(0)

        # reutn_only_mvars are prioritzed
        mvar_stub = object()
        resolve = bvar_resolver(["v"],
                                depth=1,
                                return_only_mvars={"n": mvar_stub})
        assert resolve("n") is mvar_stub

        # if return_only_mvars is not registeres, fallback to BVar
        resolve = bvar_resolver(["n", "v"],
                                depth=2,
                                return_only_mvars={"other": object()})
        assert resolve("v") == BVar(0)


class TestTypespecToCic:
    """
    Tests for ``typespec_to_cic``.
    """

    def test_tensor_spec(self):
        """
        Checks the correct binder position of resolved CIC from typespec is
        given
        """
        # simple case for "ℝ"
        assert typespec_to_cic("ℝ", [], depth=0) == _REAL_CONST
        result = typespec_to_cic(
            ("tensor", [("n", "invariant")]),
            ["n", "v"],
            depth=1,
        )
        assert result == App(App(_VEC_CONST, _REAL_CONST), BVar(0))


class TestElaborateFuncType:
    """
    Tests for ``elaborate_func_type``.
    """

    def test_elaborate_pi_function_type(self):
        """
        Checks proper elaboration of dependent and non dependent function
        types.
        """
        # non-dependent
        elab = _StubElab()
        func_def = {"params": [("x", "ℝ")], "return_type": "ℝ"}
        result = elaborate_func_type(func_def, elab)
        assert result == ForallE(
            "x",
            _REAL_CONST,
            _REAL_CONST,
            BinderInfo.DEFAULT,
        )

        # dependent type
        elab = _StubElab()
        func_def = {
            "params": [("v", ("tensor", [("n", "invariant")]))],
            "return_type": "ℝ",
        }
        result = elaborate_func_type(func_def, elab)
        assert isinstance(result, ForallE)
        assert result.binder_name == "n"
        assert result.binder_type == _NAT_CONST
        assert result.binder_info == BinderInfo.IMPLICIT
        assert result.body == ForallE(
            "v",
            App(App(_VEC_CONST, _REAL_CONST), BVar(0)),
            _REAL_CONST,
            BinderInfo.DEFAULT,
        )

        # `n` only appears in the return type, not in any param
        elab = _StubElab()
        func_def = {
            "params": [],
            "return_type": ("tensor", [("n", "invariant")])
        }
        result = elaborate_func_type(func_def, elab)
        assert result.func == App(_VEC_CONST, _REAL_CONST)
        assert isinstance(result.arg, MVar)


class TestElaborateStructKindField:
    """
    Tests for ``elaborate_struct_kind_and_ctor`` and ``struct_field_names``.
    """

    def test_elab_structor(self):
        """
        Verifies correct elaboration of struct into its class name,
        constructor type, number of parameters, dimension variable, and fields
        name.
        """
        kind, ctor_type, num_params, dim_vars = elaborate_struct_kind_and_ctor(
            "Box", [("x", "ℝ")])
        assert kind == Sort(LSucc(LZero()))
        assert ctor_type == ForallE(
            "x",
            _REAL_CONST,
            Const("Box", ()),
            BinderInfo.DEFAULT,
        )
        assert num_params == 0
        assert dim_vars == []

        # dependent class should gets nat param and implicit_binder
        kind, ctor_type, num_params, dim_vars = elaborate_struct_kind_and_ctor(
            "Vec", [("v", ("tensor", [("n", "invariant")]))])
        assert num_params == 1
        assert dim_vars == ["n"]
        assert kind == ForallE(
            "n",
            _NAT_CONST,
            Sort(LSucc(LZero())),
            BinderInfo.DEFAULT,
        )
        assert isinstance(ctor_type, ForallE)
        assert ctor_type.binder_name == "n"
        assert ctor_type.binder_info == BinderInfo.IMPLICIT

        # checks fields names
        env = Environment()
        register_struct(env, "Box")
        assert struct_field_names("Box", env) == ["n", "m"]

        assert struct_field_names("NoSuchType", Environment()) is None


class TestElaborateMethodType:
    """
    Tests for ``elaborate_method_type``.
    """

    def test_class_method_type_elab(self):
        """
        Checks a class's method type is properly elaborated.
        """
        # non dependent_class prepends "this"
        method_def = {"params": [("x", "ℝ")], "return_type": "ℝ"}
        result = elaborate_method_type("Box", 0, [], method_def)
        assert result == ForallE(
            "this",
            Const("Box", ()),
            ForallE("x", _REAL_CONST, _REAL_CONST, BinderInfo.DEFAULT),
            BinderInfo.DEFAULT,
        )

        # "this" should carry class dim vars
        method_def = {"params": [], "return_type": "ℝ"}
        result = elaborate_method_type("Vec", 1, ["n"], method_def)

        assert result.binder_name == "n"
        assert result.binder_info == BinderInfo.IMPLICIT
        this_binder = result.body

        assert this_binder.binder_name == "this"
        assert this_binder.binder_type == App(Const("Vec", ()), BVar(0))
        assert this_binder.body == _REAL_CONST

        # test method's dim var reuse class dim var name does not shadows
        method_def = {
            "params": [],
            "return_type": ("tensor", [("n", "invariant")])
        }
        result = elaborate_method_type("Vec", 1, ["n"], method_def)
        this_binder = result.body
        method_n_binder = this_binder.body
        assert method_n_binder.binder_name == "n"
        assert method_n_binder.binder_info == BinderInfo.IMPLICIT
        # References the outer class level "n" (BVar(2)), not the
        # method level one (which would be BVar(0)).
        assert method_n_binder.body == App(App(_VEC_CONST, _REAL_CONST),
                                           BVar(2))
