import pytest

from physika.core.elab.elab import (
    Elab,
    ElabError,
    ElabState,
    find_synthetic_hole_names,
    open_telescope,
    resolve_binder_names,
)
from physika.core.environment import Environment, ConstantInfo
from physika.core.expr import (App, BVar, Const, ForallE, FVar, FVarId, Lam,
                               LetE, Lit, MData, MVar, Proj, Sort, BinderInfo)
from physika.core.level import LMVar, LSucc, LZero
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext, MetaVarKind


class TestOpenTelescope:
    """
    Tests for ``open_telescope``.
    """

    def test_opens_forall_chain_into_fvars(self):
        """
        ``ForallE`` binders becomes FVar, ``current`` ends at
        the chain's final body.
        """
        elab = Elab(Environment())
        real = Const("Real", ())
        pi = ForallE("x", real, real, BinderInfo.DEFAULT)
        inner, fvar_env, _, current = open_telescope(pi, elab)
        assert list(fvar_env) == ["x"]
        assert current == real
        assert inner is not elab

    def test_non_pi_type_opens_nothing(self):
        """
        A non-``ForallE`` type opens zero binders, ``current`` is itself.
        """
        elab = Elab(Environment())
        real = Const("Real", ())
        inner, fvar_env, binder_order, current = open_telescope(real, elab)
        assert fvar_env == {}
        assert binder_order == []
        assert current == real
        assert inner is elab


class TestResolveBinderNames:
    """
    Tests for ``resolve_binder_names``.
    """

    def test_renames_conflict_dim_var(self):
        """
        A dimension variable binder where its display name is same as a non
        dim var binder is renamed to ``__dim_<name>``.
        """
        n_dim_fv = FVar(FVarId("n.0"))
        n_param_fv = FVar(FVarId("n.1"))
        binder_order = [("n", n_dim_fv), ("n", n_param_fv)]
        is_dim_var = [True, False]
        compiled_names, param_order = resolve_binder_names(
            binder_order, is_dim_var)
        assert compiled_names == {n_dim_fv.id: "__dim_n", n_param_fv.id: "n"}
        assert param_order == ["__dim_n", "n"]

    def test_original_name(self):
        """
        A dim var with conflict binder keeps its name.
        """
        m_fv = FVar(FVarId("m.0"))
        compiled_names, param_order = resolve_binder_names([("m", m_fv)],
                                                           [True])
        assert compiled_names == {m_fv.id: "m"}
        assert param_order == ["m"]


class TestFindSyntheticHoleNames:
    """
    Tests for ``find_synthetic_holew_names``.
    """

    def test_finds_unassigned_synthetic_mvar(self):
        """
        An unassigned ``SYNTHETIC`` mvar referenced in ``exprs`` is named.
        """
        elab = Elab(Environment())
        mv = elab.new_mvar("_hole",
                           Const("Real", ()),
                           kind=MetaVarKind.SYNTHETIC)
        assert find_synthetic_hole_names(elab.state.mctx, mv) == ["_hole"]

    def test_ignores_natural_mvar(self):
        """
        A ``NATURAL`` (ordinary) mvar isn't reported as a synthetic hole.
        """
        elab = Elab(Environment())
        mv = elab.new_mvar("_hole", Const("Real", ()))
        assert find_synthetic_hole_names(elab.state.mctx, mv) == []


class TestElabState:
    """
    Tests for ``ElabState``.
    """

    def test_errors_empty_list(self):
        """
        ``errors`` defaults to a fresh empty list when omitted.
        """
        state = ElabState(Environment(), LocalContext(), MetaVarContext())
        assert state.errors == []

        # errors are stored
        errs = ["boom"]
        state = ElabState(Environment(), LocalContext(), MetaVarContext(),
                          errs)
        assert state.errors is errs


class TestElab:
    """
    Tests for ``Elab``.
    """

    def test_binds_fresh_lctx_and_mctx(self):
        """
        Starts with tenv and an empty local/metavariable context.
        """
        env = Environment()
        elab = Elab(env)
        assert elab.state.env is env
        assert elab.infer_type(Lit(1.0)) == Const("Real", ())

    def test_returns_child_elab_with_fresh_fvar(self):
        """
        Returns a new ``Elab`` (not the same instance) where local context
        have FVars bound ta the given type
        """
        elab = Elab(Environment())
        child, x_fv = elab.with_local("x", Const("Real", ()))
        assert child is not elab
        assert child.state.lctx.find(x_fv.id).type == Const("Real", ())

    # tests let binding
    def test_pushes_let_binding(self):
        """
        Test for verifying corrent let binding declaration have type and
        value.
        """
        elab = Elab(Environment())
        real = Const("Real", ())
        child, x_fv = elab.with_let("x", real, Lit(1.0))
        decl = child.state.lctx.find(x_fv.id)
        assert decl.type == real
        assert decl.value == Lit(1.0)

    def test_creates_and_registers_mvar(self):
        """
        Returns a ``MVar`` registered in ``mctx`` at the given type.
        """
        elab = Elab(Environment())
        mv = elab.new_mvar("_hole", Const("Real", ()))
        assert isinstance(mv, MVar)
        assert elab.state.mctx.find_decl(mv.id).type == Const("Real", ())

    def test_creates_lmvar(self):
        """
        Returns a universe level metavariable not registered in metavar context
        """
        elab = Elab(Environment())
        lv = elab.new_level_mvar()
        assert isinstance(lv, LMVar)

    # test unifiy method
    def test_equal_consts_unify(self):
        """
        Two structurally equal terms unify.
        """
        elab = Elab(Environment())
        assert elab.unify(Const("Real", ()), Const("Real", ())) is True
        # Two different consts fail to unify, no mctx mutation on failure.

        elab = Elab(Environment())
        assert elab.unify(Const("Real", ()), Const("Nat", ())) is False

        # A unify against a MVar solves in ``mctx``.

        elab = Elab(Environment())
        mv = elab.new_mvar("_h", Const("Real", ()))
        assert elab.unify(mv, Lit(1.0)) is True

        assert elab.state.mctx.instantiate_mvars(mv) == Lit(1.0)

    # tests infer_type method
    def test_literal_types(self):
        """
        A float ``Lit`` infers ``Real``, an int ``Lit`` infers ``Nat``.
        """
        elab = Elab(Environment())
        assert elab.infer_type(Lit(1.0)) == Const("Real", ())
        assert elab.infer_type(Lit(1)) == Const("Nat", ())

    def test_sort_bumps_universe_level(self):
        """
        ``Sort(l)``'s type is ``Sort(l + 1)``.
        """
        elab = Elab(Environment())
        assert elab.infer_type(Sort(LZero())) == Sort(LSucc(LZero()))

    def test_bvar_raises(self):
        """
        A loose ``BVar`` (an unopened binder reference) raises.
        """
        elab = Elab(Environment())
        with pytest.raises(ElabError):
            elab.infer_type(BVar(0))

    def test_fvar_returns_its_declared_type(self):
        """
        A bound FVar's type is looked up from the local context.
        """
        elab = Elab(Environment())
        elab, x_fv = elab.with_local("x", Const("Real", ()))
        assert elab.infer_type(x_fv) == Const("Real", ())

    def test_unbound_fvar_raises(self):
        """
        A FVar not present in the local context raises.
        """
        elab = Elab(Environment())
        with pytest.raises(ElabError):
            elab.infer_type(FVar(FVarId("missing.0")))

    def test_mvar_from_a_different_mctx_raises(self):
        """
        A MVar minted by one Elab isn't declared in another's ``mctx``.
        """
        elab_a = Elab(Environment())
        mv = elab_a.new_mvar("_h", Const("Real", ()))
        elab_b = Elab(Environment())
        with pytest.raises(ElabError):
            elab_b.infer_type(mv)

    def test_unregistered_const_raises(self):
        """
        A ``Const`` naming an unregistered constant raises.
        """
        elab = Elab(Environment())
        with pytest.raises(ElabError):
            elab.infer_type(Const("Undefined", ()))

    def test_registered_const_returns_its_type(self):
        """
        A registered constant's type is accessed fro m evironment.
        """
        env = Environment()
        env.add_constant(ConstantInfo("pi", (), Const("Real", ()), None))
        elab = Elab(env)
        assert elab.infer_type(Const("pi", ())) == Const("Real", ())

    def test_app_instantiates_pi_body_with_argument(self):
        """
        An application's type is the Pi's body, with argument substituted.
        """
        env = Environment()
        real = Const("Real", ())
        env.add_constant(
            ConstantInfo("double", (), ForallE("x", real, real), None))
        elab = Elab(env)
        result = elab.infer_type(App(Const("double", ()), Lit(1.0)))
        assert result == real

    def test_lam_infers_forall_type(self):
        """
        A lambda's type is a ``ForallE`` whose codomain is the body's
        inferred type.
        """
        elab = Elab(Environment())
        real = Const("Real", ())
        identity = Lam("x", real, BVar(0), BinderInfo.DEFAULT)
        result = elab.infer_type(identity)
        assert isinstance(result, ForallE)
        assert result.binder_type == real
        assert result.body == real

    def test_lete_reduces_before_inferring(self):
        """
        A ``LetE``'s type is its body's type after substituting the value.
        """
        elab = Elab(Environment())
        let_expr = LetE("x", Const("Real", ()), Lit(1.0), BVar(0))
        assert elab.infer_type(let_expr) == Const("Real", ())

    def test_mdata_infers(self):
        """
        ``MData`` is transparent to type inference.
        """
        elab = Elab(Environment())
        assert elab.infer_type(MData((("line", 1), ),
                                     Lit(1.0))) == Const("Real", ())

    def test_proj_on_unregistered_struct_raises(self):
        """
        A ``Proj`` on a struct type not registered in the environment
        raises.
        """
        elab = Elab(Environment())
        with pytest.raises(ElabError):
            elab.infer_type(Proj("Ray", 0, Const("ray", ())))

    # tests Elab.check emthod
    def test_matching_type(self):
        """
        A well typed expr against its inferred type is silent.
        """
        elab = Elab(Environment())
        assert elab.check(Lit(1.0), Const("Real", ())) is None

        # a type mismatch shoud raise an error
        elab = Elab(Environment())
        with pytest.raises(ElabError):
            elab.check(Lit(1.0), Const("Nat", ()))

    # tests Elab.elaborate() emthod
    def test_elaborates_function_signature_and_body(self):
        """
        A single parameter identity function's Pi-type and body are
        elaborated and verified wit kernel.
        """
        elab = Elab(Environment())
        func_def = {
            "params": [("x", "ℝ")],
            "return_type": "ℝ",
            "body": ("var", "x"),
            "statements": []
        }
        result = elab.elaborate({
            "functions": {
                "id_real": func_def
            },
            "classes": {},
            "program": []
        })
        assert result["errors"] == []
        pi = result["functions"]["id_real"]
        assert isinstance(pi, ForallE)
        assert pi.binder_name == "x"
        assert pi.binder_type == Const("Real", ())
        assert pi.body == Const("Real", ())
        assert "id_real" in result["resolved_bodies"]

    # tests save and restore methods
    def test_restore_undoes_a_solved_mvar(self):
        """
        Restoring an earlier snapshot undoes an mvar solved after it.
        """
        elab = Elab(Environment())
        mv = elab.new_mvar("_h", Const("Real", ()))
        snap = elab.save()
        assert elab.unify(mv, Lit(1.0)) is True
        assert elab.state.mctx.instantiate_mvars(mv) == Lit(1.0)
        elab.restore(snap)
        assert elab.state.mctx.instantiate_mvars(mv) == mv
