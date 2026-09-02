from physika.core.expr import App, BVar, Const, FVar, FVarId, MVarId
from physika.core.level import LMVar, LZero, LSucc
from physika.core.local_context import LocalContext
from physika.core.metavar import (
    MetaVarContext,
    MetaVarContextState,
    MetaVarDecl,
    MetaVarKind,
    LMVarId,
    children,
    collect_all_mvars,
    collect_fvars_rec,
    expr_has_level_mvar,
    fresh_mvar_id,
)
from physika.core.expr import Sort

NAT = Const("Nat", ())


class TestFreshMvarId:
    """
    Tests for ``fresh_mvar_id``
    """

    def test_correct_hint(self):
        """
        Correct hint id uses ".".
        """
        mv_id = fresh_mvar_id("n")
        assert mv_id.id.startswith("n.")

    def test_unique_ids(self):
        """
        Consecutive calls with the same hint produce unique ids.
        """
        a = fresh_mvar_id("n")
        b = fresh_mvar_id("n")
        assert a.id != b.id


class TestLMVarId:
    """
    Tests two ``LMVarId`` instances.
    """

    def test_compare_ids(self):
        """
        Compare different and same ids for equality and hashing.
        """
        assert LMVarId("u.0") == LMVarId("u.0")
        assert LMVarId("u.0") != LMVarId("u.1")
        assert hash(LMVarId("u.0")) == hash(LMVarId("u.0"))

        # ids works as dict keys
        d = {}
        d[LMVarId("u.0")] = LZero()
        assert d.get(LMVarId("u.0")) == LZero()
        assert d.get(LMVarId("u.1")) is None


class TestMetaVarDecl:
    """
    Tests for ``MetaVarDecl`` construction.
    """

    def test_stores_all_fields(self):
        """
        Tests that all fields are stored correctly when declaring a new
        metavarioable in ``MetaVarDecl``.
        """
        decl = MetaVarDecl(
            mvar_id=MVarId("n.0"),
            user_name="n",
            lctx=LocalContext(),
            type=NAT,
            depth=2,
            kind=MetaVarKind.SYNTHETIC,
        )

        assert decl.mvar_id == MVarId("n.0")
        assert decl.user_name == "n"
        assert decl.type == NAT
        assert decl.depth == 2
        assert decl.kind is MetaVarKind.SYNTHETIC


class TestMetaVarContextState:
    """
    Tests for ``MetaVarContextState`` construction.
    """

    def test_stores_both_assignment_maps(self):
        """
        Tests that both expr_assignments and level_assignments are stored
        correctly
        """
        state = MetaVarContextState(
            expr_assignments={"m.0": NAT},
            level_assignments={LMVarId("u.0"): LZero()},
        )

        assert state.expr_assignments["m.0"] == NAT
        assert state.level_assignments[LMVarId("u.0")] == LZero()


class TestMetaVarContext:
    """
    Tests for ``MetaVarContext construction and operation with methods
    """

    def test_find_decl_mk_var(self):
        """
        Tests for .mk_var() and .find_decl() methods
        """

        mctx = MetaVarContext()
        lctx = LocalContext()
        mv = mctx.mk_mvar("n", lctx, NAT)
        decl = mctx.find_decl(mv.id)

        assert decl.user_name == "n"
        assert decl.lctx is lctx
        assert decl.type == NAT
        # default to NAT kind
        assert mctx.find_decl(mv.id).kind is MetaVarKind.NATURAL

        # test explicit kind is stored
        mv = mctx.mk_mvar("n", LocalContext(), NAT, kind=MetaVarKind.SYNTHETIC)

        assert mctx.find_decl(mv.id).kind is MetaVarKind.SYNTHETIC

        # successive calls should produce unique mvars
        a = mctx.mk_mvar("n", lctx, NAT)
        b = mctx.mk_mvar("n", lctx, NAT)
        assert a.id.id != b.id.id

        # cehcks explicit depth overrides context depth
        mctx.depth = 3
        mv = mctx.mk_mvar("n", LocalContext(), NAT, depth=0)
        assert mctx.find_decl(mv.id).depth == 0

        # find registered decl
        mv = mctx.mk_mvar("n", LocalContext(), NAT)
        assert mctx.find_decl(mv.id) is not None

        mctx = MetaVarContext()
        # returns None for unregistered mvar_id
        assert mctx.find_decl(MVarId("missing.0")) is None

    def test_instantiate_mvars_inst_expr(self):
        """
        Tests for ``MetaVarContext.instantiate_mvars`` and ``inst_expr``.
        """
        mctx = MetaVarContext()
        lctx = LocalContext()
        m = mctx.mk_mvar("m", lctx, NAT)
        n = mctx.mk_mvar("n", lctx, NAT)
        mctx.expr_assignments[n.id.id] = Const("Nat.zero", ())
        mctx.expr_assignments[m.id.id] = n

        assert mctx.instantiate_mvars(m) == Const("Nat.zero", ())

        # unassigned mvars should be returned unchanged
        mctx = MetaVarContext()
        m = mctx.mk_mvar("m", LocalContext(), NAT)

        assert mctx.instantiate_mvars(m) == m

        # no mvar present should return the same object
        term = App(Const("f", ()), Const("a", ()))
        assert mctx.instantiate_mvars(term) is term

        mctx = MetaVarContext()
        m = mctx.mk_mvar("m", LocalContext(), NAT)
        mctx.expr_assignments[m.id.id] = Const("Nat.zero", ())
        term = App(Const("f", ()), m)

        result = mctx.instantiate_mvars(term)
        # only the right branch should be replaced
        assert result == App(Const("f", ()), Const("Nat.zero", ()))
        assert result is not term

        mctx.level_assignments[LMVarId("u.0")] = LZero()
        # Sort levels are preserved
        assert mctx.instantiate_mvars(Sort(LMVar("u.0"))) == Sort(LZero())

    def test_instantiate_level_mvars(self):
        """
        Tests for ``MetaVarContext.instantiate_level_mvars``.
        """
        mctx = MetaVarContext()
        mctx.level_assignments[LMVarId("v.0")] = LZero()
        mctx.level_assignments[LMVarId("u.0")] = LMVar("v.0")

        assert mctx.instantiate_level_mvars(LMVar("u.0")) == LZero()

        # unassigned lmvar returned unchanged
        assert mctx.instantiate_level_mvars(LMVar("w.0")) == LMVar("w.0")

    def test_is_valid_assignment(self):
        """
        Tests for the conditions of ``MetaVarContext.is_valid_assignment``
        """
        # rejects assignment to unregistered mvar_id
        mctx = MetaVarContext()
        ok, reason = mctx.is_valid_assignment(MVarId("missing.0"), NAT)

        assert ok is False
        assert "?missing.0 not found in MetaVarContext" == reason

        # reject loose bvars
        mv = mctx.mk_mvar("m", LocalContext(), NAT)
        ok, reason = mctx.is_valid_assignment(mv.id, BVar(0))

        assert ok is False

        assert reason == "solution contains loose BVars"

        mv = mctx.mk_mvar("m", LocalContext(), NAT)
        ok, reason = mctx.is_valid_assignment(mv.id, mv)
        # rejects self reference occurs
        assert ok is False

        assert reason == f"occurs check: ?{mv.id.id} appears in its own solution"  # noqa: E501

        _, x_fv = LocalContext().push_local("x", NAT)
        # mvar created before x existed in the context
        mv = mctx.mk_mvar("m", LocalContext(), NAT)
        # rejects_fvar_outside_creation_scope
        ok, reason = mctx.is_valid_assignment(mv.id, x_fv)

        assert ok is False
        assert reason == f"solution mentions FVars outside ?{mv.id.id}'s creation scope"  # noqa: E501

        # accepts_fvar_inside_creation_scope
        lctx_with_x, x_fv = LocalContext().push_local("x", NAT)
        # mvar created with x already in scope
        mv = mctx.mk_mvar("m", lctx_with_x, NAT)
        ok, _ = mctx.is_valid_assignment(mv.id, x_fv)
        assert ok is True

    def test_save_restore(self):
        """
        Tests for .save and .restore methods.
        """
        # restore expr assignment
        mctx = MetaVarContext()
        mv = mctx.mk_mvar("m", LocalContext(), NAT)
        mctx.expr_assignments[mv.id.id] = Const("Nat.zero", ())

        snap = mctx.save()
        mctx.expr_assignments[mv.id.id] = Const("Nat.succ", ())
        mctx.restore(snap)

        assert mctx.expr_assignments[mv.id.id] == Const("Nat.zero", ())

        # restore level assignment
        mctx.level_assignments[LMVarId("u.0")] = LZero()

        snap = mctx.save()
        mctx.level_assignments[LMVarId("u.0")] = LSucc(LZero())
        mctx.restore(snap)

        assert mctx.level_assignments[LMVarId("u.0")] == LZero()

    def test_unassigned_mvars_all_mvars(self):
        """
        Tests for .unassigned_mvars and .all_mvars
        """
        mctx = MetaVarContext()
        lctx = LocalContext()
        m = mctx.mk_mvar("m", lctx, NAT)
        n = mctx.mk_mvar("n", lctx, NAT)
        term = App(m, n)

        assert len(mctx.unassigned_mvars(term)) == 2

        # assign m mvar
        mctx.expr_assignments[m.id.id] = Const("Nat.zero", ())

        assert mctx.unassigned_mvars(term) == [n.id]

        # all_mvars ignores assignment state
        ids = {mv_id.id for mv_id in mctx.all_mvars(term)}
        assert ids == {m.id.id, n.id.id}

        # all_mvars accounts for duplicates
        mctx = MetaVarContext()
        mv = mctx.mk_mvar("m", LocalContext(), NAT)
        term = App(mv, mv)

        assert len(mctx.all_mvars(term)) == 1


class TestMetaVarLevelHelpers:
    """
    Tests for MetaVarContext level helpers: ``children``, ``collect_all_mvars``
    , ``collect_fvars_rec``, and ``expr_has_level_mvar``.
    """

    def test_children(self):
        """
        Tests for ``children`` function.
        """
        f, a = Const("f", ()), Const("a", ())
        assert children(App(f, a)) == (f, a)
        # chilren of Const (leaf) is empty
        assert children(Const("f", ())) == ()

    def test_expr_has_level_mvar(self):
        """
        Checks ``expr_has_level_mvar`` recurses into subterms.
        """
        term = App(Const("f", ()), Sort(LMVar("u.0")))
        assert expr_has_level_mvar(term) is True

        # checks flase case when no LMVar present
        term = App(Const("f", ()), Sort(LZero()))
        assert expr_has_level_mvar(term) is False

    def test_collect_fvars_all_mvars(self):
        """"
        collect_fvars and collect_all_mvars recurses into subterms and
        collects FVarIds without duplicates.
        """
        x = FVar(FVarId("x.0"))
        result, seen = [], set()

        collect_fvars_rec(App(x, x), result, seen)

        assert result == [FVarId("x.0")]

        mctx = MetaVarContext()
        mv = mctx.mk_mvar("m", LocalContext(), NAT)
        result, seen = [], set()

        collect_all_mvars(App(mv, mv), result, seen)
        assert result == [mv.id]
