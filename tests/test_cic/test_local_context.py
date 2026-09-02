from physika.core.expr import (
    BinderInfo,
    BVar,
    Const,
    ForallE,
    FVar,
    FVarId,
    Lam,
)
from physika.core.local_context import (
    LocalContext,
    LocalDeclDef,
    LocalDeclVar,
    fresh_fvar_id,
)


class TestFreshFVarId:
    """
    Tests for ``fresh_fvar_id``
    """

    def test_uses_hint_as_prefix(self):
        """
        Checks ``FVarId`` is contains the original name.
        """
        fid = fresh_fvar_id("x")

        assert isinstance(fid, FVarId)

        prefix, _, suffix = fid.id.partition(".")
        assert prefix == "x"
        assert suffix.isdigit()

        # two calls with the same hint are different
        a = fresh_fvar_id("x")
        b = fresh_fvar_id("x")

        assert a != b
        assert a.id != b.id

        assert a.id == f"x.{int(suffix) + 1}"
        assert b.id == f"x.{int(suffix) + 2}"


class TestLocalDeclVar:
    """
    Tests for ``LocalDeclVar``
    """

    def test_construction(self):
        """
        Verifies construction of ``LocalDeclVar``.
        """
        decl = LocalDeclVar(
            fvar_id=FVarId("x.0"),
            index=0,
            user_name="x",
            type=Const("Real", ()),
            binder_info=BinderInfo.DEFAULT,
        )

        assert decl.fvar_id == FVarId("x.0")
        assert decl.index == 0
        assert decl.user_name == "x"
        assert decl.type == Const("Real", ())
        assert decl.binder_info == BinderInfo.DEFAULT


class TestLocalDeclDef:
    """
    Tests for ``LocalDeclDef``
    """

    def test_construction(self):
        """
        Checks proper construction of ``LocalDeclDef``.
        """
        decl = LocalDeclDef(
            fvar_id=FVarId("x.0"),
            index=0,
            user_name="x",
            type=Const("Real", ()),
            value=Const("one", ()),
        )

        assert decl.fvar_id == FVarId("x.0")
        assert decl.index == 0
        assert decl.user_name == "x"
        assert decl.type == Const("Real", ())
        assert decl.value == Const("one", ())
        assert decl.non_dep is False


class TestLocalContext:
    """
    Tests for ``LocalContext`` construction
    """

    def test_empty_local_context(self):
        """
        Checks a new``LocalContext`` is empty.
        """
        lctx = LocalContext()

        assert lctx.decls == []
        assert lctx.fvar_map == {}

    def test_make(self):
        """
        Checks ``make`` builds a ``LocalContext`` from a given decls list
        and fvar_map.
        """
        decl = LocalDeclVar(
            FVarId("x.0"),
            0,
            "x",
            Const("Real", ()),
            BinderInfo.DEFAULT,
        )
        lctx = LocalContext().make([decl], {"x.0": decl})

        assert lctx.decls_l() == [decl]
        assert lctx.find(FVarId("x.0")) is decl

    def test_push_local(self):
        """
        Checks ``push_local`` returns a new ``LocalContext``
        with the new ``FVar``.
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_local("x", Const("Real", ()))

        # original context should remain the same
        assert lctx.decls == []
        assert lctx.contains(x_fv.id) is False

        # should create a new LocalContext
        assert len(lctx2.decls) == 1
        assert lctx2.contains(x_fv.id) is True
        assert lctx2.fvar_type(x_fv) == Const("Real", ())

        #  implicit BinderInfo is tracked (non default)
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_local(
            "x",
            Const("Real", ()),
            bi=BinderInfo.IMPLICIT,
        )

        decl = lctx2.find(x_fv.id)
        assert decl.binder_info == BinderInfo.IMPLICIT

        # Checks each pushed declaration records its position
        lctx = LocalContext()
        lctx, x_fv = lctx.push_local("x", Const("Real", ()))
        lctx, y_fv = lctx.push_local("y", Const("Real", ()))

        assert lctx.find(x_fv.id).index == 0
        assert lctx.find(y_fv.id).index == 1

    def test_push_let(self):
        """
        Checks ``push_let`` registers a ``LocalDeclDef`` with both a type
        and a value, leaving the original context untouched.
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_let(
            "x",
            Const("Real", ()),
            Const("one", ()),
        )

        assert lctx.contains(x_fv.id) is False

        decl = lctx2.find(x_fv.id)
        assert isinstance(decl, LocalDeclDef)
        assert decl.type == Const("Real", ())
        assert decl.value == Const("one", ())
        assert decl.non_dep is False

    def test_finds_registered_decl(self):
        """
        Verify ``find`` returns the ``LocalDecl`` for a registered FVarId.
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_local("x", Const("Real", ()))

        assert lctx2.find(x_fv.id).user_name == "x"

        # must return None
        lctx = LocalContext()

        assert lctx.find(FVarId("None")) is None

    def test_contains_true(self):
        """
        Checks True and False return for contains method when checking
        FVar IDs
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_local("x", Const("Real", ()))

        assert lctx2.contains(x_fv.id) is True

        # an empty local context should not contain IDs
        lctx = LocalContext()
        assert lctx.contains(FVarId(x_fv.id)) is False

    def test_fvar_type(self):
        """
        ``fvar_type`` should work with both ``FVar`` and ``FVarId``.
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_local("x", Const("Real", ()))

        assert lctx2.fvar_type(x_fv) == Const("Real", ())
        assert lctx2.fvar_type(x_fv.id) == Const("Real", ())

    def test_decls_in_push_order(self):
        """
        Checks ``decls_l`` adds declarations in order.
        """
        lctx = LocalContext()
        lctx, _ = lctx.push_local("x", Const("Real", ()))
        lctx, _ = lctx.push_local("y", Const("Real", ()))

        names = [d.user_name for d in lctx.decls_l()]
        assert names == ["x", "y"]

        # mutating the returned list should not affect the context
        lctx.decls_l().clear()
        assert len(lctx.decls_l()) == 2

    def test_mk_lambda_closes_single_fvar(self):
        """
        Checks ``mk_lambda`` closes a single FVar back into a ``Lam``
        with a ``BVar(0)`` body.
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_local("x", Const("Real", ()))

        term = lctx2.mk_lambda([x_fv], FVar(x_fv.id))

        assert term == Lam(
            "x",
            Const("Real", ()),
            BVar(0),
            BinderInfo.DEFAULT,
        )

    def test_mk_forall_closes_single_fvar(self):
        """
        Checks ``mk_forall`` closes a single FVar back into a
        ``ForallE``.
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_local("x", Const("Real", ()))

        term = lctx2.mk_forall([x_fv], Const("Real", ()))

        assert term == ForallE(
            "x",
            Const("Real", ()),
            Const("Real", ()),
            BinderInfo.DEFAULT,
        )

    def test_mk_lambda_multiple_fvars_outermost_first(self):
        """
        Checks closing over several FVars.
        """
        lctx = LocalContext()
        lctx, x_fv = lctx.push_local("x", Const("Real", ()))
        lctx, y_fv = lctx.push_local("y", Const("Real", ()))

        body = FVar(y_fv.id)
        term = lctx.mk_lambda([x_fv, y_fv], body)

        # outermost binder is "x"
        # inner binder is "y"
        assert isinstance(term, Lam)
        assert term.binder_name == "x"
        assert isinstance(term.body, Lam)
        assert term.body.binder_name == "y"
        # y should close to BVar(0)
        assert term.body.body == BVar(0)

    def test_mk_lambda_let_bound_fvar_closes_to_lete(self):
        """
        Checks closing over a let-bound FVar produces a
        ``LetE`` node.
        """
        lctx = LocalContext()
        lctx2, x_fv = lctx.push_let(
            "x",
            Const("Real", ()),
            Const("one", ()),
        )

        term = lctx2.mk_lambda([x_fv], FVar(x_fv.id))

        assert term.binder_name == "x"
        assert term.type == Const("Real", ())
        assert term.value == Const("one", ())
        assert term.body == BVar(0)
