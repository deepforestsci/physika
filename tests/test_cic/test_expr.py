from physika.core.expr import (
    PROP,
    TYPE_0,
    TYPE_1,
    App,
    BinderInfo,
    BVar,
    Const,
    FloatLit,
    ForallE,
    FVar,
    FVarId,
    Lam,
    LetE,
    Lit,
    MData,
    MVar,
    MVarId,
    NatLit,
    Proj,
    Sort,
)
from physika.core.level import LSucc, LZero


class TestBinderInfo:
    """
    Tests for ``BinderInfo``
    """

    def test_default_binder_info(self):
        """
        Checks how a binder's argument is supplied.
        """
        # ``Lam``/``ForallE`` default to ``DEFAULT`` when not specified."""
        lam = Lam("x", Const("Real", ()), Const("x", ()))
        assert lam.binder_info == BinderInfo.DEFAULT


class TestFVarId:
    """
    Tests for free variable identifier ``FVarId``
    """

    def test_fvarid_construction(self):
        """
        Verifies that ``FVarId`` is constructed correctly.
        """
        assert FVarId("x.0").id == "x.0"
        assert str(FVarId("x.0")) == "x.0"
        assert FVarId("x.0") == FVarId("x.0")
        assert FVarId("x.0") != FVarId("x.1")

        # different FVarId
        x1 = FVar(FVarId("x.0"))
        x2 = FVar(FVarId("x.1"))
        assert x1 != x2
        assert x1.id != x2.id


class TestMVarId:
    """
    Tests for metavariable identifier ``MVarId``
    """

    def test_mvarid_construction(self):
        """
        Verifies that ``MVarId`` is constructed correctly.
        """
        assert MVarId("m.0").id == "m.0"
        assert str(MVarId("m.0")) == "m.0"

        # checks for equality and inequality
        assert MVarId("m.0") == MVarId("m.0")
        assert MVarId("m.0") != MVarId("m.1")


class TestNatLit:
    """Tests for integer literal ``NatLit``."""

    def test_natlit_construction(self):
        """
        Checks poper construction of ``NatLit`` and its value.
        """
        assert NatLit(3).val == 3

        assert NatLit(3) == NatLit(3)
        assert NatLit(3) != NatLit(4)

        # not equal to FloatLit of "same" value
        assert NatLit(3) != FloatLit(3.0)


class TestFloatLit:
    """
    Tests for floating literal ``FloatLit``
    """

    def test_floatlit_construction(self):
        """
        Checks poper construction of ``FloatLit`` and its value.
        """
        assert FloatLit(1.5).val == 1.5

        assert FloatLit(1.5) == FloatLit(1.5)
        assert FloatLit(1.5) != FloatLit(2.5)


class TestBVar:
    """Tests for bound variable ``BVar``."""

    def test_bvar_construction(self):
        """
        Checks proper construction of ``BVar`` and its index.
        """
        assert BVar(0).idx == 0

        # de Bruijn index equality and inequality
        assert BVar(0) == BVar(0)
        assert BVar(0) != BVar(1)

        # ``BVar(0)`` refers to the innermost enclosing binder
        expr = Lam("x", Const("Real", ()), BVar(0))
        assert expr.body == BVar(0)


class TestFVar:
    """Tests for free variable ``FVar``."""

    def test_fvar_construction(self):
        """
        Checks proper construction of ``FVar`` and its identifier.
        """
        assert FVar(FVarId("x.0")).id == FVarId("x.0")

        assert FVar(FVarId("x.0")) == FVar(FVarId("x.0"))
        assert FVar(FVarId("x.0")) != FVar(FVarId("x.1"))

        # Same numeric hint but different node
        assert FVar(FVarId("0")) != BVar(0)


class TestMVar:
    """Tests for metavariable ``MVar``."""

    def test_mvar_construction(self):
        assert MVar(MVarId("m.0")).id == MVarId("m.0")
        assert MVar(MVarId("m.0")) == MVar(MVarId("m.0"))
        assert MVar(MVarId("m.0")) != MVar(MVarId("m.1"))

        # An MVar and an FVar built from the "same" id string differ
        assert MVar(MVarId("x.0")) != FVar(FVarId("x.0"))


class TestSort:
    """Tests for universe ``Sort``."""

    def test_sort_construction(self):
        """
        Verifies that ``Sort`` is constructed correctly and its level is
        accessible.
        """
        assert Sort(LZero()).level == LZero()
        assert Sort(LZero()) == Sort(LZero())

        assert Sort(LZero()) != Sort(LSucc(LZero()))
        # Prop, type0, and type1 are distinct
        assert PROP == Sort(LZero())
        assert TYPE_0 == Sort(LSucc(LZero()))
        assert TYPE_1 == Sort(LSucc(LSucc(LZero())))
        assert PROP != TYPE_0
        assert TYPE_0 != TYPE_1
        assert PROP != TYPE_1


class TestConst:
    """Tests for global declaration reference ``Const``."""

    def test_const_construction(self):
        """
        Verifies that ``Const`` is constructed correctly for Real
        """
        c = Const("Real", ())
        assert c.name == "Real"
        assert c.levels == ()

    def test_const_real_and_nat(self):
        """
        Verifies that ``Const`` is constructed correctly for Real and Nat
        """
        assert Const("Real", ()) == Const("Real", ())
        assert Const("Real", ()) != Const("Nat", ())

        # test_same var name at different levels are distinct
        assert Const("f", (LZero(), )) != Const("f", (LSucc(LZero()), ))


class TestApp:
    """Tests for function application ``App``."""

    def test_app_def_args(self):
        """
        Checks that function application have function name
        and args (even if they are empty).
        """
        app = App(Const("f", ()), Const("x", ()))
        assert app.func == Const("f", ())
        assert app.arg == Const("x", ())

    def test_multi_arg_application_is_left_nested(self):
        """
        Verifies multi argument is still a nested function aplication.
        """
        # ``f(a, b)`` is ``App(App(f, a), b)``
        f, a, b = Const("f", ()), Const("a", ()), Const("b", ())
        term = App(App(f, a), b)
        assert term.func == App(f, a)
        assert term.arg == b

        # test equality and inequality of App nodes
        assert App(Const("f", ()), Const("x",
                                         ())) == App(Const("f", ()),
                                                     Const("x", ()))
        assert App(Const("f", ()), Const("x",
                                         ())) != App(Const("f", ()),
                                                     Const("y", ()))


class TestLam:
    """Tests for lambda abstraction ``Lam``."""

    def test_lam_structure(self):
        """
        Checks that a lambda abstraction has a binder name, type, body,
        and binder info.
        """
        lam = Lam("x", Const("Real", ()), BVar(0), BinderInfo.IMPLICIT)
        assert lam.binder_name == "x"
        assert lam.binder_type == Const("Real", ())
        assert lam.body == BVar(0)
        assert lam.binder_info == BinderInfo.IMPLICIT

        # two lambda constructions must be equal
        assert Lam("x", Const("Real", ()),
                   BVar(0)) == Lam("x", Const("Real", ()), BVar(0))

        # different binder info affect equality.
        explicit = Lam("x", Const("Real", ()), BVar(0), BinderInfo.DEFAULT)
        implicit = Lam("x", Const("Real", ()), BVar(0), BinderInfo.IMPLICIT)
        assert explicit != implicit


class TestForallE:
    """Tests for dependent function type (Pi type) ``ForallE``."""

    def test_foralle_constuction(self):
        """
        Checks that a dependent function abstraction has a binder name, type,
        and body.
        """
        arrow = ForallE("_", Const("Real", ()), Const("Real", ()))
        assert arrow.binder_name == "_"
        assert arrow.binder_type == Const("Real", ())
        assert arrow.body == Const("Real", ())

        a = ForallE("x", Const("Nat", ()), Const("Nat", ()))
        b = ForallE("x", Const("Nat", ()), Const("Nat", ()))
        assert a == b

        # lam is function, pforalle is its type
        pi = ForallE("x", Const("Real", ()), BVar(0))
        fn = Lam("x", Const("Real", ()), BVar(0))
        assert pi != fn


class TestLetE:
    """Tests for let-binding ``LetE`` expression."""

    def test_lete_construction(self):
        """
        Checks proper construction of LetE
        """
        term = LetE("x", Const("Real", ()), Const("one", ()), BVar(0))
        assert term.binder_name == "x"
        assert term.type == Const("Real", ())
        assert term.value == Const("one", ())
        assert term.body == BVar(0)
        assert term.non_dep is False

        # checks the body depends on LetE binding.
        assert LetE("x", Const("Real", ()), Const("one", ()),
                    BVar(0)).non_dep is False

        # non dependency affects equality.
        a = LetE("x",
                 Const("Real", ()),
                 Const("one", ()),
                 Const("y", ()),
                 non_dep=False)
        b = LetE("x",
                 Const("Real", ()),
                 Const("one", ()),
                 Const("y", ()),
                 non_dep=True)
        assert a != b


class TestLit:
    """Tests for literal value ``Lit``."""

    def test_construction_with_nat_lit(self):
        assert Lit(NatLit(3)).val == NatLit(3)
        assert Lit(FloatLit(1.5)).val == FloatLit(1.5)

        assert Lit(NatLit(3)) == Lit(NatLit(3))
        assert Lit(NatLit(3)) != Lit(NatLit(4))
        # float and nat are different Lit
        assert Lit(NatLit(3)) != Lit(FloatLit(3.0))


class TestMData:
    """Tests for metadata wrapper ``MData``."""

    def test_MData_info(self):
        """
        Checks MData is being constructed properly and carries information.
        """
        term = MData((("line", 12), ), Const("x", ()))
        assert term.kvs == (("line", 12), )
        assert term.expr == Const("x", ())


class TestProj:
    """Tests for structure field projection ``Proj``."""

    def test_proj_construction(self):
        ray = Const("ray", ())
        proj = Proj("Ray", 0, ray)
        assert proj.type_name == "Ray"
        assert proj.idx == 0
        assert proj.expr == ray
