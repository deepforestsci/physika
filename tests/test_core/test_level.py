from physika.core.level import (
    LIMax,
    LMax,
    LMVar,
    LParam,
    LSucc,
    LZero,
)


class TestLZero:
    """Tests for base universe level 0 ``LZero`."""

    def test_LZero_instances(self):
        """
        Checks LZero object initialization and equality.
        """
        assert LZero() is not None

        # any two instances are structurally equal
        assert LZero() == LZero()

        # frozen dataclasses must be hashable to be usable as dict keys
        assert hash(LZero()) == hash(LZero())
        assert len({LZero(), LZero()}) == 1

        # test that LZero is distinct from other level types
        assert LZero() != LSucc(LZero())
        assert LZero() != LParam("u")


class TestLSucc:
    """Tests for successor level (``LSucc``)."""

    def test_construction(self):
        succ = LSucc(LZero())
        assert succ.pred == LZero()

        # test LSucc equality and inequality
        assert LSucc(LZero()) == LSucc(LZero())
        assert LSucc(LZero()) != LSucc(LSucc(LZero()))

    def test_nesting(self):
        """
        Checks Level n is n nested LSucc wrapping a single LZero.
        """
        type_0 = LSucc(LZero())
        type_1 = LSucc(type_0)
        type_2 = LSucc(type_1)
        assert type_1.pred == type_0
        assert type_2.pred.pred == type_0


class TestLMax:
    """Tests for predicative maximum ``LMax``"""

    def test_LMax_construction(self):
        m = LMax(LSucc(LZero()), LZero())
        assert m.l1 == LSucc(LZero())
        assert m.l2 == LZero()

        # test instance equality
        assert LMax(LZero(), LSucc(LZero())) == LMax(LZero(), LSucc(LZero()))


class TestLIMax:
    """Tests for impredicative maximum, ``LIMax``."""

    def test_construction(self):
        im = LIMax(LSucc(LZero()), LZero())
        assert im.l1 == LSucc(LZero())
        assert im.l2 == LZero()

        # test instance equality
        assert LIMax(LZero(), LSucc(LZero())) == LIMax(LZero(), LSucc(LZero()))

    def test_distinct_from_lmax(self):
        """Same operands, different node kind must not compare equal."""
        assert LIMax(LZero(), LZero()) != LMax(LZero(), LZero())


class TestLParam:
    """Tests for named universe-polymorphism variable ``LParam``."""

    def test_construction(self):
        assert LParam("u").name == "u"

    def test_equality(self):
        assert LParam("u") == LParam("u")
        assert LParam("u") != LParam("v")


class TestLMVar:
    """Tests for universe metavariable ``LMVar``."""

    def test_construction(self):
        assert LMVar("m1").id == "m1"

    def test_equality(self):
        assert LMVar("m1") == LMVar("m1")
