from physika.core.level import LZero, LSucc, LMax, LIMax, LMVar, LParam
from physika.utils.cic_utils.level_utils import (
    mk_level_max,
    mk_level_imax,
    level_has_mvar,
    collect_params,
    eval_level,
    levels_equal,
    level_to_nat,
    instantiate_level_params,
)


class TestMkLevelMax:
    """
    Tests for ``mk_level_max``
    """

    def test_lmax_basic(self):
        """
        Basic tests checking correct lmax rule.
        """
        lvl = LSucc(LZero())
        assert mk_level_max(LZero(), lvl) == lvl

        # max(lvl, 0) returnx lvl.
        lvl = LSucc(LZero())
        assert mk_level_max(lvl, LZero()) == lvl

        # max(lvl, lvl) returns lvl
        lvl = LSucc(LSucc(LZero()))
        assert mk_level_max(lvl, lvl) == lvl

        # max(0, 0) returns LZero()
        assert mk_level_max(LZero(), LZero()) == LZero()

        l1 = LSucc(LZero())
        l2 = LSucc(LSucc(LZero()))
        # builds an LMax node
        assert mk_level_max(l1, l2) == LMax(l1, l2)


class TestMkLevelImax:
    """
    Basic tests checking correct lmax rule.
    """

    def test_level_imax_basics(self):
        """
        Checks different level configurations for imax rule.
        """
        # imax(lvl, 0) always returns 0
        assert mk_level_imax(LSucc(LZero()), LZero()) == LZero()

        # imax(0, lvl) returns lvl
        lvl = LSucc(LZero())
        assert mk_level_imax(LZero(), lvl) == lvl

        # imax(lvl, lvl) returns lvl
        lvl = LSucc(LSucc(LZero()))
        assert mk_level_imax(lvl, lvl) == lvl

        # imax(l1, l2) gives an LIMax node.
        l1 = LSucc(LZero())
        l2 = LSucc(LSucc(LZero()))
        assert mk_level_imax(l1, l2) == LIMax(l1, l2)


class TestLevelHasMvar:
    """
    Tests for ``level_has_mvar``
    """

    def test_find_mvar_without_target_id(self):
        """
        Any LMVar is found when no specific mvar_id is given.
        """
        assert level_has_mvar(LMVar("m1")) is True

        # recurse through LSucc node
        assert level_has_mvar(LSucc(LMVar("m1"))) is True

    def test_mvar_nested_max_levels(self):
        """
        Tests for nested LMVars within LMax and LIMax nodes.
        """
        # Searches both operands of LMax
        assert level_has_mvar(LMax(LZero(), LMVar("m1"))) is True
        assert level_has_mvar(LMax(LMVar("m1"), LZero())) is True

        # search recurses into both operands of LIMax.

        assert level_has_mvar(LIMax(LZero(), LMVar("m1"))) is True
        assert level_has_mvar(LIMax(LMVar("m1"), LZero())) is True

    def test_matching_specific_mvar_id(self):
        """
        Looks for LMVar with a specific given id.
        """
        assert level_has_mvar(LMVar("m1"), "m1") is True
        assert level_has_mvar(LMVar("m1"), "m2") is False

        # Looks foe MVar in LMax node
        lvl = LMax(LMVar("m1"), LMVar("m2"))
        assert level_has_mvar(lvl, "m1") is True
        assert level_has_mvar(lvl, "m2") is True
        assert level_has_mvar(lvl, "m3") is False


class TestCollectParams:
    """
    Tests for ``collect_params``
    """

    def test_collect_params(self):
        """
        Tests collect_params functionat different Levels.
        """
        assert collect_params(LParam("u")) == {"u"}

        # checks LMVar is returned with prefix
        assert collect_params(LMVar("m1")) == {"?m1"}

        # LSucc recurses into pred
        assert collect_params(LSucc(LParam("u"))) == {"u"}

        # Lmax and LImax union
        assert collect_params(LMax(LParam("u"), LParam("v"))) == {"u", "v"}
        assert collect_params(LIMax(LParam("u"), LMVar("m1"))) == {"u", "?m1"}


class TestEvalLevel:
    """
    Tests for ``eval_level``
    """

    def test_eval_level(self):
        """
        Tests for ``eval_level`` for checking appearences in different levels.
        """
        # LZero should return 0
        assert eval_level(LZero(), {}) == 0

        # LParam reads from subst dict
        assert eval_level(LParam("u"), {"u": 1}) == 1
        assert eval_level(LParam("u"), {}) == 0

        # LMVar read prefixed key from subst dict
        assert eval_level(LMVar("m1"), {"?m1": 1}) == 1
        assert eval_level(LMVar("m1"), {}) == 0

        #  ``LSucc`` evaluates its predecessor and adds 1.
        assert eval_level(LSucc(LParam("u")), {"u": 1}) == 2

        # ``LMax`` evaluates to the maximum of both operands.
        assert eval_level(LMax(LParam("u"), LParam("v")), {
            "u": 0,
            "v": 1
        }) == 1

        # ``LIMax`` forces 0 whenever the second operand
        # evaluates to 0.
        assert eval_level(LIMax(LParam("u"), LParam("v")), {
            "u": 1,
            "v": 0
        }) == 0

        # when second operand is not zero, should behaves like a regular LMax
        assert eval_level(LIMax(LParam("u"), LParam("v")), {
            "u": 1,
            "v": 1
        }) == 1


class TestLevelsEqual:
    """
    Tests for ``levels_equal``
    """

    def test_levels_equal(self):
        """
        Checks levels equals for different level setups.
        """
        # fast path parameter free levels are compared directly as
        # integers
        assert levels_equal(LSucc(LZero()), LSucc(LZero())) is True
        assert levels_equal(LSucc(LZero()), LZero()) is False

        # test max is commutative
        assert levels_equal(LMax(LParam("u"), LParam("v")),
                            LMax(LParam("v"), LParam("u"))) is True

        # LIMax for 0 value operands
        assert levels_equal(LIMax(LParam("u"), LZero()), LZero()) is True

        # checks levels are not equal
        assert levels_equal(LParam("u"), LSucc(LParam("u"))) is False


class TestLevelToNat:
    """
    Tests for ``level_to_nat``
    """

    def test_level_to_nat(self):
        """
        Checks ``LZero`` evaluates to 0.
        """
        assert level_to_nat(LZero()) == 0

        # LSucc chain
        assert level_to_nat(LSucc(LSucc(LZero()))) == 2

        # NAT is produces from LMax properly
        assert level_to_nat(LMax(LSucc(LZero()), LZero())) == 1


class TestInstantiateLevelParams:
    """
    Tests for ``instantiate_level_params``
    """

    def test_instantiate_level_params(self):
        """
        Inatntiate level parameters tests for different levels.
        """
        # Checks an ``LParam`` matching one of ``params`` is replaced by
        # its corresponding level.
        assert instantiate_level_params(LParam("u"), ["v", "u"],
                                        [LZero(), LSucc(LZero())]) == LSucc(
                                            LZero())

        # An ``LParam`` not present in ``params`` is left
        # unhanged
        assert instantiate_level_params(LParam("v"), ["u"],
                                        [LSucc(LZero())]) == LParam("v")

        # LMVar should be untouched
        mvar = LMVar("m1")

        assert instantiate_level_params(mvar, ["u"], [LSucc(LZero())]) == mvar

        # recurses into LSucc, LMax and LIMax levels
        assert instantiate_level_params(LSucc(LParam("u")), ["u"],
                                        [LZero()]) == LSucc(LZero())
        assert instantiate_level_params(LMax(LParam("u"), LZero()), ["u"],
                                        [LSucc(LZero())]) == LSucc(LZero())
        assert instantiate_level_params(LIMax(LParam("u"), LZero()), ["u"],
                                        [LSucc(LZero())]) == LZero()
