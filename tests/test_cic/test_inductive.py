from physika.core.expr import BVar, Const, ForallE, Sort
from physika.core.inductive import (
    Constructor,
    InductiveDecl,
    Recursor,
    RecursorRule,
)
from physika.core.level import LSucc, LZero


class TestConstructor:
    """
    Tests for ``Constructor``
    """

    def test_constructor(self):
        """
        Checks proper construction of Nat ``Constructor``'s and its fields.
        """
        # ``Nat.succ : Nat -> Nat``
        nat = Const("Nat", ())
        succ_type = ForallE("n", nat, nat)
        succ = Constructor("Nat.succ", succ_type)

        assert succ.name == "Nat.succ"
        assert succ.type == succ_type

        # check Nat.zero costructor
        nat = Const("Nat", ())
        zero = Constructor("Nat.zero", nat)

        assert zero.name == "Nat.zero"
        assert zero.type == nat


class TestRecursorRule:
    """
    Tests for ``RecursorRule``
    """

    def test_recursor_rule(self):
        """
        Checks proper construction of ``RecursorRule`` and its fields
        for Nat.zero constructor.
        """
        rule = RecursorRule("Nat.zero", nfields=0, rhs=BVar(0))

        assert rule.ctor_name == "Nat.zero"
        assert rule.nfields == 0
        assert rule.rhs == BVar(0)

    def test_recursor_rule_with_fields(self):
        """
        Checks ``Nat.succ`` constructor supports field arguments.
        """
        rhs = BVar(1)
        rule = RecursorRule("Nat.succ", nfields=1, rhs=rhs)

        assert rule.ctor_name == "Nat.succ"
        assert rule.nfields == 1
        assert rule.rhs == rhs


class TestRecursor:
    """
    Tests for ``Recursor``
    """

    def test_recursor(self):
        """
        Checks ``Recursor`` construction and its fields.
        """
        motive = Sort(LSucc(LZero()))
        zero_rule = RecursorRule("Nat.zero", nfields=0, rhs=BVar(0))
        succ_rule = RecursorRule("Nat.succ", nfields=1, rhs=BVar(1))

        rec = Recursor(
            name="Nat.rec",
            type=motive,
            num_params=0,
            num_indices=0,
            num_motives=1,
            num_minors=2,
            rules=(zero_rule, succ_rule),
        )

        assert rec.name == "Nat.rec"
        assert rec.type == motive
        assert rec.num_params == 0
        assert rec.num_indices == 0
        assert rec.num_motives == 1
        assert rec.num_minors == 2
        assert rec.rules == (zero_rule, succ_rule)

        # level_params defaults to an empty tuple when not provided
        assert rec.level_params == ()

    def test_recursor_level_params(self):
        """
        Checks that universe polymorphism parameters on the recursor
        are stored as given.
        """
        rec = Recursor(
            name="Vec.rec",
            type=Sort(LSucc(LZero())),
            num_params=1,
            num_indices=1,
            num_motives=1,
            num_minors=2,
            rules=(),
            level_params=("u"),
        )

        assert rec.level_params == ("u")

    def test_recursor_arity(self):
        """
        Checks ``arity()`` sums params, motives, minors, indices, and
        the major premise itself (+1).
        """
        # (Vec α n)
        # Vec.rec: 1 param (alpha) + 1 motive + 2 minors (nil/cons) +
        # 1 index (n) + 1 major premise
        rec = Recursor(
            name="Vec.rec",
            type=Sort(LSucc(LZero())),
            num_params=1,
            num_indices=1,
            num_motives=1,
            num_minors=2,
            rules=(),
            level_params=("u", ),
        )

        assert rec.arity() == 6


class TestInductiveDecl:
    """
    Tests for ``InductiveDecl``
    """

    def test_inductive_decl(self):
        """
        Checks proper construction of ``InductiveDecl`` for ``Bool``
        non recursive type.
        """
        bool_type = Sort(LSucc(LZero()))
        true_ctor = Constructor("Bool.true", Const("Bool", ()))
        false_ctor = Constructor("Bool.false", Const("Bool", ()))

        decl = InductiveDecl(
            name="Bool",
            level_params=(),
            num_params=0,
            type=bool_type,
            constructors=(true_ctor, false_ctor),
            is_recursive=False,
        )

        assert decl.name == "Bool"
        assert decl.level_params == ()
        assert decl.num_params == 0
        assert decl.type == bool_type
        assert decl.constructors == (true_ctor, false_ctor)
        assert decl.is_recursive is False

    def test_inductive_decl_recursive(self):
        """
        Checks ``is_recursive`` is stored for
        ``Nat.succ``.
        """
        nat = Const("Nat", ())
        zero_ctor = Constructor("Nat.zero", nat)
        succ_ctor = Constructor("Nat.succ", ForallE("n", nat, nat))

        decl = InductiveDecl(
            name="Nat",
            level_params=(),
            num_params=0,
            type=Sort(LSucc(LZero())),
            constructors=(zero_ctor, succ_ctor),
            is_recursive=True,
        )

        assert decl.is_recursive is True
        assert decl.constructors[1].name == "Nat.succ"
