from physika.core.expr import App, BVar, Const, ForallE, Lit, Sort
from physika.core.inductive import (
    Constructor,
    InductiveDecl,
    Recursor,
    RecursorRule,
    mk_builtin_env,
)
from physika.utils.cic_utils.inductive_utils import (verify_recursor_rules)
from physika.core.reduction import whnf, lit_nat_int
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
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


class TestMkBuiltinEnv:
    """
    Tests for ``mk_builtin_env``.

    Checks that each builtin inductive currently supported is well formed.
    The derived recursors must have the right shape and produce the correct
    CIC term.
    """

    def test_core_inductives(self):
        """
        ``mk_builtin_env()`` returns without raising an error and registers
        inductive types with a derived recursor.
        """
        env = mk_builtin_env()
        for name in ("Nat", "Int", "Bool", "Fin", "Vec", "Prod", "OfNat"):
            assert name in env.inductives

        assert env.inductives["Nat"].rec_info is not None
        # OfNat have no ι-rules
        assert env.inductives["OfNat"].rec_info is None

    def test_derived_recursor_shapes(self):
        """
        Every builtin recursor has the expected
        ``(num_params, num_indices, num_minors)`` and ``arity()``
        (``num_params + 1 motive + num_minors + num_indices + 1 major``).
        """
        env = mk_builtin_env()
        # name -> (num_params, num_indices, num_minors, arity)
        expected = {
            "Nat": (0, 0, 2, 4),
            "Int": (0, 0, 2, 4),
            "Bool": (0, 0, 2, 4),
            "Fin": (0, 1, 2, 5),
            "Vec": (1, 1, 2, 6),
            "Prod": (2, 0, 1, 5),
        }
        for name, (params, indices, minors, arity) in expected.items():
            rec = env.inductives[name].rec_info
            assert (rec.num_params, rec.num_indices,
                    rec.num_minors) == (params, indices, minors), name
            assert rec.num_motives == 1, name
            assert rec.arity() == arity, name
            assert len(
                rec.rules) == minors, name  # one reduction-rule per ctor
        assert env.inductives["OfNat"].rec_info is None

    def test_every_builtin_recursor_kernel_checks(self):
        """
        Rerun ``verify_recursor_rules`` on each builti.
        """
        env = mk_builtin_env()
        for _, ii in env.inductives.items():
            if ii.rec_info is None:
                continue
            assert verify_recursor_rules(ii.decl, ii.ctors, ii.recursor,
                                         ii.rec_info, env) is None

    def test_nat_operation_computes(self):
        """
        Check ``Nat.add`` / ``Nat.mul`` / ``Nat.sub`` have  ``Nat.rec`` based
        bodies, so the kernel reduces them on literal arguments.
        """
        env = mk_builtin_env()
        lctx, mctx = LocalContext(), MetaVarContext()
        for op, a, b, want in [("Nat.add", 2, 3, 5), ("Nat.mul", 3, 4, 12),
                               ("Nat.sub", 5, 2, 3), ("Nat.sub", 2, 5, 0)]:
            expr = App(App(Const(op, ()), Lit(a)), Lit(b))
            assert lit_nat_int(whnf(expr, env, lctx, mctx)) == want
