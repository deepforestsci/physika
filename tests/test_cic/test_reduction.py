from physika.core.expr import (
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
    NatLit,
    Proj,
    Sort,
    TYPE_0,
)
from physika.core.level import LMVar, LSucc, LZero
from physika.core.local_context import LocalContext
from physika.core.metavar import LMVarId, MetaVarContext
from physika.core.environment import ConstantInfo, Environment, InductiveInfo
from physika.core.inductive import Constructor, InductiveDecl, Recursor, RecursorRule  # noqa: E501
from physika.core.reduction import (
    is_def_eq,
    is_level_def_eq,
    lit_nat_int,
    pattern_mvar_app,
    solve_pattern,
    succ_chain_nat_int,
    try_iota,
    try_nat_literal_binop,
    whnf,
)

from physika.utils.cic_utils.expr_utils import get_app_fn_args as _fn_args

NAT = Const("Nat", ())


def nat_env(rules):
    """
    Builds an ``Environment`` with a ``Nat`` inductive type
    (``Nat.zero``/``Nat.succ`` constructors and ``Nat.rec`` )
    """
    nat_decl = InductiveDecl(
        name="Nat",
        level_params=(),
        num_params=0,
        type=TYPE_0,
        constructors=(
            Constructor("Nat.zero", NAT),
            Constructor("Nat.succ", ForallE("n", NAT, NAT)),
        ),
        is_recursive=True,
    )
    nat_rec = Recursor(
        name="Nat.rec",
        type=TYPE_0,
        num_params=0,
        num_indices=0,
        num_motives=1,
        num_minors=2,
        rules=rules,
    )
    ii = InductiveInfo(
        decl=nat_decl,
        ctors={
            "Nat.zero": ConstantInfo("Nat.zero", (), NAT, None),
            "Nat.succ": ConstantInfo("Nat.succ", (), ForallE("n", NAT, NAT),
                                     None),
        },
        recursor=ConstantInfo("Nat.rec", (), TYPE_0, None),
        rec_info=nat_rec,
    )
    env = Environment()
    env.inductives["Nat"] = ii
    for ci in ii.ctors.values():
        env.add_constant(ci)
    env.add_constant(ii.recursor)
    return env


class TestIsLevelDefEq:
    """
    Tests for ``is_level_def_eq``
    """

    def test_is_level_def_eq(self):
        """
        Checks level definitial equality for different levels and contexts
        (e.g. assigned/unassigned LMVar)
        """
        # instantiates assigned LMVar before comparing level equality
        mctx = MetaVarContext()
        mctx.level_assignments[LMVarId("u.0")] = LZero()
        ok, _ = is_level_def_eq(LMVar("u.0"), LZero(), mctx)
        assert ok is True

        # Checks an unassigned ``LMVar`` on the left is solved against
        # the rhs
        mctx = MetaVarContext()
        ok, mctx2 = is_level_def_eq(LMVar("u.0"), LZero(), mctx)

        assert ok is True
        assert mctx2.level_assignments[LMVarId("u.0")] == LZero()

        # viceversa
        mctx = MetaVarContext()

        ok, mctx2 = is_level_def_eq(LSucc(LZero()), LMVar("u.0"), mctx)
        assert ok is True
        assert mctx2.level_assignments[LMVarId("u.0")] == LSucc(LZero())

        # parameter free levels equality check
        mctx = MetaVarContext()

        ok, _ = is_level_def_eq(LSucc(LZero()), LSucc(LZero()), mctx)
        assert ok is True
        ok, _ = is_level_def_eq(LSucc(LZero()), LZero(), mctx)
        assert ok is False


class TestLitNatInt:
    """
    Tests for ``lit_nat_int``
    """

    def test_lit_nat_int(self):
        """
        Checks ``lit_nat_int`` for different ``Lit`` shapes.
        """
        # non negative Lit int returns its value
        assert lit_nat_int(Lit(3)) == 3
        assert lit_nat_int(Lit(0)) == 0

        # negative Lit int is not a valid Nat
        assert lit_nat_int(Lit(-1)) is None

        # non Lit expression should return None
        assert lit_nat_int(Const("Nat", ())) is None

        # Lit wrapping NatLit reads through to the inner int value
        assert lit_nat_int(Lit(NatLit(5))) == 5

        # Lit wrapping FloatLit should never be a valid Nat
        assert lit_nat_int(Lit(FloatLit(2.0))) is None


class TestSuccChainNatInt:
    """
    Tests for ``succ_chain_nat_int``
    """

    def test_succ_chain_nat_int(self):
        """
        Checks ``succ_chain_nat_int`` for different Nat.succ chain values.
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()

        # Nat.zero evaluates to 0
        assert succ_chain_nat_int(Const("Nat.zero", ()), env, lctx, mctx) == 0

        # Nat.succ chain counts one per layer down to Nat.zero
        two = App(Const("Nat.succ", ()),
                  App(Const("Nat.succ", ()), Const("Nat.zero", ())))
        assert succ_chain_nat_int(two, env, lctx, mctx) == 2

        # chain that ends with a Lit int adds Nat.succ count to base value
        chain = App(Const("Nat.succ", ()), Lit(1))  # succ(succ(zero)) shape
        assert succ_chain_nat_int(chain, env, lctx, mctx) == 2


class TestWhnf:
    """
    Tests for ``whnf``
    """

    def test_whnf(self):
        """
        Checks ``whnf`` reduction rules across different expressions.
        """
        env, mctx = Environment(), MetaVarContext()

        # assigned MVar follows its solution path
        lctx = LocalContext()
        mv = mctx.mk_mvar("m", lctx, NAT)
        mctx.expr_assignments[mv.id.id] = Lit(3)
        assert whnf(mv, env, lctx, mctx) == Lit(3)

        # unassigned MVar is in WHNF
        mctx2 = MetaVarContext()
        mv2 = mctx2.mk_mvar("m", LocalContext(), NAT)
        assert whnf(mv2, env, LocalContext(), mctx2) == mv2

        # zeta reduces a LetE by substituting its value into its body
        let = LetE("x", Const("Real", ()), Lit(7), BVar(0))
        assert whnf(let, env, LocalContext(), MetaVarContext()) == Lit(7)

        # zeta reduces a let-bound local (FVar with a LocalDeclDef)
        lctx3, x = LocalContext().push_let("x", Const("Real", ()), Lit(9))
        assert whnf(x, env, lctx3, MetaVarContext()) == Lit(9)

        # delta reduces a Const with a registered value
        env2 = Environment()
        env2.add_constant(ConstantInfo("one", (), NAT, Lit(1)))
        assert whnf(Const("one", ()), env2, LocalContext(),
                    MetaVarContext()) == Lit(1)

        # beta reduces (λx. x) arg to arg
        fn = Lam("x", NAT, BVar(0), BinderInfo.DEFAULT)
        assert whnf(App(fn, Lit(5)), env, LocalContext(),
                    MetaVarContext()) == Lit(5)

        # Nat.add on two literals is computed with fast path
        # instead of unfolding via Nat.rec
        add_expr = App(App(Const("Nat.add", ()), Lit(1)), Lit(2))
        assert whnf(add_expr, env, LocalContext(), MetaVarContext()) == Lit(3)

        # MData whnf gets its value
        wrapped = MData((("line", 1), ), Lit(5))
        assert whnf(wrapped, env, LocalContext(), MetaVarContext()) == Lit(5)

        # projection reduces to the matching constructor argument
        ray = Const("Ray", ())
        ray_decl = InductiveDecl(
            name="Ray",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor(
                "Ray.mk", ForallE("o", ray, ForallE("d", ray, ray))), ),
            is_recursive=False,
        )
        env4 = Environment()
        env4.add_inductive(
            InductiveInfo(
                decl=ray_decl,
                ctors={"Ray.mk": ConstantInfo("Ray.mk", (), ray, None)},
                recursor=ConstantInfo("Ray.rec", (), TYPE_0, None),
            ))
        inst = App(App(Const("Ray.mk", ()), Const("origin", ())),
                   Const("dir", ()))
        result = whnf(Proj("Ray", 1, inst), env4, LocalContext(),
                      MetaVarContext())
        assert result == Const("dir", ())

        # an application whose function is already in whnf
        term = App(Const("f", ()), Const("a", ()))
        assert whnf(term, env, LocalContext(), MetaVarContext()) == term


class TestPatternMvarApp:
    """
    Tests for ``pattern_mvar_app``
    """

    def test_pattern_mvar_app(self):
        """
        Checks ``pattern_mvar_app`` matches or rejects different application
        for Miller pattern.
        """
        lctx = LocalContext()
        lctx, x = lctx.push_local("x", Sort(LZero()))

        # unassigned MVar applied to distinct FVars should match Miller pattern
        mctx = MetaVarContext()
        mv = mctx.mk_mvar("m", lctx, Sort(LZero()))
        assert pattern_mvar_app(App(mv, x), mctx) == (mv, [x])

        # an assigned MVar head should not match
        mctx.expr_assignments[mv.id.id] = Lit(1)
        assert pattern_mvar_app(App(mv, x), mctx) is None

        # a Lit (non FVar) argument breaks should return None
        mctx2 = MetaVarContext()
        mv2 = mctx2.mk_mvar("m", LocalContext(), Sort(LZero()))
        assert pattern_mvar_app(App(mv2, Lit(1)), mctx2) is None

        # a repeated FVar argument (?m x x) has no unique solution
        mctx3 = MetaVarContext()
        mv3 = mctx3.mk_mvar("m", lctx, Sort(LZero()))
        assert pattern_mvar_app(App(App(mv3, x), x), mctx3) is None


class TestSolvePattern:
    """
    Tests for ``solve_pattern``
    """

    def test_solve_pattern(self):
        """
        Checks ``solve_pattern`` builds a correct Miller pattern lambda
        solution, or error, for different setups.
        """
        # ?m x := 3 solves to ?m := λx. 3
        lctx = LocalContext()
        lctx, x = lctx.push_local("x", Sort(LZero()))
        mctx = MetaVarContext()
        mv = mctx.mk_mvar("m", lctx, Sort(LZero()))

        ok = solve_pattern(mv, [x], Lit(3), lctx, mctx)

        assert ok is True
        assert mctx.expr_assignments[mv.id.id] == Lam("x", Sort(LZero()),
                                                      Lit(3),
                                                      BinderInfo.DEFAULT)

        # a pattern variable's own type (dependent types) is abstracted
        # properly
        lctx2 = LocalContext()
        lctx2, n = lctx2.push_local("n", NAT)
        lctx2, v = lctx2.push_local("v", App(Const("Vec", ()), n))
        mctx2 = MetaVarContext()
        mv2 = mctx2.mk_mvar("m", lctx2, Sort(LZero()))

        ok2 = solve_pattern(mv2, [n, v], Sort(LZero()), lctx2, mctx2)

        assert ok2 is True
        result = mctx2.expr_assignments[mv2.id.id]
        assert result.binder_name == "n"
        # v's own binder type must reference n via BVar(0), not the FVar
        assert result.body.binder_type == App(Const("Vec", ()), BVar(0))

        # ?m occurring in its own solution fails
        # and leaves mctx untouched
        mctx3 = MetaVarContext()
        mv3 = mctx3.mk_mvar("m", lctx, Sort(LZero()))

        ok3 = solve_pattern(mv3, [x], App(mv3, x), lctx, mctx3)

        assert ok3 is False
        assert mv3.id.id not in mctx3.expr_assignments


class TestIsDefEq:
    """
    Tests for ``is_def_eq``
    """

    def test_is_def_eq(self):
        """
        Checks ``is_def_eq`` for structural equality, WHNF reduction, Miller
        pattern solving, MVar assignment, Nat literal/succ-chain interop,
        structural comparison of each node type, and eta-expansion.
        """
        env, lctx = Environment(), LocalContext()

        # syntactically identical terms are equal without reduction
        ok, _ = is_def_eq(Lit(3), Lit(3), env, lctx, MetaVarContext())
        assert ok is True

        # both sides are reduced to WHNF before comparison
        add_lhs = App(App(Const("Nat.add", ()), Lit(1)), Lit(2))
        ok, _ = is_def_eq(add_lhs, Lit(3), env, lctx, MetaVarContext())
        assert ok is True

        # a Miller-pattern metavariable is solved during def equality check
        lctx2 = LocalContext()
        lctx2, x = lctx2.push_local("x", Sort(LZero()))
        mctx = MetaVarContext()
        mv = mctx.mk_mvar("m", lctx2, Sort(LZero()))
        ok, mctx = is_def_eq(App(mv, x), Lit(3), env, lctx2, mctx)
        assert ok is True
        assert mctx.expr_assignments[mv.id.id] == Lam("x", Sort(LZero()),
                                                      Lit(3),
                                                      BinderInfo.DEFAULT)

        # two distinct unassigned MVars are unified by assigning one
        # to the other
        mctx2 = MetaVarContext()
        m1 = mctx2.mk_mvar("m1", LocalContext(), NAT)
        m2 = mctx2.mk_mvar("m2", LocalContext(), NAT)
        ok, mctx2 = is_def_eq(m1, m2, env, LocalContext(), mctx2)
        assert ok is True
        assert mctx2.expr_assignments[m1.id.id] == m2

        # a unassigned MVar is solved against a concrete term
        mctx3 = MetaVarContext()
        mv3 = mctx3.mk_mvar("m", LocalContext(), NAT)
        ok, mctx3 = is_def_eq(mv3, Lit(5), env, LocalContext(), mctx3)
        assert ok is True
        assert mctx3.expr_assignments[mv3.id.id] == Lit(5)

        # a Lit int and an equivalent Nat.succ chain shoud be equal
        chain = App(Const("Nat.succ", ()),
                    App(Const("Nat.succ", ()), Const("Nat.zero", ())))
        ok, _ = is_def_eq(Lit(2), chain, env, lctx, MetaVarContext())
        assert ok is True

        # Sort equality solves a level metavariable
        mctx5 = MetaVarContext()
        ok, mctx5 = is_def_eq(Sort(LMVar("u.0")), Sort(LZero()), env, lctx,
                              mctx5)
        assert ok is True
        assert mctx5.level_assignments[LMVarId("u.0")] == LZero()

        # Const equality
        # same name and level args
        ok, _ = is_def_eq(Const("f", ()), Const("f", ()), env, lctx,
                          MetaVarContext())
        assert ok is True
        # same level args and different name
        ok, _ = is_def_eq(Const("f", ()), Const("g", ()), env, lctx,
                          MetaVarContext())
        assert ok is False

        # FVar equality compares by id
        fx = FVar(FVarId("x.0"))
        fy = FVar(FVarId("y.0"))
        ok, _ = is_def_eq(fx, fx, env, lctx, MetaVarContext())
        assert ok is True
        ok, _ = is_def_eq(fx, fy, env, lctx, MetaVarContext())
        assert ok is False

        # Lit compares by raw value, unwrapping NatLit
        ok, _ = is_def_eq(Lit(NatLit(4)), Lit(4), env, lctx, MetaVarContext())
        assert ok is True

        # App compares both the function and argument side
        t1 = App(Const("f", ()), Const("a", ()))
        t2 = App(Const("f", ()), Const("b", ()))
        ok, _ = is_def_eq(t1, t1, env, lctx, MetaVarContext())
        assert ok is True
        ok, _ = is_def_eq(t1, t2, env, lctx, MetaVarContext())
        assert ok is False

        # two Lam expressions are equal if their binder types matches and if
        # their bodies agree once opened with the same variable
        l1 = Lam("x", NAT, BVar(0), BinderInfo.DEFAULT)
        l2 = Lam("y", NAT, BVar(0), BinderInfo.DEFAULT)
        ok, _ = is_def_eq(l1, l2, env, lctx, MetaVarContext())
        assert ok is True

        # two ForallE expr are equal when domain and codomain (opened with
        # the same variable) agree
        f1 = ForallE("x", NAT, BVar(0), BinderInfo.DEFAULT)
        f2 = ForallE("y", NAT, BVar(0), BinderInfo.DEFAULT)
        ok, _ = is_def_eq(f1, f2, env, lctx, MetaVarContext())
        assert ok is True

        # Proj requires matching type name and idx and recurses into the
        # inner expression
        p1 = Proj("Ray", 0, Const("r", ()))
        p2 = Proj("Ray", 0, Const("r", ()))
        p3 = Proj("Ray", 1, Const("r", ()))
        ok, _ = is_def_eq(p1, p2, env, lctx, MetaVarContext())
        assert ok is True
        ok, _ = is_def_eq(p1, p3, env, lctx, MetaVarContext())
        assert ok is False

        # a Lam compared against a plain function constant is
        # eta-expanded before comparison
        f = Const("f", ())
        lam = Lam("x", NAT, App(f, BVar(0)), BinderInfo.DEFAULT)
        ok, _ = is_def_eq(lam, f, env, lctx, MetaVarContext())
        assert ok is True
        # viceversa
        ok, _ = is_def_eq(f, lam, env, lctx, MetaVarContext())
        assert ok is True

        # failing case, totally different heads
        ok, _ = is_def_eq(Const("f", ()), Lit(3), env, lctx, MetaVarContext())
        assert ok is False


class TestTryNatLiteralBinop:
    """
    Tests for ``try_nat_literal_binop``
    """

    def test_try_nat_literal_binop(self):
        """
        Checks ``try_nat_literal_binop``.
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()

        # Nat.add/Nat.mul/Nat.sub on two literals are computed directly
        add = App(App(Const("Nat.add", ()), Lit(1)), Lit(2))
        assert try_nat_literal_binop(add, env, lctx, mctx) == Lit(3)

        mul = App(App(Const("Nat.mul", ()), Lit(3)), Lit(4))
        assert try_nat_literal_binop(mul, env, lctx, mctx) == Lit(12)

        sub = App(App(Const("Nat.sub", ()), Lit(5)), Lit(2))
        assert try_nat_literal_binop(sub, env, lctx, mctx) == Lit(3)

        # Nat.sub should never be negative
        sub_clamped = App(App(Const("Nat.sub", ()), Lit(2)), Lit(5))
        assert try_nat_literal_binop(sub_clamped, env, lctx, mctx) == Lit(0)

        # non NAT_LITERAL_BINOPS  operation returns None
        non_binop = App(App(Const("Nat.div", ()), Lit(4)), Lit(2))
        assert try_nat_literal_binop(non_binop, env, lctx, mctx) is None

        # wrongly applied binop returns None
        partial = App(Const("Nat.add", ()), Lit(1))  # missing one argument
        assert try_nat_literal_binop(partial, env, lctx, mctx) is None

        # an operand that is not a literal returns None
        non_lit_operand = App(App(Const("Nat.add", ()), Const("n", ())),
                              Lit(1))
        assert try_nat_literal_binop(non_lit_operand, env, lctx, mctx) is None


class TestTryIota:
    """
    Tests for ``try_iota``
    """

    def test_try_iota(self):
        """
        Checks ``try_iota`` reduction for constructor, literal
        major premises, extra args and failing cases
        """
        # Nat.rec applied to Nat.zero reduces to the Nat.zero minor
        # premise
        zero_rule = RecursorRule("Nat.zero", nfields=0, rhs=BVar(1))
        env = nat_env((zero_rule, ))
        fn = App(App(App(Const("Nat.rec", ()), NAT), Lit(10)), NAT)

        result = try_iota(fn, Const("Nat.zero", ()), env, LocalContext(),
                          MetaVarContext())
        assert result == Lit(10)

        # Nat.rec applied to Nat.succ n reduces to a rule whose body
        # can reference the extracted field n
        succ_rule = RecursorRule("Nat.succ", nfields=1, rhs=BVar(0))
        env2 = nat_env((zero_rule, succ_rule))
        fn2 = App(App(App(Const("Nat.rec", ()), NAT), Lit(0)), NAT)
        major = App(Const("Nat.succ", ()), Lit(5))

        result = try_iota(fn2, major, env2, LocalContext(), MetaVarContext())
        assert result == Lit(5)

        # a Lit major premise is treated as Nat.succ equivalent constructor
        # form
        result = try_iota(fn2, Lit(6), env2, LocalContext(), MetaVarContext())
        assert result == Lit(NatLit(5))

        # application arguments beyond the recursor's own arity are
        # reapplied to the reduced result
        fn3 = App(App(App(Const("Nat.rec", ()), NAT), Lit(99)), NAT)
        applied = App(fn3, Const("Nat.zero", ()))
        # Nat.rec  needs 4 args (motive, 2 minor premises, major premise)
        # "extra" is given
        result = try_iota(applied, Const("extra", ()), env, LocalContext(),
                          MetaVarContext())
        assert result == App(Lit(99), Const("extra", ()))

        # fewer args than its arity returns None
        # missing one minor premise (succ)
        fn_short = App(App(Const("Nat.rec", ()), NAT), Lit(10))
        result = try_iota(fn_short, Const("Nat.zero", ()), env, LocalContext(),
                          MetaVarContext())
        assert result is None

        # a major premise that doesn't reduce to constructor form returns None
        # should be a valid ctor, like Nat.zero
        result = try_iota(fn, Const("invalid_ctor", ()), env, LocalContext(),
                          MetaVarContext())
        assert result is None

        # a constructor with no registered RecursorRule returns None
        env_no_succ_rule = nat_env((zero_rule, ))  # no rule for Nat.succ
        major_succ = App(Const("Nat.succ", ()), Lit(2))
        result = try_iota(fn, major_succ, env_no_succ_rule, LocalContext(),
                          MetaVarContext())
        assert result is None

    def test_derived_builtin_recursors_reduce(self):
        """
        Smoke-test the ``derive_recursor``-generated ι-rules for the
        builtin env: ``Nat.rec`` still reduces through a ``Nat.succ`` /
        literal major, and the indexed-family recursors ``Fin.rec`` /
        ``Vec.rec`` fire on their constructors. ``whnf``/``try_iota`` do
        no type-checking, so opaque placeholder motives/minors are fine
        here — the point is that the ι-rule pattern-matches and
        substitutes.
        """
        from physika.core.inductive import mk_builtin_env
        from physika.core.level import LZero
        env = mk_builtin_env()
        lctx, mctx = LocalContext(), MetaVarContext()

        M = Const("M", ())
        A = Const("A", ())
        z = Const("z", ())
        s = Const("s", ())

        def app(fn, *xs):
            for x in xs:
                fn = App(fn, x)
            return fn

        # Nat.rec M z s (Lit 2)  →  s 1 (Nat.rec M z s 1)   (succ ι-rule)
        nat_rec = Const("Nat.rec", (LZero(), ))
        res = whnf(app(nat_rec, M, z, s, Lit(2)), env, lctx, mctx)
        head, args = _fn_args(res)
        assert head == s and args[0] == Lit(NatLit(1))

        # Fin.rec M z s m (Fin.zero m)  →  z m   (zero ι-rule)
        fin_rec = Const("Fin.rec", (LZero(), ))
        fin_zero_major = App(Const("Fin.zero", ()), Lit(4))
        res = whnf(app(fin_rec, M, z, s, Lit(5), fin_zero_major), env, lctx,
                   mctx)
        assert res == App(z, Lit(4))

        # Fin.rec M z s m (Fin.succ m k)  →  s m k (Fin.rec ... m k)
        k = Const("k", ())
        fin_succ_major = app(Const("Fin.succ", ()), Lit(4), k)
        res = whnf(app(fin_rec, M, z, s, Lit(5), fin_succ_major), env, lctx,
                   mctx)
        head, args = _fn_args(res)
        assert head == s and args[:2] == [Lit(4), k]

        # Vec.rec A M nil cons n (Vec.cons A 0 hd Vec.nil)  →  cons branch
        vec_rec = Const("Vec.rec", (LZero(), ))
        nil, cons, hd = Const("nil", ()), Const("cons", ()), Const("hd", ())
        vec_cons_major = app(Const("Vec.cons", ()), A, Lit(0), hd,
                             App(Const("Vec.nil", ()), A))
        res = whnf(app(vec_rec, A, M, nil, cons, Lit(1), vec_cons_major), env,
                   lctx, mctx)
        head, args = _fn_args(res)
        assert head == cons and args[0] == Lit(0) and args[1] == hd
