from physika.core.expr import (
    App,
    BVar,
    Const,
    ForallE,
    FVar,
    FVarId,
    Lam,
    LetE,
    Lit,
    MData,
    MVar,
    MVarId,
    Proj,
    Sort,
    BinderInfo,
)
from physika.core.level import LParam, LSucc, LZero
from physika.utils.cic_utils.expr_utils import (
    abstract,
    abstract_fvars,
    get_app_args,
    get_app_fn,
    get_app_fn_args,
    has_mvar,
    instantiate,
    instantiate1,
    instantiate_level_params_in_expr,
    lift,
    loose_bvar_range,
    mk_arrow,
    substitute,
)


class TestGetAppFn:
    """
    Tests for ``get_app_fn``
    """

    def test_get_app_fn(self):
        """
        Checks the head of a function application.s
        """
        add = Const("Nat.add", ())
        call = App(add, Lit(2))

        assert get_app_fn(call) == add

    def test_get_app_fn_multi_arg(self):
        """
        Checks the head of a curried application chain.
        """
        add = Const("Nat.add", ())
        call = App(App(add, Lit(2)), Lit(3))

        assert get_app_fn(call) == add

        # 4 level App nesting (``Vec.cons alpha n hd tl``)
        cons = Const("Vec.cons", ())
        chain = App(App(App(App(cons, Const("Real", ())), Lit(2)), Lit(1.0)),
                    Const("u", ()))

        assert get_app_fn(chain) == cons


class TestGetAppArgs:
    """
    Tests for ``get_app_args``
    """

    def test_get_app_args(self):
        """
        Checks function application with one argument
        returns it.
        """
        add = Const("Nat.add", ())
        call = App(add, Lit(2))

        assert get_app_args(call) == [Lit(2)]

    def test_get_app_multi_args(self):
        """
        Checks 2 arguments function application returns properly
        (from left to right).
        """
        add = Const("Nat.add", ())
        call = App(App(add, Lit(2)), Lit(3))

        assert get_app_args(call) == [Lit(2), Lit(3)]

    def test_get_app_args_deeply_nested(self):
        """
        Checks a 4 argument chain function application.
        """
        cons = Const("Vec.cons", ())
        alpha = Const("Real", ())
        n = Lit(2)
        hd = Lit(1.0)
        tl = Const("u", ())
        chain = App(App(App(App(cons, alpha), n), hd), tl)

        assert get_app_args(chain) == [alpha, n, hd, tl]


class TestGetAppFnArgs:
    """
    Tests for ``get_app_fn_args``
    """

    def test_get_app_fn(self):
        """
        Checks ``get_app_fn_args`` returns the same as
        applying ``get_app_fn`` and ``get_app_args`` separately.
        """
        add = Const("Nat.add", ())
        call = App(App(add, Lit(2)), Lit(3))

        head, args = get_app_fn_args(call)

        assert head == get_app_fn(call)
        assert args == get_app_args(call)
        assert head == add
        assert args == [Lit(2), Lit(3)]


class TestMkArrow:
    """
    Tests for ``mk_arrow``
    """

    def test_builds_non_dependent_forall(self):
        """
        ``mk_arrow(A, B)`` is a ``ForallE`` with a ``"_"`` binder
        and its body shoudb be unchanged (no dependent types in the argument).
        """
        nat = Const("Nat", ())
        arrow = mk_arrow(nat, nat)  # Nat -> Nat

        assert isinstance(arrow, ForallE)
        assert arrow.binder_name == "_"
        assert arrow.binder_type == nat
        assert arrow.body == nat
        assert arrow.binder_info == BinderInfo.DEFAULT

    def test_nests_to_the_right(self):
        """
        ``mk_arrow(A, mk_arrow(B, C))`` produce ``A -> (B -> C)``.
        """
        a, b, c = Const("A", ()), Const("B", ()), Const("C", ())
        arrow = mk_arrow(a, mk_arrow(b, c))

        assert arrow.binder_type == a
        assert arrow.body == mk_arrow(b, c)


class TestAbstract:
    """
    Tests for ``abstract``
    """

    def test_abstract_fvar_hit(self):
        """
        Checks an ``FVar`` whose id is in ``fvar_ids`` becomes a
        ``BVar`` at ``depth + i``.
        """
        x = FVar(FVarId("x.0"))

        assert abstract(x, ["x.0"], 0) == BVar(0)

    def test_abstract_fvar_miss(self):
        """
        Checks an ``FVar`` whose id is not in ``fvar_ids`` is left
        unchanged.
        """
        x = FVar(FVarId("x.0"))

        assert abstract(x, ["other.9"], 0) == x

    def test_abstract_multiple_fvars_ordering(self):
        """
        Checks each ``FVar`` gets the ``BVar`` index matching its own
        position in ``fvar_ids``.
        """
        x = FVar(FVarId("x.0"))
        y = FVar(FVarId("y.1"))
        term = App(App(Const("f", ()), x), y)

        assert abstract(term, ["x.0", "y.1"],
                        0) == App(App(Const("f", ()), BVar(0)), BVar(1))

    def test_abstract_app_recurses_both_sides(self):
        """
        Checks both the function and argument side of an ``App`` are
        walked.
        """
        x = FVar(FVarId("x.0"))
        term = App(Const("f", ()), x)

        assert abstract(term, ["x.0"], 0) == App(Const("f", ()), BVar(0))

    def test_abstract_lam_increments_depth_only_in_body(self):
        """
        Checks a ``Lam``'s ``binder_type`` is abstracted at the same
        depth as the ``Lam`` itself, while its ``body`` is abstracted
        one level deeper — the type isn't under the Lam's own scope,
        only the body is.
        """
        x = FVar(FVarId("x.0"))
        lam = Lam("n", x, x, BinderInfo.DEFAULT)

        result = abstract(lam, ["x.0"], 0)

        assert result.binder_type == BVar(0)
        assert result.body == BVar(1)

    def test_abstract_forall_increments_depth_only_in_body(self):
        """
        Checks ``ForallE`` follows the same depth rule as ``Lam``.
        """
        x = FVar(FVarId("x.0"))
        forall = ForallE("n", x, x, BinderInfo.DEFAULT)

        result = abstract(forall, ["x.0"], 0)

        assert result.binder_type == BVar(0)
        assert result.body == BVar(1)

    def test_abstract_lete_increments_depth_only_in_body(self):
        """
        Checks a ``LetE``'s ``type`` and ``value`` are abstracted at
        the same depth as the ``LetE`` itself, while its ``body`` is
        one level deeper.
        """
        x = FVar(FVarId("x.0"))
        let = LetE("n", x, x, x, False)

        result = abstract(let, ["x.0"], 0)

        assert result.type == BVar(0)
        assert result.value == BVar(0)
        assert result.body == BVar(1)

    def test_abstract_mdata(self):
        """
        Checks the wrapped expression inside ``MData`` is abstracted.
        """
        x = FVar(FVarId("x.0"))
        wrapped = MData((("line", 1), ), x)

        assert abstract(wrapped, ["x.0"], 0) == MData((("line", 1), ), BVar(0))

    def test_abstract_proj(self):
        """
        Checks the struct instance inside ``Proj`` is abstracted.
        """
        x = FVar(FVarId("x.0"))
        proj = Proj("Ray", 0, x)

        assert abstract(proj, ["x.0"], 0) == Proj("Ray", 0, BVar(0))

    def test_abstract_leaf_nodes_unchanged(self):
        """
        Checks nodes that can never contain an ``FVar`` (``BVar``,
        ``Const``) are returned as the same object, not a copy.
        """
        b = BVar(0)
        real = Const("Real", ())

        assert abstract(b, ["x.0"], 0) is b
        assert abstract(real, ["x.0"], 0) is real

    def test_abstract_depth_tracks_binder_scopes_not_call_depth(self):
        """
        Checks depth only increases when the walk actually crosses
        into a binder's body — passing through an ``App`` on the way
        to a nested ``Lam`` does not, by itself, add to depth.
        """
        x = FVar(FVarId("x.0"))
        real = Const("Real", ())
        # App(f, Lam(n : Real, x)) -- one binder (the Lam) encloses x,
        # even though reaching x also means recursing through the App.
        wrapped = App(Const("f", ()), Lam("n", real, x, BinderInfo.DEFAULT))

        result = abstract(wrapped, ["x.0"], 0)

        assert result == App(Const("f", ()),
                             Lam("n", real, BVar(1), BinderInfo.DEFAULT))


class TestAbstractFvars:
    """
    Tests for ``abstract_fvars``
    """

    def test_abstract_fvars_basic(self):
        """
        Checks a single ``FVar`` is abstracted to ``BVar(0)``.
        """
        x = FVar(FVarId("x.0"))
        term = App(Const("f", ()), x)

        assert abstract_fvars(term, [x]) == App(Const("f", ()), BVar(0))

    def test_abstract_fvars_multiple_ordering(self):
        """
        Checks ``fvars[0]`` becomes the innermost ``BVar(0)`` and
        ``fvars[-1]`` the outermost.
        """
        x = FVar(FVarId("x.0"))
        y = FVar(FVarId("y.1"))
        body = App(App(Const("g", ()), x), y)

        assert abstract_fvars(body,
                              [x, y]) == App(App(Const("g", ()), BVar(0)),
                                             BVar(1))

    def test_abstract_fvars_empty_list_is_noop(self):
        """
        Checks an empty ``fvars`` list returns the exact same object,
        untouched — there's no new binder for any FVar to be
        abstracted into.
        """
        x = FVar(FVarId("x.0"))
        body = App(Const("f", ()), x)

        assert abstract_fvars(body, []) is body

    def test_abstract_fvars_delegates_to_abstract(self):
        """
        Checks ``abstract_fvars`` produces the same result as calling
        ``abstract`` directly with the matching ids and depth 0.
        """
        x = FVar(FVarId("x.0"))
        y = FVar(FVarId("y.1"))
        body = App(App(Const("g", ()), x), y)

        assert abstract_fvars(body,
                              [x, y]) == abstract(body, ["x.0", "y.1"], 0)


class TestLooseBvarRange:
    """
    Tests for ``loose_bvar_range``
    """

    def test_loose_bvars(self):
        """
        Checks loose BVars in different expression contexts
        """
        # checks a bare ``BVar(k)`` needs ``k + 1`` wrapping binders.
        assert loose_bvar_range(BVar(0)) == 1
        assert loose_bvar_range(BVar(2)) == 3

        # checks closed expressions do not have loose BVars.
        assert loose_bvar_range(Const("Nat", ())) == 0
        assert loose_bvar_range(Lit(2)) == 0

    def test_app_loose_bvars(self):
        """
        Checks ``App`` reports the deepest loose reference from either
        side.
        """
        assert loose_bvar_range(App(BVar(0), BVar(2))) == 3

    def test_lam_binder(self):
        """
        Checks ``λx. x`` is closed.
        """
        id_fn = Lam("x", Const("Nat", ()), BVar(0), BinderInfo.DEFAULT)

        assert loose_bvar_range(id_fn) == 0

    def test_lam_body_deeper(self):
        """
        Checks a body referencing one binder further out have a loose range of
        1.
        """
        frag = Lam("x", Const("Nat", ()), BVar(1), BinderInfo.DEFAULT)

        assert loose_bvar_range(frag) == 1

    def test_forall(self):
        """
        Checks ``ForallE`` treats ``binder_type``/``body`` the same way
        ``Lam`` does.
        """
        frag = ForallE("x", BVar(0), BVar(1), BinderInfo.DEFAULT)

        assert loose_bvar_range(frag) == 1

    def test_lete_body(self):
        """
        Checks ``LetE`` are measured at its own depth while ``body``
        is one level deeper.
        """
        frag = LetE("n", BVar(0), BVar(0), BVar(1), False)

        assert loose_bvar_range(frag) == 1

        closed_body = LetE("n", Const("Nat", ()), Const("Nat", ()), BVar(0),
                           False)

        assert loose_bvar_range(closed_body) == 0

    def test_mdata_proj(self):
        """
        Checks ``MData`` and ``Proj`` reports its wrapped expression's range.
        """
        assert loose_bvar_range(MData((("line", 1), ), BVar(0))) == 1
        assert loose_bvar_range(Proj("Ray", 0, BVar(0))) == 1


class TestHasMvar:
    """
    Tests for ``has_mvar``
    """

    def test_mvar(self):
        """
        Basic tests for checking if an expression contains a metavariable.
        """
        # checks a term with no ``MVar`` is False.
        assert has_mvar(Const("Nat", ())) is False

        # checks ``MVar`` is found when no specific id is given.
        m = MVar(MVarId("m.0"))
        assert has_mvar(App(Const("f", ()), m)) is True

        # checks a mvar_id is found
        m = MVar(MVarId("m.0"))
        assert has_mvar(m, MVarId("m.0")) is True
        assert has_mvar(m, MVarId("n.0")) is False

    def test_mvar_lam_binder(self):
        """
        Checks both ``binder_type`` and ``body`` of a ``Lam``expression are
        searched.
        """
        m = MVar(MVarId("m.0"))

        assert has_mvar(Lam("n", m, Const("Nat", ()),
                            BinderInfo.DEFAULT)) is True
        assert has_mvar(Lam("n", Const("Nat", ()), m,
                            BinderInfo.DEFAULT)) is True

    def test_mvar_forall(self):
        """
        Checks ``ForallE`` is searched same as ``Lam``.
        """
        m = MVar(MVarId("m.0"))

        assert has_mvar(ForallE("n", Const("Nat", ()), m,
                                BinderInfo.DEFAULT)) is True

    def test_mvar_lete(self):
        """
        Checks ``LetE`` ``type``, ``value``, and ``body`` searches for MVars.
        """
        m = MVar(MVarId("m.0"))
        nat = Const("Nat", ())

        assert has_mvar(LetE("n", m, nat, nat, False)) is True
        assert has_mvar(LetE("n", nat, m, nat, False)) is True
        assert has_mvar(LetE("n", nat, nat, m, False)) is True

    def test_mvar_mdata_and_proj(self):
        """
        Checks the wrapped expression inside ``MData``/``Proj`` looks for
        metavariables.
        """
        m = MVar(MVarId("m.0"))

        assert has_mvar(MData((), m)) is True
        assert has_mvar(Proj("Ray", 0, m)) is True


class TestLift:
    """
    Tests for ``lift``
    """

    def test_lift_bvar(self):
        """
        Checks a BVar is being shifting (up and down) properly.
        """
        # ``BVar`` at or above ``depth`` is shifted up by n.
        n = 2
        assert lift(BVar(0), 0, n) == BVar(n)

        # below depth unchanged
        assert lift(BVar(0), n - 1, n) == BVar(0)

        # app recurses both_sides
        term = App(BVar(0), BVar(1))
        assert lift(term, 0, 1) == App(BVar(1), BVar(2))

        # lam increments depth only in body
        lam = Lam("x", BVar(0), BVar(0), BinderInfo.DEFAULT)

        result = lift(lam, 0, 1)

        assert result.binder_type == BVar(1)
        assert result.body == BVar(0)

        # forallE increments depth only in body
        forall = ForallE("x", BVar(0), BVar(0), BinderInfo.DEFAULT)

        result = lift(forall, 0, 1)

        assert result.binder_type == BVar(1)
        assert result.body == BVar(0)

        # letE increments depth in body
        let = LetE("n", BVar(0), BVar(0), BVar(0), False)

        result = lift(let, 0, 1)

        assert result.type == BVar(1)
        assert result.value == BVar(1)
        assert result.body == BVar(0)

        # lift mdata and proj
        assert lift(MData((("line", 1), ), BVar(0)), 0, 1) == MData(
            (("line", 1), ), BVar(1))
        assert lift(Proj("Ray", 0, BVar(0)), 0, 1) == Proj("Ray", 0, BVar(1))


class TestSubstitute:
    """
    Tests for ``substitute``
    """

    def test_substitute(self):
        """
        Verifies that substitution is being applied correctly dependending
        on bound variables.

        """
        # bvar bound inside untouched
        assert substitute(BVar(0), [Lit(5)], 1) == BVar(0)

        # Checks a ``BVar`` is replaced by the matching entry in ``subst``
        assert substitute(BVar(0), [Lit(5)], 0) == Lit(5)

        # bvar beyond range should shifts down
        assert substitute(BVar(2), [Lit(1), Lit(2)], 0) == BVar(0)

        # crossing Lam's own binder brings depth to 1 by the time
        # the substitution reaches BVar(1) inside its body.
        lam = Lam("x", Const("T", ()), BVar(1), BinderInfo.DEFAULT)
        result = substitute(lam, [BVar(5)], 0)
        assert result.body == BVar(6)

        # app recurses both side
        term = App(BVar(0), BVar(1))

        assert substitute(term, [Lit(1), Lit(2)], 0) == App(Lit(1), Lit(2))

        # forallE increments depth only in body
        forall = ForallE("x", BVar(0), BVar(0), BinderInfo.DEFAULT)
        result = substitute(forall, [Lit(3)], 0)

        assert result.binder_type == Lit(3)
        assert result.body == BVar(0)  # bound by the ForallE's own binder

        # letE increments depth only in body
        let = LetE("n", BVar(0), BVar(0), BVar(0), False)

        result = substitute(let, [Lit(3)], 0)

        assert result.type == Lit(3)
        assert result.value == Lit(3)
        assert result.body == BVar(0)  # bound by the LetE's own binder

        # MData and Proj substitution
        assert substitute(MData((("line", 1), ), BVar(0)), [Lit(9)],
                          0) == MData((("line", 1), ), Lit(9))
        assert substitute(Proj("Ray", 0, BVar(0)), [Lit(9)],
                          0) == Proj("Ray", 0, Lit(9))


class TestInstantiate:
    """
    Tests for ``instantiate``
    """

    def test_instantiate_(self):
        """
        Tests ``instatiate`` replaces BVars with correct values.
        """
        # e innermost binder.
        assert instantiate(BVar(0), [Lit(1), Lit(2)]) == Lit(1)

        # ``BVar`` beyond ``len(subst)`` range shifts index by one
        assert instantiate(BVar(2), [Lit(1), Lit(2)]) == BVar(0)

        # implicit depth should begin at 0
        term = App(BVar(0), BVar(1))
        subst = [Lit(1), Lit(2)]

        assert instantiate(term, subst) == substitute(term, subst, 0)


class TestInstantiate1:
    """
    Tests for ``instantiate1``
    """

    def test_instantiate1_replaces_bvar0(self):
        """
        Checks ``BVar(0)`` is replaced by ``s``.
        """
        assert instantiate1(BVar(0), Lit(7)) == Lit(7)

        # checks beta reduction
        body = App(Const("f", ()), BVar(0))
        assert instantiate1(body, Lit(9)) == App(Const("f", ()), Lit(9))

        # shifts bvars beyond 0
        assert instantiate1(BVar(1), Lit(9)) == BVar(0)


class TestInstantiateLevelParamsInExpr:
    """
    Tests for ``instantiate_level_params_in_expr``
    """

    def test_instantiate_level_params_in_expr(self):
        """
        Tests LParam instantiation in different expression contexts.
        """
        # replaces LParam inside Sort
        result = instantiate_level_params_in_expr(Sort(LParam("u")), ["u"],
                                                  [LSucc(LZero())])

        assert result == Sort(LSucc(LZero()))

        # LParams in Const
        const = Const("Vec", (LParam("u"), LParam("v")))

        result = instantiate_level_params_in_expr(
            const, ["u", "v"], [LZero(), LSucc(LZero())])

        assert result == Const("Vec", (LZero(), LSucc(LZero())))

        # LParam instantiation in App, Lam, and Forall should recurse to their
        # children
        sort = Sort(LParam("u"))
        app = App(Const("f", (LParam("u"), )), sort)
        lam = Lam("x", sort, sort, BinderInfo.DEFAULT)
        forall = ForallE("x", sort, sort, BinderInfo.DEFAULT)

        params, levels = ["u"], [LZero()]
        expected_sort = Sort(LZero())

        result_app = instantiate_level_params_in_expr(app, params, levels)
        assert result_app == App(Const("f", (LZero(), )), expected_sort)

        result_lam = instantiate_level_params_in_expr(lam, params, levels)
        assert result_lam.binder_type == expected_sort
        assert result_lam.body == expected_sort

        result_forall = instantiate_level_params_in_expr(
            forall, params, levels)
        assert result_forall.binder_type == expected_sort
        assert result_forall.body == expected_sort
