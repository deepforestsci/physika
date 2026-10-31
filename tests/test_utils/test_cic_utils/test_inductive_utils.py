from physika.core.expr import (
    App,
    BVar,
    Const,
    ForallE,
    Lam,
    LetE,
    MData,
    Proj,
    TYPE_0,
)
import pytest

from physika.core.expr import BinderInfo, FVar, FVarId, PROP
from physika.core.inductive import (Constructor, InductiveDecl, Recursor,
                                    RecursorRule, mk_builtin_env)
from physika.utils.cic_utils.expr_utils import get_app_fn_args
from physika.utils.cic_utils.inductive_utils import (
    app_all,
    check_positivity_for_inductive,
    cons_case_returning,
    const_vec_motive,
    decl_is_prop_sorted,
    derive_recursor,
    dim_vec_nth,
    int_eqb_case,
    int_lit,
    int_rec_app,
    mk_bool_decl,
    mk_dim_vec_literal,
    mk_fin_decl,
    mk_int_decl,
    mk_int_eqb_value,
    mk_nat_decl,
    mk_nat_eqb_value,
    mk_prod_decl,
    mk_vec_decl,
    mk_vec_literal,
    name_appears,
    natrec,
    open_index_tele,
    read_dim_vec_literal,
    reg_autodiff,
    reg_dim_ops,
    reg_int_ops,
    reg_mat_ops,
    reg_nat_ops,
    reg_ofnat,
    reg_real_ops,
    reg_vec_ops,
    self_reference_indices,
    strict_positive_check,
    vec_rec_app,
    verify_recursor_rules,
)
from physika.core.environment import ConstantInfo, Environment, InductiveInfo
from physika.core.level import LZero
from physika.core.expr import NatLit, Lit
from physika.core.reduction import is_def_eq, lit_nat_int, whnf
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext


class TestNameAppears:
    """
    Tests for ``name_appears``
    """

    def test_name_appears(self):
        """
        Checks ``Const`` node with ``name``.
        """
        nat = Const("Nat", ())

        assert name_appears("Nat", nat) is True

        # Not registered name should fail
        nat = Const("Nat", ())

        assert name_appears("Bool", nat) is False

    def test_name_appears_in_app_func(self):
        """
        Checks ``name`` is found inside a function's application
        ``App``.
        """
        nat = Const("Nat", ())
        call = App(nat, Const("x", ()))

        assert name_appears("Nat", call) is True

        # should also find "name" when is applied as an arg
        nat = Const("Nat", ())
        call = App(Const("f", ()), nat)

        assert name_appears("Nat", call) is True

    def test_name_appears_in_lam(self):
        """
        Checks `name` is inside a ``Lam``.
        """

        nat = Const("Nat", ())
        lam = Lam("x", nat, Const("x", ()))

        # should be in Lam's binder type
        assert name_appears("Nat", lam) is True

        # should also be able to found "name" in body
        lam = Lam("x", Const("Real", ()), nat)

        assert name_appears("Nat", lam) is True

    def test_name_appears_in_forall(self):
        """
        Checks `name` is found in a ``ForallE``.
        """
        nat = Const("Nat", ())
        pi = ForallE("x", nat, Const("Real", ()))

        # should be in ForallE binder type
        assert name_appears("Nat", pi) is True

        # should also be able to found "name" in body
        nat = Const("Nat", ())
        pi = ForallE("x", Const("Real", ()), nat)

        assert name_appears("Nat", pi) is True

    def test_name_appears_in_lete(self):
        """
        Checks `name` is found in a ``LetE``
        """
        nat = Const("Nat", ())
        let = LetE("x", nat, Const("v", ()), Const("b", ()))

        # should be in LetE declared type
        assert name_appears("Nat", let) is True

        # should also appear in bound type
        let = LetE("x", Const("Real", ()), nat, Const("b", ()))
        assert name_appears("Nat", let) is True

        # should also appear in body
        let = LetE("x", Const("Real", ()), Const("v", ()), nat)
        assert name_appears("Nat", let) is True

    def test_name_appears_in_mdata(self):
        """
        Checks `name` is found in``MData`` wrapper.
        """
        nat = Const("Nat", ())
        wrapped = MData((("line", 1), ), nat)

        assert name_appears("Nat", wrapped) is True

    def test_name_appears_in_proj(self):
        """
        Checks `name` is in an instance of ``Proj``.
        """
        nat = Const("Nat", ())
        proj = Proj("Ray", 0, nat)

        assert name_appears("Nat", proj) is True


class TestStrictPositiveCheck:
    """
    Tests for ``strict_positive_check``
    """

    def test_direct_self_application(self):
        """
        Checks ``Vec.cons``'s field ``tl : Vec alpha n``
        """
        vec = Const("Vec", ())
        alpha = Const("Real", ())
        n = Const("n", ())
        # field: Vec α n
        field = App(App(vec, alpha), n)
        # self application must be positive
        assert strict_positive_check("Vec", field,
                                     lambda name: name == "Vec") is True

    def test_negative_in_arrow_domai(self):
        """
        Checks a type name appearing in the domain of an arrow
        type is rejected.
        """
        bad = Const("Bad", ())
        nat = Const("Nat", ())

        domain = ForallE("_", bad, nat)
        # bad type at the left should fail positivity check
        assert strict_positive_check("Bad", domain,
                                     lambda name: name == "Bad") is False

        domain = ForallE("_", nat, bad)
        # bad after nat in arrow type should be positive
        assert strict_positive_check("Bad", domain,
                                     lambda name: name == "Bad") is True

    def test_nested_positive_via_another_inductive_former(self):
        """
        Checks ``List Tree`` is positive when ``List`` is an inductive type
        former.
        """
        lst = Const("List", ())
        tree = Const("Tree", ())
        field = App(lst, tree)
        is_former = lambda name: name in ("List", "Tree")  # noqa: E731

        assert strict_positive_check("Tree", field, is_former) is True

    def test_rejected_when_head_is_not_an_inductive_former(self):
        """
        Checks ``f (Fix f)``, where ``f`` is a bound variable fails positivity
        test.
        """
        fix = Const("Fix", ())
        f_param = BVar(0)
        fix_f = App(fix, f_param)
        weird = App(f_param, fix_f)

        assert strict_positive_check("Fix", weird,
                                     lambda name: name == "Fix") is False


class TestCheckPositivityForInductive:
    """
    Tests for ``check_positivity_for_inductive``
    """

    def test_nat_passes(self):
        """
        Checks positivity for ``Nat`` inductive type
        """
        nat = Const("Nat", ())
        nat_decl = InductiveDecl(
            name="Nat",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(
                Constructor("Nat.zero", nat),
                Constructor("Nat.succ", ForallE("n", nat, nat)),
            ),
            is_recursive=True,
        )

        assert check_positivity_for_inductive(nat_decl) is None

    def test_bad_inductive_type(self):
        """
        Checks ``Bad.bad : (Bad -> Nat) -> Bad`` fails and an error message
        is reported.
        """
        bad = Const("Bad", ())
        nat = Const("Nat", ())
        bad_ctor_type = ForallE("x", ForallE("_", bad, nat), bad)
        bad_decl = InductiveDecl(
            name="Bad",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Bad.bad", bad_ctor_type), ),
            is_recursive=True,
        )

        result = check_positivity_for_inductive(bad_decl)
        assert result == "constructor 'Bad.bad' violates strict positivity: 'Bad' appears in a negative position in a field type"  # noqa: E501

    def test_custom_predicate_allows_nested_inductive(self):
        """
        Checks positivity ``Tree.node : List Tree -> Tree`` passes if
        ``List`` is an inductive type former.
        """
        lst = Const("List", ())
        tree = Const("Tree", ())
        tree_ctor_type = ForallE("x", App(lst, tree), tree)
        tree_decl = InductiveDecl(
            name="Tree",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Tree.node", tree_ctor_type), ),
            is_recursive=True,
        )

        def is_former(name):
            return name in ("List", "Tree")

        assert check_positivity_for_inductive(tree_decl, is_former) is None


# Helper functions for testing inductive utls
def list_decl() -> InductiveDecl:
    """
    Builds a new inductive type ``List α``.

    One uniform param, a nullary and a directly recursive
    constructor."""
    lst = Const("List", ())
    nil_t = ForallE("α", TYPE_0, App(lst, BVar(0)), BinderInfo.IMPLICIT)
    cons_t = ForallE(
        "α", TYPE_0,
        ForallE(
            "hd", BVar(0),
            ForallE("tl", App(lst, BVar(1)), App(lst, BVar(2)),
                    BinderInfo.DEFAULT), BinderInfo.DEFAULT),
        BinderInfo.IMPLICIT)
    return InductiveDecl(name="List",
                         level_params=(),
                         num_params=1,
                         type=ForallE("α", TYPE_0, TYPE_0, BinderInfo.DEFAULT),
                         constructors=(Constructor("List.nil", nil_t),
                                       Constructor("List.cons", cons_t)),
                         is_recursive=True)


def whnf_builtin(expr):
    """
    Reduce``expr`` to whnf in ``mk_builtin_env()``.
    """
    return whnf(expr, mk_builtin_env(), LocalContext(), MetaVarContext())


def apply(fn, *xs):
    """
    Apply ``fn`` to each ``xs`` fom left to right.
    """
    for x in xs:
        fn = App(fn, x)
    return fn


def def_eq_builtin(a, b):
    """
    Helper testing fucntion for checking if two values are definitionally
    equal in the same CIC environment.
    """
    return is_def_eq(a, b, mk_builtin_env(), LocalContext(),
                     MetaVarContext())[0]


def nat_info():
    """
    ``(env, InductiveInfo)`` for ``Nat`` created registered from
    mk_builitin_env.
    """
    env = mk_builtin_env()
    return env, env.inductives["Nat"]


def recursor_with_rules(valid: Recursor, rules) -> Recursor:
    """
    A copy of recursor ``valid`` with its ι-rules replaced by ``rules``.
    """
    return Recursor(valid.name, valid.type, valid.num_params,
                    valid.num_indices, valid.num_motives, valid.num_minors,
                    tuple(rules), valid.level_params)


def inductives_only_env() -> Environment:
    """
    An Environment with the builtin inductives and derived recursors
    registered.
    """
    env = Environment()
    for mk in (mk_nat_decl, mk_int_decl, mk_bool_decl, mk_fin_decl,
               mk_vec_decl, mk_prod_decl):
        decl = mk()
        rec = derive_recursor(decl)
        env.add_constant(
            ConstantInfo(decl.name, decl.level_params, decl.type, None))
        ctors = {
            c.name: ConstantInfo(c.name, decl.level_params, c.type, None)
            for c in decl.constructors
        }
        env.add_inductive(
            InductiveInfo(decl=decl,
                          ctors=ctors,
                          recursor=ConstantInfo(rec.name, rec.level_params,
                                                rec.type, None),
                          rec_info=rec))
    return env


class TestDeriveRecursor:
    """
    Tests for ``derive_recursor``.

    Recursor shape metadata plus a kernel check of every derived eliminiation
    rule via ``verify_recursor_rules``.
    """

    def verify(self, decl: InductiveDecl):
        """
        Helper function for deriving valid recursor rules for an inductive
        declaration.
        """
        rec = derive_recursor(decl)
        env = mk_builtin_env()
        if decl.name in env.inductives:
            ii = env.inductives[decl.name]
            assert verify_recursor_rules(
                decl,
                ii.ctors,
                ii.recursor,
                ii.rec_info,  # type: ignore[arg-type]
                env) is None
            return ii.rec_info
        ctors = {
            c.name: ConstantInfo(c.name, (), c.type, None)
            for c in decl.constructors
        }
        rec_ci = ConstantInfo(rec.name, rec.level_params, rec.type, None)
        env.add_inductive(
            InductiveInfo(decl=decl,
                          ctors=ctors,
                          recursor=rec_ci,
                          rec_info=rec))
        return rec

    def test_list_non_indexed(self):
        """
        ``List`` indcutve type should have:
        1 param, 0 indices, 2 minorxs.
        """
        rec = self.verify(list_decl())
        assert (rec.num_params, rec.num_indices, rec.num_minors) == (1, 0, 2)
        assert rec.arity() == 1 + 1 + 2 + 0 + 1

    def test_nat_derived_shape(self):
        """
        ``Nat.rec`` should have:
        0 params, 2 minors, and the succ rule
        """
        rec = self.verify(mk_nat_decl())
        assert (rec.num_params, rec.num_indices, rec.num_minors) == (0, 0, 2)
        assert rec.arity() == 4
        succ = next(r for r in rec.rules if r.ctor_name == "Nat.succ")
        assert succ.nfields == 1

    def test_bool_derived_shape(self):
        """
        ``Bool.rec`` must have:
        0 params and one minor premise per constructor (2 in total).
        """
        rec = self.verify(mk_bool_decl())
        assert (rec.num_params, rec.num_indices, rec.num_minors) == (0, 0, 2)

    def test_fin_indexed_family(self):
        """
        ``Fin.rec`` should have:
        0 params, 1 index, 2 minors.
        """
        rec = self.verify(mk_fin_decl())
        assert (rec.num_params, rec.num_indices, rec.num_minors) == (0, 1, 2)
        assert rec.arity() == 5

    def test_vec_indexed_family(self):
        """
        ``Vec.rec`` shoudl have:
        1 param, 1 index, 2 minors.
        """
        rec = self.verify(mk_vec_decl())
        assert (rec.num_params, rec.num_indices, rec.num_minors) == (1, 1, 2)
        assert rec.arity() == 6

    # Check proper iota reduction of derived recursors

    def test_nat_recursor_reduces(self):
        """
        Checks that the derived ``Nat.rec`` acts on both constructors:
        ``Nat.rec M z s 0 -> z`` and ``Nat.rec M z s 2 -> s 1 (Nat.rec … 1)``.
        """
        M, z, s = Const("M", ()), Const("z", ()), Const("s", ())
        natrec = Const("Nat.rec", (LZero(), ))
        assert whnf_builtin(apply(natrec, M, z, s, Lit(0))) == z
        r2 = whnf_builtin(apply(natrec, M, z, s, Lit(2)))
        head, args = get_app_fn_args(r2)
        assert head == s and args[0] == Lit(NatLit(1))

    def test_vec_recursor_reduces(self):
        """
        The derived ``Vec.rec`` must compute on ``Vec.nil`` and ``Vec.cons``
        """

        A = Const("A", ())
        nilc, consc, hd = Const("nilc", ()), Const("consc",
                                                   ()), Const("hd", ())
        vecrec = Const("Vec.rec", (LZero(), ))
        nil_major = App(Const("Vec.nil", ()), A)
        assert whnf_builtin(
            apply(vecrec, A, Const("M", ()), nilc, consc, Lit(0),
                  nil_major)) == nilc
        cons_major = apply(Const("Vec.cons", ()), A, Lit(0), hd, nil_major)
        r = whnf_builtin(
            apply(vecrec, A, Const("M", ()), nilc, consc, Lit(1), cons_major))
        head, args = get_app_fn_args(r)
        assert head == consc and args[1] == hd  # consc n hd tl ih
        ih_head, _ = get_app_fn_args(args[3])
        assert ih_head == Const("Vec.rec", (LZero(), ))  # the IH call

    def test_fin_recursor_reduces(self):
        """
        The derived ``Fin.rec`` should work on both ``Fin.zero`` and
        ``Fin.succ``.
        """

        M, z, s = Const("M", ()), Const("z", ()), Const("s", ())
        finrec = Const("Fin.rec", (LZero(), ))
        # Fin.rec M z s m (Fin.zero m) -> z m
        assert whnf_builtin(
            apply(finrec, M, z, s, Lit(3), App(Const("Fin.zero", ()),
                                               Lit(2)))) == App(z, Lit(2))
        # Fin.rec M z s m (Fin.succ m k) -> s m k (Fin.rec ... m k)
        k = Const("k", ())
        r = whnf_builtin(
            apply(finrec, M, z, s, Lit(3),
                  apply(Const("Fin.succ", ()), Lit(2), k)))
        head, args = get_app_fn_args(r)
        assert head == s and args[:2] == [Lit(2), k]

    def test_nested_occurrence_is_rejected(self):
        """
        A inductive type nested inside another type former is not a shape that
        recursor derivation supports.
        """
        tree = Const("Tree", ())
        node_t = ForallE("x", App(Const("List", ()), tree), tree,
                         BinderInfo.DEFAULT)
        decl = InductiveDecl(name="Tree",
                             level_params=(),
                             num_params=0,
                             type=TYPE_0,
                             constructors=(Constructor("Tree.node", node_t), ),
                             is_recursive=True)
        with pytest.raises(ValueError):
            derive_recursor(decl)


class TestSelfReferenceIndices:
    """
    Tests for ``self_reference_indices``.
    Should classify a constructor field as a recursive occurrence or return
    None.
    """

    def test_indexed_recursive_field(self):
        """
        Checks Vec.cons's ``tl : Vec a n`` field as a recursive occurrence of
        the inductive and returns its index list [n].
        """
        a, n = FVar(FVarId("a.0")), FVar(FVarId("n.1"))
        field = App(App(Const("Vec", ()), a), n)
        assert self_reference_indices("Vec", [a], field) == [n]

    def test_non_indexed_recursive_field(self):
        """
        ``tl : List a`` is recursive with no indices.
        """
        a = FVar(FVarId("a.0"))
        assert self_reference_indices("List", [a], App(Const("List", ()),
                                                       a)) == []

    def test_field_is_not_self_reference(self):
        """
        ``hd : a`` is not an occurrence of the inductive.
        """
        a = FVar(FVarId("a.0"))
        assert self_reference_indices("Vec", [a], a) is None

    def test_param_mismatch(self):
        """
        ``Vec b n`` when the uniform parameter is ``a`` -> ``None``.
        """
        a, b, n = (FVar(FVarId("a.0")), FVar(FVarId("b.1")),
                   FVar(FVarId("n.2")))
        field = App(App(Const("Vec", ()), b), n)
        assert self_reference_indices("Vec", [a], field) is None

    def test_different_head(self):
        """
        A field headed by a different constant -> ``None``
        """
        a = FVar(FVarId("a.0"))
        field = App(Const("Other", ()), a)
        assert self_reference_indices("Vec", [a], field) is None


class TestDeclIsPropSorted:
    """
    Tests for ``decl_is_prop_sorted``.
    """

    def test_type_sorted_is_false(self):
        """
        ``Nat : Type0`` is not Prop sort.
        """
        assert decl_is_prop_sorted(mk_nat_decl()) is False

    def test_prop_sorted_is_true(self):
        """A declaration whose type is ``Prop``."""
        decl = InductiveDecl(name="MyProp",
                             level_params=(),
                             num_params=0,
                             type=PROP,
                             constructors=(Constructor("MyProp.mk",
                                                       Const("MyProp", ())), ),
                             is_recursive=False)
        assert decl_is_prop_sorted(decl) is True

    def test_remove_telescope_before_checking_the_sort(self):
        """``∀ (a : Nat), Prop`` counts as Prop sort."""
        decl = InductiveDecl(name="MyProp",
                             level_params=(),
                             num_params=0,
                             type=ForallE("a", Const("Nat", ()), PROP,
                                          BinderInfo.DEFAULT),
                             constructors=(),
                             is_recursive=False)
        assert decl_is_prop_sorted(decl) is True


class TestOpenIndexTele:
    """
    Tests for ``open_index_tele``.

    Open ``K`` telescope binders as typed FVars.
    """

    TELE = ForallE("n", Const("Nat", ()), TYPE_0, BinderInfo.DEFAULT)

    def test_opens_k_fresh_typed_fvars(self):
        """
        ``K=1`` gives one FVar and a context with its type.
        """
        lctx, idx = open_index_tele(LocalContext(), self.TELE, 1)
        assert len(idx) == 1
        assert lctx.fvar_type(idx[0]) == Const("Nat", ())

    def test_k_zero(self):
        """``K=0`` returns the context untouched and no FVars."""
        lctx0 = LocalContext()
        lctx, idx = open_index_tele(lctx0, self.TELE, 0)
        assert idx == [] and lctx is lctx0


class TestAppAll:
    """
    Tests for ``app_all``.
    """

    def test_no_args_returns_fn(self):
        """No arguments shpuld leave the head expression unchanged."""
        f = Const("f", ())
        assert app_all(f, []) == f

    def test_left_nests_applications(self):
        """
        ``app_all(f, [a, b])`` == ``App(App(f, a), b)``.
        """
        f, a, b = Const("f", ()), Const("a", ()), Const("b", ())
        assert app_all(f, [a, b]) == App(App(f, a), b)


class TestVerifyRecursorRules:
    """
    Tests for ``verify_recursor_rules``.
    """

    def test_accepts_nat_recursor(self):
        """
         Drrived ``Nat.rec`` should pass.
         """
        env, ii = nat_info()
        assert verify_recursor_rules(ii.decl, ii.ctors, ii.recursor,
                                     ii.rec_info, env) is None

    def test_rejects_a_wrong_rhs(self):
        """A rule whose rhs is a loose ``BVar` is wrong."""
        env, ii = nat_info()
        bad = recursor_with_rules(ii.rec_info, (RecursorRule(
            "Nat.zero", 0, BVar(7)) if r.ctor_name == "Nat.zero" else r
                                                for r in ii.rec_info.rules))
        assert isinstance(
            verify_recursor_rules(ii.decl, ii.ctors, ii.recursor, bad, env),
            str)

    def test_wrong_nfields(self):
        """A rule with more fields than the constructor is an error."""
        env, ii = nat_info()
        bad = recursor_with_rules(ii.rec_info, (RecursorRule(
            "Nat.succ", 5, r.rhs) if r.ctor_name == "Nat.succ" else r
                                                for r in ii.rec_info.rules))
        assert isinstance(
            verify_recursor_rules(ii.decl, ii.ctors, ii.recursor, bad, env),
            str)

    def test_rejects_a_missing_rule(self):
        """A recursor missing a constructor's rule shoudl produce an error."""
        env, ii = nat_info()
        bad = recursor_with_rules(
            ii.rec_info,
            (r for r in ii.rec_info.rules if r.ctor_name == "Nat.zero"))
        assert isinstance(
            verify_recursor_rules(ii.decl, ii.ctors, ii.recursor, bad, env),
            str)


class TestInductiveDecls:
    """
    Tests for delaration builders.

    Checks genreal shape of:
    name, arity, constructor list.
    Each declaration shpuld derive a recursor without error.
    """

    def test_nat(self):
        """Test for Nat inductive type"""
        d = mk_nat_decl()
        assert (d.name, d.num_params, d.is_recursive) == ("Nat", 0, True)
        assert [c.name for c in d.constructors] == ["Nat.zero", "Nat.succ"]

    def test_int(self):
        """Test for Int inductive declaration"""
        d = mk_int_decl()
        assert (d.name, d.is_recursive) == ("Int", False)
        assert [c.name for c in d.constructors] == ["Int.ofNat", "Int.negSucc"]

    def test_bool(self):
        """Test for Bool inductive type"""
        d = mk_bool_decl()
        assert [c.name for c in d.constructors] == ["Bool.false", "Bool.true"]

    def test_fin(self):
        """Test for Fin inductive type"""
        d = mk_fin_decl()
        assert (d.name, d.num_params) == ("Fin", 0)
        assert [c.name for c in d.constructors] == ["Fin.zero", "Fin.succ"]

    def test_vec(self):
        """Test for Vec inductive type"""
        d = mk_vec_decl()
        assert (d.name, d.num_params, d.is_recursive) == ("Vec", 1, True)
        assert [c.name for c in d.constructors] == ["Vec.nil", "Vec.cons"]

    def test_prod(self):
        """Test for Prod inductive type"""
        d = mk_prod_decl()
        assert (d.name, d.num_params) == ("Prod", 2)
        assert [c.name for c in d.constructors] == ["Prod.mk"]

    def test_decl_derives_a_recursor(self):
        """Every builtin declaration is a shape ``derive_recursor`` handles."""
        for mk in (mk_nat_decl, mk_int_decl, mk_bool_decl, mk_fin_decl,
                   mk_vec_decl, mk_prod_decl):
            assert derive_recursor(mk()).name.endswith(".rec")


class TestNatValues:
    """
    Tests for ``mk_nat_add/mul/pred/sub_value``.
    """

    def test_add(self):
        """Test Nat.add is properly reduced to whnf"""
        expr = App(App(Const("Nat.add", ()), Lit(2)), Lit(3))
        assert lit_nat_int(whnf_builtin(expr)) == 5

    def test_mul(self):
        """Test Nat.mul is properly reduced to whnf"""
        expr = App(App(Const("Nat.mul", ()), Lit(3)), Lit(4))
        assert lit_nat_int(whnf_builtin(expr)) == 12

    def test_pred(self):
        """Test Nat.pred is properly reduced to whnf"""
        assert lit_nat_int(whnf_builtin(App(Const("Nat.pred", ()),
                                            Lit(5)))) == 4
        # pred 0 = 0 , which is Nat.zero constructor
        assert whnf_builtin(App(Const("Nat.pred", ()),
                                Lit(0))) == Const("Nat.zero", ())

    def test_sub_saturates_at_zero(self):
        """Test Nat.sub is properly reduced to whnf"""
        assert lit_nat_int(
            whnf_builtin(App(App(Const("Nat.sub", ()), Lit(5)), Lit(2)))) == 3
        assert lit_nat_int(
            whnf_builtin(App(App(Const("Nat.sub", ()), Lit(2)), Lit(5)))) == 0

    def test_natrec(self):
        """``natrec`` applies ``Nat.rec``: base case at 0, step otherwise."""
        nat = Const("Nat", ())
        motive = Lam("_", nat, nat, BinderInfo.DEFAULT)
        step = Lam("k", nat, Lam("ih", nat, BVar(0), BinderInfo.DEFAULT),
                   BinderInfo.DEFAULT)
        assert lit_nat_int(whnf_builtin(natrec(motive, Lit(7), step,
                                               Lit(0)))) == 7
        assert lit_nat_int(whnf_builtin(natrec(motive, Lit(7), step,
                                               Lit(3)))) == 7

    def test_eqb(self):
        """``Nat.eqb`` reduces to ``Bool.true`` / ``Bool.false``."""

        def eqb(a, b):
            return App(App(Const("Nat.eqb", ()), a), b)

        true, false = Const("Bool.true", ()), Const("Bool.false", ())
        assert whnf_builtin(eqb(Lit(3), Lit(3))) == true
        assert whnf_builtin(eqb(Lit(0), Lit(0))) == true
        assert whnf_builtin(eqb(Lit(4), Lit(5))) == false
        assert whnf_builtin(eqb(Lit(0), Lit(2))) == false
        # operands are reduced first
        two_plus_three = App(App(Const("Nat.add", ()), Lit(2)), Lit(3))
        assert whnf_builtin(eqb(two_plus_three, Lit(5))) == true
        # applying the value directly gives the same result
        assert whnf_builtin(App(App(mk_nat_eqb_value(), Lit(2)),
                                Lit(2))) == true


class TestIntLit:
    """
    Tests for ``int_lit``.
    """

    def test_int_of_nat_neg_succ(self):
        """``k >= 0`` encodes as ``Int.ofNat k`` and ``k < 0`` should be a
        ``Int.negSucc (-k - 1)`` expression."""
        assert int_lit(3) == App(Const("Int.ofNat", ()), Lit(3))
        assert int_lit(0) == App(Const("Int.ofNat", ()), Lit(0))

        # k <0
        assert int_lit(-1) == App(Const("Int.negSucc", ()), Lit(0))
        assert int_lit(-2) == App(Const("Int.negSucc", ()), Lit(1))


class TestReadDimVecLiteral:
    """
    Tests for ``read_dim_vec_literal``.
    """

    def test_reads_a_literal_chain(self):
        """
        A ``Vec Int`` cons/nil chain should produce a correct ``list[int]``.
        """
        v = mk_vec_literal(Const("Int", ()), [int_lit(4), int_lit(-1)])

        assert read_dim_vec_literal(v, mk_builtin_env(), LocalContext(),
                                    MetaVarContext()) == [4, -1]

    def test_term_reduction(self):
        """An unreduced ``dim_mul`` application is WHNF reduced."""
        a = mk_dim_vec_literal([1, 0, -2, 0, 0, 0, 0])
        b = mk_dim_vec_literal([0, 1, 0, 0, 0, 0, 0])
        prod = App(App(Const("dim_mul", ()), a), b)

        assert read_dim_vec_literal(prod, mk_builtin_env(), LocalContext(),
                                    MetaVarContext()) == [
                                        1, 1, -2, 0, 0, 0, 0
                                    ]

    def test_non_vector(self):
        """A non ``Vec.cons`` / ``Vec.nil`` term gives ``None``."""
        assert read_dim_vec_literal(Const("dim_mul", ()), mk_builtin_env(),
                                    LocalContext(), MetaVarContext()) is None


class TestMkVecLiteral:
    """
    Tests for ``mk_vec_literal``.
    """

    def test_empty_is_vec_nil(self):
        """An empty element list is just ``Vec.nil elem_type``."""
        v = mk_vec_literal(Const("Int", ()), [])

        assert v == App(Const("Vec.nil", ()), Const("Int", ()))

    def test_cons_chain(self):
        """``Vec.cons`` chain should produce a list of Int elements."""
        v = mk_vec_literal(
            Const("Int", ()),
            [int_lit(1), int_lit(0), int_lit(-2)])

        assert read_dim_vec_literal(v, mk_builtin_env(), LocalContext(),
                                    MetaVarContext()) == [1, 0, -2]

    def test_cons_matches_tail_length(self):
        """Each ``Vec.cons`` is indexed by the length of its tail."""
        v = mk_vec_literal(Const("Int", ()), [int_lit(7), int_lit(8)])
        head, args = get_app_fn_args(v)

        # tail length should be 1
        assert head == Const("Vec.cons", ()) and args[1] == Lit(1)


class TestMkDimVecLiteral:
    """
    Tests for ``mk_dim_vec_literal``.
    """

    def test_SI_exponents(self):
        """A SI exponent sequence should output an equal object."""
        si_dims = [1, 1, -2, 0, 0, 0, 0]
        v = mk_dim_vec_literal(si_dims)  # kg*m*s^-2

        assert len(si_dims) == 7
        assert read_dim_vec_literal(v, mk_builtin_env(), LocalContext(),
                                    MetaVarContext()) == si_dims

    def test_wrong_si_rejected(self):
        """A sequence that is not 7 entries long is an error."""
        non_si_dims = [1, 2, 3]
        assert len(non_si_dims) != 7
        with pytest.raises(ValueError):
            mk_dim_vec_literal(non_si_dims)


class TestConstVecMotive:
    """
    Tests for ``const_vec_motive``.
    """

    def test_binds_length_then_vector(self):
        """The motive is ``Λ (n : Nat) (xs : Vec Int n). result_type``."""
        m = const_vec_motive(Const("Int", ()))

        assert (m.binder_name, m.body.binder_name) == ("n", "xs")

    def test_body_result_type(self):
        """Body ignores both binders and is ``result_type``."""
        m = const_vec_motive(Const("Int", ()))

        assert m.body.body == Const("Int", ())


class TestConsCaseReturning:
    """
    Tests for ``cons_case_returning``.
    """

    def test_pick_head(self):
        """``pick=2`` returns the ``hd`` binder under nested lambda
        applications ``n, hd, tl, ih``."""
        cc = cons_case_returning(Const("Int", ()), pick=2)

        assert cc.body.body.body.body == BVar(2)

    def test_pick_tail(self):
        """``pick=1`` returns the ``tl`` binder."""
        cc = cons_case_returning(Const("Int", ()), pick=1)

        assert cc.body.body.body.body == BVar(1)


class TestVecRecApp:
    """
    Tests for ``vec_rec_app``.
    """

    def test_reduces_vec_nil(self):
        """``Vec.rec`` of an empty vector returns ``nil``."""
        nil = App(Const("Vec.nil", ()), Const("Int", ()))
        # 7 is Vec.nil case and 9 is Vec.cons case
        t = vec_rec_app(const_vec_motive(Const("Int", ())), Lit(7), Lit(9),
                        Lit(0), nil)

        assert whnf_builtin(t) == Lit(7)

    def test_reduces_on_vec_cons(self):
        """``Vec.rec`` on a cons chain uses ``cons_case`` premise."""
        v = mk_vec_literal(Const("Int", ()), [int_lit(5), int_lit(6)])
        t = vec_rec_app(const_vec_motive(Const("Int", ())), int_lit(0),
                        cons_case_returning(Const("Int", ()), pick=2), Lit(2),
                        v)

        assert whnf_builtin(t) == int_lit(5)


class TestDimVecNth:
    """
    Tests for ``dim_vec_nth``.
    """

    def test_reads_each_position(self):
        """``dim_vec_nth`` extracts the ``k`` indexed ``Int`` exponent."""
        v = mk_dim_vec_literal([1, 0, -2, 0, 0, 0, 0])  # kg*s^-2

        assert whnf_builtin(dim_vec_nth(v, 0)) == int_lit(1)
        assert whnf_builtin(dim_vec_nth(v, 1)) == int_lit(0)
        # a negative exponent is ``Int.negSucc``
        assert whnf_builtin(dim_vec_nth(v, 2)) == int_lit(-2)


class TestIntRecApp:
    """
    Tests for ``int_rec_app``.
    """

    NAT = Const("Nat", ())

    def branches(self):
        """``(on_ofnat, on_negsucc)`` returning ``0`` and ``1``."""
        return (Lam("n", self.NAT, Lit(0), BinderInfo.DEFAULT),
                Lam("n", self.NAT, Lit(1), BinderInfo.DEFAULT))

    def test_selects_negsucc_branch(self):
        """The ``on_negsucc`` branch fires for an ``Int.negSucc`` major."""
        on_ofnat, on_negsucc = self.branches()
        t = int_rec_app(on_ofnat, on_negsucc,
                        App(Const("Int.negSucc", ()), Lit(3)))

        assert whnf_builtin(t) == Lit(1)

    def test_selects_ofnat_branch(self):
        """The ``on_ofnat`` branch fires for an ``Int.ofNat`` major."""
        on_ofnat, on_negsucc = self.branches()
        t = int_rec_app(on_ofnat, on_negsucc,
                        App(Const("Int.ofNat", ()), Lit(3)))

        assert whnf_builtin(t) == Lit(0)


class TestIntValues:
    """
    Tests for ``mk_int_neg/subNatNat/add/sub/mul_value`` Int operators.
    """

    def test_neg(self):
        """
        ``Int.neg`` should produce a negative int if input is positve
        and viceversa.
        """
        assert def_eq_builtin(App(Const("Int.neg", ()), int_lit(2)),
                              int_lit(-2))
        assert def_eq_builtin(App(Const("Int.neg", ()), int_lit(-3)),
                              int_lit(3))

    def test_subnatnat(self):
        """``Int.subNatNat m n`` is ``m - n```."""
        nat_subs = App(App(Const("Int.subNatNat", ()), Lit(2)), Lit(5))

        assert def_eq_builtin(nat_subs, int_lit(-3))

        # Int.add uses ``Int.subNatNat``
        s = App(App(Const("Int.add", ()), int_lit(2)), int_lit(-5))

        assert def_eq_builtin(s, int_lit(-3))

    def test_sub(self):
        """``Int.sub t1 t2`` should produce a correct int result."""
        s = App(App(Const("Int.sub", ()), int_lit(1)), int_lit(3))
        assert def_eq_builtin(s, int_lit(-2))

    def test_mul_negative_operand(self):
        """``Int.mul`` should get the correct sign for negative values."""
        m = App(App(Const("Int.mul", ()), int_lit(-2)), int_lit(3))

        assert def_eq_builtin(m, int_lit(-6))

    def test_int_eqb_case(self):
        """``int_eqb_case`` compares only same-constructor payloads."""
        true, false = Const("Bool.true", ()), Const("Bool.false", ())
        # first argument was ofNat 2
        ofnat = int_eqb_case(True)
        assert whnf_builtin(App(App(ofnat, Lit(2)), int_lit(2))) == true
        assert whnf_builtin(App(App(ofnat, Lit(2)), int_lit(3))) == false
        assert whnf_builtin(App(App(ofnat, Lit(2)), int_lit(-3))) == false
        # first argument was negSucc 2 (= -3)
        negsucc = int_eqb_case(False)
        assert whnf_builtin(App(App(negsucc, Lit(2)), int_lit(-3))) == true
        assert whnf_builtin(App(App(negsucc, Lit(2)), int_lit(2))) == false

    def test_eqb(self):
        """``Int.eqb`` reduces to ``Bool.true`` / ``Bool.false``."""

        def eqb(a, b):
            return App(App(Const("Int.eqb", ()), int_lit(a)), int_lit(b))

        true, false = Const("Bool.true", ()), Const("Bool.false", ())
        assert whnf_builtin(eqb(-2, -2)) == true
        assert whnf_builtin(eqb(0, 0)) == true
        assert whnf_builtin(eqb(3, -3)) == false
        assert whnf_builtin(eqb(-1, 0)) == false
        # applying the value directly gives the same result
        value = App(App(mk_int_eqb_value(), int_lit(-4)), int_lit(-4))
        assert whnf_builtin(value) == true


class TestDimValues:
    """
    Tests for ``mk_dim_componentwise_value`` / ``mk_dim_pow_value`` from
    ``mk_dim_vec_literal``.
    """

    def read(self, expr):
        """
        Helper method for kernel reduction of an expression and return it back
        as a ``list[int]``.
        """
        return read_dim_vec_literal(expr, mk_builtin_env(), LocalContext(),
                                    MetaVarContext())

    def test_dim_mul_adds_exponents(self):
        """``dim_mul`` adds ``Int`` exponents componentwise."""
        a = mk_dim_vec_literal([1, 1, -2, 0, 0, 0, 0])
        b = mk_dim_vec_literal([0, -1, 1, 0, 0, 0, 0])

        assert self.read(App(App(Const("dim_mul", ()), a),
                             b)) == [1, 0, -1, 0, 0, 0, 0]

    def test_dim_div_subtracts_exponents(self):
        """``dim_div`` subtracts ``Int`` exponents componentwise."""
        a = mk_dim_vec_literal([1, 1, -2, 0, 0, 0, 0])
        b = mk_dim_vec_literal([0, 1, 0, 0, 0, 0, 0])

        assert self.read(App(App(Const("dim_div", ()), a),
                             b)) == [1, 0, -2, 0, 0, 0, 0]

    def test_dim_pow_scales_every_exponent(self):
        """``dim_pow u p`` multiplies each exponent of ``u`` by ``p``."""
        u = mk_dim_vec_literal([1, 1, -2, 0, 0, 0, 0])  # kg*m*s^-2

        assert self.read(App(App(Const("dim_pow", ()), u),
                             int_lit(2))) == [2, 2, -4, 0, 0, 0, 0]
        # a power of -1 negates every exponent
        assert self.read(App(App(Const("dim_pow", ()), u),
                             int_lit(-1))) == [-1, -1, 2, 0, 0, 0, 0]


class TestRegOps:
    """
    Tests for helpers that register operations of inductive types.
    """

    def test_reg_nat_ops(self):
        """
        ``reg_nat_ops`` registers ``Nat.add`` with a real body and
        comparisons (``Nat.ite``) as axioms.
        """
        env = inductives_only_env()
        reg_nat_ops(env)
        assert env.constants["Nat.add"].value is not None
        assert env.constants["Nat.eqb"].value is not None
        assert env.constants["Nat.ltb"].value is None  # axiom
        assert "Nat.ite" in env.constants

    def test_reg_real_ops(self):
        """
        ``reg_real_ops`` registers ``Real``, its operators and functions
        as opaque axioms."""
        env = inductives_only_env()
        reg_real_ops(env)
        for n in ("Real", "Real.add", "Real.ite", "log", "Nat.toReal"):
            assert n in env.constants
        assert env.constants["Real.add"].value is None

    def test_reg_autodiff(self):
        """``reg_autodiff`` registers the ``grad`` / ``Vec.grad`` axioms."""
        env = inductives_only_env()
        reg_real_ops(env)
        reg_autodiff(env)
        assert "grad" in env.constants and "Vec.grad" in env.constants

    def test_reg_ofnat(self):
        """``reg_ofnat`` registers the ``OfNat`` struct plus a reducible
        ``instOfNatNat`` and an opaque ``instOfNatReal``."""
        env = inductives_only_env()
        reg_real_ops(env)
        reg_ofnat(env)
        assert "OfNat" in env.inductives
        assert env.constants["instOfNatNat"].value is not None
        assert env.constants["instOfNatReal"].value is None

    def test_reg_vec_ops(self):
        """
        ``reg_vec_ops`` registers vector operators (``Vec.get`` /
        ``Vec.tabulate`` / ``Vec.foldl``) and ``Fin.ofNat``.
        """
        env = inductives_only_env()
        reg_nat_ops(env)
        reg_real_ops(env)
        reg_vec_ops(env)
        for n in ("Vec.dot", "Vec.get", "Vec.tabulate", "Vec.foldl",
                  "Fin.ofNat"):
            assert n in env.constants

    def test_reg_mat_ops(self):
        """
        ``reg_mat_ops`` registers four matrix operators.
        """
        env = inductives_only_env()
        reg_nat_ops(env)
        reg_real_ops(env)
        reg_mat_ops(env)
        for n in ("Mat.matmul", "Mat.madd", "Mat.add_scalar",
                  "Mat.concat_rows"):
            assert n in env.constants

    def test_reg_int_ops(self):
        """
        ``reg_int_ops`` registers ``Int`` arithmetic (``Int.neg`` /
        ``Int.subNatNat`` / ``Int.add`` / ``Int.sub`` / ``Int.mul``) with
        reducible bodies.
        """
        env = inductives_only_env()
        reg_int_ops(env)
        for n in ("Int.neg", "Int.subNatNat", "Int.add", "Int.sub", "Int.mul",
                  "Int.eqb"):
            assert env.constants[n].value is not None

    def test_reg_dim_ops(self):
        """
        ``reg_dim_ops`` registers the ``DimVec`` alias, ``dim.one`` and the
        componentwise ``dim_mul`` / ``dim_div`` / ``dim_pow`` operators.
        """
        env = inductives_only_env()
        reg_nat_ops(env)
        reg_int_ops(env)
        reg_dim_ops(env)
        assert "DimVec" in env.constants
        assert env.constants["dim.one"].value is not None
        for n in ("dim_mul", "dim_div", "dim_pow"):
            assert env.constants[n].value is not None
