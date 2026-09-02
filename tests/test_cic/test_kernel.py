from physika.core.expr import (
    App,
    BinderInfo,
    BVar,
    Const,
    Expr,
    FloatLit,
    ForallE,
    Lam,
    LetE,
    Lit,
    MData,
    NatLit,
    Proj,
    Sort,
    TYPE_0,
)
from physika.core.level import LParam, LSucc, LZero
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.environment import ConstantInfo, Environment, InductiveInfo
from physika.core.inductive import Constructor, InductiveDecl
from physika.core.kernel import KernelException, check, infer_type, proj_field_type  # noqa: E501

NAT = Const("Nat", ())


def vec_t(a: Expr, n: Expr) -> Expr:
    """
    ``Vec a n`` application that correspondes to ``Vec`` type constructor.
    """
    return App(App(Const("Vec", ()), a), n)


class TestProjFieldType:
    """
    Tests for ``proj_field_type``
    """

    def test_proj_field_type(self):
        """
        Checks ``proj_field_type`` for class fields.
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        env.add_constant(ConstantInfo("Nat", (), TYPE_0, None))

        # independent fields have num_params=0
        # Box:
        #   n: ℕ
        #   m: ℕ
        # b = Box(3, 5)
        # b.n
        # b.m
        box_ctor = ForallE("n", NAT, ForallE("m", NAT, Const("Box", ())))
        box_decl = InductiveDecl(
            name="Box",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Box.mk", box_ctor), ),
            is_recursive=False,
        )
        env.add_inductive(
            InductiveInfo(
                decl=box_decl,
                ctors={"Box.mk": ConstantInfo("Box.mk", (), box_ctor, None)},
                recursor=ConstantInfo("Box.rec", (), TYPE_0, None),
            ))
        b = App(App(Const("Box.mk", ()), Lit(NatLit(3))), Lit(NatLit(5)))
        assert proj_field_type("Box", 0, b, env, lctx, mctx) == NAT  # b.n
        assert proj_field_type("Box", 1, b, env, lctx, mctx) == NAT  # b.m

    def test_missing_proj_decl(self):
        """
        Test an error is raised for a missing class.
        """
        b = App(App(Const("Box.mk", ()), Lit(NatLit(3))), Lit(NatLit(5)))
        try:
            env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
            env.add_constant(ConstantInfo("Nat", (), TYPE_0, None))
            proj_field_type("Missing", 0, b, env, lctx, mctx)
            assert False
        except KernelException:
            pass

    def test_idx_beyond_arity(self):
        """
        Test an error is raised when idx is greater than constructor's number
        of parameters
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        box_ctor = ForallE("n", NAT, ForallE("m", NAT, Const("Box", ())))
        box_decl = InductiveDecl(
            name="Box",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Box.mk", box_ctor), ),
            is_recursive=False,
        )
        env.add_inductive(
            InductiveInfo(
                decl=box_decl,
                ctors={"Box.mk": ConstantInfo("Box.mk", (), box_ctor, None)},
                recursor=ConstantInfo("Box.rec", (), TYPE_0, None),
            ))
        b = App(App(Const("Box.mk", ()), Lit(NatLit(3))), Lit(NatLit(5)))
        try:
            proj_field_type("Box", 5, b, env, lctx,
                            mctx)  # Box just have 2 fields
            assert False, "expected KernelException"
        except KernelException:
            pass

    def test_dependent_field(self):
        """
        Test for inferring the type of a dependently typed field
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        real = Const("Real", ())
        vec3_ctor = ForallE(
            "n",
            NAT,
            ForallE("data", vec_t(real, BVar(0)), Const("Vec3", ())),
        )
        vec3_decl = InductiveDecl(
            name="Vec3",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Vec3.mk", vec3_ctor), ),
            is_recursive=False,
        )
        env.add_inductive(
            InductiveInfo(
                decl=vec3_decl,
                ctors={
                    "Vec3.mk": ConstantInfo("Vec3.mk", (), vec3_ctor, None)
                },
                recursor=ConstantInfo("Vec3.rec", (), TYPE_0, None),
            ))
        lctx3, data_fv = lctx.push_local("data", vec_t(real, Lit(NatLit(3))))
        v = App(App(Const("Vec3.mk", ()), Lit(NatLit(3))), data_fv)
        result = proj_field_type("Vec3", 1, v, env, lctx3, mctx)
        assert result == vec_t(real, Proj("Vec3", 0, v))


class TestInferType:
    """
    Tests for ``infer_type``
    """

    def test_mdata(self):
        """
        MData infers to the wrapped exprssion.
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        wrapped = MData((("line", 1), ), Lit(NatLit(5)))
        assert infer_type(wrapped, env, lctx, mctx) == NAT

    def test_sort(self):
        """
        Sort u : Sort(u+1).
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        assert infer_type(Sort(LZero()), env, lctx,
                          mctx) == Sort(LSucc(LZero()))

    def test_bvar(self):
        """
        Inferring BVar type inferrence tests for errors and proper type
        checking.
        """
        # Kernel infering a loose BVar should return error
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        try:
            infer_type(BVar(0), env, lctx, mctx)
            assert False, "expected KernelException"
        except KernelException:
            pass

        # An MVar not declared in mctx raises an error
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        try:
            infer_type(mctx.mk_mvar("ghost", lctx, NAT), env, lctx,
                       MetaVarContext())
            assert False, "expected KernelException"
        except KernelException:
            pass

        # assigned MVar infers type

        env, lctx = Environment(), LocalContext()
        mctx1 = MetaVarContext()
        mv = mctx1.mk_mvar("m", lctx, NAT)
        mctx1.expr_assignments[mv.id.id] = Lit(NatLit(3))
        assert infer_type(mv, env, lctx, mctx1) == NAT

    def test_mvar_unassigned(self):
        """
        An unassigned MVar is refused by the trusted kernel.
        """
        env, lctx = Environment(), LocalContext()
        mctx2 = MetaVarContext()
        mv2 = mctx2.mk_mvar("m", lctx, NAT)
        try:
            infer_type(mv2, env, lctx, mctx2)
            assert False, "expected KernelException"
        except KernelException:
            pass

    def test_fvar(self):
        """
        Infer type of free variables in and not in Local Context
        """
        # An FVar found in lctx should returns its declared type.
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        lctx_fv, fv = lctx.push_local("x", NAT)
        assert infer_type(fv, env, lctx_fv, mctx) == NAT

        # An FVar not present in the given lctx should raise an error
        try:
            infer_type(fv, env, lctx, mctx)  # fv not pushed into this lctx
            assert False, "expected KernelException"
        except KernelException:
            pass

    def test_const(self):
        """
        Infer type of Constant
        """
        # An unknown constant raises an error
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        try:
            infer_type(Const("Unknown", ()), env, lctx, mctx)
            assert False, "expected KernelException"
        except KernelException:
            pass

        # A universe-polymorphic constant applied with the wrong number of
        # level arguments raises.
        lctx, mctx = LocalContext(), MetaVarContext()
        env2 = Environment()
        env2.add_constant(ConstantInfo("id", ("u", ), Sort(LParam("u")), None))
        try:
            infer_type(Const("id", ()), env2, lctx,
                       mctx)  # expects 1 level, got 0
            assert False, "expected KernelException"
        except KernelException:
            pass

    def test_app(self):
        """
       Infer type of function applications (App)
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        env3 = Environment()
        env3.add_constant(ConstantInfo("n0", (), NAT, None))
        #  App function that has a non-Pi type should return error
        try:
            infer_type(App(Const("n0", ()), Lit(NatLit(1))), env3, lctx, mctx)
            assert False, "expected KernelException"
        except KernelException:
            pass

        # argument type mismatch
        lctx, mctx = LocalContext(), MetaVarContext()
        env4 = Environment()
        real = Const("Real", ())
        env4.add_constant(ConstantInfo("f", (), ForallE("x", NAT, real), None))
        try:
            infer_type(App(Const("f", ()), Lit(1.0)), env4, lctx, mctx)
            assert False, "expected KernelException"
        except KernelException:
            pass

        # substitute argument into the dependent codomain
        lctx, mctx = LocalContext(), MetaVarContext()
        real = Const("Real", ())
        env5 = Environment()
        env5.add_constant(
            ConstantInfo("vecOf", (), ForallE("n", NAT, vec_t(real, BVar(0))),
                         None))
        applied = App(Const("vecOf", ()), Lit(NatLit(3)))
        assert infer_type(applied, env5, lctx,
                          mctx) == vec_t(real, Lit(NatLit(3)))

    def test_lam(self):
        """
        Verifies proper infer of b's type of identity funciton:
        Lam x:A. b : Pi x:A.
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        identity = Lam("x", NAT, BVar(0), BinderInfo.DEFAULT)
        assert infer_type(identity, env, lctx,
                          mctx) == ForallE("x", NAT, NAT, BinderInfo.DEFAULT)

    def test_forallE(self):
        """
        Infers type of ForallE expression for dependent and non-dependet type
        cases.
        """
        # non dependent case
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        non_dep = ForallE("_", Sort(LZero()), Sort(LZero()),
                          BinderInfo.DEFAULT)
        assert infer_type(non_dep, env, lctx, mctx) == Sort(LSucc(LZero()))

        # dependent codomain (Pi (A:Type), A))
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        dependent = ForallE("A", Sort(LZero()), BVar(0), BinderInfo.DEFAULT)
        assert infer_type(dependent, env, lctx, mctx) == Sort(LZero())

    def test_let(self):
        """
        Infer type of LetE expression.
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        real = Const("Real", ())
        let_expr = LetE("x", real, Lit(2.0), BVar(0))
        assert infer_type(let_expr, env, lctx, mctx) == real

    def test_lit(self):
        """
        Test FloatLit and NatLit expressions inference
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        real = Const("Real", ())
        assert infer_type(Lit(3), env, lctx, mctx) == NAT
        assert infer_type(Lit(NatLit(3)), env, lctx, mctx) == NAT
        assert infer_type(Lit(2.0), env, lctx, mctx) == real
        assert infer_type(Lit(FloatLit(2.0)), env, lctx, mctx) == real

    def test_proj(self):
        """
        Verifies proper inference of Proj expression for class fields.
        """
        lctx, mctx = LocalContext(), MetaVarContext()
        box_ctor = ForallE("n", NAT, ForallE("m", NAT, Const("Box", ())))
        box_decl = InductiveDecl(
            name="Box",
            level_params=(),
            num_params=0,
            type=TYPE_0,
            constructors=(Constructor("Box.mk", box_ctor), ),
            is_recursive=False,
        )
        env6 = Environment()
        env6.add_inductive(
            InductiveInfo(
                decl=box_decl,
                ctors={"Box.mk": ConstantInfo("Box.mk", (), box_ctor, None)},
                recursor=ConstantInfo("Box.rec", (), TYPE_0, None),
            ))
        b = App(App(Const("Box.mk", ()), Lit(NatLit(3))), Lit(NatLit(5)))
        assert infer_type(Proj("Box", 1, b), env6, lctx, mctx) == NAT


class TestCheck:
    """
    Tests for ``check``
    """

    def test_check(self):
        """
        Tests for kernel ``check``. When a type matches, should return None
        if not raises an error.
        """
        env, lctx, mctx = Environment(), LocalContext(), MetaVarContext()
        assert check(Lit(NatLit(5)), NAT, env, lctx, mctx) is None

        # Nat != Real, shpuld raise an error
        try:
            check(Lit(NatLit(5)), Const("Real", ()), env, lctx, mctx)
            assert False, "expected KernelException"
        except KernelException:
            pass

        # Bad App expression construction should raise an error
        try:
            # App checks Lit(1) to be ForallE (Pi/function type)
            check(App(Lit(1), Lit(2)), NAT, env, lctx, mctx)
            assert False, "expected KernelException"
        except KernelException:
            pass
