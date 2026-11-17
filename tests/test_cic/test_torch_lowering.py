import pytest

from physika.core.expr import (
    App,
    BVar,
    BinderInfo,
    Const,
    FVar,
    FVarId,
    ForallE,
    Lam,
    Lit,
    NatLit,
    Proj,
)
from physika.core.environment import ConstantInfo, Environment
from physika.core.torch_lowering import (
    apply_dim_rename,
    dim_extraction_stmt,
    dim_var_name,
    fold_gen,
    grad_gen,
    implicit_binders,
    lower,
    lower_call,
    lower_expr,
    lower_function_body,
    lower_vec_literal,
    stack_elems,
    body_mutates_in_place,
)

NAT = Const("Nat", ())
REAL = Const("Real", ())


# Helper functions for testing torch lowering
def c(name: str) -> Const:
    """
    Represents a elbaorated ``Const`` term from CIC.
    """
    return Const(name, ())


def fv(name: str) -> FVar:
    """
    Elaborated free variable with ``name``.
    """
    return FVar(FVarId(name))


def app(fn, *xs):
    """
     Elaborated ``fn`` application to  ``xs`` starting from left argument.
     """
    for x in xs:
        fn = App(fn, x)
    return fn


def vec_cons(elem_type, length, hd, tl):
    """
    Elaborated ``Vec.cons`` application node.
    """
    return app(c("Vec.cons"), elem_type, length, hd, tl)


def real_vec_literal(*vals):
    """
    ``Vec.cons``/``Vec.nil`` chain of ``Real`` literals ``vals``.
    """
    chain = App(c("Vec.nil"), REAL)
    for i, v in enumerate(reversed(vals)):
        chain = vec_cons(REAL, Lit(i), Lit(v), chain)
    return chain


def env_with(name: str, tp) -> Environment:
    """
    ``Environment`` with one constant ``name : tp``.
    """
    env = Environment()
    env.add_constant(ConstantInfo(name, (), tp, None))
    return env


def dot_env() -> Environment:
    """
    Env with a defined function  ``dot : {n : Nat} -> Vec Real n -> Real``
    (one implicit binder).
    """
    tp = ForallE(
        "n", NAT,
        ForallE("v", app(c("Vec"), REAL, c("n")), REAL, BinderInfo.DEFAULT),
        BinderInfo.IMPLICIT)
    return env_with("dot", tp)


class TestBodyMutatesInPlace:
    """
    ``body_mutates_in_place``  fall back to raw-AST codegen for statements
    reassignments inside functions.
    """

    def test_body_mutate_in_place(self):
        """
        Verifies statenent reassingments present from statements list
        """
        # should return true for ""body_index_assign" tag
        fd = {
            "statements": [("body_index_assign", "x", ("num", 1), ("num", 3))]
        }
        assert body_mutates_in_place(fd) is True

        # should find assignments nested in a loop
        fd = {
            "statements":
            [("body_for", "i", ("var", "n"), [("loop_index_assign_nd", "a", [
                ("index_item", ("var", "i"))
            ], ("num", 0))])]
        }
        assert body_mutates_in_place(fd) is True

        # no body reassingment should return False
        fd = {"statements": [("return", ("var", "x"))]}
        assert body_mutates_in_place(fd) is False


class TestImplicitBinders:
    """
    Tests for ``implicit_binders``.
    """

    def test_unknown_name(self):
        """An unregistered name havw no implicit binders."""
        assert implicit_binders("not.registered", Environment()) == 0

    def test_counts_implicit_binders(self):
        """``{n} -> Vec Real n -> Real`` has one leading implicit binder."""
        assert implicit_binders("dot", dot_env()) == 1

        # if first arg is explicit, then should return 0 implict binder
        env = env_with("f", ForallE("x", REAL, REAL, BinderInfo.DEFAULT))
        assert implicit_binders("f", env) == 0


class TestDimVarName:
    """
    Tests for ``dim_var_name``.
    """

    def test_dim_symbolic_name(self):
        """A bare dim-var name returns itself."""
        assert dim_var_name("n") == "n"
        # should catch k dim var from adding as dependent type
        assert dim_var_name(("add_dim", "k", 1)) == "k"

        # "n" dim var should be catched
        assert dim_var_name(("mul_dim_id", "n", "n")) == "n"

        # no dim var, getting an int dim value whould return None
        assert dim_var_name(3) is None


class TestApplyDimRename:
    """
    Tests for ``apply_dim_rename``.
    """

    def test_renames_a_bare_name(self):
        """A dim-var name is substituted."""
        assert apply_dim_rename("n", {"n": "__dim_n"}) == "__dim_n"

        # Variable names inside a derived-dim tuple are substituted; tag stays.
        assert apply_dim_rename(
            ("add_dim_id", "n", "m"),
            {"n": "__dim_n"}) == (("add_dim_id", "__dim_n", "m"))

        # An ``int`` dimension is returned unchanged
        assert apply_dim_rename(3, {"n": "__dim_n"}) == 3


class TestDimExtractionStmt:
    """
    Tests for ``dim_extraction_stmt``.
    """

    def test_dim_extraction(self):
        """A dim var matches shape axis at codegen"""
        assert dim_extraction_stmt("n", 0, "x") == "    n = int(x.shape[0])"

        # when computing with dependent types ``n + 1`` inverts to
        # ``shape - 1``
        assert dim_extraction_stmt(("add_dim", "k", 1), 1,
                                   "X") == ("    k = int(X.shape[1]) - 1")
        # ``n * 2`` to ``shape // 2``
        assert dim_extraction_stmt(("mul_dim", "n", 2), 0,
                                   "x") == ("    n = int(x.shape[0]) // 2")


class TestStackElems:
    """
    Tests for ``stack_elems``.
    """

    def test_wraps_each_element(self):
        """Each element is wrapped as a tensor under torch.stack."""
        assert stack_elems([
            "a", "1.0"
        ]) == ("torch.stack([torch.as_tensor(a), torch.as_tensor(1.0)])")

        # an empty list should return an empty tensor
        assert stack_elems([]) == "torch.tensor([])"


class TestFoldGen:
    """
    Tests for ``fold_gen``.
    """

    def test_emits_self_contained_fold(self):
        """a fold loop should be generated from properly"""
        assert fold_gen("n", "k", "acc", "(acc + k)",
                        "0.0") == ("(lambda acc: ([acc := (acc + k) "
                                   "for k in range(int(n))], acc)[1])(0.0)")


class TestGradGen:
    """
    Tests for ``grad_gen``.
    """

    def test_scalar_grad(self):
        """``grad(f(x), x)`` lowers to a ``compute_grad`` callable."""
        x = fv("x0")
        assert grad_gen(App(c("f"), x), x, {x.id: "x"}, Environment(),
                        ()) == ("compute_grad(lambda _dx: f(_dx), x)")

    def test_rename_fixed_arguments(self):
        """Variable being differenetiated wrt should be renamed renamed."""
        x, th = fv("x0"), fv("th")
        got = grad_gen(app(c("g"), x, th), x, {
            x.id: "state",
            th.id: "theta"
        }, Environment(), ())
        assert got == "compute_grad(lambda _dstate: g(_dstate, theta), state)"


class TestLowerVecLiteral:
    """
    Tests for ``lower_vec_literal``.
    """

    def test_scalar_chain(self):
        """A ``Vec.cons`` chain of literals flattens to its element sources."""
        chain = real_vec_literal(1.0, 2.0)
        assert lower_vec_literal(chain, {}, Environment(),
                                 ()) == ["1.0", "2.0"]

        # A nested row is lowered to torch.stack
        vec_real = app(c("Vec"), REAL)
        row = real_vec_literal(1.0, 2.0)
        mat = vec_cons(vec_real, Lit(0), row, App(c("Vec.nil"), vec_real))
        assert lower_vec_literal(mat, {}, Environment(), ()) == [
            "torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0)])"
        ]

        # An expression that is not a ``Vec.cons``/``Vec.nil`` chain should
        # return ``None``
        assert lower_vec_literal(Lit(3), {}, Environment(), ()) is None


class TestLower:
    """
    Tests for ``lower``.
    """

    def test_nat_literal(self):
        """A ``NatLit`` wrapped ``Lit`` lowers to a number."""
        assert lower(Lit(NatLit(5)), {}, Environment(), ()) == "5"

    def test_free_variable(self):
        """
        ``FVar`` lowers to its name from ``names``
        """
        x = fv("x0")
        assert lower(x, {x.id: "x"}, Environment(), ()) == "x"

    def test_unbound_free_variable(self):
        """An ``FVar`` missing from ``names`` is a lowering error."""
        with pytest.raises(NotImplementedError):
            lower(fv("z"), {}, Environment(), ())

    def test_binop(self):
        """A ``BUILTIN_BINOP`` application lowers to its related expression."""
        expr = app(c("Real.add"), Lit(1.0), Lit(2.0))
        assert lower(expr, {}, Environment(), ()) == "(1.0 + 2.0)"

    def test_mat_inv_skips_implicit_dim(self):
        """
        ``Mat.inv`` lowers to ``torch.linalg.inv`` of its matrix argument,
        dropping the implicit ``{n}`` that torch reads off the tensor.
        """
        mat = fv("A0")
        expr = app(c("Mat.inv"), Lit(3), mat)
        assert lower(expr, {mat.id: "A"}, Environment(),
                     ()) == "torch.linalg.inv(A)"

    def test_vec_get_wraps_index(self):
        """
        ```Vec.get`` should lower to ``v[int(i)]``
        """
        v = fv("v")
        expr = app(c("Vec.get"), REAL, Lit(3), v, Lit(0))
        assert lower(expr, {v.id: "v"}, Environment(), ()) == "v[int(0)]"

    def test_fin_constructors(self):
        """
        ``Fin.zero`` is index 0 and ``Fin.succ k`` is ``1 + k``
        """
        assert lower(app(c("Fin.zero"), Lit(2)), {}, Environment(), ()) == "0"
        assert lower(app(c("Fin.succ"), Lit(2), Lit(0)), {}, Environment(),
                     ()) == "(1 + 0)"

    def test_ite(self):
        """``Real.ite`` lowers to conditional expression."""
        cond = fv("cnd")
        expr = app(c("Real.ite"), cond, Lit(1.0), Lit(2.0))
        assert lower(expr, {cond.id: "cond"}, Environment(),
                     ()) == ("(1.0 if cond else 2.0)")

    def test_array_literal_stacks(self):
        """A ``Vec.cons`` chain lowers to a single ``torch.stack``."""
        expr = real_vec_literal(1.0, 2.0)
        assert lower(expr, {}, Environment(), ()) == (
            "torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0)])")

    def test_tabulate(self):
        """
        ``Vec.tabulate`` should lower to a stacked with ``range(n)``.
        """
        body = app(c("Real.mul"), Lit(2.0), Lit(1.0))
        expr = app(c("Vec.tabulate"), REAL, Lit(3), Lam("i", NAT, body))
        assert lower(expr, {}, Environment(),
                     ()) == ("torch.stack([torch.as_tensor((2.0 * 1.0)) "
                             "for i in range(int(3))]).float()")

    def test_foldl_emits_a_fold(self):
        """
        ``Vec.foldl`` opens its curried step Lam and emits a ``fold_gen`` fold.
        """
        step = Lam("k", NAT,
                   Lam("acc", REAL, App(App(c("Real.add"), BVar(0)), BVar(1))))
        expr = app(c("Vec.foldl"), REAL, Lit(3), step, Lit(0.0))
        assert lower(expr, {}, Environment(),
                     ()) == ("(lambda acc: ([acc := (acc + k) "
                             "for k in range(int(3))], acc)[1])(0.0)")

    def test_sin_function_operand(self):
        """
        ``sin`` lowers to ``torch.sin`` with the operand coerced to a tensor
        """
        x = fv("x0")
        assert lower(app(c("sin"), x), {x.id: "x"}, Environment(),
                     ()) == ("torch.sin(torch.as_tensor(x).float())")

    def test_abs_does_not_float(self):
        """``abs`` keeps its dtype so a complex tensor is not truncated."""
        x = fv("x0")
        assert lower(app(c("abs"), x), {x.id: "x"}, Environment(),
                     ()) == ("torch.abs(torch.as_tensor(x))")

    def test_grad(self):
        """A ``grad`` elaborated term uses ``grad_gen`` to ``compute_grad``."""
        x = fv("x0")
        expr = app(c("grad"), App(c("f"), x), x)
        assert lower(expr, {x.id: "x"}, Environment(),
                     ()) == ("compute_grad(lambda _dx: f(_dx), x)")

    def test_proj_of_prod_is_tuple_index(self):
        """
        A ``Prod`` projection should lowers to a positional index, not
        attribute access
        ."""
        p = fv("p")
        assert lower(Proj("Prod", 0, p), {p.id: "p"}, Environment(),
                     ()) == "p[0]"

    def test_ofnat_proj_folds_to_a_number(self):
        """An ``OfNat`` numeral projection lowers to its literal value."""
        nat = Proj("OfNat", 0, App(c("instOfNatNat"), Lit(3)))
        real = Proj("OfNat", 0, App(c("instOfNatReal"), Lit(3)))
        assert lower(nat, {}, Environment(), ()) == "3"
        assert lower(real, {}, Environment(), ()) == "3"

    def test_arity_mismatch(self):
        """A builtin applied to the wrong number of arguments is an error."""
        with pytest.raises(NotImplementedError):
            lower(app(c("Vec.sum"), Lit(1), Lit(2), Lit(3)), {}, Environment(),
                  ())


class TestLowerCall:
    """
    Tests for ``lower_call``.
    """

    def test_lower_function(self):
        """
        A function call keeps its explicit arguments in order
        """
        assert lower_call("f", [Lit(1), Lit(2)], {}, Environment(),
                          ()) == ("f(1, 2)")

    def test_class_constructor(self):
        """``Box.mk`` lowers to a class instance ``Box(...)``."""
        assert lower_call("Box.mk", [Lit(3), Lit(5)], {}, Environment(),
                          ()) == "Box(3, 5)"

    def test_prod_mk_is_a_tuple(self):
        """``Prod.mk`` lowers to a tuple literal."""
        assert lower_call("Prod.mk", [Lit(1), Lit(2)], {}, Environment(),
                          ()) == "(1, 2)"

    def test_method_without_instance(self):
        """A method term with no receiver argument should return error."""
        with pytest.raises(NotImplementedError):
            lower_call("Vec2.add", [], {}, Environment(), ())

    def test_implicit_arg__for_unresolved_callee(self):
        """
        A callee not in ``resolved_names`` loses its implicit dim-var arg and
        cannot comput with dependent types.
        """
        v = fv("v")
        got = lower_call("dot", [Lit(3), v], {v.id: "v"}, dot_env(), ())
        assert got == "dot(v)"

    def test_implicit_arg_kept_for_resolved_callee(self):
        """A resolved callee keeps the implicit arg"""
        v = fv("v")
        got = lower_call("dot", [Lit(3), v], {v.id: "v"}, dot_env(), ("dot", ))
        assert got == "dot(v, 3)"


class TestLowerFunctionBodyExpr:
    """
    Tests for ``lower_function_body``.
    """

    def test_emits_locals_then_return(self):
        """
        Local declarations should become an assignment line and if present
        should also lower a ``return`` line.
        """
        got = lower_function_body(Lit(0), {}, Environment(), [("x", Lit(1))])
        assert got == "    x = 1\n    return 0"

    def test_lowers_a_top_level_expression(self):
        """
        A  CIC elaborated and verified term for an additino should
        be lowered.
        """
        expr = app(c("Real.add"), Lit(1.0), Lit(2.0))
        assert lower_expr(expr, {}, Environment(), ()) == "(1.0 + 2.0)"
