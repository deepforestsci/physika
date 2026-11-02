from physika.features.modifier_letters import ModifierLetters
from physika.utils.types import TTensor, T_REAL
from tests.test_utils.test_infer_expr import make_ctx
from physika.utils.infer_expr import infer_expr
from physika.utils.ast_utils import ast_to_torch_expr

type_rules = ModifierLetters().type_rules()
expr_transpose = type_rules["transpose"]

fwd_rules = ModifierLetters().forward_rules()
emit_transpose = fwd_rules["transpose"]


class TestModifierLetters:
    """
    Checks parser is emitting the correct AST nodes,
    that the forward rules have the correct keys and type
    rules catches proper errors.
    """

    def test_elf_name(self):
        """
        ELF is registered under the name 'modifier_letters'.
        """
        assert ModifierLetters.name == "modifier_letters"

    def test_parser_rules(self):
        """
        parser_rules should return two handlers.
        """
        rules = ModifierLetters().parser_rules()
        assert len(rules) == 2
        names = [r.__name__ for r in rules]
        assert "p_factor_transpose" in names
        assert "p_func_factor_transpose" in names

    def test_forward_rules_keys(self):
        """
        forward rules should return one rule.
        """
        rules = ModifierLetters().forward_rules()
        assert len(rules) == 1
        assert set(rules.keys()) == {"transpose"}

    def test_type_rules_keys(self):
        """
        type rules should return one rule.
        """
        rules = ModifierLetters().type_rules()
        assert len(rules) == 1
        assert set(rules.keys()) == {"transpose"}


class TestExprTranspose:
    """
    Tests for ``expr_transpose``
    """

    def test_transpose_functionality(self):
        """Tests for basic matrix transpose"""
        # rectangular: ℝ[3, 4] -> ℝ[4, 3]
        ctx = make_ctx(
            env={"A": TTensor(((3, "invariant"), (4, "invariant")))})
        t, _ = expr_transpose(("transpose", ("var", "A")), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, ctx.add_error,
                              infer_expr)
        assert t == TTensor(((4, "invariant"), (3, "invariant")))

        # square matrices (shape will be same)
        ctx = make_ctx(
            env={"S": TTensor(((5, "invariant"), (5, "invariant")))})
        t, _ = expr_transpose(("transpose", ("var", "S")), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, ctx.add_error,
                              infer_expr)
        assert t == TTensor(((5, "invariant"), (5, "invariant")))

        # column matrix/vector ℝ[2, 1] -> row matrix/vector ℝ[1, 2]
        ctx = make_ctx(
            env={"c": TTensor(((2, "invariant"), (1, "invariant")))})
        t, _ = expr_transpose(("transpose", ("var", "c")), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, ctx.add_error,
                              infer_expr)
        assert t == TTensor(((1, "invariant"), (2, "invariant")))

    def test_transpose_wrong_rank_errors(self):
        """Tests for 1D and 3D tensors"""
        # 1D Tensor
        errors = []
        ctx = make_ctx(env={"v": TTensor(((4, "invariant"), ))})
        t, _ = expr_transpose(("transpose", ("var", "v")), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, errors.append,
                              infer_expr)
        assert t is None
        assert len(errors) == 1
        assert "Transpose requires a 2D tensor, got 1 dimensions" in errors[0]

        # 3D tensor
        errors = []
        ctx = make_ctx(env={
            "T":
            TTensor(((2, "invariant"), (3, "invariant"), (4, "invariant")))
        })
        t, _ = expr_transpose(("transpose", ("var", "T")), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, errors.append,
                              infer_expr)
        assert t is None
        assert len(errors) == 1
        assert "Transpose requires a 2D tensor, got 3 dimensions" in errors[0]

    def test_transpose_non_tensor_errors(self):
        """Tranpose on non-tensor value"""
        errors = []
        ctx = make_ctx(env={"x": T_REAL})
        t, _ = expr_transpose(("transpose", ("var", "x")), ctx.env, ctx.s,
                              ctx.func_env, ctx.class_env, errors.append,
                              infer_expr)
        assert t is None
        assert errors == ["Transpose requires a tensor"]


class TestEmitTranspose:
    """
    Tests for ``emit_transpose`` function.
    """

    def test_emit_transpose_variable(self):
        """test for codegen ``torch.transpose``"""
        node = ("transpose", ("var", "A"))

        code = emit_transpose(node, ast_to_torch_expr)

        assert code == "torch.transpose(A, 0, 1)"
