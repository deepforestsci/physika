import torch
from physika.features.randomness import (extract_dist_args, sample,
                                         normal_dist, uniform_dist, beta_dist,
                                         gamma_dist, bernoulli_dist,
                                         get_shape_args, get_dim,
                                         RandomnessFeature)
from physika.utils.ast_utils import ast_to_torch_expr, build_unified_ast
from physika.utils.types import T_REAL, TTensor, Substitution
from tests.conftest import exec_phyk
from physika.codegen import from_ast_to_torch


def parse_physika(src):
    """Parse a Physika source string and return the unified AST."""
    import physika.parser as pm
    from physika.lexer import lexer
    pm.symbol_table.clear()
    lexer.lexer.lineno = 1  # reset PLY line counter for deterministic output
    program_ast = pm.parser.parse(src, lexer=lexer)
    return build_unified_ast(program_ast, pm.symbol_table)


class TestRandomessHelpers:
    """
    Tests for helper functions used by randomness ELF.
    - extract_dist_args
    - sample
    """

    def test_extract_dist_args(self):
        """
        Verify that extract_dist_args correctly gives distribution arguments
        for supported distributions.
        """
        # Test normal distribution
        args = [("num", 0.0), ("num", 1.0)]
        extracted_args = extract_dist_args(args, n_params=2)
        assert len(extracted_args) == 3
        assert extracted_args[0] == args
        assert extracted_args[1] == []
        assert extracted_args[2] == 'none'

        # case multiple shape arguments
        args = [("num", 0.0), ("num", 1.0), ("num", 20.0), ("num", 1.0)]
        n_params = 2
        extracted_args = extract_dist_args(args, n_params)
        assert len(extracted_args) == 3
        assert extracted_args[0] == args[:n_params]
        assert extracted_args[1] == args[n_params:]
        assert extracted_args[2] == 'none'

        # 'reparam' as estimator
        args = [("num", 0.0), ("num", 1.0), ("string", "reparam")]
        extracted_args = extract_dist_args(args, n_params)
        assert len(extracted_args) == 3
        assert extracted_args[0] == args[:n_params]
        assert extracted_args[1] == []
        assert extracted_args[2] == 'reparam'

        # Test bernoulli distribution with score function estimator
        args = [("num", 0.5), ("string", "score")]
        n_params = 1
        extracted_args = extract_dist_args(args, n_params)
        assert len(extracted_args) == 3
        assert extracted_args[0] == args[:n_params]
        assert extracted_args[2] == 'score'

    def test_sample(self):
        """
        Checks generated pytorch code is valid given a distribution expression
        with arguments, shape arguments, and estimator type.
        """
        # Scalar sample, Normal(0.0, 1.0) dist
        to_expr = ast_to_torch_expr
        estimator = "reparam"
        mean, std = "0.0", "1.0"
        result = sample("torch.distributions.Normal(%s, %s)" % (mean, std), [],
                        mode=estimator,
                        default_mode=estimator,
                        to_expr=to_expr)
        assert result == 'torch.distributions.Normal(%s, %s).rsample()' % (
            mean, std)

        # estimator node is none, but default reparam used
        result = sample("torch.distributions.Normal(%s, %s)" % (mean, std), [],
                        mode="none",
                        default_mode=estimator,
                        to_expr=to_expr)
        assert result == 'torch.distributions.Normal(%s, %s).rsample()' % (
            mean, std)

        # sample a tensor with shape ℝ[20, 1]
        shape_args = [20, 1]
        result = sample("torch.distributions.Normal(%s, %s)" % (mean, std),
                        shape_args=shape_args,
                        mode="none",
                        default_mode=estimator,
                        to_expr=to_expr)
        assert result == 'torch.distributions.Normal(%s, %s).rsample((int(%s), int(%s),))' % (  # noqa: E501
            mean, std, shape_args[0], shape_args[1])

        # sample a tensor with shape ℝ[20, 1, 3] and mean/std 3, 0.5
        shape_args = [20, 1, 3]
        mean_d, std_d = "3", "0.5"
        result = sample("torch.distributions.Normal(%s, %s)" % (mean_d, std_d),
                        shape_args=shape_args,
                        mode="none",
                        default_mode=estimator,
                        to_expr=to_expr)
        assert result == 'torch.distributions.Normal(%s, %s).rsample((int(%s), int(%s), int(%s),))' % (  # noqa: E501
            mean_d, std_d, shape_args[0], shape_args[1], shape_args[2])

        # none estimator produces .sample()
        result = sample("torch.distributions.Normal(%s, %s)" % (mean, std), [],
                        mode="none",
                        default_mode="none",
                        to_expr=to_expr)
        assert result == 'torch.distributions.Normal(%s, %s).sample()' % (mean,
                                                                          std)

        # Bernoulli distribution with score function estimator
        p = "0.5"
        result = sample("torch.distributions.Bernoulli(%s)" % (p), [],
                        mode="score",
                        default_mode="none",
                        to_expr=to_expr)
        assert result == 'torch.distributions.Bernoulli(%s).sample().detach()' % (  # noqa: E501
            p)


class TestProbDistributionsCodegen:
    """
    Verify correctens in generated pytorch code for supported
    probability distributions in randomness ELF.
    - normal_dist
    - uniform_dist
    - beta_dist
    - gamma_dist,
    - bernoulli_dist
    """

    def test_normal_dist(self):
        """
        Verify normal_dist emits correct PyTorch code for
        Normal(μ, σ)
        """
        to_expr = ast_to_torch_expr
        mu, sigma = ("num", 0.0), ("num", 1.0)

        # scalar sample
        result = normal_dist([mu, sigma], to_expr)
        assert result == "torch.distributions.Normal(0.0, 1.0).rsample()"

        # vector sample ℝ[20]
        n = ("num", 20.0)
        result = normal_dist([mu, sigma, n], to_expr)
        assert result == "torch.distributions.Normal(0.0, 1.0).rsample((int(20.0),))"  # noqa: E501

        # vector sample ℝ[20, 3, 3]
        n0, n1, n2 = ("num", 20.0), ("num", 3.0), ("num", 3.0)
        result = normal_dist([mu, sigma, n0, n1, n2], to_expr)
        assert result == "torch.distributions.Normal(0.0, 1.0).rsample((int(20.0), int(3.0), int(3.0),))"  # noqa: E501

        # grad mode ooverride
        result = normal_dist([mu, sigma, ("string", "score")], to_expr)
        assert result == "torch.distributions.Normal(0.0, 1.0).sample().detach()"  # noqa: E501

    def test_uniform_dist(self):
        """
        Verify uniform_dist emits correct PyTorch code for
        Uniform(lo, hi)
        """
        to_expr = ast_to_torch_expr
        lo, hi = ("num", 0.0), ("num", 1.0)

        # scalar sample — reparam by default
        result = uniform_dist([lo, hi], to_expr)
        assert result == "torch.distributions.Uniform(0.0, 1.0).rsample()"

        # vector sample ℝ[10]
        n = ("num", 10.0)
        result = uniform_dist([lo, hi, n], to_expr)
        assert result == "torch.distributions.Uniform(0.0, 1.0).rsample((int(10.0),))"  # noqa: E501

        # vector sample ℝ[20, 3, 3]
        n0, n1, n2 = ("num", 20.0), ("num", 3.0), ("num", 3.0)
        result = uniform_dist([lo, hi, n0, n1, n2], to_expr)
        assert result == "torch.distributions.Uniform(0.0, 1.0).rsample((int(20.0), int(3.0), int(3.0),))"  # noqa: E501

        # explicit score override
        result = uniform_dist([lo, hi, ("string", "score")], to_expr)
        assert result == "torch.distributions.Uniform(0.0, 1.0).sample().detach()"  # noqa: E501

    def test_beta_dist(self):
        """
        Verify beta_dist emits correct PyTorch code for
        Beta(α, β)
        """
        to_expr = ast_to_torch_expr
        alpha, beta = ("num", 0.5), ("num", 0.5)

        # scalar sample — reparam by default
        result = beta_dist([alpha, beta], to_expr)
        assert result == "torch.distributions.Beta(0.5, 0.5).rsample()"

        # vector sample ℝ[8]
        n = ("num", 8.0)
        result = beta_dist([alpha, beta, n], to_expr)
        assert result == "torch.distributions.Beta(0.5, 0.5).rsample((int(8.0),))"  # noqa: E501

        # vector sample ℝ[20, 3, 3]
        n0, n1, n2 = ("num", 20.0), ("num", 3.0), ("num", 3.0)
        result = beta_dist([alpha, beta, n0, n1, n2], to_expr)
        assert result == "torch.distributions.Beta(0.5, 0.5).rsample((int(20.0), int(3.0), int(3.0),))"  # noqa: E501

        # explicit score override
        result = beta_dist([alpha, beta, ("string", "score")], to_expr)
        assert result == "torch.distributions.Beta(0.5, 0.5).sample().detach()"  # noqa: E501

    def test_gamma_dist(self):
        """
        Verify gamma_dist emits correct PyTorch code for
        Gamma(concentration, rate)
        """
        to_expr = ast_to_torch_expr
        concentration, rate = ("num", 1.0), ("num", 1.0)

        # scalar sample — reparam by default
        result = gamma_dist([concentration, rate], to_expr)
        assert result == "torch.distributions.Gamma(1.0, 1.0).rsample()"

        # vector sample ℝ[5]
        n = ("num", 5.0)
        result = gamma_dist([concentration, rate, n], to_expr)
        assert result == "torch.distributions.Gamma(1.0, 1.0).rsample((int(5.0),))"  # noqa: E501

        # vector sample ℝ[20, 3, 3]
        n0, n1, n2 = ("num", 20.0), ("num", 3.0), ("num", 3.0)
        result = gamma_dist([concentration, rate, n0, n1, n2], to_expr)
        assert result == "torch.distributions.Gamma(1.0, 1.0).rsample((int(20.0), int(3.0), int(3.0),))"  # noqa: E501

        # explicit score override
        result = gamma_dist([concentration, rate, ("string", "score")],
                            to_expr)
        assert result == "torch.distributions.Gamma(1.0, 1.0).sample().detach()"  # noqa: E501

    def test_bernoulli_dist(self):
        """
        Verify bernoulli_dist emits the score function estimator
        (.sample().detach()).
        """
        to_expr = ast_to_torch_expr
        p = ("num", 0.5)

        # scalar sample
        result = bernoulli_dist([p], to_expr)
        assert result == "torch.distributions.Bernoulli(0.5).sample().detach()"

        # vector sample ℝ[20]
        n = ("num", 20.0)
        result = bernoulli_dist([p, n], to_expr)
        assert result == "torch.distributions.Bernoulli(0.5).sample((int(20.0),)).detach()"  # noqa: E501

        # vector sample ℝ[20, 3, 3]
        n0, n1, n2 = ("num", 20.0), ("num", 3.0), ("num", 3.0)
        result = bernoulli_dist([p, n0, n1, n2], to_expr)
        assert result == "torch.distributions.Bernoulli(0.5).sample((int(20.0), int(3.0), int(3.0),)).detach()"  # noqa: E501

        # score function estimator
        result = bernoulli_dist([p, ("string", "score")], to_expr)
        assert result == "torch.distributions.Bernoulli(0.5).sample().detach()"
        # reparam grad is ignored
        result = bernoulli_dist([p, ("string", "reparam")], to_expr)
        assert result == "torch.distributions.Bernoulli(0.5).sample().detach()"


class TestGetShapeArgs:
    """
    Tests for ``get_shape_args``.

    Verify that shape arguments are collected from distribution calls starting
    from the right.
    """

    def test_shape_arg(self):
        """
        Test that trailing num literals are collected as shape args for single
        and multiple shape arguments.
        """
        # case single shape arg
        result = get_shape_args([("num", 0.0), ("num", 1.0), ("num", 100)], {})
        assert result == [("num", 100)]

        # multiple shapes
        result = get_shape_args([("num", 0.0), ("num", 1.0), ("num", 20),
                                 ("num", 3)], {})
        assert result == [("num", 20), ("num", 3)]

    def test_var_shape_arg(self):
        """
        Test that variables are collected as shape args when they have tracked
        values.
        """
        # Variable n with value 100 is registered in env
        env = {("__val__", "n"): 100}
        result = get_shape_args([("var", "mu"), ("var", "sigma"),
                                 ("var", "n")], env)
        assert result == [("var", "n")]

        # a var that is not registered in env is treated as a distribution
        # param
        result = get_shape_args([("var", "mu"), ("var", "sigma"),
                                 ("var", "n")], {})
        assert result == []


class TestGetDim:
    """
    Tests for ``get_dim`` which resolves an AST node with "num" or "var"
    to an integer dimension size or a symbolic string name.
    """

    def test_num_node(self):
        """
        AST Nodes with "num" keys are converted to integer for dimensions.
        """
        assert get_dim(("num", 20), {}) == 20
        # even in the value is a float, it should be converted to int
        # get_dim is used for type checking
        assert get_dim(("num", 5.0), {}) == 5

    def test_var_node(self):
        """
        Verifies that "var" nodes are resolved to tracked values in the env
        when present, and left as symbolic names when not present.
        """
        # Variable n with value 100 is registered in env
        env = {("__val__", "n"): 100}
        assert get_dim(("var", "n"), env) == 100

        # a var that is not registered in env is treated as a symbolic dim
        assert get_dim(("var", "m"), {}) == "m"


class TestLexerParserRules:
    """
    Test grammar rules in ``parser_rules`` for RandomnessFeature.

    Tests that parses a Physika program and verifies the structure
    of the AST node.
    """

    def test_seed_parser(self):
        """
        Test for parsing physika.seed function calls in different
        contexts
        """
        # at top level
        ast = parse_physika("physika.seed(42)\n")
        node = ast["program"][0]
        assert node == ("seed", ("num", 42))

        # physika.seed with a variable argument
        ast = parse_physika("n : ℕ = 7\nphysika.seed(n)\n")
        node = ast["program"][1]
        assert node == ("seed", ("var", "n"))

        # physika.seed inside a function body
        ast = parse_physika("def f(s: ℕ) : ℝ:\n"
                            "    physika.seed(s)\n"
                            "    return 0.0\n")

        stmts = ast["functions"]["f"]["statements"]
        assert stmts == [("seed", ("var", "s"))]

    def test_parser_rules_structure(self):
        """parser_rules() returns 11 callable PLY functions"""
        rules = RandomnessFeature().parser_rules()
        assert isinstance(rules, list)
        assert len(rules) == 11
        for rule in rules:
            assert callable(rule)
            # functions must be p_-prefixed
            assert rule.__name__.startswith("p_")

    def test_lexer_rules(self):
        """TILDE token and distribution alias functions are registered."""
        rules = RandomnessFeature().lexer_rules()
        assert "TILDE" in rules["tokens"]
        t_tilde = rules["token_funcs"][0]
        assert t_tilde.__name__ == "t_TILDE"
        assert t_tilde.__doc__ == "~"

        assert "PHYSIKA" in rules["tokens"]
        assert rules["reserved"].get("physika") == "PHYSIKA"

        func_names = [f.__name__ for f in rules["token_funcs"]]
        dists = ["Normal", "Uniform", "Beta", "Gamma"]
        for dist in dists:
            assert "t_DIST_%s" % dist.upper() in func_names

    def test_sample_untyped(self):
        """
        Test for parsing a untyped sample physika statement.
        """
        ast = parse_physika("x ~ Normal(0.0, 1.0)\n")
        stmt = ast["program"][0]
        assert ('sample', 'x', ('call', 'Normal', [('num', 0.0),
                                                   ('num', 1.0)]), 1) == stmt

    def test_sample_typed(self):
        """
        Test for parsing a typed sample physika statement.
        """
        ast = parse_physika("x : ℝ ~ Normal(0.0, 1.0)\n")
        stmt = ast["program"][0]
        print(stmt)
        assert ('typed_sample', 'x', 'ℝ', ('call', 'Normal', [('num', 0.0),
                                                              ('num', 1.0)]),
                1) == stmt

    def test_func_body_stmt_sample(self):
        """
        Test for parsing an untyped sample physika statement inside a function.
        """
        ast = parse_physika("def f(μ: ℝ) : ℝ:\n"
                            "    z ~ Normal(μ, 1.0)\n"
                            "    return z\n")
        stmts = ast["functions"]["f"]["statements"]
        assert [('sample', 'z', ('call', 'Normal', [('var', 'μ'),
                                                    ('num', 1.0)]))] == stmts

    def test_func_body_stmt_sample_typed(self):
        """
        Test for parsing an typed sample physika statement inside a function.
        """
        ast = parse_physika("def f(μ: ℝ) : ℝ[5]:\n"
                            "    z : ℝ[5] ~ Normal(μ, 1.0, 5)\n"
                            "    return z\n")
        stmts = ast["functions"]["f"]["statements"]
        print(stmts)
        assert [('typed_sample', 'z', ('tensor', [(5, 'invariant')]),
                 ('call', 'Normal', [('var', 'μ'), ('num', 1.0),
                                     ('num', 5)]))] == stmts

    def test_for_sample(self):
        """
        Test for parsing an typed sample physika statement used in in-line
        for-loops.
        """
        ast = parse_physika(
            "z : ℝ[10, 2] = for i : ℕ(10) → ε : ℝ[2] ~ Normal(0.0, 1.0, 2)\n")
        stmt = ast["program"][0]

        print(stmt)
        assert (
            'decl',
            'z',
            ('tensor', [(10, 'invariant'), (2, 'invariant')]),
            # z : ℝ[10, 2]
            (
                'for_expr',
                'i',
                ('num', 10),  # for i : ℕ(10)
                (
                    'typed_sample_expr',
                    'ε',
                    ('tensor', [(2, 'invariant')]),  # ε : ℝ[2]
                    (
                        'call',
                        'Normal',  # Normal(...)
                        [('num', 0.0), ('num', 1.0), ('num', 2)]))),
            1) == stmt  # Normal(0.0, 1.0, 2)

    def test_func_factor_sample_expr(self):
        """
        Test for parsing a sample physika statement used inside a function
        and passed direclty as return.
        """
        ast = parse_physika("def f(mu: ℝ) : ℝ[2]:\n"
                            "    return ε ~ Normal(mu, 1.0, 2.0)\n")
        f_statements = ast["functions"]["f"]["statements"]
        f_return_expr = ast["functions"]["f"]["body"]
        assert f_statements == []
        assert ('sample_expr', 'ε', ('call', 'Normal', [('var', 'mu'),
                                                        ('num', 1.0),
                                                        ('num', 2.0)
                                                        ])) == f_return_expr


class TestTypeRules:
    """
    Tests for RandomnessFeature type rules.
    """

    def test_typed_sample(self):
        """
        Check types of sample statements.
        """
        # declared ℝ with scalar sampling
        check = RandomnessFeature().type_rules()["typed_sample"]
        errors = []
        node = ("typed_sample", "x", "ℝ", ("call", "Normal", [("num", 0.0),
                                                              ("num", 1.0)]))
        t, _ = check(node, {}, Substitution(), {}, {}, errors.append, None)
        assert errors == []
        assert t is T_REAL

        # declared ℝ[100] with normal distribution sampling
        check = RandomnessFeature().type_rules()["typed_sample"]
        errors = []
        env = {("__val__", "n"): 100}
        node = ("typed_sample", "x", ("tensor", [(100, "invariant")]),
                ("call", "Normal", [("num", 0.0), ("num", 1.0), ("num", 100)]))
        t, _ = check(node, env, Substitution(), {}, {}, errors.append, None)
        assert errors == []
        assert isinstance(t, TTensor)

    def test_typed_sample_rank_mismatch(self):
        """
        Test that type checkercatch errors properly.
        """
        # declared ℝ but Normal(mu, sigma, n) produces ℝ[n]
        check = RandomnessFeature().type_rules()["typed_sample"]
        errors = []
        node = ("typed_sample", "x", "ℝ", ("call", "Normal", [("num", 0.0),
                                                              ("num", 1.0),
                                                              ("num", 10)]))
        check(node, {}, Substitution(), {}, {}, errors.append, None)
        assert len(errors) == 1
        assert errors[
            0] == "'x': declared ℝ but Normal(...) produces a ℝ[n] sample"

        # declared ℝ[n] but Normal(mu, sigma) produces scalar
        check = RandomnessFeature().type_rules()["typed_sample"]
        errors = []
        node = ("typed_sample", "x", ("tensor", [(100, "invariant")]),
                ("call", "Normal", [("num", 0.0), ("num", 1.0)]))
        check(node, {}, Substitution(), {}, {}, errors.append, None)
        assert len(errors) == 1
        assert errors[
            0] == "'x': declared ℝ[100] but Normal(...) produces a ℝ sample"

        # declared ℝ[2] but produces ℝ[3]
        check = RandomnessFeature().type_rules()["typed_sample"]
        errors = []
        node = ("typed_sample", "x", ("tensor", [(2, "invariant")]),
                ("call", "Normal", [("num", 0.0), ("num", 1.0), ("num", 3)]))
        check(node, {}, Substitution(), {}, {}, errors.append, None)
        assert len(errors) == 1
        assert errors[
            0] == "'x': declared ℝ[2]. Normal(...) in dim[0] infers 3 but declared 2"  # noqa: E501

    def test_sample_expr(self):
        """
        Test sample expr types are inferred correctly since
        there is no declared type to check against.
        """
        # scalar case
        check = RandomnessFeature().type_rules()["sample_expr"]
        node = ("sample_expr", "ε", ("call", "Normal", [("num", 0.0),
                                                        ("num", 1.0)]))
        t, _ = check(node, {}, Substitution(), {}, {}, None, None)
        assert t is T_REAL

        # vector case
        check = RandomnessFeature().type_rules()["sample_expr"]
        node = ("sample_expr", "ε", ("call", "Normal", [("num", 0.0),
                                                        ("num", 1.0),
                                                        ("num", 20)]))
        t, _ = check(node, {}, Substitution(), {}, {}, None, None)
        assert isinstance(t, TTensor)


class TestForwardRules:
    """
    Tests for RandomnessFeature forward_rules code generation.

    Each test parses a Physika source string and calls the forward rule
    handler and checks the emitted PyTorch code.
    """

    def test_sample_stmt_emit(self):
        """
        Check top-level sample statement emits correct PyTorch code for Normal
        distribution.
        """
        rules = RandomnessFeature().forward_rules()
        ast = parse_physika("x ~ 𝒩(0.0, 1.0)\n")
        node = ast["program"][0]
        assert node[0] == "sample"
        result = rules["sample"](node, ast_to_torch_expr)
        assert result == "x = torch.distributions.Normal(0.0, 1.0).rsample()"

    def test_typed_sample_stmt_emit(self):
        """
        Check top-level typed sample statement emits correct PyTorch code
        for Normal distribution.
        """
        rules = RandomnessFeature().forward_rules()
        ast = parse_physika("x : ℝ[5] ~ 𝒩(0.0, 1.0, 5)\n")
        node = ast["program"][0]
        assert node[0] == "typed_sample"
        result = rules["typed_sample"](node, ast_to_torch_expr)
        assert result == "x = torch.distributions.Normal(0.0, 1.0).rsample((int(5),))"  # noqa: E501

    def test_sample_expr_emit(self):
        """
        Check sample statement inside functions as return expr emits correct
        PyTorch code for Normal distribution.
        """
        rules = RandomnessFeature().forward_rules()
        ast = parse_physika("def f(mu: ℝ) : ℝ[2]:\n"
                            "    return ε ~ 𝒩(mu, 1.0, 2.0)\n")
        node = ast["functions"]["f"]["body"]
        assert node[0] == "sample_expr"
        result = rules["sample_expr"](node, ast_to_torch_expr)
        assert result == "torch.distributions.Normal(mu, 1.0).rsample((int(2.0),))"  # noqa: E501

    def test_seed_emit(self):
        """
        Verifies emitted code for physika.seed
        """
        rules = RandomnessFeature().forward_rules()
        result = rules["seed"](("seed", ("num", 42)), ast_to_torch_expr)
        assert result == "torch.manual_seed(int(42))"

        # physika.seed(n) where n is a symbolic variable
        rules = RandomnessFeature().forward_rules()
        result = rules["seed"](("seed", ("var", "n")), ast_to_torch_expr)
        assert result == "torch.manual_seed(int(n))"

        # physika.seed(0) is generated as torch.manual_seed(int(0))
        ast = parse_physika("physika.seed(0)\nx : ℝ ~ Normal(0.0, 1.0)\n")
        code = from_ast_to_torch(ast, print_code=False)
        assert "torch.manual_seed(int(0))" in code

    def test_dual_sample_emit_scalar(self):
        """
        dual_sample with scalar declared type infers 0 shape args from _type1,
        so all args after stripping the mode string are treated as dist params.
        """
        rules = RandomnessFeature().forward_rules()
        # b : ℝ, lp : ℝ ~ Bernoulli(0.5)  — scalar sample, scalar log_prob
        node = ("dual_sample", "b", "ℝ", "lp", "ℝ", ("call", "Bernoulli",
                                                     [("num", 0.5)]))
        result = rules["dual_sample"](node, ast_to_torch_expr)
        assert result == ("_dist_b = torch.distributions.Bernoulli(0.5)\n"
                          "b = _dist_b.sample().detach()\n"
                          "lp = _dist_b.log_prob(b)")

    def test_dual_sample_emit_vector(self):
        """
        dual_sample with vector declared type uses tensor rank to split
        the last declared rank args as shape args.
        """
        rules = RandomnessFeature().forward_rules()
        # b_s : ℝ[10], lp : ℝ[10] ~ Bernoulli(0.5, 10)
        # vector log_prob
        node = ("dual_sample", "b_s", ("tensor", [(10, "invariant")]), "lp",
                ("tensor", [(10, "invariant")]), ("call", "Bernoulli", [
                    ("num", 0.5), ("num", 10), ("string", "score-function")
                ]))
        result = rules["dual_sample"](node, ast_to_torch_expr)
        assert result == ("_dist_b_s = torch.distributions.Bernoulli(0.5)\n"
                          "b_s = _dist_b_s.sample((int(10),)).detach()\n"
                          "lp = _dist_b_s.log_prob(b_s)")

    def test_typed_sample_expr_emit(self):
        """
        Checks that the generated PyTorch code for a for-expr with a
        typed sample expression is correct.
        """
        ast = parse_physika(
            "z : ℝ[3, 3] = for i : ℕ(3) → ε : ℝ[3] ~ 𝒩(0.0, 1.0, 3)\n")
        code = from_ast_to_torch(ast, print_code=False)
        expected = (
            "z = torch.stack(["
            "torch.distributions.Normal(0.0, 1.0).rsample((int(3),)) "
            # ε : ℝ[3] ~ 𝒩(0.0, 1.0, 3)
            "for _fi_i in range(int(3)) "  # for i : ℕ(3)
            "for i in [torch.tensor(float(_fi_i))]])")
        assert expected in code


class TestRandomnessIntegration:
    """
    Integration tests for ``RandomnessFeature`` using examples/randomness.phyk.

    Executes the full Physika pipeline (lexer, parser, type checking, codegen,
    exec) and checks output are correct.
    """

    def test_randomness_example(self):
        """
        Verifies randomness.phyk executes without error and verifies outputs.
        """
        ns_dict = exec_phyk("randomness")
        # Physika code:
        # z: ℝ[10,2]= for i : ℕ(10) → ε : ℝ[2] ~ Normal(μ, σ, 2)
        z = ns_dict["z"]
        assert z.shape == torch.Size([10, 2])
        # Physika code:
        # z_3d: ℝ[10, 5, 2]= for i : ℕ(10) → for j : ℕ(5) → ε : ℝ[2]  # noqa: E501
        # ~ Normal(μ, σ, 2)
        z_3d = ns_dict["z_3d"]
        assert z_3d.shape == torch.Size([10, 5, 2])
        # Physika code:
        # x : ℝ = 3.0
        # y : ℝ[3, 2] = sample_normal2D(x) where x=3
        y = ns_dict["y"]
        assert y.shape == torch.Size([3, 2])

        loss = ns_dict["loss"]
        # after 300 epochs loss should be close to 0
        assert loss.item() <= 1e-03
        # learned parameters
        assert ns_dict["μ"].item() - (
            -1) <= 1e-02  # learned μ should be close to -1
        assert ns_dict["σ"].item() - (
            0) <= 1e-02  # learned σ should be close to 0

        # u : ℝ ~ 𝒩(0.0, 1.0)
        u = ns_dict["u"]
        assert u.shape == torch.Size([])
        # u_vec : ℝ[5] ~ 𝒩(0.0, 1.0, 5)
        u_vec = ns_dict["u_vec"]
        assert u_vec.shape == torch.Size([5])

        # v : ℝ ~ 𝒰(0.0, 1.0)
        v = ns_dict["v"]
        assert v.shape == torch.Size([])
        assert 0.0 <= v <= 1.0
        # v_vec : ℝ[5] ~ 𝒰(0.0, 1.0, 5)
        v_vec = ns_dict["v_vec"]
        assert v_vec.shape == torch.Size([5])
        assert v_vec.min() >= 0.0
        assert v_vec.max() <= 1.0

        # bs : ℝ ~ ℬ(α, β_b)  with α=2.0, β_b=5.0
        bs = ns_dict["bs"]
        assert bs.shape == torch.Size([])
        assert 0.0 < bs < 1.0
        # bs_vec : ℝ[5] ~ ℬ(α, β_b, 5)
        bs_vec = ns_dict["bs_vec"]
        assert bs_vec.shape == torch.Size([5])
        assert bs_vec.min() >= 0.0
        assert bs_vec.max() <= 1.0

        # gs : ℝ ~ Gamma(conc, rate)  with conc=2.0, rate=1.0
        gs = ns_dict["gs"]
        assert gs.shape == torch.Size([])
        assert gs > 0.0
        # gs_vec : ℝ[5] ~ Gamma(conc, rate, 5)
        gs_vec = ns_dict["gs_vec"]
        assert gs_vec.shape == torch.Size([5])
        assert gs_vec.min() > 0.0

        # coin : ℝ ~ Bernoulli(p)  with p=0.4
        coin = ns_dict["coin"]
        assert coin.shape == torch.Size([])
        assert float(coin) in {0.0, 1.0}
        # coin_vec : ℝ[10] ~ Bernoulli(p, 10)
        coin_vec = ns_dict["coin_vec"]
        assert coin_vec.shape == torch.Size([10])
        for s in coin_vec:
            assert s == 0.0 or s == 1.0
