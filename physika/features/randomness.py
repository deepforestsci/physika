from typing import Tuple, List, Callable, Optional, Union
from physika.elf import ELF
from physika.utils.types import Substitution


def extract_dist_args(args: List[Tuple],
                      n_params: int) -> Tuple[List, List, str]:
    """
    Split distribution args into (param_args, shape_args, mode).

    Returns ``(param_args, shape_args, mode)`` where *mode* is one of
    ``"reparam"``, ``"score"``, or ``"none"``. ``param_args`` are
    distribution parameters related to sampling like mean (μ) and standard
    deviation (σ) for Normal distribution. ``shape_args`` are related to size
    of output sampled vector (empty mean to sample one element).

    ``"reparam"`` and ``"score"`` refers to two estimators used in stochastic
    graph computation described in [1]_.

    Parameters
    ----------
    args: List[Union[Tuple, Tuple]]
        List that contains the arguments passed to a probability
        distribution.

    References
    ----------
    .. [1] John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel.
           Gradient estimation using stochastic computation graphs. Advances
           in neural information processing systems, 28, 2015.

    Example
    -------
    >>> from physika.features.randomness import extract_dist_args
    >>> # Normal distribution (mean = 0, std = 1),
    >>> # 1 sample (ℝ),
    >>> # none grad mode
    >>> args = [("num", 0.0), ("num", 1.0)]
    >>> extract_dist_args(args, n_params=2)
    ([('num', 0.0), ('num', 1.0)], [], 'none')

    >>> # Normal distribution (mean = 0, std = 1),
    >>> # 20 samples (ℝ[20]),
    >>> # none grad mode
    >>> args = [("num", 0.0), ("num", 1.0), ("num", 20.0), ("num", 1.0)]
    >>> extract_dist_args(args, n_params=2)
    ([('num', 0.0), ('num', 1.0)], [('num', 20.0), ('num', 1.0)], 'none')

    >>> # Normal distribution (mean = 0, std = 1),
    >>> # 1 sample (ℝ),
    >>> # 'reparam' as grad mode
    >>> args = [("num", 0.0), ("num", 1.0), ("string", "reparam")]
    >>> extract_dist_args(args, n_params=2)
    ([('num', 0.0), ('num', 1.0)], [], 'reparam')
    """

    # strip grad mode from args passed to a probability distribution
    if args and isinstance(
            args[-1], tuple) and args[-1][0] in ("string", "equation_string"):
        remaining = list(args[:-1])
        mode = args[-1][1]
    else:
        remaining = list(args)
        mode = "none"
    param_args = remaining[:n_params]
    shape_args = remaining[n_params:]
    return param_args, shape_args, mode


def sample(dist_expr: str, shape_args: List[Tuple], mode: str,
           default_mode: str, to_expr: Callable) -> str:
    """
    Emit PyTorch source code for a stochastic node in a Physika program.

    Physika models probabilistic programs as Stochastic Computation Graphs
    (SCGs) [1]_.  In an SCG, random variables are known as stochastic node and
    other operations, not related with randomness, are deterministic nodes.
    Gradients flow through deterministic nodes by backpropagation, but stochastic
    nodes require a dedicated estimator to propagate gradients when sampling from
    a distribution.

    Physika supports two estimators:

    - Pathwise Estimator (``"reparam"``, default for continuous
      distributions): the sample is written as a deterministic
      transformation of a noise variable, e.g.
      ``z: ℝ = μ + σ·ε`` where ``ε : ℝ ~ N(0,1)``.
      PyTorch's ``rsample()`` allows gradients to flow through ``μ``
      and ``σ``.

    - Score Function Estimator (``"score"``, for non-continous distributions):
        ``∇ log p(x, θ)`` is used to estimate the gradient without needing a
        differentiable sampler.  The sample is detached from the tape
        (``sample().detach()``) and a differentiable ``log_prob`` term in
        the loss is needed so that the gradient is computed.

    Parameters
    ----------
    dist_expr : str
        Emitted PyTorch source code (``torch.distributions.Dist(...)``) expression.
    shape_args : list
        Tuple containing the output shape dimensions. Empty values means scalar sample.
    mode : str
        Explicit grad mode from the source.
    default_mode : str
        Fallback mode when *mode* is ``"none"``. ``"reparam"`` for continuous
        distributions, ``"score"`` for non-continuous distributions such as Bernoulli.
    to_expr : callable
        ``ast_to_torch_expr`` used to emit sub-expression for dims and shapes.

    Returns
    -------
    str
        A Python code string that evaluates to a sampled tensor.

    Example
    -------
    >>> from physika.features.randomness import sample
    >>> # Scalar reparam sample
    >>> sample("torch.distributions.Normal(0.0, 1.0)", [], "none", "reparam", str)
    'torch.distributions.Normal(0.0, 1.0).rsample()'

    >>> shape_nodes = [("num", 20.0), ("num", 1.0)]
    >>> to_expr = lambda node: str(node[1])
    >>> # 2D reparam normal sample, shape (20, 1)
    >>> sample("torch.distributions.Normal(0.0, 1.0)", shape_nodes, "none", "reparam", to_expr)  # noqa: E501
    'torch.distributions.Normal(0.0, 1.0).rsample((int(20.0), int(1.0),))'

    >>> # Bernoulli (score function sample)
    >>> sample("torch.distributions.Bernoulli(0.3)", [], "score", "score", str)
    'torch.distributions.Bernoulli(0.3).sample().detach()'
    """
    if mode != "none":
        effective = mode
    else:
        effective = default_mode
    if shape_args:
        dims = ", ".join(f"int({to_expr(node)})" for node in shape_args)
        shape = f"({dims},)"
    else:
        shape = None

    if effective == "reparam":
        return f"{dist_expr}.rsample({shape or ''})"
    elif effective == "score":
        return f"{dist_expr}.sample({shape or ''}).detach()"
    else:
        return f"{dist_expr}.sample({shape or ''})"


def normal_dist(args: List[Tuple], to_expr: Callable, **ctx) -> str:
    """
    Emit Pytorch code for sampling from a Normal distribution based on args
    (mean, std).

    Parameters
    ----------
    args: List[Union[Tuple, Tuple]]
        List that contains the arguments passed to a probability
        distribution.
    to_expr : callable
        ``ast_to_torch_expr`` to transform AST elements for normal distribution
        to valid torch code as strings.

    Example
    -------
    >>> from physika.features.randomness import normal_dist
    >>> from physika.utils.ast_utils import ast_to_torch_expr
    >>> args = [('num', 0.0), ('num', 1.0)]
    >>> normal_dist(args, ast_to_torch_expr)
    'torch.distributions.Normal(0.0, 1.0).rsample()'
    """
    param_args, shape_args, mode = extract_dist_args(args, n_params=2)
    mu = to_expr(param_args[0])
    sigma = to_expr(param_args[1])
    dist = f"torch.distributions.Normal({mu}, {sigma})"
    return sample(dist, shape_args, mode, "reparam", to_expr)


def uniform_dist(args: List[Tuple], to_expr: Callable, **ctx) -> str:
    """
    Emit Pytorch code for sampling from a Uniform distribution based on args
    (lo, hi).

    Parameters
    ----------
    args: List[Union[Tuple, Tuple]]
        List that contains the arguments passed to a probability
        distribution.
    to_expr : callable
        ``ast_to_torch_expr`` to transform AST elements for normal distribution
        to valid torch code as strings.

    Example
    -------
    >>> from physika.features.randomness import normal_dist
    >>> from physika.utils.ast_utils import ast_to_torch_expr
    >>> args = [('num', 0.0), ('num', 1.0)]
    >>> normal_dist(args, ast_to_torch_expr)
    'torch.distributions.Normal(0.0, 1.0).rsample()'
    """
    param_args, shape_args, mode = extract_dist_args(args, n_params=2)
    lo = to_expr(param_args[0])
    hi = to_expr(param_args[1])
    dist = f"torch.distributions.Uniform({lo}, {hi})"
    return sample(dist, shape_args, mode, "reparam", to_expr)


def beta_dist(args: List[Tuple], to_expr: Callable, **ctx) -> str:
    """
    Emit Pytorch code for sampling from a Beta distribution based on args
    (alpha, beta).

    Parameters
    ----------
    args: List[Union[Tuple, Tuple]]
        List that contains the arguments passed to a probability
        distribution.
    to_expr : callable
        ``ast_to_torch_expr`` to transform AST elements for beta distribution
        to valid torch code as strings.

    Example
    -------
    >>> from physika.features.randomness import beta_dist
    >>> from physika.utils.ast_utils import ast_to_torch_expr
    >>> args = [('num', 0.0), ('num', 1.0)]
    >>> beta_dist(args, ast_to_torch_expr)
    'torch.distributions.Beta(0.0, 1.0).rsample()'
    """
    param_args, shape_args, mode = extract_dist_args(args, n_params=2)
    alpha = to_expr(param_args[0])
    beta = to_expr(param_args[1])
    dist = f"torch.distributions.Beta({alpha}, {beta})"
    return sample(dist, shape_args, mode, "reparam", to_expr)


def gamma_dist(args: List[Tuple], to_expr: Callable, **ctx) -> str:
    """
    Emit Pytorch code for sampling from a Gamma distribution based on args
    (concentration, rate).

    Parameters
    ----------
    args: List[Union[Tuple, Tuple]]
        List that contains the arguments passed to a probability
        distribution.
    to_expr : callable
        ``ast_to_torch_expr`` to transform AST elements for gamma distribution
        to valid torch code as strings.

    Example
    -------
    >>> from physika.features.randomness import gamma_dist
    >>> from physika.utils.ast_utils import ast_to_torch_expr
    >>> args = [('num', 0.0), ('num', 1.0)]
    >>> gamma_dist(args, ast_to_torch_expr)
    'torch.distributions.Gamma(0.0, 1.0).rsample()'
    """
    param_args, shape_args, mode = extract_dist_args(args, n_params=2)
    conc = to_expr(param_args[0])
    rate = to_expr(param_args[1])
    dist = f"torch.distributions.Gamma({conc}, {rate})"
    return sample(dist, shape_args, mode, "reparam", to_expr)


def bernoulli_dist(args: List[Tuple], to_expr: Callable, **ctx) -> str:
    """
    Emit Pytorch code for sampling from a Bernoulli distribution based
    on args (p).

    Parameters
    ----------
    args: List[Union[Tuple, Tuple]]
        List that contains the arguments passed to a probability
        distribution.
    to_expr : callable
        ``ast_to_torch_expr`` to transform AST elements for bernoulli
        distribution to valid torch code as strings.

    Example
    -------
    >>> from physika.features.randomness import bernoulli_dist
    >>> from physika.utils.ast_utils import ast_to_torch_expr
    >>> args = [('num', 0.0)]
    >>> bernoulli_dist(args, ast_to_torch_expr)
    'torch.distributions.Bernoulli(0.0).sample().detach()'
    """
    # Bernoulli has no reparametrization trick sampling (non-continours)
    # always score function estimator mode
    param_args, shape_args, _ = extract_dist_args(args, n_params=1)
    p = to_expr(param_args[0])
    dist = f"torch.distributions.Bernoulli({p})"
    return sample(dist, shape_args, "score", "score", to_expr)


def get_shape_args(call_args: List[Tuple], env: dict) -> List[Tuple]:
    """
    Extract shape arguments from distribution arguments.

    Distribution calls in Physika have leading args
    as distribution parameters, and the optional trailing
    args specify the output shape.  This function is a helper for type checker
    algorithm to get shape args without requiring explicit knowledge of how
    many params each distribution takes.

    Starts from the end of distribution args and stopping at the first
    distribution parameter rather than a size.

    A ``("string", ...)`` or ``("equation_string", ...)`` arg
    (a gradient mode hint such as ``"reparam"`` or ``"score"``) is stripped
    before looking for shape arguments.

    Parameters
    ----------
    call_args : List[Tuple]
        Argument list from a distribution ``"call"`` AST node.
    env : dict
        Type environment accumulated by the type checker.  Must include
        ``("__val__", name)`` entries for variables whose literal integer
        value was tracked during declaration.

    Returns
    -------
    List[Tuple]
        Shape arg AST nodes in order.  Empty when
        the call produces a scalar sample.

    Examples
    --------
    >>> from physika.features.randomness import get_shape_args
    >>> env = {}
    >>> # Normal(0.0, 1.0) — float params, no size args
    >>> get_shape_args([("num", 0.0), ("num", 1.0)], env)
    []

    >>> # Normal(0.0, 1.0, 100)
    >>> get_shape_args([("num", 0.0), ("num", 1.0), ("num", 100)], env)
    [('num', 100)]

    >>> # Normal(mu, sigma, n)
    >>> env_n = {("__val__", "n"): 100}
    >>> get_shape_args([("var", "mu"), ("var", "sigma"), ("var", "n")], env_n)
    [('var', 'n')]

    >>> get_shape_args([("num", 0.0), ("num", 1.0), ("num", 50), ("string", "reparam")], env)  # noqa: E501
    [('num', 50)]
    """
    args = list(call_args)
    if args and isinstance(
            args[-1], tuple) and args[-1][0] in ("string", "equation_string"):
        args = args[:-1]
    shape: List[Tuple] = []
    for a in reversed(args):
        if isinstance(a, tuple) and a[0] == "num" and isinstance(a[1], int):
            shape.insert(0, a)
        elif isinstance(a, tuple) and a[0] == "var" and isinstance(
                env.get(("__val__", a[1])), int):
            shape.insert(0, a)
        else:
            break
    return shape


def get_dim(val: Tuple, env: dict) -> Optional[Union[int, str]]:
    """
    Get the dimension value, int or string, from a distribution shape
    argument.

    Parameters
    ----------
    val: Tuple
        AST node ("num"/"var") or a  dim (int or str) from
        declared_type.dims[i][0].
    env: dict
        Enviroment dictionary with variables, classes, functions,
        and types accumulated so far.

    Example
    -------
    >>> from physika.features.randomness import get_dim
    >>> env = {("__val__", "n"): 100}
    >>> get_dim(("var", "n"), env)
    100
    """
    if isinstance(val, tuple):
        if val[0] == "num":
            return int(val[1])
        if val[0] == "var":
            tracked = env.get(("__val__", val[1]))
            return int(tracked) if isinstance(tracked, int) else val[1]
    elif isinstance(val, int):
        return val
    elif isinstance(val, str):
        tracked = env.get(("__val__", val))
        return int(tracked) if isinstance(tracked, int) else val
    return None


class RandomnessFeature(ELF):
    """
    Differentiable probabilistic sampling for Physika.

    ``RandomnessFeature``, as an ELF subclass, adds support for sampling from
    Pytorch probability distributions in Physika programs being fully
    differentiable. Physika random sampling uses tilde syntax ``~`` to
    draw from a distribution (e.g. ``x ~ Normal(0, 1)``). Each
    distribution recieves its own set of parameters (e.g. mean and std
    for Normal). There are two general arguments: 1) shape parameters to
    specify the number of samples and their shapes, 2) a string argument
    to specify the gradient estimator to use (``"reparam"``, ``"score"``,
    or ``"none"``).

    Supported distributions
    -----------------------
    - Normal(µ, σ, n, mode)
    - Uniform(a, b, n, mode)
    - Beta(α, β, n, mode)
    - Gamma(concentration, rate, n, mode)
    - Bernoulli(p, n, mode)

    Examples
    --------
    >>> import torch
    >>> from physika.lexer import lexer
    >>> from physika.parser import parser, symbol_table
    >>> from physika.utils.ast_utils import build_unified_ast
    >>> from physika.codegen import from_ast_to_torch
    >>> def run_phyk(src):
    ...     symbol_table.clear()
    ...     lexer.lexer.lineno = 1
    ...     ast = build_unified_ast(parser.parse(src, lexer=lexer), symbol_table)  # noqa: E501
    ...     exec(from_ast_to_torch(ast, print_code=False), {})

    >>> # Physika scalar Normal and Bernoulli samples
    >>> src = '''
    ... μ : ℝ = 0.0
    ... σ : ℝ = 1.0
    ... x : ℝ ~ Normal(μ, σ)
    ... coin : ℝ ~ Bernoulli(0.5)
    ... '''
    >>> # Execute code
    >>> run_phyk(src)
    """

    name = "randomness"

    def lexer_rules(self) -> dict:
        """
        Adds ``TILDE`` token (``"~"``) for stochastic sampling syntax
        and includes ``PHYSIKA`` reserved keyword so that
        ``physika.seed(n)`` parses.
        Also, includes greek letters aliases for mapping with torch
        distributions:

        - ``𝒩`` → ``Normal``
        - ``𝒰`` → ``Uniform``
        - ``Γ`` → ``Gamma``
        - ``ℬ`` → ``Beta``

        Returns
        -------
        dict
            Dictionary with ``tokens`` (``["TILDE", "PHYSIKA"]``) and
            ``token_funcs`` (``t_TILDE``, ``t_DIST_NORMAL``, ``t_DIST_GAMMA``,
            ``t_DIST_BETA``, ``t_DIST_UNIFORM``).

        Examples
        --------
        >>> from physika.features.randomness import RandomnessFeature
        >>> rules = RandomnessFeature().lexer_rules()
        >>> rules["tokens"]
        ['TILDE', 'PHYSIKA']
        >>> rules["reserved"]
        {'physika': 'PHYSIKA'}
        >>> rules["token_funcs"][1].__name__
        't_DIST_NORMAL'
        >>> rules["token_funcs"][2].__name__
        't_DIST_GAMMA'
        >>> rules["token_funcs"][3].__name__
        't_DIST_BETA'
        >>> rules["token_funcs"][4].__name__
        't_DIST_UNIFORM'
        """

        def t_TILDE(t):
            r"~"
            return t

        # Distribution aliases
        def t_DIST_NORMAL(t):
            # syntax: x ~ 𝒩(0, 1)
            # sample x from Normal distribution with mean 0 and std 1
            r"𝒩"
            t.type = "ID"
            t.value = "Normal"
            return t

        def t_DIST_UNIFORM(t):
            # syntax: x ~ 𝒰(0, 1)
            # Uniform distribution between 0 and 1
            r"𝒰"
            t.type = "ID"
            t.value = "Uniform"
            return t

        def t_DIST_GAMMA(t):
            # syntax: x ~ Γ(2, 3)
            # Gamma distribution with concentration 2 and rate 3
            r"Γ"
            t.type = "ID"
            t.value = "Gamma"
            return t

        def t_DIST_BETA(t):
            # syntax: x ~ ℬ(0.5, 0.5)
            # Beta distribution with alpha 0.5 and beta 0.5
            r"ℬ"
            t.type = "ID"
            t.value = "Beta"
            return t

        return {
            "reserved": {
                "physika": "PHYSIKA"
            },
            "tokens": ["TILDE", "PHYSIKA"],
            "token_funcs": [
                t_TILDE, t_DIST_NORMAL, t_DIST_GAMMA, t_DIST_BETA,
                t_DIST_UNIFORM
            ]
        }

    def parser_rules(self) -> list:
        """
        Handler for new grammar rules.

        Eleven new PLY grammar functions:

        - Seven for random sampling at top-level, function/method bodies,
          and for-loops.
        - Two for score-function estimator
          ``name1 : T1, name2 : T2 ~ Dist(args)`` syntax.
        - Two for ``physika.seed(n)`` at top-level and inside function bodies.

        Returns
        -------
        list
            List of PLY grammar functions to be injected into
            ``physika.parser``.

        Examples
        --------
        >>> from physika.features import RandomnessFeature
        >>> rules = RandomnessFeature().parser_rules()
        >>> len(rules)
        11
        >>> rules[0].__name__
        'p_sample_untyped'
        """

        def p_sample_untyped(p):
            """sample : ID TILDE func_factor"""
            p[0] = ("sample_rhs", p[1], None, p[3], p.lineno(1))

        def p_sample_typed(p):
            """sample : ID COLON type_spec TILDE func_factor"""
            p[0] = ("sample_rhs", p[1], p[3], p[5], p.lineno(1))

        def p_statement_sample(p):
            """statement : sample NEWLINE"""
            _tag, name, type_spec, call_node, lineno = p[1]
            if type_spec is None:
                p[0] = ("sample", name, call_node, lineno)
            else:
                p[0] = ("typed_sample", name, type_spec, call_node, lineno)

        def p_func_body_stmt_sample(p):
            """func_body_stmt : sample NEWLINE"""
            _tag, name, type_spec, call_node, _lineno = p[1]
            if type_spec is None:
                p[0] = ("sample", name, call_node)
            else:
                p[0] = ("typed_sample", name, type_spec, call_node)

        def p_for_statement_sample(p):
            """for_statement : sample NEWLINE"""
            _tag, name, type_spec, call_node, _lineno = p[1]
            if type_spec is None:
                p[0] = ("sample", name, call_node)
            else:
                p[0] = ("typed_sample", name, type_spec, call_node)

        def p_func_factor_sample_expr(p):
            """func_factor : ID TILDE func_factor"""
            p[0] = ("sample_expr", p[1], p[3])

        def p_for_sample(p):
            """func_factor : FOR ID COLON TYPE LPAREN func_expr RPAREN ARROW sample
               factor : FOR ID COLON TYPE LPAREN func_expr RPAREN ARROW sample"""  # noqa: E501
            _, samp_name, type_spec, call_node, _ = p[9]
            if type_spec:
                body = ("typed_sample_expr", samp_name, type_spec, call_node)
            else:
                body = ("sample_expr", samp_name, call_node)
            p[0] = ("for_expr", p[2], p[6], body)

        def p_statement_dual_sample(p):
            """statement : ID COLON type_spec COMMA ID COLON type_spec TILDE func_factor NEWLINE"""  # noqa: E501
            p[0] = ("dual_sample", p[1], p[3], p[5], p[7], p[9])

        def p_func_body_stmt_dual_sample(p):
            """func_body_stmt : ID COLON type_spec COMMA ID COLON type_spec TILDE func_factor NEWLINE"""  # noqa: E501
            p[0] = ("dual_sample", p[1], p[3], p[5], p[7], p[9])

        def p_statement_seed(p):
            """statement : PHYSIKA DOT ID LPAREN func_expr RPAREN NEWLINE"""
            p[0] = ("seed", p[5])

        def p_func_body_stmt_seed(p):
            """func_body_stmt : PHYSIKA DOT ID LPAREN func_expr RPAREN NEWLINE"""  # noqa: E501
            p[0] = ("seed", p[5])

        return [
            p_sample_untyped,
            p_sample_typed,
            p_statement_sample,
            p_func_body_stmt_sample,
            p_for_statement_sample,
            p_func_factor_sample_expr,
            p_for_sample,
            p_statement_dual_sample,
            p_func_body_stmt_dual_sample,
            p_statement_seed,
            p_func_body_stmt_seed,
        ]

    def type_rules(self) -> dict:
        """
        Adds two type checker rules that verifies the declared and inferred
        type of random sampling:

        - ``typed_sample_type``: checks for statements, declarations,
          and assignments.
        - ``sample_expr_type``: intended for expressions
          (e.g., inside inline for-loops).

        Returns
        -------
        dict
            Dispatch table mapping ``"typed_sample_type"`` and
            ``"sample_expr_type"`` AST tags to their type
            inference handlers.

        Examples
        --------
        >>> from physika.features import RandomnessFeature
        >>> rules = RandomnessFeature().type_rules()
        >>> sorted(rules.keys())
        ['dual_sample', 'sample_expr', 'typed_sample', 'typed_sample_expr']
        """

        def typed_sample_type(node: tuple, env: dict, s: Substitution,
                              func_env: dict, class_env: dict,
                              add_error: Callable, infer_expr: Callable):
            """
            Checks correct types of a sample statement like:
                ``name : type ~ Dist(...)``.

            Validates that the declared type is same as the type produced by
            the distribution call.

            ``typed_sample_type`` checks for rank and dimension correctness.
            Rank check verifies the number of size arguments in the
            distribution call must match the rank of the declared type.  A
            scalar declaration (``ℝ``) requires no size args. A vector
            declaration (``ℝ[n]``) requires exactly one.  A mismatch is
            registered as a type error. Dimension type check verifies declared
            and inferred shape args are consistent.

            Parameters
            ----------
            node : tuple
                AST node of the form
                ``("typed_sample", name, type_spec, call_node [, lineno])``.
            env : dict
                Type environment mapping variable names to their inferred
                types.
            s : Substitution
                Accumulated substitutions so far.
            add_error : Callable[[str], None]
                Callback to record a type error message.

            Returns
            -------
            tuple[Type, Substitution]
                The declared type (converted from the annotation via
                ``from_typespec``) and the unchanged substitution ``s``.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.types import Substitution, TTensor
            >>> rules = RandomnessFeature().type_rules()
            >>> check = rules["typed_sample"]
            >>> s = Substitution()
            >>> errors = []
            >>> env = {("__val__", "n"): 100}
            >>> node = ("typed_sample", "x", ("tensor", [(100, "invariant")]),
            ...         ("call", "Normal", [("var", "mu"), ("var", "sigma"), ("var", "n")]))  # noqa: E501
            >>> t, _ = check(node, env, s, errors.append)
            >>> isinstance(t, TTensor)
            True
            >>> errors
            []
            """
            from physika.utils.type_checker_utils import from_typespec
            from physika.utils.types import TTensor
            name = node[1]
            type_spec = node[2]
            call_node = node[3]
            declared_type = from_typespec(type_spec)

            if isinstance(call_node, tuple) and call_node[0] == "call":
                func_name = call_node[1]
                dist_args = list(call_node[2])
                # after stripping any mode string, the last declared_rank
                # args are the shape args.
                if dist_args and isinstance(
                        dist_args[-1], tuple) and dist_args[-1][0] in (
                            "string", "equation_string"):  # noqa: E501
                    dist_args = dist_args[:-1]

                actual_shape_args = get_shape_args(dist_args, env)
                actual_rank = len(actual_shape_args)
                if isinstance(declared_type, TTensor):
                    declared_rank = len(declared_type.dims)
                    if actual_rank > 0 and declared_rank == 0:
                        # concrete shape args present but scalar declared
                        add_error(
                            f"'{name}': declared ℝ but {func_name}(...) produces a ℝ[n] sample"  # noqa: E501
                        )
                    elif actual_rank == 0 and declared_rank > 0:
                        # no shape args detected
                        # error when all declared dims are concrete ints
                        declared_dims = [d[0] for d in declared_type.dims]
                        if all(isinstance(d, int) for d in declared_dims):
                            add_error(
                                f"'{name}': declared {declared_type} but {func_name}(...) produces a ℝ sample"  # noqa: E501
                            )
                        else:
                            # symbolic dim case
                            shape_args = dist_args[-declared_rank:]
                            for i, shape_arg in enumerate(shape_args):
                                actual = get_dim(shape_arg, env)
                                declared = get_dim(declared_type.dims[i][0],
                                                   env)
                                if declared != actual:
                                    if isinstance(declared,
                                                  int) and isinstance(
                                                      actual, int):
                                        add_error(
                                            f"'{name}': declared {declared_type}. "  # noqa: E501
                                            f"{func_name}(...) in dim[{i}] infers {actual} but declared {declared}"  # noqa: E501
                                        )
                                    elif isinstance(declared,
                                                    str) and isinstance(
                                                        actual, str):
                                        add_error(
                                            f"'{name}': declared {declared_type}. "  # noqa: E501
                                            f"{func_name}(...) in dim[{i}] infers {actual} but declared {declared}"  # noqa: E501
                                        )
                    else:
                        # check each dimension for concrete mismatches
                        shape_args = dist_args[
                            -declared_rank:] if declared_rank > 0 else []
                        for i, shape_arg in enumerate(shape_args):
                            actual = get_dim(shape_arg, env)
                            declared = get_dim(declared_type.dims[i][0], env)
                            if declared != actual:
                                if isinstance(declared, int) and isinstance(
                                        actual, int):
                                    add_error(
                                        f"'{name}': declared {declared_type}. "
                                        f"{func_name}(...) in dim[{i}] infers {actual} but declared {declared}"  # noqa: E501
                                    )
                                elif isinstance(declared, str) and isinstance(
                                        actual, str):
                                    add_error(
                                        f"'{name}': declared {declared_type}. "
                                        f"{func_name}(...) in dim[{i}] infers {actual} but declared {declared}"  # noqa: E501
                                    )
                else:
                    # declared ℝ
                    if actual_rank > 0:
                        add_error(
                            f"'{name}': declared ℝ but {func_name}(...) produces a ℝ[n] sample"  # noqa: E501
                        )

            env[name] = declared_type
            return declared_type, s

        def sample_expr_type(node: tuple, env: dict, s: Substitution,
                             func_env: dict, class_env: dict,
                             add_error: Callable, infer_expr: Callable):
            """
            Type checking for sampling inside an expression rather than an
            assignment statement.

            The type is determined by shape args passed to distribution call.
            If no shape args, sample is a scalar and returns ``ℝ``. The inferred
            type is not registered in ``env`` because ``sample_expr`` is an
            expression, not a declaration.

            Parameters
            ----------
            node : tuple
                AST node of the form ``("sample_expr", name, distb_call)``.
                ``dist_call_node`` is the distribution ``"call"`` node.
            env : dict
                Type environment, used to get size
                args.
            s : Substitution
                Accumulated type substitutions.
            func_env : dict
                Function signatures in scope.  Not used by this handler.
            class_env : dict
                Class definitions in scope.  Not used by this handler.
            add_error : Callable[[str], None]
                Error callback.  Not used by this handler.
            infer_expr : Callable
                Recursive expression type-inference.  Not used by this handler.

            Returns
            -------
            tuple[Type, Substitution]
                ``(T_REAL, s)`` for a scalar sample, or
                ``(TTensor(((dim, "invariant"),)), s)`` for a 1-D vector sample.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.types import Substitution, T_REAL, TTensor
            >>> rules = RandomnessFeature().type_rules()
            >>> check = rules["sample_expr"]
            >>> s = Substitution()
            >>> # Scalar: ε ~ Normal(0.0, 1.0)
            >>> node = ("sample_expr", "ε", ("call", "Normal", [("num", 0.0), ("num", 1.0)]))  # noqa: E501
            >>> t, _ = check(node, {}, s, {}, {}, None, None)
            >>> t is T_REAL
            True
            >>> # Vector: ε ~ Normal(0.0, 1.0, 20) — size arg present
            >>> node = ("sample_expr", "ε", ("call", "Normal", [("num", 0.0), ("num", 1.0), ("num", 20)]))  # noqa: E501
            >>> t, _ = check(node, {}, s, {}, {}, None, None)
            >>> isinstance(t, TTensor)
            True
            """
            from physika.utils.types import T_REAL, TTensor
            from physika.utils.type_checker_utils import from_typespec
            # ("typed_sample_expr", name, type_spec, call_node)
            # type is declared
            if node[0] == "typed_sample_expr":
                return from_typespec(node[2]), s
            # sample_expr: ("sample_expr", name, call_node)
            # infer type from shape args
            call_node = node[2]
            if isinstance(call_node, tuple) and call_node[0] == "call":
                shape_args = get_shape_args(call_node[2], env)
                if shape_args:
                    if shape_args[0][0] in ("num", "var"):
                        dim = shape_args[0][1]
                    else:
                        dim = shape_args[0]
                    return TTensor(((dim, "invariant"), )), s
            return T_REAL, s

        def check_dual_sample(node: tuple, env: dict, s: Substitution,
                              func_env: dict, class_env: dict,
                              add_error: Callable, infer_expr: Callable):
            """
            Check ``name1 : T1, name2 : T2 ~ Dist(...)`` is
            well typed.

            Compare ``name1`` against the distribution shape args.  Validates
            ``name2`` (log_prob) against ``name1``'s type: must be ``ℝ``
            or have the same shape as the sample.  Reports a type
            error for concrete dimension mismatches like ``b_s : ℝ[2]`` with
            ``Bernoulli(p, n)``.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.types import Substitution, TTensor
            >>> rules = RandomnessFeature().type_rules()
            >>> check = rules["dual_sample"]
            >>> s = Substitution()
            >>> # Correct: both ℝ[n]
            >>> env = {}
            >>> errors = []
            >>> node = ("dual_sample",
            ...         "b_s", ("tensor", [("n", "invariant")]),
            ...         "log_prob", ("tensor", [("n", "invariant")]),
            ...         ("call", "Bernoulli",
            ...          [("var", "p"), ("var", "n"), ("string", "score-function")]))
            >>> t, _ = check(node, env, s, {}, {}, errors.append, None)
            >>> errors
            []
            >>> # Error: b_s : ℝ[2] declared but distribution produces ℝ[n]
            >>> env2 = {}
            >>> errors2 = []
            >>> node2 = ("dual_sample",
            ...          "b_s", ("tensor", [(2, "invariant")]),
            ...          "log_prob", ("tensor", [(2, "invariant")]),
            ...          ("call", "Bernoulli",
            ...           [("var", "p"), ("var", "n"), ("string", "score-function")]))
            >>> _, _ = check(node2, env2, s, {}, {}, errors2.append, None)
            >>> len(errors2) > 0
            True
            >>> # Error: log_prob shape mismatches sample shape
            >>> env3 = {("__val__", "n"): 10}
            >>> errors3 = []
            >>> node3 = ("dual_sample",
            ...          "b_s", ("tensor", [(10, "invariant")]),
            ...          "log_prob", ("tensor", [(5, "invariant")]),
            ...          ("call", "Bernoulli",
            ...           [("var", "p"), ("var", "n"), ("string", "score-function")]))  # noqa: E501
            >>> _, _ = check(node3, env3, s, {}, {}, errors3.append, None)
            >>> len(errors3) > 0
            True
            """
            from physika.utils.type_checker_utils import from_typespec
            from physika.utils.types import TTensor
            _, name1, type_spec1, name2, type_spec2, call_node = node

            # Validate name1 via typed_sample_type
            fake1 = ("typed_sample", name1, type_spec1, call_node)
            type1, s = typed_sample_type(fake1, env, s, func_env, class_env,
                                         add_error, infer_expr)

            # Register name2
            type2 = from_typespec(type_spec2)
            env[name2] = type2

            # Check name2
            # name2 must have the same type shape as name1
            if isinstance(type2, TTensor) and isinstance(type1, TTensor):
                if len(type2.dims) != len(type1.dims):
                    add_error(
                        f"'{name2}': declared {type2} but '{name1}' has "
                        f"{len(type1.dims)} dimension(s) — log_prob shape must "  # noqa: E501
                        f"match the sample or be declared ℝ for scalar sum")
                else:
                    for i, (d2, d1) in enumerate(zip(type2.dims, type1.dims)):
                        v2, v1 = d2[0], d1[0]
                        if (isinstance(v2, int) and isinstance(v1, int)
                                and v2 != v1):
                            add_error(
                                f"'{name2}': declared {type2} but '{name1}' "
                                f"dim[{i}] is {v1}")
            return type1, s

        return {
            "typed_sample": typed_sample_type,
            "sample_expr": sample_expr_type,
            "typed_sample_expr": sample_expr_type,
            "dual_sample": check_dual_sample,
        }

    def forward_rules(self) -> dict:
        """
        Code generation handler for emiting random sampling nodes
        using Pytorch as backend.

        ``sample_stmt_emit`` emits ``name = <dist>.rsample(...)`` for
        statement-level sample nodes (``"sample"``, ``"typed_sample"``,
        ``"body_sample"``, ``"body_typed_sample"``, ``"for_sample"``,
        ``"for_typed_sample"``). For inline sample expressions nodes,
        ``sample_expr_emit`` emits call expression  (``"sample_expr"``,
        ``"typed_sample_expr"``). ``make_call_emit`  wraps each distribution
        function (e.g. ``normal_dist``) so it can be dispatched by
        ``"call:Name"`` key.

        Returns
        -------
        dict
            Dispatch table mapping AST node tags and ``"call:Name"`` keys to
            their code generation emiters.

        Examples
        --------
        >>> from physika.features import RandomnessFeature
        >>> from physika.utils.ast_utils import ast_to_torch_expr
        >>> rules = RandomnessFeature().forward_rules()

        >>> # Physika code:
        >>> # x ~ Normal(0.0, 1.0)
        >>> node = ("sample", "x", ("call", "Normal", [("num", 0.0), ("num", 1.0)]))  # noqa: E501
        >>> rules["sample"](node, ast_to_torch_expr)
        'x = torch.distributions.Normal(0.0, 1.0).rsample()'
        """

        def sample_expr_emit(node: Tuple, to_expr: Callable, **ctx):
            """
            Emit code for sample expression.

            Handles both ``"sample_expr"`` and ``"typed_sample_expr"`` AST
            nodes by extracting the distribution call node and delegating
            to ``to_expr``.

            Parameters
            ----------
            node : tuple
                ``("sample_expr", name, call_node)`` or
                ``("typed_sample_expr", name, type_spec, call_node)``.
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit the distribution call.

            Returns
            -------
            str
                PyTorch distribution sampling string.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = RandomnessFeature().forward_rules()
            >>> node = ("sample_expr", "ε", ("call", "Normal", [("num", 0.0), ("num", 1.0), ("num", 2)]))  # noqa: E501
            >>> rules["sample_expr"](node, ast_to_torch_expr)
            'torch.distributions.Normal(0.0, 1.0).rsample((int(2),))'
            """
            if node[0] == "typed_sample_expr":
                call_node = node[3]
            else:
                call_node = node[2]
            return to_expr(call_node)

        def sample_stmt_emit(node: Tuple, to_expr: Callable, **ctx):
            """
            Emit distribution sampling Pytorch code from sample
            statement nodes.

            Handles statement sample nodes in AST with or without
            a type annotation.

            Parameters
            ----------
            node : tuple
                - ``("sample",       name, call_node [, lineno])``
                - ``("typed_sample", name, type_spec, call_node [, lineno])``
                Body and for-loops support (``"body_sample"``, ``"for_sample"``,
                ``"body_typed_sample"``, ``"for_typed_sample"``).
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit the distribution call.

            Returns
            -------
            str
                Pytorch source string.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = RandomnessFeature().forward_rules()
            >>> node = ("sample", "x", ("call", "Normal", [("num", 0.0), ("num", 1.0)]))  # noqa: E501
            >>> rules["sample"](node, ast_to_torch_expr)
            'x = torch.distributions.Normal(0.0, 1.0).rsample()'
            """
            name = node[1]
            typed = node[0] in ("typed_sample", "body_typed_sample",
                                "for_typed_sample")
            call_node = node[3] if typed else node[2]
            return f"{name} = {to_expr(call_node)}"

        def dual_sample_emit(node: Tuple, to_expr: Callable, **ctx):
            """
            Emit a sample paired with its log-probability for computing
            score-function estimator following SCGs.

            The syntax is as follows:

            ``name1 : T1, name2 : T2 ~ Dist(args)`` where ``Dist`` is
            a non-differentiable distribution. At runtime, the distribution
            object, the detached sample, and the accumulated log-probability
            are emitted.

            Parameters
            ----------
            node : tuple
                ``("dual_sample", name1, type1, name2, type2, call_node)``
            to_expr : Callable
                ``ast_to_torch_expr`` used to emit sub-expressions.

            Returns
            -------
            str
                Pytorch source string for sampling and log_prob.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = RandomnessFeature().forward_rules()
            >>> # log_prob : ℝ[n] (per element)
            >>> node = ("dual_sample", "b_s", ("tensor", [(10, "invariant")]),
            ...         "lp", ("tensor", [(10, "invariant")]),
            ...         ("call", "Bernoulli", [("num", 0.5), ("num", 10),
            ...                                ("string", "score-function")]))
            >>> print(rules["dual_sample"](node, ast_to_torch_expr))
            _dist_b_s = torch.distributions.Bernoulli(0.5)
            b_s = _dist_b_s.sample((int(10),)).detach()
            lp = _dist_b_s.log_prob(b_s)
            >>> node = ("dual_sample", "b_s", ("tensor", [(10, "invariant")]),
            ...         "lp", "ℝ",
            ...         ("call", "Bernoulli", [("num", 0.5), ("num", 10),
            ...                                ("string", "score-function")]))
            >>> print(rules["dual_sample"](node, ast_to_torch_expr))
            _dist_b_s = torch.distributions.Bernoulli(0.5)
            b_s = _dist_b_s.sample((int(10),)).detach()
            lp = _dist_b_s.log_prob(b_s).sum()
            """
            _, name1, _type1, name2, _type2, call_node = node
            dist_name = call_node[1]
            raw_args = list(call_node[2])
            if raw_args and isinstance(
                    raw_args[-1], tuple) and raw_args[-1][0] in (
                        "string", "equation_string"):  # noqa: E501
                raw_args = raw_args[:-1]
            declared_rank = len(_type1[1]) if isinstance(
                _type1, tuple) and _type1[0] == "tensor" else 0  # noqa: E501
            shape_args = raw_args[-declared_rank:] if declared_rank > 0 else []
            param_args = raw_args[:
                                  -declared_rank] if declared_rank > 0 else raw_args  # noqa: E501
            params_str = ", ".join(to_expr(a) for a in param_args)
            dist_cls = f"torch.distributions.{dist_name}"
            dist_var = f"_dist_{name1}"
            shape = ""
            if shape_args:
                dims = ", ".join(f"int({to_expr(a)})" for a in shape_args)
                shape = f"({dims},)"
            logprob_expr = f"{dist_var}.log_prob({name1})"
            return "\n".join([
                f"{dist_var} = {dist_cls}({params_str})",
                f"{name1} = {dist_var}.sample({shape}).detach()",
                f"{name2} = {logprob_expr}",
            ])

        def make_call_emit(fn: Callable):
            """
            Wraps a supported distribution function for dispatch under a
            ``"call:Name"`` key.

            Parameters
            ----------
            fn : Callable
                Distribution emitter such as ``normal_dist`` or
                ``bernoulli_dist``.

            Returns
            -------
            Callable
                Handler with signature ``(node, to_expr, **ctx) -> str``
                suitable for ELF forward dispatch table.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> from physika.features.randomness import normal_dist
            >>> rules = RandomnessFeature().forward_rules()
            >>> node = ("call", "Normal", [("num", 0.0), ("num", 1.0)])
            >>> rules["call:Normal"](node, ast_to_torch_expr)
            'torch.distributions.Normal(0.0, 1.0).rsample()'
            """

            def wrapper(node: Tuple, to_expr: Callable, **ctx):
                """
                 ``node[2]`` is the arg list. this wrapper unpacks
                distribution args and forward to ``fn``.

                Parameters
                ----------
                node : tuple
                    ``("call", dist_name, args)`` AST node.
                to_expr : Callable
                    ``ast_to_torch_expr`` for emitting sub-expressions.

                Returns
                -------
                str
                    PyTorch sampling expression produced by ``fn``.
                """
                return fn(node[2], to_expr, **ctx)

            return wrapper

        def seed_emit(node: Tuple, to_expr: Callable, **ctx) -> str:
            """
            Emit ``torch.manual_seed(int(n))`` for ``physika.seed(n)``.

            Examples
            --------
            >>> from physika.features import RandomnessFeature
            >>> from physika.utils.ast_utils import ast_to_torch_expr
            >>> rules = RandomnessFeature().forward_rules()
            >>> rules["seed"](("seed", ("num", 42.0)), ast_to_torch_expr)
            'torch.manual_seed(int(42.0))'
            """
            _, expr = node
            return f"torch.manual_seed(int({to_expr(expr)}))"

        return {
            "sample": sample_stmt_emit,
            "body_sample": sample_stmt_emit,
            "for_sample": sample_stmt_emit,
            "sample_expr": sample_expr_emit,
            "typed_sample_expr": sample_expr_emit,
            "typed_sample": sample_stmt_emit,
            "body_typed_sample": sample_stmt_emit,
            "for_typed_sample": sample_stmt_emit,
            "dual_sample": dual_sample_emit,
            "call:Normal": make_call_emit(normal_dist),
            "call:Uniform": make_call_emit(uniform_dist),
            "call:Beta": make_call_emit(beta_dist),
            "call:Gamma": make_call_emit(gamma_dist),
            "call:Bernoulli": make_call_emit(bernoulli_dist),
            "seed": seed_emit,
        }
