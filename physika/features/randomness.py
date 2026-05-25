from typing import Tuple, List, Callable


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
    graph computation described in [1].

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
    (SCGs) [1].  In an SCG, random variables are known as stochastic node and
    other operations, not related with randomness, are deterministic nodes.
    Gradients flow through deterministic nodes by backpropagation, but stochastic
    nodes require a dedicated estimator to propagate gradients when sampling from
    a distribution.

    Physika supports two estimators as described in [1]:

    - Pathwise Estimator (``"reparam"``, default for continuous
    distributions):
        The sample is written as a deterministic transformation of a
        random variable. For example,
          ``z: ℝ = μ + σ·ε``, where ``ε : ℝ ~ N(0,1)``.
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

    References
    ----------
    .. [1] John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel.
           Gradient estimation using stochastic computation graphs. Advances
           in neural information processing systems, 28, 2015.

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
