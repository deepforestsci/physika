from physika.features.randomness import (extract_dist_args, sample,
                                         normal_dist, uniform_dist, beta_dist,
                                         gamma_dist, bernoulli_dist)
from physika.utils.ast_utils import ast_to_torch_expr


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
