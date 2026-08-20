Normalizing Flows
=======================

This tutorial is to introduce Normalizing Flow models and how to implement them in Physika.
Normalizing Flows are a class of generative models that allow for density estimation and sampling by transforming a simple base distribution into a more complex target distribution.
NICE (Non-linear Independent Components Estimation) is a Normalizing Flow model. 
It works by splitting the input into two halves: one half stays fixed, while the other is transformed using a neural network conditioned on the fixed half. 
Stacking several of these layers produces a transformation that is easy to invert and whose likelihood is easy to compute,
the two properties that make Normalizing Flows useful for both generating data and estimating density.

By the end of this tutorial, you will learn how to train a NICE Normalizing Flow model for density estimation and image generation.
This tutorial is based on the Deep Generative Models (CS236) course notes `[1] <https://deepgenerativemodels.github.io/notes/flow/>`__.


What are Normalizing Flows?
---------------------------

Normalizing Flows are a class of generative models that transform a simple base distribution (e.g., Gaussian) into a more complex target distribution which can model a real world data distribution through a series of invertible transformations.
The key idea is to model the probability density function of the target distribution by applying a sequence of **bijective mappings** to the base distribution.

A bijective mapping is a function :math:`f: \mathbb{R}^N \to \mathbb{R}^N` that is both *injective* (distinct inputs always produce distinct outputs, i.e. :math:`f(a) = f(b) \implies a = b`) and *surjective* (every point in the output space is the image of some input).
Equivalently, a bijective function establishes a one-to-one correspondence between input and output spaces, so it is guaranteed to have a well-defined inverse :math:`f^{-1}` satisfying :math:`f^{-1}(f(z)) = z`.
This invertibility is essential for normalizing flows: it lets us map freely between the simple base distribution and the complex target, and crucially it allows us to compute exact probability densities via the change-of-variables formula.

.. figure:: /_static/tutorial_files/norm_flow/norm_flow_basic.png
   :alt: Illustration of a normalizing flow transforming a simple Gaussian distribution into a complex multi-modal distribution through a sequence of invertible mappings.
   :align: center
   :width: 500px

   **Figure 1.** A normalizing flow transforms a simple base density :math:`p_z(z)` (left) into a complex target density :math:`p(x)` (right) through a chain of invertible transformations :math:`f_1, f_2, \ldots, f_K`. Figure from `[2] <https://lilianweng.github.io/posts/2018-10-13-flow-models/>`__.

Setup and Notation
^^^^^^^^^^^^^^^^^^

Let's consider a probability distribution :math:`\mathcal{P}` over :math:`\mathbb{R}^N`.
We represent this distribution by its **probability density function** :math:`p: \mathbb{R}^N \to \mathbb{R}`, a function that takes a point :math:`x \in \mathbb{R}^N` (a vector of shape :math:`(N,)`) and returns a non-negative real number such that the integral over any region gives the probability of landing in that region.

We begin by sampling from a base distribution:

.. math::
    z \sim \mathcal{N}(\mu, \Sigma)

Here, :math:`\mu \in \mathbb{R}^N` is the **mean vector** (the center of the distribution) and :math:`\Sigma \in \mathbb{R}^{N \times N}` is the **covariance matrix** (encoding variance along each axis and correlations between dimensions).
The notation :math:`z \sim \mathcal{N}(\mu, \Sigma)` means that :math:`z` is a random variable drawn from this :math:`N`-dimensional Gaussian.

In Physika, this sampling can be expressed as:

.. code-block:: text

    μ : ℝ[N] = ...    # mean vector
    z : ℝ[N] ~ for i : ℕ(N) → ε : ℝ ~ Normal(μ, σ)

We then define an invertible transformation :math:`f: \mathbb{R}^N \to \mathbb{R}^N` that maps latent samples to the data space:

.. math::
    x = f(z)

where :math:`f` is a bijective function (and therefore invertible), taking a latent vector :math:`z` of shape :math:`(N,)` and producing a data-space vector :math:`x` of the same shape :math:`(N,)`.

Change of Variables
^^^^^^^^^^^^^^^^^^^

Because :math:`f` is invertible, we can express the **probability density of** :math:`x` using the **change-of-variables formula**.
The density :math:`p(x)` tells us how likely a particular point :math:`x \in \mathbb{R}^N` is under the transformed distribution its input is a vector of shape :math:`(N,)` and its output is a non-negative scalar:

.. math::
    p(x) = p_z\!\left(f^{-1}(x)\right) \left| \det \mathcal{J}\!\left(f^{-1}\right) \right|

The term :math:`p_z(f^{-1}(x))` appears because we first map :math:`x` back to the latent space to get :math:`z = f^{-1}(x)`, then look up the density of that :math:`z` in the known distribution :math:`p_z` (e.g. a standard Gaussian).
Since :math:`f` is bijective, every :math:`x` maps to exactly one such :math:`z`, so the density is always well-defined.

The factor :math:`\left| \det \mathcal{J}(f^{-1}) \right|` is the **absolute value of the determinant of the Jacobian matrix** of :math:`f^{-1}`.
The Jacobian describes how a function changes the region around its input, in our case how :math:`f` changes the input space :math:`x`. 
In normalizing flows, we need this because transforming one distribution into another changes the shape of the space the probability lives in, and the Jacobian lets us account for that.
Also, since :math:`f` is invertible, the determinant of the Jacobian is guaranteed to be non-zero, and therefore it is always well-defined.
The **Jacobian matrix** :math:`\mathcal{J}(f^{-1}) \in \mathbb{R}^{N \times N}` contains all first-order partial derivatives of the mapping:

.. math::
    \mathcal{J}(f^{-1})_{ij} = \frac{\partial [f^{-1}]_i}{\partial x_j}

Its **determinant** is a single scalar that measures how the transformation locally scales volumes.
If :math:`|\det \mathcal{J}| > 1` the mapping stretches a region of space, spreading probability over a larger volume (decreasing density); if :math:`|\det \mathcal{J}| < 1` it compresses a region (increasing density).
Taking the absolute value ensures the density remains non-negative regardless of whether the transformation reverses orientation.

Composing Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^

A single bijective function may not be expressive enough to model complex distributions.
The power of normalizing flows comes from *composing* multiple simple bijective transformations into a chain, where each individual transformation is easy to invert and has a tractable Jacobian determinant, but the overall composition can represent highly flexible mappings:

.. math::
    z &\sim \mathcal{N}(\mu, \Sigma) \\
    x_1 &= f_1(z) \\
    x_2 &= f_2(x_1) \\
    &\;\vdots \\
    x_K &= f_K(x_{K-1})

In Physika, a single step of this chain could be written as:

.. code-block:: text

    # scale
    def f1(x: ℝ[N]): ℝ[N]:
        return x * 2.0

    # shift
    def f2(x: ℝ[N]): ℝ[N]:
        return x + 1.0

    # Sample z from N-dimensional Gaussian
    z : ℝ[N] ~ for i : ℕ(N) → ε : ℝ ~ Normal(μ, σ)

    # Compose transformations
    x1 : ℝ[N] = f1(z)
    x2 : ℝ[N] = f2(x1)

Applying the change-of-variables formula repeatedly, the density of the final output is:

.. math::
    p(x_K) = p_z(z) \prod_{i=1}^{K} \left| \det \mathcal{J}(f_i) \right|^{-1}

or equivalently, using the inverse form:

.. math::
    p(x_K) = p_z\!\left(f_1^{-1} \circ \cdots \circ f_K^{-1}(x_K)\right) \prod_{i=1}^{K} \left| \det \mathcal{J}\!\left(f_i^{-1}\right) \right|

Each Jacobian determinant in the product accounts for the local volume change introduced by the corresponding transformation.

Training via Log-Likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An **unbiased estimator** is a quantity whose expected value equals the true value it is estimating: if :math:`\hat{m}_\theta` is our estimate of the true model parameters :math:`m_\theta`, then :math:`\mathbb{E}[\hat{m}_\theta] = m_\theta`, where :math:`\mathbb{E}` denotes the expected value and :math:`\hat{\cdot}` (hat) marks an estimated quantity. **Maximum likelihood estimation (MLE)** finds the model parameters :math:`\theta` that maximize the probability the model assigns to the observed data.
By pushing the model to assign high probability to data it has seen, the learned parameters generalize to accurately capture the underlying distribution.
The MLE is an unbiased estimator of the true parameters as the dataset grows large.
For a dataset :math:`\{x_1, \ldots, x_M\}`, this amounts to maximizing: :math:`\prod_{j=1}^{M} p_\theta(x_j)`, where :math:`p_\theta(x_j)` is the probability the model with parameters :math:`\theta` assigns to the :math:`j`-th data point, and :math:`\prod` denotes the product over all :math:`M` samples.
Taking the logarithm turns the product into a sum, and negating it gives us a loss to minimize: 

.. math::
    \mathcal{L}(\theta) = -\sum_{j=1}^{M} \log p_\theta(x_j)

We train a normalizing flow by **maximum likelihood estimation**: we adjust the parameters of the transformations :math:`f_1, \ldots, f_K` so that the model assigns high probability to observed data.

The **log-likelihood** is the logarithm of the probability the model assigns to a data point :math:`x`.
Working with log-probabilities instead of raw probabilities is standard practice for two reasons: it converts products into sums (which are numerically stabler and cheaper to compute), and it avoids the underflow that occurs when multiplying many small probabilities together.

Applying the logarithm to the change-of-variables formula gives the loss (negative log-likelihood):

.. math::
    \mathcal{L} = -\log p_z\!\left[f_K^{-1}(\cdots(f_1^{-1}(x)))\right] - \sum_{i=1}^{K} \log \left| \det \mathcal{J}(f_i^{-1}) \right|

The first term, :math:`-\log p_z[\cdot]`, penalizes mappings that send data points to low-density regions of the base distribution.
The second term, :math:`-\sum_i \log |\det \mathcal{J}(f_i^{-1})|`, penalizes transformations that excessively compress volume (which would artificially inflate density).
We use the negative sign because we *minimize* a loss function during training, which is equivalent to *maximizing* the log-likelihood.

For deep-network-based flows, it is also important that the Jacobian determinant of each :math:`f_i` is efficient to compute ideally :math:`O(N)` rather than the :math:`O(N^3)` cost of a general determinant which motivates architectural choices such as coupling layers and autoregressive transforms.

In Physika, this entire pipeline is differentiable end-to-end.
Gradients flow through all transformations :math:`f_1, \ldots, f_K` and the log-density computation automatically, so ``grad()`` can backpropagate from the loss to every learnable parameter.
Sampling operations such as ``z: ℝ[N] ~ 𝒩(μ, Σ)`` are also differentiable.
Physika follows the `Stochastic Computation Graphs (SCG) framework <https://physika.readthedocs.io/en/latest/elf.html#id2>`__, using the reparameterization trick for continuous distributions (Normal, Uniform, Beta, Gamma) by default.
Normally, sampling is a non-differentiable operation because the random draw has no gradient with respect to the distribution's parameters.
The reparameterization trick sidesteps this by expressing the sample as a deterministic function of the parameters plus independent noise: :math:`z = \mu + \sigma \cdot \varepsilon` where :math:`\varepsilon \sim \mathcal{N}(0, I)` is fixed random noise.
Since :math:`z` is now a smooth function of :math:`\mu` and :math:`\sigma`, gradients flow through :math:`z` to these parameters as with any other computation, without special handling by the user.

A more detailed treatment of the change of variables formula and of flow-based models in general can be found in `[1] <https://deepgenerativemodels.github.io/notes/flow/>`__ and `[2] <https://lilianweng.github.io/posts/2018-10-13-flow-models/>`__.


Types of Normalizing Flows
---------------------------

There are various methods to implement normalizing flows, some more complex than others.
This is not an exhaustive list, but below are some popular methods.

1. Planar Flow
    The Planar Flow `[3] <https://arxiv.org/abs/1505.05770>`__ introduces the following invertible transformation

        .. math::
            x = f(z) = z + u\, h(w^\top z + b)

    The absolute value of the determinant of the Jacobian is given by

        .. math::
            \left|\det\left(\frac{\partial f(z)}{\partial z}\right)\right| = \left|1 + h'(w^\top z + b)\,u^\top w\right|

    The planar flow while simple, runs into the following issues:

        - The learned parameters :math:`u,w,b` and :math:`h`, need to be restricted to be invertible.
        - Computing :math:`f^{-1}(z)` could be difficult analytically.
          Because :math:`h` is typically a nonlinear activation (e.g. :math:`\tanh`), inverting :math:`f(z) = z + u\,h(w^\top z + b)` for :math:`z` requires solving an implicit nonlinear equation there is generally no closed-form expression, and iterative numerical methods (e.g. fixed-point iteration) must be used instead.

    The below two methods address this by ensuring that the forward and inverse is easy to compute.

2. NICE (Nonlinear Independent Components Estimation) model
    The NICE `[4] <https://arxiv.org/abs/1410.8516>`__ coupling layer partitions :math:`z` into 2 disjoint subsets :math:`z_1, z_2`.
    :math:`m` denotes a neural network.

    - Forward Mapping (:math:`x \to z`):
        .. math::
            x_1 &= z_1 \\
            x_2 &= z_2 + m(z_1)

    - Inverse Mapping (:math:`z \to x`):
        .. math::
            z_1 &= x_1 \\
            z_2 &= x_2 - m(x_1)

    The Jacobian of the forward mapping is lower triangular, which means its determinant is simply the product of its diagonal entries.
    To see this, consider :math:`z = (a, b, c, d) \in \mathbb{R}^4` split into a pass-through half :math:`(a, b)` and a shifted half :math:`(c, d)`.
    The forward mapping gives:

    .. math::
        x_a &= a, \quad x_b = b \qquad &\text{(copied unchanged)} \\
        x_c &= c + m_1, \quad x_d = d + m_2 \qquad &\text{(shifted by the network)}

    where :math:`m(a, b) = (m_1,\, m_2)^\top` is a neural network outputting one shift per component of the shifted half.

    The Jacobian is (rows = outputs, columns = inputs):

    .. math::
        \mathcal{J} =
        \begin{array}{c|cccc}
          & \partial a & \partial b & \partial c & \partial d \\ \hline
        x_a & 1 & 0 & 0 & 0 \\
        x_b & 0 & 1 & 0 & 0 \\
        x_c & \frac{\partial m_1}{\partial a} & \frac{\partial m_1}{\partial b} & 1 & 0 \\
        x_d & \frac{\partial m_2}{\partial a} & \frac{\partial m_2}{\partial b} & 0 & 1
        \end{array}

    The first two rows are identity because :math:`x_a, x_b` depend only on :math:`a, b` directly.
    The bottom-left entries (:math:`\partial m / \partial a`, :math:`\partial m / \partial b`) are nonzero, the network depends on the pass-through inputs but :math:`c` and :math:`d` each appear only in their own output row with coefficient 1 (since the coupling just *adds* :math:`m` to them), giving the 1s on the diagonal and 0s to their right.
    The diagonal is all 1s, so :math:`|\det \mathcal{J}| = 1`.
    The property :math:`|\det \mathcal{J}| = 1` makes the coupling layer **volume preserving**, it reshapes the density without stretching or compressing any region of space.

    The inverse is equally straightforward: since :math:`x_1 = z_1` is already known, we can evaluate :math:`m(x_1)` and recover :math:`z_2 = x_2 - m(x_1)` in a single forward pass of the network, making it easy to compute unlike the planar flow above.

    **Diagonal scaling layer:**
    Since a stack of additive coupling layers alone is always volume preserving, the full NICE model adds one final diagonal scaling layer
    after all the coupling layers, so the model can rescale the overall distribution to match real data.

    In the NICE paper `[4] <https://arxiv.org/abs/1410.8516>`__, the scaling layer applies a diagonal matrix :math:`S \in \mathbb{R}^{N \times N}`,
    multiplying each dimension :math:`i` of the intermediate vector :math:`h \in \mathbb{R}^N` by the corresponding diagonal entry :math:`S_{ii}`:

    .. math::
        x_i = S_{ii} \, h_i \qquad \text{for } i = 1, \ldots, N

    Written as a matrix operation, this is simply :math:`x = S\,h`, where :math:`S = \text{diag}(S_{11}, \ldots, S_{NN})`.

    In Physika, the matrix and the operation can be expressed as:

    .. code-block:: text

        S : ℝ[N, N] = ...          # diagonal scaling matrix
        h : ℝ[N]    = ...          # output of final coupling layer
        x : ℝ[N]    = S @ h        # scaled output: x_i = S_ii * h_i

    **Jacobian of the scaling layer:**
    Because :math:`S` is diagonal, the Jacobian :math:`\partial x / \partial h` is itself the diagonal matrix :math:`S`.
    The determinant of a diagonal matrix is the product of its diagonal entries:

    .. math::
        \det \frac{\partial x}{\partial h} = \det S = \prod_{i=1}^{N} S_{ii}

    Taking the log of the absolute value (as required by the change-of-variables formula):

    .. math::
        \log \left| \det \frac{\partial x}{\partial h} \right| = \sum_{i=1}^{N} \log |S_{ii}|

    This decomposition from a product into a sum is the key computational advantage: it reduces an :math:`O(N^3)` determinant to an :math:`O(N)` sum.

    **Parameterization:**
    In practice, :math:`S_{ii}` is parameterized as :math:`S_{ii} = e^{a_i}`, where :math:`a_i`
    (rather than :math:`S_{ii}` directly) is the learned parameter, as is done in RealNVP `[5] <https://arxiv.org/abs/1605.08803>`__ and the normflows library `[6] <https://github.com/VincentStimper/normalizing-flows>`__.
    The exponential guarantees :math:`S_{ii} > 0` for all values of :math:`a_i` (since :math:`e^{a_i} > 0\;\forall\, a_i \in \mathbb{R}`), avoiding sign flips during optimization.
    Substituting into the log-determinant:

    .. math::
        \log \left| \det \frac{\partial x}{\partial h} \right| = \sum_{i=1}^{N} \log\, e^{a_i} = \sum_{i=1}^{N} a_i

    The absolute value is no longer needed because :math:`e^{a_i}` is always positive, and the log-exp pair cancels to give a simple sum of the learned parameters.

    **Loss:**
    The model is trained by maximum likelihood. Applying the change-of-variables formula and taking the logarithm, the log-density
    of a data point :math:`x` under the full model is:

    .. math::
        \log p_X(x) = \log p_Z(z) + \log |\det \mathcal{J}_{\text{total}}|

    The total Jacobian determinant decomposes across layers. Each additive coupling layer contributes :math:`\log|\det \mathcal{J}| = 0`
    (since its determinant is 1), and the scaling layer contributes :math:`\sum_i a_i`. Therefore:

    .. math::
        \log p_X(x) = \log p_Z(z) + \sum_{i=1}^{N} a_i

    where :math:`\log p_Z(z)` is the log-density of the base distribution (e.g. a standard Gaussian)
    evaluated at :math:`z = f^{-1}(x)` (the result of passing :math:`x` backward through all layers).
    The training objective is the negative log-likelihood, averaged over the dataset:

    .. math::
        \mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log p_X(x)\right]
        = -\mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log p_Z(f^{-1}(x)) + \sum_{i=1}^{N} a_i\right]

    Minimizing :math:`\mathcal{L}` pushes the model to (a) map data points to high-density regions of the base distribution (via the :math:`\log p_Z` term) and (b) learn appropriate per-dimension scaling (via the :math:`\sum_i a_i` term).

3. RealNVP (Real Non-Volume Preserving) model
    RealNVP `[5] <https://arxiv.org/abs/1605.08803>`__ adds scaling factors to the transformation, which makes it non volume preserving.
    :math:`s,m` are both neural networks that have been conditioned on :math:`x_1`, acting as scale and shift factors respectively.

    .. math::
        x_2 = \exp(s(z_1)) \odot z_2 + m(z_1)


.. note::

    In Physika, sampling operations in flows are differentiable via the `SCG framework <https://physika.readthedocs.io/en/latest/elf.html#id2>`__: continuous distributions use the reparameterization trick, discrete ones use score function estimators.
    Gradients propagate through the full chain of transformations automatically.

Implementing the NICE Normalizing Flow in Physika
------------------------------------------------------

The code block below contains the core components of the NICE model implemented in Physika.
The coupling layer is implemented as a simple feedforward neural network with one hidden layer and ReLU activation function.
The rescale layer is implemented as a diagonal scaling layer, where the scaling factors are learned parameters.
In the next section, we will show how to train the NICE model on a simple image classification task.


Note: the code below is for pedagogical purposes.
Please refer to the next section for the complete standalone implementation for image classification.

.. code-block:: text

    class NICE:
        W1a: ℝ[h, n]
        b1a: ℝ[h]
        W2a: ℝ[n, h]
        b2a: ℝ[n]
        W1b: ℝ[h, n]
        b1b: ℝ[h]
        W2b: ℝ[n, h]
        b2b: ℝ[n]
        a: ℝ[d]
        n: ℕ
        d: ℕ
        # couple(x) = (x₁, x₂ + m(x₁))
        # m(x₁) = W₂ ReLU(W₁ x₁ + b₁) + b₂
        def coupling(x: ℝ[d], W1: ℝ[h, n], b1: ℝ[h], W2: ℝ[n, h], b2: ℝ[n]): ℝ[d]:
            x1: ℝ[n] = x[:this.n]
            x2: ℝ[n] = x[this.n:]
            m: ℝ[n] = linear(relu(linear(x1, W1, b1)), W2, b2)
            return concat(x1, x2 + m)
        # couple⁻¹(y) = (y₁, y₂ − m(y₁))
        def coupling_inv(y: ℝ[d], W1: ℝ[h, n], b1: ℝ[h], W2: ℝ[n, h], b2: ℝ[n]): ℝ[d]:
            y1: ℝ[n] = y[:this.n]
            y2: ℝ[n] = y[this.n:]
            m: ℝ[n] = linear(relu(linear(y1, W1, b1)), W2, b2)
            return concat(y1, y2 - m)
        # swap(x₁, x₂) = (x₂, x₁)
        def swap(x: ℝ[d]): ℝ[d]:
            return concat(x[this.n:], x[:this.n])
        # zᵢ = eᵃⁱ hᵢ
        def rescale(h: ℝ[d]): ℝ[d]:
            return h * exp(this.a)
        # hᵢ = e⁻ᵃⁱ zᵢ
        def rescale_inv(z: ℝ[d]): ℝ[d]:
            return z * exp(-this.a)
        # log|det ∂z/∂h| = Σᵢ aᵢ
        def log_det(): ℝ:
            return sum(this.a)
        # z = rescale ∘ swap ∘ coupleB ∘ swap ∘ coupleA(x)
        def forward_z(x: ℝ[d]): ℝ[d]:
            h: ℝ[d] = this.coupling(x, this.W1a, this.b1a, this.W2a, this.b2a)
            h = this.swap(h)
            h = this.coupling(h, this.W1b, this.b1b, this.W2b, this.b2b)
            h = this.swap(h)
            return this.rescale(h)
        # log p(x) = log pZ(z) + Σᵢ aᵢ
        def λ(x: ℝ[d]) -> ℝ:
            z: ℝ[d] = this.forward_z(x)
            return log_pz(z, this.d) + this.log_det()
        # f⁻¹ = coupleA⁻¹ ∘ swap ∘ coupleB⁻¹ ∘ swap ∘ rescale⁻¹
        def inverse(z: ℝ[d]): ℝ[d]:
            h: ℝ[d] = this.rescale_inv(z)
            h = this.swap(h)
            h = this.coupling_inv(h, this.W1b, this.b1b, this.W2b, this.b2b)
            h = this.swap(h)
            return this.coupling_inv(h, this.W1a, this.b1a, this.W2a, this.b2a)
        # z ~ 𝒩(0, I),  x = f⁻¹(z)
        def sample(): ℝ[d]:
            z: ℝ[d] ~ Normal(0.0, 1.0, this.d)
            return this.inverse(z)

Putting it all together: Training a model with Normalizing Flow - Full code
---------------------------------------------------------------------------------


This is the complete code for training a model on the MNIST dataset using the NICE Normalizing Flow.


.. note::
   ``create_dataset`` is not a built-in Physika function. To use it,
   add the following helper to ``physika/runtime.py``:

    .. code-block:: python

        def create_dataset(train_test_split = 80, total_dataset_size = 40):
            import torch
            from torchvision import datasets, transforms

            transform = transforms.ToTensor()

            mnist = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transform
            )

            X = []
            y = []

            # take first total_dataset_size samples
            for i in range(total_dataset_size):
                image, label = mnist[i]

                # [1,28,28] -> [28,28]
                image = image.squeeze(0)
                X.append(image)
                y.append(label)
            X = torch.stack(X)
            y = torch.tensor(y)

            # split index
            split_index = int(
                (train_test_split / 100.0)
                *
                total_dataset_size
            )

            # train split
            X_train = X[:split_index]
            y_train = y[:split_index]

            # test split
            X_test = X[split_index:]
            y_test = y[split_index:]
            train_data = [X_train, y_train]
            test_data = [X_test, y_test]
            return [train_data, test_data]

Code in the Physika (.phyk) file

.. code-block:: python

    # NICE on MNIST
    physika.seed(0)


    # Helpers

    def len1d(x: ℝ[n]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def relu(x: ℝ[n]): ℝ[n]:
        return (x + abs(x)) * 0.5

    # y = Wx + b
    def linear(x: ℝ[li], W: ℝ[lo, li], b: ℝ[lo]): ℝ[lo]:
        in_dim: ℝ = len1d(x)
        col: ℝ[li, 1] = for i:ℕ(in_dim) -> for j:ℕ(1) -> j*0
        col[:, 0] = x
        res: ℝ[lo, 1] = W @ col
        return res[:, 0] + b

    def flatten(img: ℝ[H, W], rows: ℝ, cols: ℝ): ℝ[d]:
        d: ℝ = rows * cols
        out: ℝ[d] = for i:ℕ(d) -> i*0
        for i:ℕ(rows):
            out[i * cols : (i + 1) * cols] = img[i, :]
        return out

    def unflatten(x: ℝ[d], rows: ℝ, cols: ℝ): ℝ[rows, cols]:
        img: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        for i:ℕ(rows):
            img[i, :] = x[i * cols : (i + 1) * cols]
        return img

    def dequantize(x: ℝ[d], d: ℕ): ℝ[d]:
        u: ℝ[d] ~ 𝒰(0.0, 1.0, d)
        return (x + u) / 256.0

    def log_pz(x: ℝ[d], d: ℕ): ℝ:
        return sum(-0.5 * x * x) - d * 0.5 * log(2.0 * 3.14159265)


    # NICE model

    class NICE:
        W1a: ℝ[h, n]
        b1a: ℝ[h]
        W2a: ℝ[n, h]
        b2a: ℝ[n]
        W1b: ℝ[h, n]
        b1b: ℝ[h]
        W2b: ℝ[n, h]
        b2b: ℝ[n]
        a: ℝ[d]
        n: ℕ
        d: ℕ
        def coupling(x: ℝ[d], W1: ℝ[h, n], b1: ℝ[h], W2: ℝ[n, h], b2: ℝ[n]): ℝ[d]:
            x1: ℝ[n] = x[:this.n]
            x2: ℝ[n] = x[this.n:]
            m: ℝ[n] = linear(relu(linear(x1, W1, b1)), W2, b2)
            return concat(x1, x2 + m)
        def coupling_inv(y: ℝ[d], W1: ℝ[h, n], b1: ℝ[h], W2: ℝ[n, h], b2: ℝ[n]): ℝ[d]:
            y1: ℝ[n] = y[:this.n]
            y2: ℝ[n] = y[this.n:]
            m: ℝ[n] = linear(relu(linear(y1, W1, b1)), W2, b2)
            return concat(y1, y2 - m)
        def swap(x: ℝ[d]): ℝ[d]:
            return concat(x[this.n:], x[:this.n])
        def rescale(h: ℝ[d]): ℝ[d]:
            return h * exp(this.a)
        def rescale_inv(z: ℝ[d]): ℝ[d]:
            return z * exp(-this.a)
        def log_det(): ℝ:
            return sum(this.a)
        def forward_z(x: ℝ[d]): ℝ[d]:
            h: ℝ[d] = this.coupling(x, this.W1a, this.b1a, this.W2a, this.b2a)
            h = this.swap(h)
            h = this.coupling(h, this.W1b, this.b1b, this.W2b, this.b2b)
            h = this.swap(h)
            return this.rescale(h)
        def λ(x: ℝ[d]) -> ℝ:
            z: ℝ[d] = this.forward_z(x)
            return log_pz(z, this.d) + this.log_det()
        def inverse(z: ℝ[d]): ℝ[d]:
            h: ℝ[d] = this.rescale_inv(z)
            h = this.swap(h)
            h = this.coupling_inv(h, this.W1b, this.b1b, this.W2b, this.b2b)
            h = this.swap(h)
            return this.coupling_inv(h, this.W1a, this.b1a, this.W2a, this.b2a)
        def sample(): ℝ[d]:
            z: ℝ[d] ~ Normal(0.0, 1.0, this.d)
            return this.inverse(z)


    def neg_loglik(log_px: ℝ): ℝ:
        return -log_px

    def nll(model: NICE, X: ℝ[k, d], count: ℝ): ℝ:
        total: ℝ = 0
        for i:ℕ(count):
            total += neg_loglik(model(X[i]))
        return total / count


    # Dataset

    dataset = create_dataset(80, 100)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    train_X = train_dataset[0]
    test_X = test_dataset[0]

    len_train: ℝ, len_test: ℝ = len1d(train_X), len1d(test_X)

    # Dimensions

    d: ℕ = 784
    h: ℕ = 128
    n: ℕ = 392

    # He init, near-zero output so coupling starts near identity
    s1: ℝ, s2: ℝ = sqrt(2.0 / n), sqrt(2.0 / h) * 0.01

    W1a: ℝ[h, n] = for i:ℕ(h) -> ε: ℝ[n] ~ Normal(0.0, s1, n)
    b1a: ℝ[h] = for i:ℕ(h) -> i*0
    W2a: ℝ[n, h] = for i:ℕ(n) -> ε: ℝ[h] ~ Normal(0.0, s2, h)
    b2a: ℝ[n] = for i:ℕ(n) -> i*0

    W1b: ℝ[h, n] = for i:ℕ(h) -> ε: ℝ[n] ~ Normal(0.0, s1, n)
    b1b: ℝ[h] = for i:ℕ(h) -> i*0
    W2b: ℝ[n, h] = for i:ℕ(n) -> ε: ℝ[h] ~ Normal(0.0, s2, h)
    b2b: ℝ[n] = for i:ℕ(n) -> i*0

    a0: ℝ[d] = for i:ℕ(d) -> i*0

    model: NICE = NICE(W1a, b1a, W2a, b2a, W1b, b1b, W2b, b2b, a0, n, d)

    # Sanity check

    x0: ℝ[d] = dequantize(flatten(train_X[0], 28, 28), d)
    print(model(x0))
    print(DEVICE)

    # Preprocess

    train_flat: ℝ[len_train, d] = for i:ℕ(len_train) -> for j:ℕ(d) -> j*0
    for i:ℕ(len_train):
        train_flat[i, :] = dequantize(flatten(train_X[i], 28, 28), d)

    test_flat: ℝ[len_test, d] = for i:ℕ(len_test) -> for j:ℕ(d) -> j*0
    for i:ℕ(len_test):
        test_flat[i, :] = dequantize(flatten(test_X[i], 28, 28), d)

    # Training SGD

    epochs: ℕ = 20
    lr: ℝ = 0.001

    losses: ℝ[epochs] = for i:ℕ(epochs) -> i*0

    for i:ℕ(epochs):
        for j:ℕ(len_train):
            L = neg_loglik(model(train_flat[j]))
            nW1a = model.W1a - lr * grad(L, model.W1a)
            nb1a = model.b1a - lr * grad(L, model.b1a)
            nW2a = model.W2a - lr * grad(L, model.W2a)
            nb2a = model.b2a - lr * grad(L, model.b2a)
            nW1b = model.W1b - lr * grad(L, model.W1b)
            nb1b = model.b1b - lr * grad(L, model.b1b)
            nW2b = model.W2b - lr * grad(L, model.W2b)
            nb2b = model.b2b - lr * grad(L, model.b2b)
            na = model.a - lr * grad(L, model.a)
            model = NICE(nW1a, nb1a, nW2a, nb2a, nW1b, nb1b, nW2b, nb2b, na, model.n, model.d)
        epoch_loss = nll(model, train_flat, len_train)
        losses[i] = epoch_loss
        print(epoch_loss)
        bits = epoch_loss / (d * log(2.0)) + 8.0
        print(bits)

    # Test

    test_loss: ℝ = nll(model, test_flat, len_test)
    print(test_loss)
    bits_per_dim: ℝ = test_loss / (d * log(2.0)) + 8.0
    print(bits_per_dim)

    # Generate sample

    gen_flat: ℝ[d] = model.sample()
    gen_img: ℝ[28, 28] = unflatten(gen_flat, 28, 28)
    print(gen_img)

Training plots
---------------

After running the code below, you should see the training loss decrease over epochs, indicating that the model is learning to better fit the data distribution.
Note that these plots are very linear due to the small scale of the data used (80 vs 60000) and the small number of epochs (20 vs 1500), in the original NICE paper.

.. figure:: /_static/tutorial_files/norm_flow/nice_train_curve.png
   :alt:
   :align: center
   :width: 750px



References
----------------

1. `Deep Generative Models: Normalizing Flow Models <https://deepgenerativemodels.github.io/notes/flow/>`_
2. `Flow-based Deep Generative Models <https://lilianweng.github.io/posts/2018-10-13-flow-models/>`_
3. `Variational Inference with Normalizing Flows <https://arxiv.org/abs/1505.05770>`_
4. `NICE: Non-linear Independent Components Estimation <https://arxiv.org/abs/1410.8516>`_
5. `Density estimation using Real NVP <https://arxiv.org/abs/1605.08803>`_
6. `normflows: A PyTorch Package for Normalizing Flows <https://github.com/VincentStimper/normalizing-flows>`_
