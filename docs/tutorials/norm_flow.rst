Normalizing Flows
=======================

This tutorial is to introduce Normalizing Flow models and how to implement them in Physika.
Suppose we are given samples from an unknown data distribution (e.g. images of handwritten digits) and we want to both estimate how likely any given point is under that distribution and generate new samples that resemble the data. This is the problem of generative modeling with exact density estimation, and it is the problem a normalizing flow like RealNVP is designed to solve.
Normalizing Flows are a class of generative models that allow for density estimation and sampling by transforming a simple base distribution into a more complex target distribution.
RealNVP (Real-valued Non-Volume Preserving transformations) is a Normalizing Flow model.
It works by splitting the input into two halves: one half stays fixed, while the other is scaled and shifted using neural networks conditioned on the fixed half.
Stacking several of these layers produces a transformation that is easy to invert and whose likelihood is easy to compute,
the two properties that make Normalizing Flows useful for both generating data and estimating density.

By the end of this tutorial, you will learn how to train a RealNVP Normalizing Flow model for density estimation and image generation.
This tutorial is based on the Deep Generative Models (CS236) course notes [DeepGenModels]_.


What are Normalizing Flows?
---------------------------

Normalizing Flows are a class of generative models that transform a simple base distribution (e.g., Gaussian) into a more complex target distribution which can model a real world data distribution through a series of invertible transformations [DeepGenModels]_.
The key idea is to model the probability density function of the target distribution by applying a sequence of **bijective mappings** to the base distribution.

A bijective mapping is a function :math:`f: \mathbb{R}^N \to \mathbb{R}^N` that is both *injective* (distinct inputs always produce distinct outputs, i.e. :math:`f(a) = f(b) \implies a = b`) and *surjective* (every point in the output space is the image of some input) [Wikipedia_Bijection]_.
Equivalently, a bijective function establishes a one-to-one correspondence between input and output spaces, so it is guaranteed to have a well-defined inverse :math:`f^{-1}` satisfying :math:`f^{-1}(f(z)) = z`.
This invertibility is essential for normalizing flows: it lets us map freely between the simple base distribution and the complex target, and crucially it allows us to compute exact probability densities via the change-of-variables formula [DeepGenModels]_.

.. figure:: /_static/tutorial_files/norm_flow/norm_flow_basic.png
   :alt: Illustration of a normalizing flow transforming a simple Gaussian distribution into a complex multi-modal distribution through a sequence of invertible mappings.
   :align: center
   :width: 500px

   **Figure 1.** A normalizing flow transforms a simple base density :math:`p_z(z)` (left) into a complex target density :math:`p(x)` (right) through a chain of invertible transformations :math:`f_1, f_2, \ldots, f_K`. Figure from [Weng2018]_.

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

General Derivation of the Loss in a Normalizing Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Maximum likelihood estimation (MLE)** [Wikipedia_MLE]_ finds the model parameters :math:`\theta` that make the observed data as likely as possible under the model.
For a dataset :math:`\{x_1, \ldots, x_M\}`, this means choosing :math:`\theta` to maximize the **likelihood** :math:`\prod_{j=1}^{M} p_\theta(x_j)`, where :math:`p_\theta(x_j)` is the probability the model assigns to the :math:`j`-th data point and :math:`\prod` denotes the product over all :math:`M` samples.
Multiplying many probabilities together is numerically awkward, so we take the logarithm, which turns the product into a sum. Maximizing this **log-likelihood** is the same as minimizing its negative, which gives us a loss to minimize: 

.. math::
    \mathcal{L}(\theta) = -\sum_{j=1}^{M} \log p_\theta(x_j)

We train a normalizing flow by **maximum likelihood estimation**: we adjust the parameters of the transformations :math:`f_1, \ldots, f_K` so that the model assigns high probability to observed data.

The **log-likelihood** is the logarithm of the probability the model assigns to a data point :math:`x`.
Working with log-probabilities instead of raw probabilities is standard practice for two reasons: it converts products into sums (which are numerically stabler and cheaper to compute), and it avoids the underflow that occurs when multiplying many small probabilities together.

Applying the logarithm to the change-of-variables formula gives the loss (negative log-likelihood):

.. math::
    \mathcal{L} = -\log p_z\!\left[f_K^{-1}(\cdots(f_1^{-1}(x)))\right] - \sum_{i=1}^{K} \log \left| \det \mathcal{J}(f_i^{-1}) \right|

In Physika, the loss mirrors the two terms of the equation directly: it maps :math:`x` back to :math:`z` by undoing the transformations in reverse order, then adds the base log-density :math:`\log p_z(z)` and each layer's log-determinant of the Jacobian. 
The ``loss`` method below is derived from the chain of transformations ``f1``, ``f2`` which we applied on our sample ``z`` above.

.. code-block:: text

    # log-density of z under a standard Gaussian N(0, I)
    def log_pz(z: ℝ[N]): ℝ:
        return sum(-0.5 * z * z) - N * 0.5 * log(2.0 * 3.14159265)

    # Inverse transformations (x to z)
    def f2_inv(x: ℝ[N]): ℝ[N]:
        return x - 1.0

    def f1_inv(x: ℝ[N]): ℝ[N]:
        return x / 2.0

    # Loss: map x back to z, then add log p_z(z) and each layer's log|det J|
    def loss(x: ℝ[N]): ℝ:
        z1: ℝ[N] = f2_inv(x)
        J2: ℝ[N, N] = grad(z1, x)
        z: ℝ[N] = f1_inv(z1)
        J1: ℝ[N, N] = grad(z, z1)
        return -log_pz(z) - log(abs(det(J2))) - log(abs(det(J1)))

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

A more detailed treatment of the change of variables formula and of flow-based models in general can be found in [DeepGenModels]_ and [Weng2018]_.
The above derivation is a general formulation of the loss in a normalizing flow, and it applies to any choice of invertible transformations :math:`f_1, \ldots, f_K`.
This crucial to understand before proceeding to the next section, as we derive the complete loss function for RealNVP in more detail, where we substitute the series of transformations into this general expression.


Types of Normalizing Flows
---------------------------

There are various methods to implement normalizing flows, some more complex than others.
This is not an exhaustive list, but below are some popular methods.

1. Planar Flow
    The Planar Flow [RezendeMohamed2015]_ introduces the following invertible transformation

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
    The NICE [DinhNICE2014]_ coupling layer partitions :math:`z` into 2 disjoint subsets :math:`z_1, z_2`.
    :math:`m` denotes a neural network.

    - Forward Mapping (:math:`x \to z`):
        .. math::
            x_1 &= z_1 \\
            x_2 &= z_2 + m(z_1)

    - Inverse Mapping (:math:`z \to x`):
        .. math::
            z_1 &= x_1 \\
            z_2 &= x_2 - m(x_1)

    The Jacobian of the forward mapping is lower triangular with all 1s on the diagonal, so :math:`|\det \mathcal{J}| = 1`.
    This makes the NICE coupling layer **volume preserving**: it reshapes the density without stretching or compressing any region of space.
    Because a stack of additive coupling layers alone is always volume preserving, the full NICE model adds a final diagonal scaling layer
    after all the coupling layers so the model can rescale the overall distribution to match real data.

3. RealNVP (Real Non-Volume Preserving) model
    RealNVP [DinhRealNVP2016]_ extends the NICE coupling layer by adding learned scaling factors to the transformation.
    The coupling layer partitions :math:`z` into 2 disjoint subsets :math:`z_1, z_2`.
    :math:`s,m` are both neural networks that have been conditioned on :math:`z_1`, acting as scale and shift factors respectively.

    - Forward Mapping (:math:`x \to z`):
        .. math::
            x_1 &= z_1 \\
            x_2 &= \exp(s(z_1)) \odot z_2 + m(z_1)

        This is implemented as the ``coupling`` method in Physika shown below: the first half ``x1`` passes through unchanged, and the second half ``x2`` is scaled by ``exp(s)`` and shifted by ``m``. Here ``s`` and ``m`` are two one-hidden-layer networks conditioned on ``x1``, built from the helpers ``linear`` (which computes ``x @ W + b``) and the activation function ``relu`` which computes ``(x + abs(x)) * 0.5``.

        .. code-block:: text

            def coupling(x: ℝ[d]): ℝ[d]:
                x1: ℝ[n] = x[:n]
                x2: ℝ[n] = x[n:]
                s: ℝ[n] = linear(relu(linear(x1, W1_s, b1_s)), W2_s, b2_s)
                m: ℝ[n] = linear(relu(linear(x1, W1_m, b1_m)), W2_m, b2_m)
                return concat(x1, exp(s) * x2 + m)

    - Inverse Mapping (:math:`z \to x`):
        .. math::
            z_1 &= x_1 \\
            z_2 &= (x_2 - m(x_1)) \odot \exp(-s(x_1))

        The ``coupling_inv`` method is a mirror image of ``coupling``. Since ``y1`` is passed through unchanged, we recompute the exact same ``s`` and ``m`` from it and undo the affine map on ``y2`` by subtracting ``m``, then multiply by ``exp(-s)``:

        .. code-block:: text

            def coupling_inv(y: ℝ[d]): ℝ[d]:
                y1: ℝ[n] = y[:n]
                y2: ℝ[n] = y[n:]
                s: ℝ[n] = linear(relu(linear(y1, W1_s, b1_s)), W2_s, b2_s)
                m: ℝ[n] = linear(relu(linear(y1, W1_m, b1_m)), W2_m, b2_m)
                return concat(y1, (y2 - m) * exp(-s))

    The inverse is straightforward: since :math:`x_1 = z_1` is already known, we can evaluate :math:`s(x_1)` and :math:`m(x_1)` and recover :math:`z_2 = (x_2 - m(x_1)) \odot \exp(-s(x_1))` in a single forward pass of the networks, making it easy to compute unlike the planar flow above.

    The Jacobian of the forward mapping is lower triangular, which means its determinant is simply the product of its diagonal entries.
    To see this, consider :math:`z = (a, b, c, d) \in \mathbb{R}^4` split into a pass-through half :math:`(a, b)` and an affine-transformed half :math:`(c, d)`.
    The forward mapping gives:

    .. math::
        x_a &= a, \quad x_b = b \qquad &\text{(copied unchanged)} \\
        x_c &= e^{s_1}\,c + m_1, \quad x_d = e^{s_2}\,d + m_2 \qquad &\text{(scaled and shifted by the networks)}

    where :math:`s(a, b) = (s_1,\, s_2)^\top` and :math:`m(a, b) = (m_1,\, m_2)^\top` are neural networks, each outputting one value per component of the transformed half.

    The Jacobian is (rows = outputs, columns = inputs):

    .. math::
        \mathcal{J} =
        \begin{array}{c|cccc}
          & \partial a & \partial b & \partial c & \partial d \\ \hline
        x_a & 1 & 0 & 0 & 0 \\
        x_b & 0 & 1 & 0 & 0 \\
        x_c & \frac{\partial (e^{s_1}c + m_1)}{\partial a} & \frac{\partial (e^{s_1}c + m_1)}{\partial b} & e^{s_1} & 0 \\
        x_d & \frac{\partial (e^{s_2}d + m_2)}{\partial a} & \frac{\partial (e^{s_2}d + m_2)}{\partial b} & 0 & e^{s_2}
        \end{array}

    The first two rows are identity because :math:`x_a, x_b` depend only on :math:`a, b` directly.
    The bottom-left entries are nonzero because both :math:`s` and :math:`m` depend on the pass-through inputs, but :math:`c` and :math:`d` each appear only in their own output row with coefficient :math:`e^{s_1}` and :math:`e^{s_2}` respectively (since the coupling *scales* them by :math:`e^{s_i}` and then *adds* :math:`m_i`), giving the exponentials on the diagonal and 0s to their right.
    The diagonal is :math:`(1, 1, e^{s_1}, e^{s_2})`, so:

    .. math::
        |\det \mathcal{J}| = 1 \cdot 1 \cdot e^{s_1} \cdot e^{s_2} = \exp(s_1 + s_2)

    **Non-volume preserving property:**
    Unlike NICE, where the diagonal is all 1s and :math:`|\det \mathcal{J}| = 1`, the RealNVP coupling layer has :math:`|\det \mathcal{J}| = \exp(\sum_i s_i)` which is in general not equal to 1.
    When :math:`s_i > 0` the transformation stretches along dimension :math:`i`; when :math:`s_i < 0` it compresses.
    This makes the coupling layer **non-volume preserving**, it can both reshape *and* rescale the density, which is where the model's name comes from.
    Because scaling is built into the coupling layers themselves, RealNVP does not need a separate diagonal scaling layer (unlike NICE).

    **Log-determinant:**
    Taking the log of the absolute value (as required by the change-of-variables formula):

    .. math::
        \log \left| \det \mathcal{J} \right| = \log \exp\!\left(\sum_{i=1}^{n} s_i\right) = \sum_{i=1}^{n} s_i

    where the sum runs over the :math:`n` components of the transformed half.
    The log-exp pair cancels, reducing the computation to a simple sum of the scale network outputs.
    This is :math:`O(n)` rather than the :math:`O(N^3)` cost of a general determinant, the same computational advantage enjoyed by NICE.

    In Physika, this is implemented as the ``log_det`` method: it runs just the scale network on ``x1`` and returns ``sum(s)``, reading the log-determinant straight off the scale outputs instead of forming the full Jacobian:

    .. code-block:: text

        def log_det(x: ℝ[d]): ℝ:
            x1: ℝ[n] = x[:n]
            s: ℝ[n] = linear(relu(linear(x1, W1_s, b1_s)), W2_s, b2_s)
            return sum(s)

    Note that since :math:`s_i` can be any real number (positive or negative), the absolute value is accounted for by the exponential: :math:`e^{s_i} > 0\;\forall\, s_i \in \mathbb{R}`, so the determinant is always positive and the absolute value is no longer needed.

    **Loss:**
    The model is trained by maximum likelihood. Applying the change-of-variables formula and taking the logarithm, the log-density
    of a data point :math:`x` under the full model is:

    .. math::
        \log p_X(x) = \log p_Z(z) + \log |\det \mathcal{J}_{\text{total}}|

    The total Jacobian determinant decomposes across layers. Each affine coupling layer :math:`k` contributes :math:`\log|\det \mathcal{J}_k| = \sum_i s_i^{(k)}`,
    where :math:`s_i^{(k)}` is the :math:`i`-th output of the scale network in the :math:`k`-th coupling layer. Therefore:

    .. math::
        \log p_X(x) = \log p_Z(z) + \sum_{k=1}^{K} \sum_{i=1}^{n} s_i^{(k)}

    where :math:`\log p_Z(z)` is the log-density of the base distribution (e.g. a standard Gaussian)
    evaluated at :math:`z = f^{-1}(x)` (the result of passing :math:`x` backward through all layers).
    The training objective is the negative log-likelihood, averaged over the dataset:

    .. math::
        \mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log p_X(x)\right]
        = -\mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log p_Z(f^{-1}(x)) + \sum_{k=1}^{K} \sum_{i=1}^{n} s_i^{(k)}\right]

    In Physika these two terms are the ``λ`` method , it maps ``x`` to ``z`` with ``forward_z`` and adds the base log-density to the log-determinant  while ``loss`` just negates it for a single sample:

    .. code-block:: text

        def forward_z(x: ℝ[d]): ℝ[d]:
            return coupling(x)
        def λ(x: ℝ[d]) -> ℝ:
            z: ℝ[d] = forward_z(x)
            return log_pz(z, d) + log_det(x)
        def loss(x: ℝ[784]): ℝ:
            return -λ(x)
        
    Minimizing :math:`\mathcal{L}` pushes the model to (a) map data points to high-density regions of the base distribution (via the :math:`\log p_Z` term) and (b) learn appropriate per-dimension scaling and shifting (via the :math:`\sum_k \sum_i s_i^{(k)}` term).


.. note::

    In Physika, sampling operations in flows are differentiable via the `SCG framework <https://physika.readthedocs.io/en/latest/elf.html#id2>`__: continuous distributions use the reparameterization trick, discrete ones use score function estimators.
    Gradients propagate through the full chain of transformations automatically.

Training a RealNVP Normalizing Flow in Physika
----------------------------------------------

To train the model we need a RealNVP class (shown as a snippet below), which we use to instantiate a RealNVP model allowing us to refer to its methods (eg: ``this.loss``) and parameters (eg: ``this.W1_s``) with ``this.`` representing the instance of the class. 
The coupling layer is implemented using two simple feedforward neural networks with one hidden layer and ReLU activation function: one for the scale :math:`s` and one for the shift :math:`m`.
Unlike NICE, no separate rescale layer is needed because scaling is built into the affine coupling layers.
When calling the class (``this(x)``), runs the ``λ`` method, so ``this(x)`` is equivalent to ``λ(x)``. The above sections are implemented as individual methods, to illustrate the derivations in a simple manner.
The Full Code section contains the complete Physika code, which can be run as-is.

.. code-block:: text

    class RealNVP(W1_s: ℝ[h, n], b1_s: ℝ[h], W2_s: ℝ[n, h], b2_s: ℝ[n], W1_m: ℝ[h, n], b1_m: ℝ[h], W2_m: ℝ[n, h], b2_m: ℝ[n], n: ℕ, d: ℕ):
        def coupling(x: ℝ[d]): ℝ[d]:
            x1: ℝ[n] = x[:this.n]
            x2: ℝ[n] = x[this.n:]
            s: ℝ[n] = linear(relu(linear(x1, this.W1_s, this.b1_s)), this.W2_s, this.b2_s)
            m: ℝ[n] = linear(relu(linear(x1, this.W1_m, this.b1_m)), this.W2_m, this.b2_m)
            return concat(x1, exp(s) * x2 + m)
        def coupling_inv(y: ℝ[d]): ℝ[d]:
            y1: ℝ[n] = y[:this.n]
            y2: ℝ[n] = y[this.n:]
            s: ℝ[n] = linear(relu(linear(y1, this.W1_s, this.b1_s)), this.W2_s, this.b2_s)
            m: ℝ[n] = linear(relu(linear(y1, this.W1_m, this.b1_m)), this.W2_m, this.b2_m)
            return concat(y1, (y2 - m) * exp(-s))
        def log_det(x: ℝ[d]): ℝ:
            x1: ℝ[n] = x[:this.n]
            s: ℝ[n] = linear(relu(linear(x1, this.W1_s, this.b1_s)), this.W2_s, this.b2_s)
            return sum(s)
        def forward_z(x: ℝ[d]): ℝ[d]:
            return this.coupling(x)
        def λ(x: ℝ[d]) -> ℝ:
            z: ℝ[d] = this.forward_z(x)
            return log_pz(z, this.d) + this.log_det(x)
        def inverse(z: ℝ[d]): ℝ[d]:
            return this.coupling_inv(z)
        def sample(): ℝ[d]:
            z: ℝ[d] ~ Normal(0.0, 1.0, this.d)
            return this.inverse(z)
        def loss(x: ℝ[784]): ℝ:
            return -this(x)
        def train(X: ℝ[160, 784], epochs: ℕ, lr: ℝ, len_train: ℝ): ℝ[epochs]:
            losses: ℝ[epochs] = for i:ℕ(epochs) -> i*0
            for epoch: ℕ(epochs):
                for i: ℕ(len_train):
                    L = this.loss(X[i])
                    learnable_grads = grad(L, this.learnable_params)
                    this.update_params(lr, learnable_grads)
                total = 0
                for i:ℕ(len_train):
                    total += this.loss(X[i])
                epoch_loss = total/len_train
                losses[epoch] = epoch_loss
                print(epoch_loss)
                bits = this.evaluate(epoch_loss, len_train)
                print(bits)
            return losses
        def test(Y: ℝ[40, 784], len_test: ℝ): ℝ:
            total: ℝ = 0
            for i:ℕ(len_test):
                total += this.loss(Y[i])
            return total / len_test
        def evaluate(num: ℝ, len: ℝ): ℝ:
            return num / (this.d * log(2.0)) + 8.0
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W1_s = this.W1_s - lr * learnable_grads[0]
            this.b1_s = this.b1_s - lr * learnable_grads[1]
            this.W2_s = this.W2_s - lr * learnable_grads[2]
            this.b2_s = this.b2_s - lr * learnable_grads[3]
            this.W1_m = this.W1_m - lr * learnable_grads[4]
            this.b1_m = this.b1_m - lr * learnable_grads[5]
            this.W2_m = this.W2_m - lr * learnable_grads[6]
            this.b2_m = this.b2_m - lr * learnable_grads[7]

In the ``train`` method, the loss is computed for each image, the gradient of the loss with respect to the learnable parameters is computed with ``grad``, and the parameters are updated with a simple gradient descent step.
For convenient gradient computation ``this.learnable_params`` is a built-in that collects all the parameters of the class that are differentiable, so ``grad`` returns a list of gradients in the same order as the parameters.
The ``update_params`` method uses simple gradient descent, more sophisticated optimizers can be used as well.

So when training the model the entire training looks like the snippet below, shown is a simplified example on dummy data:

.. code-block:: text

    # dummy data all zeros
    train_X: ℝ[160, 28, 28] = for i:ℕ(160) -> for j:ℕ(28) -> for k:ℕ(28) -> k*0
    test_X: ℝ[40, 28, 28] = for i:ℕ(40) -> for j:ℕ(28) -> for k:ℕ(28) -> k*0
    len_train: ℝ, len_test: ℝ = 160, 40

    # Dimensions
    d: ℝ, h: ℝ, n: ℝ = 784, 128, 392
    # He init, near-zero output so coupling starts near identity
    s1: ℝ, s2: ℝ = sqrt(2.0 / n), sqrt(2.0 / h) * 0.01
    W1_s, W1_m = for i:ℕ(h) -> ε: ℝ[n] ~ Normal(0.0, s1, n), for i:ℕ(h) -> ε: ℝ[n] ~ Normal(0.0, s1, n)
    b1_s, b1_m = for i:ℕ(h) -> i*0, for i:ℕ(h) -> i*0
    W2_s, W2_m = for i:ℕ(n) -> ε: ℝ[h] ~ Normal(0.0, s2, h), for i:ℕ(n) -> ε: ℝ[h] ~ Normal(0.0, s2, h)
    b2_s, b2_m = for i:ℕ(n) -> i*0, for i:ℕ(n) -> i*0

    realnvp: RealNVP = RealNVP(W1_s, b1_s, W2_s, b2_s, W1_m, b1_m, W2_m, b2_m, n, d)

    train_flat: ℝ[len_train, d] = for i:ℕ(len_train) -> for j:ℕ(d) -> j*0
    X: ℝ[160,784] = train_flat
    Y: ℝ[40,784] = test_flat

    epochs: ℕ = 20
    lr: ℝ = 0.00015
    losses: ℝ[epochs] = realnvp.train(X, epochs, lr, len_train)
    test_loss: ℝ = realnvp.test(Y, len_test)
    bits: ℝ = realnvp.evaluate(test_loss, len_test)
    gen_flat: ℝ[d] = realnvp.sample()


We also compute bits-per-dimension through the ``evaluate`` method, to measure how well the model can represent some given data. 
If this goes below 8, the model has learnt a representation of the train data which is better than an image created from pure random noise.
The ``test`` method lets us compute the loss on the test set, which is useful to check if the model is able to generalize to unseen data.
The ``sample`` method lets us generate new samples from the model. In this image is a flat vector of size 784, which can be reshaped to a square image.

Dataset
-------

MNIST is a dataset of handwritten digits, each a :math:`28 \times 28` grayscale image, when these images are flattened or expressed as a 1D vector we get a vector of size :math:`784`.
``create_dataset`` returns the first ``total_dataset_size`` MNIST digits as a list containing a train an test split, where ``train_test_split`` is the percentage of images to keep in the train split. 
In our case ``train_test_split = 80, total_dataset_size = 200`` so 160 images in the train split, and 40 in the test split.

.. note::
   ``create_dataset`` is not a built-in Physika function. To use it, add the following helper to ``physika/runtime.py``:

    .. code-block:: python

        def create_dataset(train_test_split = 80, total_dataset_size = 200):
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

Plotting Graphs
---------------
Once training is done it is useful to visualize loss curves, as it tells us if the model is training properly. 
The note below contains the helper to add in ``physika/runtime.py``.

.. note::
   To plot the training curve and the bits per dimension, add the ``plot_losses_bits_per_dim`` helper to ``physika/runtime.py``:

    .. code-block:: python

        def plot_losses_bits_per_dim(losses):
            import matplotlib.pyplot as plt
            import numpy as np
        
            bits_per_dim = losses / (784 * np.log(2)) + 8
            epochs = range(1, len(losses) + 1)
        
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
            ax1.plot(epochs, losses.cpu().detach().numpy())
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("RealNVP Normalizing Flow on 160 Samples of MNIST — Training Curve\nLoss by epoch")
            ax1.set_xticks(epochs)
        
            ax2.plot(epochs, bits_per_dim.cpu().detach().numpy())
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Bits per dim")
            ax2.set_title("Bits per dim by epoch (lower is better)")
            ax2.set_xticks(epochs)
        
            plt.tight_layout()
            plt.savefig("realnvp_train_curve.png", dpi=300, bbox_inches="tight")
            plt.show()

Full Code
---------------------------------------------------------------------------------

Here we load 160 train images of MNIST, 40 images for test, train the RealNVP model for 20 epochs, and plot the bits per dimension and train loss.
We keep the learning rate as ``0.00015``, and hidden size as ``128`` as the hyperparameters. 
The first line ``physika.seed(0)`` ensures that training runs can be reproduced exactly, without variations across runs. 
More details on this can be found in the `Sampling documentation <https://physika.readthedocs.io/en/latest/elf.html#random-sampling>`__.
Additionally, since Normalizing flows only work with continuous distributions, we dequantize the MNIST images to make them continuous by adding some noise sampled from uniform distribution.

.. code-block:: text

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
        a: ℝ[1] = [0.0]
        b: ℝ[1] = [1.0]
        u: ℝ[d] ~ 𝒰(a[0], b[0], d)
        return (x + u) / 256.0

    def log_pz(x: ℝ[d], d: ℕ): ℝ:
        return sum(-0.5 * x * x) - d * 0.5 * log(2.0 * 3.14159265)

    def concat(a: ℝ[na], b: ℝ[nb]): ℝ[d]:
        la: ℝ = len1d(a)
        lb: ℝ = len1d(b)
        d: ℝ = la + lb
        out: ℝ[d] = for i:ℕ(d) -> i*0
        out[:la] = a
        out[la:] = b
        return out

    # RealNVP, two networks scale (s) and shift (m)
    class RealNVP(W1_s: ℝ[h, n], b1_s: ℝ[h], W2_s: ℝ[n, h], b2_s: ℝ[n], W1_m: ℝ[h, n], b1_m: ℝ[h], W2_m: ℝ[n, h], b2_m: ℝ[n], n: ℕ, d: ℕ):
        def coupling(x: ℝ[d]): ℝ[d]:
            x1: ℝ[n] = x[:this.n]
            x2: ℝ[n] = x[this.n:]
            s: ℝ[n] = linear(relu(linear(x1, this.W1_s, this.b1_s)), this.W2_s, this.b2_s)
            m: ℝ[n] = linear(relu(linear(x1, this.W1_m, this.b1_m)), this.W2_m, this.b2_m)
            return concat(x1, exp(s) * x2 + m)
        def coupling_inv(y: ℝ[d]): ℝ[d]:
            y1: ℝ[n] = y[:this.n]
            y2: ℝ[n] = y[this.n:]
            s: ℝ[n] = linear(relu(linear(y1, this.W1_s, this.b1_s)), this.W2_s, this.b2_s)
            m: ℝ[n] = linear(relu(linear(y1, this.W1_m, this.b1_m)), this.W2_m, this.b2_m)
            return concat(y1, (y2 - m) * exp(-s))
        def log_det(x: ℝ[d]): ℝ:
            x1: ℝ[n] = x[:this.n]
            s: ℝ[n] = linear(relu(linear(x1, this.W1_s, this.b1_s)), this.W2_s, this.b2_s)
            return sum(s)
        def forward_z(x: ℝ[d]): ℝ[d]:
            return this.coupling(x)
        def λ(x: ℝ[d]) -> ℝ:
            z: ℝ[d] = this.forward_z(x)
            return log_pz(z, this.d) + this.log_det(x)
        def inverse(z: ℝ[d]): ℝ[d]:
            return this.coupling_inv(z)
        def sample(): ℝ[d]:
            z: ℝ[d] ~ Normal(0.0, 1.0, this.d)
            return this.inverse(z)
        def loss(x: ℝ[784]): ℝ:
            return -this(x)
        def train(X: ℝ[160, 784], epochs: ℕ, lr: ℝ, len_train: ℝ): ℝ[epochs]:
            losses: ℝ[epochs] = for i:ℕ(epochs) -> i*0
            for epoch: ℕ(epochs):
                for i: ℕ(len_train):
                    L = this.loss(X[i])
                    learnable_grads = grad(L, this.learnable_params)
                    this.update_params(lr, learnable_grads)
                total = 0
                for i:ℕ(len_train):
                    total += this.loss(X[i])
                epoch_loss = total/len_train
                losses[epoch] = epoch_loss
                print(epoch_loss)
                bits = this.evaluate(epoch_loss, len_train)
                print(bits)
            return losses
        def test(Y: ℝ[40, 784], len_test: ℝ): ℝ:
            total: ℝ = 0
            for i:ℕ(len_test):
                total += this.loss(Y[i])
            return total / len_test
        def evaluate(num: ℝ, len: ℝ): ℝ:
            return num / (this.d * log(2.0)) + 8.0
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W1_s = this.W1_s - lr * learnable_grads[0]
            this.b1_s = this.b1_s - lr * learnable_grads[1]
            this.W2_s = this.W2_s - lr * learnable_grads[2]
            this.b2_s = this.b2_s - lr * learnable_grads[3]
            this.W1_m = this.W1_m - lr * learnable_grads[4]
            this.b1_m = this.b1_m - lr * learnable_grads[5]
            this.W2_m = this.W2_m - lr * learnable_grads[6]
            this.b2_m = this.b2_m - lr * learnable_grads[7]


    # Dataset -- add in runtime.py to run with real data
    dataset = create_dataset(80, 200)
    train_dataset = dataset[0]
    test_dataset = dataset[1]
    train_X: ℝ[160, 28, 28] = train_dataset[0]
    test_X: ℝ[40, 28, 28] = test_dataset[0]
    len_train: ℝ, len_test: ℝ = 160, 40

    # Dimensions
    d: ℝ, h: ℝ, n: ℝ = 784, 128, 392
    # He init, near-zero output so coupling starts near identity
    s1: ℝ, s2: ℝ = sqrt(2.0 / n), sqrt(2.0 / h) * 0.01
    W1_s, W1_m = for i:ℕ(h) -> ε: ℝ[n] ~ Normal(0.0, s1, n), for i:ℕ(h) -> ε: ℝ[n] ~ Normal(0.0, s1, n)
    b1_s, b1_m = for i:ℕ(h) -> i*0, for i:ℕ(h) -> i*0
    W2_s, W2_m = for i:ℕ(n) -> ε: ℝ[h] ~ Normal(0.0, s2, h), for i:ℕ(n) -> ε: ℝ[h] ~ Normal(0.0, s2, h)
    b2_s, b2_m = for i:ℕ(n) -> i*0, for i:ℕ(n) -> i*0

    realnvp: RealNVP = RealNVP(W1_s, b1_s, W2_s, b2_s, W1_m, b1_m, W2_m, b2_m, n, d)
    print(DEVICE)

    # Preprocess
    train_flat: ℝ[len_train, d] = for i:ℕ(len_train) -> for j:ℕ(d) -> j*0
    for i:ℕ(len_train):
        train_flat[i, :] = dequantize(flatten(train_X[i], 28, 28), d)

    test_flat: ℝ[len_test, d] = for i:ℕ(len_test) -> for j:ℕ(d) -> j*0
    for i:ℕ(len_test):
        test_flat[i, :] = dequantize(flatten(test_X[i], 28, 28), d)

    # Training
    epochs: ℕ = 20
    lr: ℝ = 0.00015
    X: ℝ[160,784] = train_flat
    Y: ℝ[40,784] = test_flat
    losses: ℝ[epochs] = realnvp.train(X, epochs, lr, len_train)
    plot_losses_bits_per_dim(losses)

    # Test
    test_loss: ℝ = realnvp.test(Y, len_test)
    print(test_loss)
    bits: ℝ = realnvp.evaluate(test_loss, len_test)
    print(bits)

    # Generate sample
    gen_flat: ℝ[d] = realnvp.sample()
    gen_img: ℝ[28, 28] = unflatten(gen_flat, 28, 28)
    print(gen_img)


Training plots
---------------

After running the code below (~30 mins), you should see the training loss and bits-per-dimension decrease over epochs, indicating that the model is learning to better fit the data distribution.

.. figure:: /_static/tutorial_files/norm_flow/realnvp_train_curve.png
   :alt:
   :align: center
   :width: 750px


References
-----------------

.. [DeepGenModels] A. Grover and S. Ermon,
    *Deep Generative Models: Normalizing Flow Models*.
    https://deepgenerativemodels.github.io/notes/flow/

.. [Weng2018] L. Weng,
    *Flow-based Deep Generative Models*.
    https://lilianweng.github.io/posts/2018-10-13-flow-models/

.. [RezendeMohamed2015] D. Rezende and S. Mohamed,
    *Variational Inference with Normalizing Flows*.
    https://arxiv.org/abs/1505.05770

.. [DinhNICE2014] L. Dinh, D. Krueger, and Y. Bengio,
    *NICE: Non-linear Independent Components Estimation*.
    https://arxiv.org/abs/1410.8516

.. [DinhRealNVP2016] L. Dinh, J. Sohl-Dickstein, and S. Bengio,
    *Density Estimation using Real NVP*.
    https://arxiv.org/abs/1605.08803

.. [Wikipedia_Bijection] Wikipedia,
    *Bijection*.
    https://en.wikipedia.org/wiki/Bijection

.. [Wikipedia_MLE] Wikipedia,
    *Maximum Likelihood Estimation*.
    https://en.wikipedia.org/wiki/Maximum_likelihood_estimation