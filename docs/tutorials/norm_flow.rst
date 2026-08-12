Normalizing Flows
=======================

This tutorial is to introduce the concept of Normalizing Flows and how to implement them in Physika. 
Normalizing Flows are a powerful class of generative models that allow for flexible density estimation and sampling by transforming a simple base distribution into a more complex target distribution.
By the end of this tutorial you will learn how to use the NICE Normalizing Flow to train a simple image classifier.


What are Normalizing Flows?
----------------

Normalizing Flows are a class of generative models that transform a simple base distribution (e.g., Gaussian) into a more complex target distribution which can model a real world data distribution through a series of invertible transformations. 
The key idea is to model the probability density function of the target distribution by applying a sequence of bijective mappings to the base distribution. 

Let's consider a probability distribution :math:`\mathcal{P}` over :math:`\mathbb{R}^N`. 
For now, we can represent this probability distribution by a probability density function :math:`p: \mathbb{R}^n \to \mathbb{R}`.
and consider the transformation function to be :math:`g`. Suppose that

.. figure:: /_static/tutorial_files/norm_flow/norm_flow_basic.png
   :alt: 
   :align: center
   :width: 500px

.. math::
    z &\sim \mathcal{N}(\mu, \Sigma) \\
    x &= f(z)

Then the probability density of :math:`x` is given by (also known as the change in variables formula)

.. math::
    P(x) = P_z(f(x))\left | \det \mathcal{J}(f) \right |

We constrain the function :math:`g` to be one-to-one and onto, which makes :math:`g` bijective and invertible. 
A core idea of using a deep network to train a normalizing flow is that we can consider a chain of such transformations 

.. math::
    z &\sim \mathcal{N}(\mu, \Sigma) \\
    x_1 &= f_1(z) \\
    x_2 &= f_2(x_1) \\

Then the final distribution is given by

.. math::
    P(x_2) = P_z(f(x))\left | \det \mathcal{J}(f_2) \right | \left | \det \mathcal{J}(f_1) \right |

The loss for the normalizing flow is simply given by the log-likelihood

.. math::
    \mathcal{L} = -\log P_z[f_n(\dotsc (f_1(x))] - \sum_i \log \left | \det \mathcal{J}(f_i) \right |

For deep networks, it is also important that the Jacobian determinant, :math:`\mathcal{J}(f)` is easy to compute.


Types of Normalizing Flows
----------------

There are various methods to implement normalizing flows, some more complex than others.
This is not an exhaustive list, but below are some popular methods.

1. Planar Flow
    The Planar Flow [] introduces the following invertible transformation

        .. math::
            x = f(z) = z + u\, h(w^\top z + b)

    The absolute value of the determinant of the Jacobian is given by

        .. math::
            \left|\det\left(\frac{\partial f(z)}{\partial z}\right)\right| = \left|1 + h'(w^\top z + b)\,u^\top w\right|

    The planar flow while simple, runs into the following issues:

        - The learned parameters :math:`u,w,b` and :math:`h`, need to be restricted to be invertible.
        - Computing :math:`f^{-1}(z)` could be difficult analytically.

    The below to methods address this by ensuring that the forward and inverse is easy to compute

2. NICE (Nonlinear Independent Components Estimation) model
    The NICE coupling layer partitions :math:`z` into 2 disjoint subsets :math:`z_1, z_2`.
    :math:`m` denotes a neural network.

    - Forward Mapping (:math:`x \to z`):
        .. math::
            x_1 &= z_1 \\ 
            x_2 &= z_2 + m(z_1) 
    
    - Inverse Mapping (:math:`z \to x`):
        .. math::
            z_1 &= x_1 \\
            z_2 &= x_2 - m(x_1)

    Therefore, the Jacobian of the forward mapping is lower triangular, whose determinant is simply the product of the elements on the diagonal, which is 1. 
    Therefore, this defines a volume preserving transformation.

    Since a stack of additive coupling layers alone is always volume preserving, the full NICE model adds one final diagonal scaling layer
    after all the coupling layers, so the model can rescale the overall distribution to match real data
    In the NICE paper, the scaling layer applies a diagonal matrix :math:`S`,
    multiplying each dimension by :math:`S_{ii}`:

    .. math::
        x_i = S_{ii} \, h_i

    The Jacobian of this layer is diagonal, so its determinant is :math:`\prod_i S_{ii}`, giving:

    .. math::
        \log \left| \det \frac{\partial x}{\partial h} \right| = \sum_i \log |S_{ii}|

    In practice, :math:`S_{ii}` is parameterized as :math:`S_{ii} = e^{a_i}`, where :math:`a_i`
    (rather than :math:`S_{ii}` directly) is the learned parameter [normflow cite] [realnvp cite]. This guarantees
    :math:`S_{ii} > 0` during optimization and simplifies the log-determinant to a plain sum:

    .. math::
        \log \left| \det \frac{\partial x}{\partial h} \right| = \sum_i a_i

    - Loss:
        The model is trained by maximum likelihood. Using the change of variables formula, the log-density
        of a data point :math:`x` under the model is:

        .. math::
            \log p_X(x) = \log p_Z(z) + \sum_i a_i

        where :math:`\log p_Z(z)` is the log-density of the base distribution (e.g. a standard Gaussian)
        evaluated at :math:`z = f(x)`, and :math:`\sum_i a_i` is the log-determinant contributed by the
        final scaling layer (all coupling layers contribute :math:`0`, as shown above). The training
        objective is the negative log-likelihood, averaged over the dataset:

        .. math::
            \mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}\left[\log p_X(x)\right]
        
.. 3. RealNVP (Real Non-Volume Preserving) model
..     RealNVP adds scaling factors to the transformation, which makes it non volume preserving.
..     :math:`s,t` are both neural networks that have been conditioned on :math:`x_1`, acting as scale and shift factors respectively.

..     .. math::
..         x_2 = \exp(s(z_1)) \odot z_2 + m(z_1)



Implementing the NICE Normalizing Flow in Physika
----------------
The code block below contain the core components of the NICE model implemented in Physika. 
The coupling layer is implemented as a simple feedforward neural network with one hidden layer and ReLU activation function. 
The rescale layer is implemented as a diagonal scaling layer, where the scaling factors are learned parameters.
In the next section, we will show how to train the NICE model on a simple image classification task.

Note: the code below is for pedagogical purposes. Please refer to the next section for the complete standalone implementation for image classification. 

.. code-block:: python

    class CouplingLayer:
        w1: ℝ[H, n]
        b1: ℝ[H]
        w2: ℝ[n, H]
        b2: ℝ[n]

        # forward: x1 = z1 ; x2 = z2 + m(z1)
        def forward(z1: ℝ[n], z2: ℝ[n]): ℝ[n], ℝ[n]:
            m: ℝ[n] = linear(relu1d(linear(z1, this.w1, this.b1)), this.w2, this.b2)
            return z1, z2 + m

        # inverse: z1 = x1 ; z2 = x2 - m(x1)
        def inverse(x1: ℝ[n], x2: ℝ[n]): ℝ[n], ℝ[n]:
            m: ℝ[n] = linear(relu1d(linear(x1, this.w1, this.b1)), this.w2, this.b2)
            return x1, x2 - m

    class RescaleLayer:
        a: ℝ[m] # log scale

        # y_i = s_i * x_i ,  s_i = exp(a_i)
        def forward(x: ℝ[m]): ℝ[m]:
            s = exp(this.a)
            y = x * s
            return y

        # inverse: x_i = y_i / s_i
        def inverse(y: ℝ[m]): ℝ[m]:
            s = exp(-this.a)
            x = y * s
            return x

        # sum_i log|s_i| = sum_i a_i
        def log_det(): ℝ:
            return sum(this.a)

    class NICE:
        layers: [CouplingLayer]
        rescale: RescaleLayer

        # x -> z, log p_X(x)
        def λ(x: ℝ[m]): ℝ:
            z1, z2 = split(x)
            for layer in this.layers:
                z1, z2 = layer.forward(z1, z2)
                z1, z2 = z2, z1              # alternate which half is transformed

            z: ℝ[m] = this.rescale.forward(concat(z1, z2))

            # log p_X(x) = log p_Z(z) + sum_i a_i     [coupling log-dets are 0]
            log_pz: ℝ = gaussian_log_prob(z)
            log_px: ℝ = log_pz + this.rescale.log_det()
            return log_px

    # L(θ) = - E_x[ log p_X(x) ]
    def neg_loglik(log_px: ℝ): ℝ:
        return -log_px

Putting it all together: Training a model with Normalizing Flow
----------------


References
----------------

- https://lilianweng.github.io/posts/2018-10-13-flow-models/
- https://arxiv.org/abs/1410.8516
- https://arxiv.org/abs/1505.05770
- https://arxiv.org/abs/1605.08803
