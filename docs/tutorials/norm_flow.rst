Normalizing Flows
=======================

This tutorial is to introduce the concept of Normalizing Flows and how to implement them in Physika.
Normalizing Flows are a powerful class of generative models that allow for flexible density estimation and sampling by transforming a simple base distribution into a more complex target distribution.
By the end of this tutorial you will learn how to use the NICE Normalizing Flow to train a simple image classifier.
This tutorial is based on the Deep Generative Models (CS236) course notes `[1] <https://deepgenerativemodels.github.io/notes/flow/>`__.


What are Normalizing Flows?
---------------------------

Normalizing Flows are a class of generative models that transform a simple base distribution (e.g., Gaussian) into a more complex target distribution which can model a real world data distribution through a series of invertible transformations.
The key idea is to model the probability density function of the target distribution by applying a sequence of bijective mappings to the base distribution. Figure below (taken from `[2] <https://lilianweng.github.io/posts/2018-10-13-flow-models/>`__) depicts a normalizing flow.

.. figure:: /_static/tutorial_files/norm_flow/norm_flow_basic.png
   :alt:
   :align: center
   :width: 500px

Let's consider a probability distribution :math:`\mathcal{P}` over :math:`\mathbb{R}^N`.
For now, we can represent this probability distribution by a probability density function :math:`p: \mathbb{R}^n \to \mathbb{R}`.
and consider the transformation function to be :math:`g`. Suppose that

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

    Therefore, the Jacobian of the forward mapping is lower triangular, whose determinant is simply the product of the elements on the diagonal, which is 1.
    Therefore, this defines a volume preserving transformation.

    Since a stack of additive coupling layers alone is always volume preserving, the full NICE model adds one final diagonal scaling layer
    after all the coupling layers, so the model can rescale the overall distribution to match real data
    In the NICE paper `[4] <https://arxiv.org/abs/1410.8516>`__, the scaling layer applies a diagonal matrix :math:`S`,
    multiplying each dimension by :math:`S_{ii}`:

    .. math::
        x_i = S_{ii} \, h_i

    The Jacobian of this layer is diagonal, so its determinant is :math:`\prod_i S_{ii}`, giving:

    .. math::
        \log \left| \det \frac{\partial x}{\partial h} \right| = \sum_i \log |S_{ii}|

    In practice, :math:`S_{ii}` is parameterized as :math:`S_{ii} = e^{a_i}`, where :math:`a_i`
    (rather than :math:`S_{ii}` directly) is the learned parameter as is done in RealNVP `[5] <https://arxiv.org/abs/1605.08803>`__, and the normflows `[6] <https://github.com/VincentStimper/normalizing-flows>`__; a popular library for normalizing flows implemented in PyTorch.
    This guarantees :math:`S_{ii} > 0` during optimization and simplifies the log-determinant to a plain sum:

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

3. RealNVP (Real Non-Volume Preserving) model
    RealNVP `[5] <https://arxiv.org/abs/1605.08803>`__ adds scaling factors to the transformation, which makes it non volume preserving.
    :math:`s,m` are both neural networks that have been conditioned on :math:`x_1`, acting as scale and shift factors respectively.

    .. math::
        x_2 = \exp(s(z_1)) \odot z_2 + m(z_1)



Implementing the NICE Normalizing Flow in Physika
------------------------------------------------------

The code block below contains the core components of the NICE model implemented in Physika.
The coupling layer is implemented as a simple feedforward neural network with one hidden layer and ReLU activation function.
The rescale layer is implemented as a diagonal scaling layer, where the scaling factors are learned parameters.
In the next section, we will show how to train the NICE model on a simple image classification task.


Note: the code below is for pedagogical purposes.
Please refer to the next section for the complete standalone implementation for image classification.

.. code-block:: python

    class NICE:
        w1a: ℝ[H, n]
        b1a: ℝ[H]
        w2a: ℝ[n, H]
        b2a: ℝ[n]
        w1b: ℝ[H, n]
        b1b: ℝ[H]
        w2b: ℝ[n, H]
        b2b: ℝ[n]
        a: ℝ[m]
        n_half: ℕ
        ndim: ℕ
        def coupling(x: ℝ[m], w1: ℝ[H, n], b1: ℝ[H], w2: ℝ[n, H], b2: ℝ[n], half: ℕ): ℝ[m]:
            # Forward additive coupling:
            #   y_1 = x_1
            #   y_2 = x_2 + m(x_1)
            x1: ℝ[n] = x[:half]
            x2: ℝ[n] = x[half:]
            hidden_pre: ℝ[n] = linear(x1, w1, b1)
            shift: ℝ[n] = linear(relu1d(hidden_pre), w2, b2)
            y2: ℝ[n] = x2 + shift
            return concat(x1, y2)
        def coupling_inv(y: ℝ[m], w1: ℝ[H, n], b1: ℝ[H], w2: ℝ[n, H], b2: ℝ[n], half: ℕ): ℝ[m]:
            # Inverse additive coupling (same network m, subtract instead of add):
            #   x_1 = y_1
            #   x_2 = y_2 - m(y_1)
            y1: ℝ[n] = y[:half]
            y2: ℝ[n] = y[half:]
            hidden_pre: ℝ[n] = linear(y1, w1, b1)
            shift: ℝ[n] = linear(relu1d(hidden_pre), w2, b2)
            x2: ℝ[n] = y2 - shift
            return concat(y1, x2)
        def swap(x: ℝ[m], half: ℕ): ℝ[m]:
            return concat(x[half:], x[:half])
        def rescale(x: ℝ[m]): ℝ[m]:
            # z_i = S_ii * h_i,  S_ii = exp(a_i)
            s: ℝ[m] = exp(this.a)
            return x * s
        def rescale_inv(z: ℝ[m]): ℝ[m]:
            # h_i = z_i / S_ii = z_i * exp(-a_i)
            s: ℝ[m] = exp(-this.a)
            return z * s
        def rescale_log_det(): ℝ:
            # log |det dz/dh| = sum_i a_i
            return sum(this.a)
        def forward_z(x: ℝ[m]): ℝ[m]:
            h: ℝ[m] = this.coupling(x, this.w1a, this.b1a, this.w2a, this.b2a, this.n_half)
            h = this.swap(h, this.n_half)
            h = this.coupling(h, this.w1b, this.b1b, this.w2b, this.b2b, this.n_half)
            h = this.swap(h, this.n_half)
            return this.rescale(h)
        def λ(x: ℝ[m]) -> ℝ:
            z: ℝ[m] = this.forward_z(x)
            log_pz: ℝ = gaussian_log_prob(z, this.ndim)
            log_px: ℝ = log_pz + this.rescale_log_det()
            return log_px
        def inverse(z: ℝ[m]): ℝ[m]:
            # f = rescale . swap . coupling_b . swap . coupling_a, so
            # f^{-1} = coupling_a^{-1} . swap . coupling_b^{-1} . swap . rescale^{-1}
            # (swap is its own inverse)
            h: ℝ[m] = this.rescale_inv(z)
            h = this.swap(h, this.n_half)
            h = this.coupling_inv(h, this.w1b, this.b1b, this.w2b, this.b2b, this.n_half)
            h = this.swap(h, this.n_half)
            h = this.coupling_inv(h, this.w1a, this.b1a, this.w2a, this.b2a, this.n_half)
            return h
        def sample(): ℝ[m]:
            # draw z from the base distribution and push it through the
            # inverse flow to get a generated (flattened, dequantized-scale) image
            z: ℝ[m] ~ Normal(0.0, 1.0, this.ndim)
            return this.inverse(z)
        def loss(pred: ℝ, target: ℝ): ℝ:
            return -pred


    def gaussian_log_prob(x: ℝ[m], n: ℕ): ℝ:
        return sum(-0.5 * x * x) - n * 0.5 * log(2.0 * 3.14159265)

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

    # NICE (Nonlinear Independent Components Estimation) on MNIST - vectorized
    physika.seed(0)


    # Helper functions
    def get_1d_array_length(x: ℝ[gm]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    # relu(x) == (x + |x|) / 2 exactly, elementwise, no loop
    def relu1d(x: ℝ[m]): ℝ[m]:
        return (x + abs(x)) * 0.5

    # matmul rejects mixed rank, so lift x to a column with a slice assign
    def linear(x: ℝ[li], weight: ℝ[lo, li], bias: ℝ[lo]): ℝ[lo]:
        in_dim: ℝ = get_1d_array_length(x)
        col: ℝ[li, 1] = for i:ℕ(in_dim) -> for j:ℕ(1) -> j*0
        col[:, 0] = x
        res: ℝ[lo, 1] = weight @ col
        return res[:, 0] + bias

    # no reshape builtin, so one loop over rows instead of rows*cols
    def flatten(img: ℝ[H, W], rows: ℝ, cols: ℝ): ℝ[n]:
        n: ℝ = rows * cols
        results: ℝ[n] = for i:ℕ(n) -> i*0
        for i:ℕ(rows):
            results[i * cols : (i + 1) * cols] = img[i, :]
        return results

    # inverse of flatten: undo the row-major packing so a generated 1-D
    # sample can be viewed as a 28x28 image
    def unflatten(x: ℝ[n], rows: ℝ, cols: ℝ): ℝ[rows, cols]:
        img: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        for i:ℕ(rows):
            img[i, :] = x[i * cols : (i + 1) * cols]
        return img

    # dequantize: x̃ = (x + u) / 256, u ~ U(0,1)
    def dequantize(x: ℝ[n], n: ℕ): ℝ[n]:
        ε: ℝ[n] ~ 𝒰(0.0, 1.0, n)
        return (x + ε) / 256.0

    # sum(-0.5 x^2) - n*0.5*log(2pi), one reduction
    def gaussian_log_prob(x: ℝ[m], n: ℕ): ℝ:
        return sum(-0.5 * x * x) - n * 0.5 * log(2.0 * 3.14159265)

    def neg_loglik(log_px: ℝ): ℝ:
        return -log_px


    # NICE model: 2 coupling layers

    class NICE:
        w1a: ℝ[H, n]
        b1a: ℝ[H]
        w2a: ℝ[n, H]
        b2a: ℝ[n]
        w1b: ℝ[H, n]
        b1b: ℝ[H]
        w2b: ℝ[n, H]
        b2b: ℝ[n]
        a: ℝ[m]
        n_half: ℕ
        ndim: ℕ
        def coupling(x: ℝ[m], w1: ℝ[H, n], b1: ℝ[H], w2: ℝ[n, H], b2: ℝ[n], half: ℕ): ℝ[m]:
            # Forward additive coupling:
            #   y_1 = x_1
            #   y_2 = x_2 + m(x_1)
            x1: ℝ[n] = x[:half]
            x2: ℝ[n] = x[half:]
            hidden_pre: ℝ[n] = linear(x1, w1, b1)
            shift: ℝ[n] = linear(relu1d(hidden_pre), w2, b2)
            y2: ℝ[n] = x2 + shift
            return concat(x1, y2)
        def coupling_inv(y: ℝ[m], w1: ℝ[H, n], b1: ℝ[H], w2: ℝ[n, H], b2: ℝ[n], half: ℕ): ℝ[m]:
            # Inverse additive coupling (same network m, subtract instead of add):
            #   x_1 = y_1
            #   x_2 = y_2 - m(y_1)
            y1: ℝ[n] = y[:half]
            y2: ℝ[n] = y[half:]
            hidden_pre: ℝ[n] = linear(y1, w1, b1)
            shift: ℝ[n] = linear(relu1d(hidden_pre), w2, b2)
            x2: ℝ[n] = y2 - shift
            return concat(y1, x2)
        def swap(x: ℝ[m], half: ℕ): ℝ[m]:
            return concat(x[half:], x[:half])
        def rescale(x: ℝ[m]): ℝ[m]:
            # z_i = S_ii * h_i,  S_ii = exp(a_i)
            s: ℝ[m] = exp(this.a)
            return x * s
        def rescale_inv(z: ℝ[m]): ℝ[m]:
            # h_i = z_i / S_ii = z_i * exp(-a_i)
            s: ℝ[m] = exp(-this.a)
            return z * s
        def rescale_log_det(): ℝ:
            # log |det dz/dh| = sum_i a_i
            return sum(this.a)
        def forward_z(x: ℝ[m]): ℝ[m]:
            h: ℝ[m] = this.coupling(x, this.w1a, this.b1a, this.w2a, this.b2a, this.n_half)
            h = this.swap(h, this.n_half)
            h = this.coupling(h, this.w1b, this.b1b, this.w2b, this.b2b, this.n_half)
            h = this.swap(h, this.n_half)
            return this.rescale(h)
        def λ(x: ℝ[m]) -> ℝ:
            z: ℝ[m] = this.forward_z(x)
            log_pz: ℝ = gaussian_log_prob(z, this.ndim)
            log_px: ℝ = log_pz + this.rescale_log_det()
            return log_px
        def inverse(z: ℝ[m]): ℝ[m]:
            # f = rescale . swap . coupling_b . swap . coupling_a, so
            # f^{-1} = coupling_a^{-1} . swap . coupling_b^{-1} . swap . rescale^{-1}
            # (swap is its own inverse)
            h: ℝ[m] = this.rescale_inv(z)
            h = this.swap(h, this.n_half)
            h = this.coupling_inv(h, this.w1b, this.b1b, this.w2b, this.b2b, this.n_half)
            h = this.swap(h, this.n_half)
            h = this.coupling_inv(h, this.w1a, this.b1a, this.w2a, this.b2a, this.n_half)
            return h
        def sample(): ℝ[m]:
            # draw z from the base distribution and push it through the
            # inverse flow to get a generated (flattened, dequantized-scale) image
            z: ℝ[m] ~ Normal(0.0, 1.0, this.ndim)
            return this.inverse(z)
        def loss(pred: ℝ, target: ℝ): ℝ:
            return -pred

    # average NLL over a dataset: (1/count) Σ -log p(x)
    def nll(model: NICE, X: ℝ[k, d], count: ℝ): ℝ:
        total: ℝ = 0
        for i:ℕ(count):
            total += neg_loglik(model(X[i]))
        return total / count


    # Dataset
    # create_dataset is NOT a built-in Physika function.
    # Add the helper from runtime.py to physika/runtime.py
    # before running this script.

    dataset = create_dataset(80, 100)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    train_X = train_dataset[0]
    test_X = test_dataset[0]

    len_train_X: ℝ = get_1d_array_length(train_X)
    len_test_X: ℝ = get_1d_array_length(test_X)

    # Initialize model parameters
    # ℕ so they stay integers through field -> method param -> slice bound

    ndim: ℕ = 784
    hidden: ℕ = 128
    half: ℕ = 392

    # He scale on the input layer, near-zero on the output layer so each
    # coupling starts as the identity map (y2 = x2 + shift, shift ~ 0)

    s1: ℝ = sqrt(2.0 / half)
    s2: ℝ = sqrt(2.0 / hidden) * 0.01

    w1a: ℝ[hidden, half] = for i:ℕ(hidden) -> ε: ℝ[half] ~ Normal(0.0, s1, half)
    b1a: ℝ[hidden] = for i:ℕ(hidden) -> i*0
    w2a: ℝ[half, hidden] = for i:ℕ(half) -> ε: ℝ[hidden] ~ Normal(0.0, s2, hidden)
    b2a: ℝ[half] = for i:ℕ(half) -> i*0

    w1b: ℝ[hidden, half] = for i:ℕ(hidden) -> ε: ℝ[half] ~ Normal(0.0, s1, half)
    b1b: ℝ[hidden] = for i:ℕ(hidden) -> i*0
    w2b: ℝ[half, hidden] = for i:ℕ(half) -> ε: ℝ[hidden] ~ Normal(0.0, s2, hidden)
    b2b: ℝ[half] = for i:ℕ(half) -> i*0

    a_init: ℝ[ndim] = for i:ℕ(ndim) -> i*0

    nice_object: NICE = NICE(w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, a_init, half, ndim)

    # Debug: sanity-check the forward pass on one sample

    debug_input = dequantize(flatten(train_X[0], 28, 28), ndim)
    debug_log_px = nice_object(debug_input)
    print(debug_log_px)
    print(DEVICE)

    # Flatten + dequantize the dataset once, up front
    # inner element loop replaced by a row slice assignment

    train_X_flat: ℝ[len_train_X, ndim] = for i:ℕ(len_train_X) -> for j:ℕ(ndim) -> j*0
    for i:ℕ(len_train_X):
        train_X_flat[i, :] = dequantize(flatten(train_X[i], 28, 28), ndim)

    test_X_flat: ℝ[len_test_X, ndim] = for i:ℕ(len_test_X) -> for j:ℕ(ndim) -> j*0
    for i:ℕ(len_test_X):
        test_X_flat[i, :] = dequantize(flatten(test_X[i], 28, 28), ndim)

    # NICE is unsupervised
    dummy_y: ℝ[len_train_X] = for i:ℕ(len_train_X) -> i*0


    # Training loop - via Physika's built-in train()
    # train() loops over samples internally, that loop cannot be vectorized here

    epochs: ℕ = 20
    lr: ℝ = 0.001

    losses: ℝ[epochs] = for i:ℕ(epochs) -> i*0

    for i:ℕ(epochs):
        nice_object = train(nice_object, train_X_flat, dummy_y, 1, lr)
        epoch_loss = nll(nice_object, train_X_flat, len_train_X)
        losses[i] = epoch_loss
        print(epoch_loss)
        epoch_bits_per_dim = epoch_loss / (ndim * log(2.0)) + 8.0
        print(epoch_bits_per_dim)

    # Testing the Model: average negative log-likelihood

    test_loss: ℝ = nll(nice_object, test_X_flat, len_test_X)
    print(test_loss)
    bits_per_dim: ℝ = test_loss / (ndim * log(2.0)) + 8.0
    print(bits_per_dim)

    # draw z ~ N(0, I) via NICE.sample(), push through the inverse flow,result is a 28x28 image

    gen_flat = nice_object.sample()
    gen_img = unflatten(gen_flat, 28, 28)
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
