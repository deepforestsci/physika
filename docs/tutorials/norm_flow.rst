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
The inverse of each layer, and the full inverse map :math:`f^{-1}: z \to x`, are also shown; they are what would be used for sampling from the model.
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
        def coupling(x: ℝ[m], w1: ℝ[H, n], b1: ℝ[H], w2: ℝ[n, H], b2: ℝ[n]): ℝ[m]:
            # Forward additive coupling:
            #   y_1 = x_1
            #   y_2 = x_2 + m(x_1)
            x1: ℝ[n] = first_half(x)
            x2: ℝ[n] = second_half(x)
            shift: ℝ[n] = linear(relu1d(linear(x1, w1, b1)), w2, b2)
            y2: ℝ[n] = x2 + shift
            return concat_1d_array(x1, y2)
        def coupling_inv(y: ℝ[m], w1: ℝ[H, n], b1: ℝ[H], w2: ℝ[n, H], b2: ℝ[n]): ℝ[m]:
            # Inverse additive coupling (same network m, subtract instead of add):
            #   x_1 = y_1
            #   x_2 = y_2 - m(y_1)
            y1: ℝ[n] = first_half(y)
            y2: ℝ[n] = second_half(y)
            shift: ℝ[n] = linear(relu1d(linear(y1, w1, b1)), w2, b2)
            x2: ℝ[n] = y2 - shift
            return concat_1d_array(y1, x2)
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
        def λ(x: ℝ[m]) -> ℝ:
            h: ℝ[m] = x
            h = this.coupling(h, this.w1a, this.b1a, this.w2a, this.b2a)
            h = swap_halves(h)
            h = this.coupling(h, this.w1b, this.b1b, this.w2b, this.b2b)
            h = swap_halves(h)
            z: ℝ[m] = this.rescale(h)
            log_pz: ℝ = gaussian_log_prob(z)
            log_px: ℝ = log_pz + this.rescale_log_det()
            return log_px
        def inverse(z: ℝ[m]): ℝ[m]:
            # f = rescale . swap . coupling_b . swap . coupling_a, so
            # f^{-1} = coupling_a^{-1} . swap^{-1} . coupling_b^{-1} . swap^{-1} . rescale^{-1}
            # (swap_halves is its own inverse)
            h: ℝ[m] = this.rescale_inv(z)
            h = swap_halves(h)
            h = this.coupling_inv(h, this.w1b, this.b1b, this.w2b, this.b2b)
            h = swap_halves(h)
            h = this.coupling_inv(h, this.w1a, this.b1a, this.w2a, this.b2a)
            return h
        def loss(pred: ℝ, target: ℝ): ℝ:
            return -pred


    def gaussian_log_prob(x: ℝ[m]): ℝ:
        len_x: ℝ = get_1d_array_length(x)
        total: ℝ = 0
        for i:ℕ(len_x):
            total += -0.5 * x[i] * x[i] - 0.5 * log(2 * 3.14159265)
        return total

Putting it all together: Training a model with Normalizing Flow - Full code
---------------------------------------------------------------------------------


This is the complete code for training a model on the MNIST dataset using the NICE Normalizing Flow.
Note for simplicity the inverse transform is not included in this implementation, it is used for generative modelling, outside of the scope of this tutorial.
The code is also unoptimized, batching and GPU acceleration is not included for simplicity.


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

    # NICE (Nonlinear Independent Components Estimation) on MNIST

    # Helper functions
    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def get_2d_array_num_rows(x: ℝ[m, n]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols: ℝ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results

    def relu(x: ℝ): ℝ:
        if x > 0:
            return x
        else:
            return 0.0

    def relu1d(x: ℝ[m]): ℝ[m]:
        len_x: ℝ = get_1d_array_length(x)
        results: ℝ[len_x] = zero_1d_array(len_x)
        for i:ℕ(len_x):
            results[i] = relu(x[i])
        return results

    def linear(x: ℝ[n], weight: ℝ[m, n], bias: ℝ[m]): ℝ[m]:
        out: ℝ = get_1d_array_length(bias)
        inp: ℝ = get_1d_array_length(x)
        results: ℝ[out] = zero_1d_array(out)
        for i:ℕ(out):
            acc = 0
            for j:ℕ(inp):
                acc += weight[i, j] * x[j]
            results[i] = acc + bias[i]
        return results

    def flatten_image(img: ℝ[H, W]): ℝ[n]:
        rows: ℝ = get_2d_array_num_rows(img)
        cols: ℝ = get_1d_array_length(img[0])
        n: ℝ = rows * cols
        results: ℝ[n] = zero_1d_array(n)
        for i:ℕ(rows):
            for j:ℕ(cols):
                results[i*cols + j] = img[i, j] / 256.0 + 0.5 / 256.0
        return results

    def first_half(x: ℝ[m]): ℝ[n]:
        half: ℝ = get_1d_array_length(x) / 2
        results: ℝ[half] = zero_1d_array(half)
        for i:ℕ(half):
            results[i] = x[i]
        return results

    def second_half(x: ℝ[m]): ℝ[n]:
        len_x: ℝ = get_1d_array_length(x)
        half: ℝ = len_x / 2
        results: ℝ[half] = zero_1d_array(half)
        for i:ℕ(half):
            results[i] = x[i + half]
        return results

    def concat_1d_array(x1: ℝ[m], x2: ℝ[n]): ℝ[p]:
        len1: ℝ = get_1d_array_length(x1)
        len2: ℝ = get_1d_array_length(x2)
        total: ℝ = len1 + len2
        results: ℝ[total] = zero_1d_array(total)
        for i:ℕ(len1):
            results[i] = x1[i]
        for i:ℕ(len2):
            results[i + len1] = x2[i]
        return results

    def swap_halves(x: ℝ[m]): ℝ[m]:
        x1: ℝ[n] = first_half(x)
        x2: ℝ[n] = second_half(x)
        return concat_1d_array(x2, x1)

    def gaussian_log_prob(x: ℝ[m]): ℝ:
        len_x: ℝ = get_1d_array_length(x)
        total: ℝ = 0
        for i:ℕ(len_x):
            total += -0.5 * x[i] * x[i] - 0.5 * log(2 * 3.14159265)
        return total

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
        def coupling(x: ℝ[m], w1: ℝ[H, n], b1: ℝ[H], w2: ℝ[n, H], b2: ℝ[n]): ℝ[m]:
            x1: ℝ[n] = first_half(x)
            x2: ℝ[n] = second_half(x)
            shift: ℝ[n] = linear(relu1d(linear(x1, w1, b1)), w2, b2)
            y2: ℝ[n] = x2 + shift
            return concat_1d_array(x1, y2)
        def rescale(x: ℝ[m]): ℝ[m]:
            s: ℝ[m] = exp(this.a)
            return x * s
        def rescale_log_det(): ℝ:
            return sum(this.a)
        def λ(x: ℝ[m]) -> ℝ:
            h: ℝ[m] = x
            h = this.coupling(h, this.w1a, this.b1a, this.w2a, this.b2a)
            h = swap_halves(h)
            h = this.coupling(h, this.w1b, this.b1b, this.w2b, this.b2b)
            h = swap_halves(h)
            z: ℝ[m] = this.rescale(h)
            log_pz: ℝ = gaussian_log_prob(z)
            log_px: ℝ = log_pz + this.rescale_log_det()
            return log_px
        def loss(pred: ℝ, target: ℝ): ℝ:
            return -pred


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

    ndim: ℝ = 784
    hidden: ℝ = 16
    half: ℝ = ndim / 2

    w1a: ℝ[hidden, half] = for i:ℕ(hidden) -> for j:ℕ(half) -> sin(3.14 * i / hidden) * cos(3.14 * j / half) * 0.01
    b1a: ℝ[hidden] = for i:ℕ(hidden) -> i*0
    w2a: ℝ[half, hidden] = for i:ℕ(half) -> for j:ℕ(hidden) -> cos(3.14 * i / half) * sin(3.14 * j / hidden) * 0.01
    b2a: ℝ[half] = for i:ℕ(half) -> i*0

    w1b: ℝ[hidden, half] = for i:ℕ(hidden) -> for j:ℕ(half) -> sin(3.14 * i / hidden) * cos(3.14 * j / half) * 0.01
    b1b: ℝ[hidden] = for i:ℕ(hidden) -> i*0
    w2b: ℝ[half, hidden] = for i:ℕ(half) -> for j:ℕ(hidden) -> cos(3.14 * i / half) * sin(3.14 * j / hidden) * 0.01
    b2b: ℝ[half] = for i:ℕ(half) -> i*0

    a_init: ℝ[ndim] = for i:ℕ(ndim) -> i*0

    nice_object: NICE = NICE(w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, a_init)

    # Debug: sanity-check the forward pass on one sample

    debug_input = flatten_image(train_X[0])
    debug_log_px = nice_object(debug_input)
    print(debug_log_px)
    print(DEVICE)

    # Flatten + dequantize the dataset once, up front

    train_X_flat: ℝ[len_train_X, ndim] = zero_2d_array(len_train_X, ndim)
    for i:ℕ(len_train_X):
        flat = flatten_image(train_X[i])
        for j:ℕ(ndim):
            train_X_flat[i, j] = flat[j]

    test_X_flat: ℝ[len_test_X, ndim] = zero_2d_array(len_test_X, ndim)
    for i:ℕ(len_test_X):
        flat = flatten_image(test_X[i])
        for j:ℕ(ndim):
            test_X_flat[i, j] = flat[j]

    # NICE is unsupervised
    dummy_y: ℝ[len_train_X] = zero_1d_array(len_train_X)


    # Training loop - via Physika's built-in train()

    epochs: ℕ = 10
    lr: ℝ = 0.005

    losses: ℝ[epochs] = zero_1d_array(epochs)

    for i:ℕ(epochs):
        nice_object = train(nice_object, train_X_flat, dummy_y, 1, lr)
        epoch_loss = 0
        for j:ℕ(len_train_X):
            log_px = nice_object(train_X_flat[j])
            epoch_loss += neg_loglik(log_px)
        epoch_loss = epoch_loss / len_train_X
        losses[i] = epoch_loss
        print(epoch_loss)

    # Testing the Model: average negative log-likelihood

    test_loss: ℝ = 0
    for i:ℕ(len_test_X):
        log_px = nice_object(test_X_flat[i])
        test_loss += neg_loglik(log_px)
    test_loss = test_loss / len_test_X
    print(test_loss)


References
----------------

1. `Deep Generative Models: Normalizing Flow Models <https://deepgenerativemodels.github.io/notes/flow/>`_
2. `Flow-based Deep Generative Models <https://lilianweng.github.io/posts/2018-10-13-flow-models/>`_
3. `Variational Inference with Normalizing Flows <https://arxiv.org/abs/1505.05770>`_
4. `NICE: Non-linear Independent Components Estimation <https://arxiv.org/abs/1410.8516>`_
5. `Density estimation using Real NVP <https://arxiv.org/abs/1605.08803>`_
6. `normflows: A PyTorch Package for Normalizing Flows <https://github.com/VincentStimper/normalizing-flows>`_
