Deep Equilibrium Models
=======================

This tutorial introduces Deep Equilibrium Models (DEQs) and shows how to implement one in Physika.
Suppose we want the representational power of a very deep network, but without paying to store and backpropagate through every one of its layers.
A Deep Equilibrium Model gives us exactly this: instead of stacking many distinct layers, it applies **one** layer over and over until its output stops changing, and treats that settled value as the network's answer.
That settled value is called an *equilibrium* (or *fixed point*), and finding it turns the forward pass of a neural network into a **root-finding problem**.

By the end of this tutorial you will understand what a fixed point is, why a weight-tied "infinite depth" network can be summarized by one, how to solve for that fixed point with a quasi-Newton root finder, and how the whole thing is differentiated in Physika.
You will then train a small DEQ that reconstructs handwritten digits from the MNIST dataset.
This tutorial is based on Bai, Kolter, and Koltun's *Deep Equilibrium Models* paper [BaiDEQ2019]_ and their Deep Implicit Layers tutorial [ImplicitLayers]_.


What are Deep Equilibrium Models?
---------------------------------

A conventional deep network computes a sequence of hidden states, one per layer:

.. math::
    h_1 = f_1(h_0, x), \quad h_2 = f_2(h_1, x), \quad \ldots, \quad h_T = f_T(h_{T-1}, x)

Each layer :math:`f_t` usually has its own parameters, and the memory needed for training grows with the number of layers :math:`T`, because every intermediate :math:`h_t` must be kept for the backward pass.

A Deep Equilibrium Model makes two changes.
First, it **ties the weights**: every layer is the *same* function :math:`f(\cdot, x, \theta)`.
Second, it asks what happens as the depth goes to infinity.
If repeatedly applying :math:`f` drives the hidden state toward a value that no longer changes, then that limiting value :math:`h^\star` satisfies

.. math::
    h^\star = f(h^\star, x, \theta).

A point that maps to itself under :math:`f` is called a **fixed point** (or **equilibrium point**).
Rather than run :math:`f` a fixed number of times, a DEQ directly *solves* for this fixed point.
The infinite stack of identical layers is replaced by a single object, the equilibrium, and the entire forward pass becomes "find the :math:`h^\star` that :math:`f` leaves unchanged" [BaiDEQ2019]_.

.. figure:: /_static/tutorial_files/deq/deq.jpg
   :alt: An infinitely deep weight-tied network whose hidden state converges to a single fixed point h-star.
   :align: center
   :width: 500px

   **Figure 1.** A weight-tied network applies the same layer :math:`f(\cdot, x, \theta)` repeatedly. As depth grows, the hidden state settles onto an equilibrium :math:`h^\star` that satisfies :math:`h^\star = f(h^\star, x, \theta)`. Figure from [BaiDEQ2019]_.

Setup and Notation
^^^^^^^^^^^^^^^^^^

We work with three objects throughout.

The **input** :math:`x \in \mathbb{R}^{d}` is the data the network is given (for us, a flattened :math:`28 \times 28 = 784`-dimensional MNIST image, a vector of shape :math:`(d,)`).

The **hidden state** :math:`h \in \mathbb{R}^{n}` is the internal representation the network refines, a vector of shape :math:`(n,)` (here :math:`n = 16`).

The **parameters** :math:`\theta` collect every learnable weight and bias in the layer.
In our model :math:`\theta = \{W, U, b, W_o, b_o\}`.

The layer itself is a function :math:`f: \mathbb{R}^{n} \times \mathbb{R}^{d} \to \mathbb{R}^{n}`, which takes the current hidden state and the (fixed) input and returns the next hidden state.
The concrete choice used in this tutorial is a single fully connected layer with a :math:`\tanh` nonlinearity:

.. math::
    f(h, x, \theta) = \tanh\!\left(h\,W + x\,U + b\right).

Here :math:`W \in \mathbb{R}^{n \times n}` mixes the hidden state with itself, :math:`U \in \mathbb{R}^{d \times n}` injects the input, and :math:`b \in \mathbb{R}^{1 \times n}` is a bias.
Note that :math:`x` enters :math:`f` but never changes while we iterate: it is a constant *drive* term that anchors the equilibrium.

In Physika the layer is written exactly as the equation reads:

.. code-block:: text

    def f(h: ℝ[1,n], x: ℝ[1,d]): ℝ[1,n]:
        return tanh(h @ W + x @ U + b)


Fixed Points and the Banach Fixed-Point Theorem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **fixed point** of a function :math:`g` is any input that the function returns unchanged: a point :math:`h^\star` with :math:`g(h^\star) = h^\star`.
For a DEQ, the function is :math:`f(\cdot, x, \theta)` with :math:`x` and :math:`\theta` held fixed, and the equilibrium is a fixed point of that map.

Two questions immediately arise: does such a point exist, and is it unique?
The **Banach fixed-point theorem** (also called the contraction mapping theorem) answers both under one condition [Wikipedia_Banach]_.

We first need the notion of a **contraction**.
A map :math:`f` is a contraction if it always brings pairs of points *closer together* by at least a constant factor: there exists a **Lipschitz constant** :math:`L < 1` such that

.. math::
    \left\| f(a, x, \theta) - f(b, x, \theta) \right\| \le L \, \left\| a - b \right\| \qquad \text{for all } a, b,

where :math:`\|\cdot\|` denotes the Euclidean distance between two vectors.
Intuitively, applying a contraction shrinks distances, so it cannot spread points apart.

The Banach fixed-point theorem states that a contraction on a complete space has **exactly one** fixed point :math:`h^\star`, and that the simple iteration :math:`h_{k+1} = f(h_k, x, \theta)` converges to it from any starting point.
This repeated-application scheme is called **Picard iteration**, and its error shrinks geometrically as :math:`L^k`.

The theorem is what guarantees that "an infinitely deep weight-tied network" is a meaningful object: as long as :math:`f` is a contraction, the equilibrium exists and is unique.
Our layer makes this easy to arrange. The slope of :math:`\tanh` is :math:`\tanh'(z) = 1 - \tanh^2(z)`, which is largest at :math:`z = 0` where it equals :math:`1` and is smaller everywhere else, so :math:`\tanh` is *1-Lipschitz*: it never stretches a distance. We can see this by computing the slope directly:

.. code-block:: text

    t: ℝ[1,n] = tanh(z)
    slope: ℝ[1,n] = 1.0 - t * t        # 1 - tanh^2, always in (0, 1]

Because :math:`\tanh` never stretches and :math:`W` only rescales, the layer satisfies :math:`\|f(a,x,\theta) - f(b,x,\theta)\| \le \|W\|\,\|a - b\|`, so keeping :math:`\|W\|` below :math:`1` makes :math:`f` a contraction.
This is why the weights are initialized small: it keeps the equilibrium unique and the solver well behaved. (The same :math:`1 - \tanh^2` slope returns below, where it builds the Jacobian.)

Solving for the Equilibrium (the Forward Pass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finding :math:`h^\star` is a **root-finding problem**.
Define the *residual*, the amount by which a point fails to be a fixed point:

.. math::
    g(h) = f(h, x, \theta) - h,

so that the equilibrium is exactly the value that makes the residual vanish, :math:`g(h^\star) = 0`.
In Physika the residual is the expression it looks like:

.. code-block:: text

    g = this.f(this.h_star, x) - this.h_star      # residual  g(h) = f(h, x) - h

**Newton's method** solves :math:`g(h) = 0` by repeatedly replacing :math:`g` with its straight-line approximation.
Near the current iterate :math:`h_k`, the residual is well approximated by its first-order Taylor expansion,

.. math::
    g(h_k - \delta) \approx g(h_k) - J\,\delta,

where :math:`J = \partial g / \partial h` is the **residual Jacobian**, the matrix of partial derivatives of :math:`g` with respect to :math:`h`.
Newton picks the step :math:`\delta` that makes this linear approximation zero, that is it solves

.. math::
    J\,\delta = g(h_k), \qquad h_{k+1} = h_k - \delta.

The important thing to notice is that the first equation is a **linear system** :math:`J\delta = g`: each Newton step *solves a linear system*, it does not form a matrix inverse.
Solving :math:`J\delta = g` is both cheaper and more numerically stable than building :math:`J^{-1}` and multiplying, and it is exactly what the linear-solve helper in the next section does.

We still need :math:`J = \partial g/\partial h`. Since :math:`g = f - h`, differentiating the :math:`-h` term gives a :math:`-I`, so :math:`J = \partial f/\partial h - I`.
The layer Jacobian :math:`\partial f/\partial h` has a closed form here, no autodiff or finite differences required. Differentiating :math:`f = \tanh(hW + xU + b)` brings down the slope :math:`\tanh'(z) = 1 - f^2` on each unit, times the linear weight, so :math:`\partial f/\partial h` is :math:`W` with each column scaled by :math:`1 - f^2`:

.. math::
    \frac{\partial f}{\partial h} = W \odot (1 - f^2), \qquad J = \frac{\partial f}{\partial h} - I.

That column scaling is the ``df_dh`` helper, one entry at a time:

.. code-block:: text

    def df_dh(W: ℝ[n,n], tanh_prime: ℝ[1,n]): ℝ[n,n]:
        J: ℝ[n,n] = zeros2d(16, 16)
        for c:ℕ(16):
            for r:ℕ(16):
                J[r, c] = W[r, c] * tanh_prime[0, c]
        return J

With :math:`J` in hand, one Newton step is the linear solve followed by the update, which maps line for line onto the math (:math:`\delta` solves :math:`J\delta = g`, then :math:`h \leftarrow h - \delta`):

.. code-block:: text

    delta = linsolve(J, g[0])                     # solve  J δ = g
    this.h_star = this.h_star - [delta]           # update h ← h - δ

The variant used in the code is the simplest quasi-Newton scheme, the **chord method** (modified Newton): it forms :math:`J` *once* at the starting point :math:`h_0 = 0` and reuses it for every step, rather than rebuilding it each iteration.
Freezing :math:`J` is what makes it quasi-Newton, and it is justified here because the layer is a contraction, so the equilibrium stays close to :math:`h_0` and one Jacobian is a good enough model for all the steps.
For comparison, the simplest solver of all is Picard iteration :math:`h_{k+1} = f(h_k, x, \theta)`, which needs no Jacobian at all but converges only linearly.

Differentiating Through the Equilibrium (the Backward Pass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model we need the gradient of the loss with respect to the parameters, which requires the gradient of the equilibrium :math:`h^\star` with respect to :math:`\theta`.
Differentiating the equilibrium condition itself gives a closed form, called **implicit differentiation** [BaiDEQ2019]_.
Starting from :math:`h^\star = f(h^\star, x, \theta)` and differentiating both sides with respect to :math:`\theta` (the right-hand side needs the chain rule because :math:`h^\star` depends on :math:`\theta`):

.. math::
    \frac{\partial h^\star}{\partial \theta}
    = \frac{\partial f}{\partial h^\star}\,\frac{\partial h^\star}{\partial \theta}
    + \frac{\partial f}{\partial \theta}.

Collecting the :math:`\partial h^\star/\partial \theta` terms produces the same residual Jacobian :math:`\partial f/\partial h^\star - I` the forward solve used:

.. math::
    \left(\frac{\partial f}{\partial h^\star} - I\right)\frac{\partial h^\star}{\partial \theta}
    = -\frac{\partial f}{\partial \theta}
    \quad\Longrightarrow\quad
    \frac{\partial h^\star}{\partial \theta}
    = -\left(\frac{\partial f}{\partial h^\star} - I\right)^{-1}\frac{\partial f}{\partial \theta}.

Chaining once more with the loss gives the gradient we want:

.. math::
    \boxed{\;\frac{\partial \mathcal{L}}{\partial \theta}
    = -\,\frac{\partial \mathcal{L}}{\partial h^\star}
      \left(\frac{\partial f}{\partial h^\star} - I\right)^{-1}
      \frac{\partial f}{\partial \theta}\;}

This references :math:`h^\star` only: the solver's trajectory has dropped out entirely.

**Worked scalar example.**
Take a one-dimensional equilibrium with :math:`f(h, x) = \tanh(w h + u x + b)`, the scalar version of the layer.
Using :math:`\tanh' = 1 - \tanh^2` and :math:`h^\star = \tanh(\cdot)`, the layer derivative at the fixed point is :math:`\partial f/\partial h^\star = w\,(1 - {h^\star}^2)`, so

.. math::
    \frac{\partial h^\star}{\partial b}
    = -\bigl(w(1 - {h^\star}^2) - 1\bigr)^{-1}\,(1 - {h^\star}^2).

With :math:`w = 0.5` and :math:`h^\star = 0.4`, the layer derivative is :math:`0.42`, so the factor :math:`-(0.42 - 1)^{-1} = 1.724` multiplies the one-layer gradient :math:`0.84`, giving :math:`\partial h^\star/\partial b \approx 1.448`.
The :math:`-(\partial f/\partial h^\star - I)^{-1}` term is what an *infinitely deep* network contributes: in the scalar case it is the geometric series :math:`1 + f' + f'^2 + \cdots = (1 - f')^{-1}`, the summed influence of the layer applied over and over.

Differentiability in Physika
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Physika class compiles to a differentiable module, and ``grad()`` backpropagates through its methods with automatic differentiation.
We do not write a custom backward pass. Because the forward solver is an ordinary loop, ``grad(L, this.params)`` differentiates straight through the unrolled quasi-Newton iteration and into every parameter.

At first glance this looks like it would give the wrong answer, since the solver uses a Jacobian, so surely that has to be handled specially?
It does not, and the reason is the boxed formula above.
The frozen Jacobian :math:`J` acts only as a *preconditioner*: it changes how fast the iteration converges, not where it converges to.
When the iteration has settled on :math:`h^\star`, the gradient of the unrolled loop satisfies the same linear relation as the implicit-differentiation result, and the preconditioner cancels.
So autograd through the unrolled solve recovers the exact implicit gradient :math:`-\frac{\partial \mathcal{L}}{\partial h^\star}(\partial f/\partial h^\star - I)^{-1}\frac{\partial f}{\partial \theta}` without our ever coding that formula.
The only cost is memory: the unrolled version stores its iterates, whereas a hand-written backward using the closed form would not.


Methods for Solving the Fixed Point
-----------------------------------

The forward pass of a DEQ is only as good as the solver that produces :math:`h^\star`.
This is not an exhaustive list, but below are the solvers most commonly used, from simplest to most powerful.

1. Picard (Fixed-Point) Iteration
    The most direct solver simply iterates the layer, :math:`h_{k+1} = f(h_k, x, \theta)`.
    It requires nothing beyond evaluating :math:`f`, and by the Banach theorem it converges whenever :math:`f` is a contraction, but its convergence is only linear and it diverges if :math:`f` is not a contraction.

2. Newton's Method
    Newton's method uses the residual Jacobian :math:`J = \partial f/\partial h - I` to take much larger steps, solving :math:`J \delta = g(h_k)` and setting :math:`h_{k+1} = h_k - \delta`.
    Near the solution it converges quadratically, so it needs very few iterations, at the cost of forming and solving with the :math:`n \times n` Jacobian every step.

3. Quasi-Newton
    Quasi-Newton methods keep Newton's fast convergence while avoiding a fresh exact Jacobian every step.
    The implementation below uses the simplest such scheme, the **chord method**, which forms the residual Jacobian once at the initial iterate and reuses it for every step.
    Production DEQs use stronger variants such as **Broyden's method**, which maintains a low-rank running approximation of the Jacobian, and **Anderson acceleration**, which forms each iterate as a least-squares-optimal mix of the last few [BaiDEQ2019]_.

Regardless of which solver is chosen, the backward pass is unchanged: the implicit-differentiation formula depends only on the converged :math:`h^\star`, so improving the solver never changes how gradients are computed.

.. figure:: /_static/tutorial_files/deq/chord_method.png
   :alt: Left, the residual g(h) with parallel fixed-slope chord steps marching to the equilibrium; right, distance to the equilibrium versus iteration for Picard, chord, and Newton on a log scale.
   :align: center
   :width: 750px

   **Figure 2.** The chord (frozen-Jacobian quasi-Newton) solve on a scalar residual :math:`g(h) = f(h,x) - h`. *Left:* every step follows a line of the same slope :math:`g'(h_0)`, computed once at the start, down to the axis, so the steps are parallel and march :math:`h_0 \to h_1 \to \cdots \to h^\star`. *Right:* distance to the equilibrium per iteration; chord (frozen :math:`J`) is much faster than Picard and nearly as fast as Newton, at the cost of a single Jacobian build. A far start :math:`h_0` is used here only to separate the steps visually; the DEQ starts from :math:`h_0 = 0`, already close to :math:`h^\star` because the layer is a contraction.

Solving a Linear System in Physika
----------------------------------

Each Newton step is the linear system :math:`J \delta = g`, so we need a way to **solve** :math:`A x = b` for :math:`x`.
Physika has no built-in linear solver, so we write a small ``linsolve`` using **Gaussian elimination**, the same primitive used in the Physika linear-solve tutorials.

The idea is to place the right-hand side next to the matrix, forming the augmented block :math:`[\,A \mid b\,]`, and then apply row operations that reduce the left block to the identity.
The right column is carried along and becomes the solution:

.. math::
    [\,A \mid b\,] \;\xrightarrow{\ \text{row ops}\ }\; [\,I \mid x\,].

Concretely, we sweep the columns one at a time. For column :math:`i` we take the diagonal entry as the **pivot**, divide that row by the pivot so the pivot becomes :math:`1`, and subtract the right multiple of the pivot row from every other row so the column is zero elsewhere.
A final **back-substitution** reads the solution off the reduced system.
Note that this returns the solution vector :math:`x` directly; it never forms :math:`A^{-1}`, which would be more work and less stable.

Two small points make the elimination fit the language cleanly.
First, ``eye`` builds the identity by filling a zero matrix and setting the diagonal entries to :math:`1` in a loop.
Second, each column sweep rebuilds the augmented matrix as a fresh array (``aug_next``) rather than writing into it in place, which keeps automatic differentiation happy when the solve is differentiated during training.

.. code-block:: text

    def eye(n: ℝ): ℝ[n, n]:
        I: ℝ[n, n] = for i:ℕ(n) → for j:ℕ(n) → j * 0.0
        for i:ℕ(n):
            I[i, i] = 1.0
        return I

    def linsolve(A: ℝ[16, 16], b: ℝ[16]): ℝ[16]:
        aug: ℝ[16, 17] = zeros2d(16, 17)
        for i:ℕ(16):
            for c:ℕ(16):
                aug[i, c] = A[i, c]
            aug[i, 16] = b[i]
        for i:ℕ(16):
            piv = zeros1d(17)
            for c:ℕ(17):
                piv[c] = aug[i, c]
            aug_next = zeros2d(16, 17)
            for r:ℕ(16):
                if r == i:
                    for c:ℕ(17):
                        aug_next[r, c] = piv[c] / piv[i]
                else:
                    fac = aug[r, i] / piv[i]
                    for c:ℕ(17):
                        aug_next[r, c] = aug[r, c] - fac * piv[c]
            aug = aug_next
        x: ℝ[16] = zeros1d(16)
        for i:ℕ(16):
            idx = 15 - i
            total = aug[idx, 16]
            for j:ℕ(idx + 1, 16):
                total = total - aug[idx, j] * x[j]
            x_next = zeros1d(16)
            for c:ℕ(16):
                if c == idx:
                    x_next[c] = total / aug[idx, idx]
                else:
                    x_next[c] = x[c]
            x = x_next
        return x


Implementing a DEQ in Physika
-----------------------------

We now have every piece: the layer :math:`f`, its Jacobian ``df_dh``, and the linear solve ``linsolve``.
The DEQ class puts them together.
It is an autoencoding DEQ: an MNIST image :math:`x` drives the layer to an equilibrium hidden state :math:`h^\star`, and a linear decoder maps :math:`h^\star` back to a :math:`784`-dimensional reconstruction :math:`\hat{x}`, trained to match :math:`x`.

The ``equilibrium`` method is where the pieces meet. It computes :math:`f_h = f(h_0, x)` and :math:`1 - f_h^2`, freezes the residual Jacobian :math:`J = \partial f/\partial h - I` once with ``df_dh(...) - eye(16)``, then runs the chord iteration: residual, linear solve, update.

.. code-block:: text

    def equilibrium(x: ℝ[1,d]): ℝ[1,n]:
        num_solver_steps: ℕ = 3
        this.h_star = zeros2d(1, 16)
        f_h: ℝ[1,n] = this.f(this.h_star, x)
        tanh_prime: ℝ[1,n] = 1.0 - f_h * f_h
        J: ℝ[n,n] = df_dh(W, tanh_prime) - eye(16)
        for k:ℕ(num_solver_steps):
            g = this.f(this.h_star, x) - this.h_star
            delta = linsolve(J, g[0])
            this.h_star = this.h_star - [delta]
        return this.h_star

The **call operator** ``λ`` runs the solver and decodes the equilibrium into data space, :math:`\hat{x} = h^\star W_o + b_o`, and the **loss** is the squared reconstruction error :math:`\|\,x - \hat{x}\,\|^2` against the input image itself:

.. code-block:: text

    def λ(x: ℝ[1,d]) → ℝ[1,d]:
        h_star: ℝ[1,n] = this.equilibrium(x)
        return h_star @ Wo + bo

    def loss(target: ℝ[1,784], x_hat: ℝ[1,784]): ℝ:
        diff: ℝ[1,784] = target - x_hat
        return sum(diff * diff)

Training is ordinary gradient descent: ``train`` computes the loss and calls ``grad(L, this.params)``, which differentiates through the unrolled solver as explained above, then updates the parameters.

.. note::

    The DEQ is differentiable end to end with no hand-written backward. Gradients flow through the decoder, through the unrolled quasi-Newton solve (including ``linsolve``), and into :math:`W, U, b, W_o, b_o` automatically. Because the frozen Jacobian cancels at the fixed point, this recovers the exact implicit gradient.


Training a DEQ on the MNIST Dataset
-----------------------------------

This is the complete program for training the DEQ on MNIST. 
``load_mnist`` returns the first ``n`` MNIST digits as a ``ℝ[n, 784]`` array of flattened images. It is not a built-in; add this helper to ``physika/runtime.py``:

.. code-block:: python

    def load_mnist(n=1000):
        import torch
        from torchvision import datasets, transforms
        mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
        return torch.stack([mnist[i][0].view(784) for i in range(int(n))]).to(DEVICE)


Full Code
---------

.. code-block:: text

    physika.seed(0)
    def tanh(a: ℝ[p,q]): ℝ[p,q]:
        num: ℝ[p,q] = exp(a) - exp(-a)
        denom: ℝ[p,q] = exp(a) + exp(-a)
        return num / denom

    def rand_array(n: ℝ, m: ℝ, μ: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → ε: ℝ[m] ~ Normal(0.0, μ, m)

    def zeros2d(n: ℝ, m: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → j * 0.0

    def zeros1d(k: ℝ): ℝ[k]:
        return for i:ℕ(k) → i * 0.0

    def eye(n: ℝ): ℝ[n, n]:
        I: ℝ[n, n] = for i:ℕ(n) → for j:ℕ(n) → j * 0.0
        for i:ℕ(n):
            I[i, i] = 1.0
        return I

    def df_dh(W: ℝ[n,n], tanh_prime: ℝ[1,n]): ℝ[n,n]:
        return for r:ℕ(n) → for c:ℕ(n) → W[r, c] * tanh_prime[0, c]

    def linsolve(A: ℝ[16, 16], b: ℝ[16]): ℝ[16]:
        aug: ℝ[16, 17] = zeros2d(16, 17)
        for i:ℕ(16):
            for c:ℕ(16):
                aug[i, c] = A[i, c]
            aug[i, 16] = b[i]
        for i:ℕ(16):
            piv = zeros1d(17)
            for c:ℕ(17):
                piv[c] = aug[i, c]
            aug_next = zeros2d(16, 17)
            for r:ℕ(16):
                if r == i:
                    for c:ℕ(17):
                        aug_next[r, c] = piv[c] / piv[i]
                else:
                    fac = aug[r, i] / piv[i]
                    for c:ℕ(17):
                        aug_next[r, c] = aug[r, c] - fac * piv[c]
            aug = aug_next
        x: ℝ[16] = zeros1d(16)
        for i:ℕ(16):
            idx = 15 - i
            total = aug[idx, 16]
            for j:ℕ(idx + 1, 16):
                total = total - aug[idx, j] * x[j]
            x_next = zeros1d(16)
            for c:ℕ(16):
                if c == idx:
                    x_next[c] = total / aug[idx, idx]
                else:
                    x_next[c] = x[c]
            x = x_next
        return x

    class DEQ(W: ℝ[n,n], U: ℝ[d,n], b: ℝ[1,n], Wo: ℝ[n,d], bo: ℝ[1,d]):
        h_star: ℝ[1,n]
        def f(h: ℝ[1,n], x: ℝ[1,d]): ℝ[1,n]:
            return tanh(h @ W + x @ U + b)
        def equilibrium(x: ℝ[1,d]): ℝ[1,n]:
            num_solver_steps: ℕ = 3
            this.h_star = zeros2d(1, 16)
            f_h: ℝ[1,n] = this.f(this.h_star, x)
            tanh_prime: ℝ[1,n] = 1.0 - f_h * f_h
            J: ℝ[n,n] = df_dh(W, tanh_prime) - eye(16)
            for k:ℕ(num_solver_steps):
                g = this.f(this.h_star, x) - this.h_star
                delta = linsolve(J, g[0])
                this.h_star = this.h_star - [delta]
            return this.h_star
        def λ(x: ℝ[1,d]) → ℝ[1,d]:
            h_star: ℝ[1,n] = this.equilibrium(x)
            return h_star @ Wo + bo
        def loss(target: ℝ[1,784], x_hat: ℝ[1,784]): ℝ:
            diff: ℝ[1,784] = target - x_hat
            return sum(diff * diff)
        def train(X: ℝ[50,784], epochs: ℕ, lr: ℝ, images: ℝ):
            for epoch:ℕ(epochs):
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    preds = this(x)
                    L = this.loss(x, preds)
                    grads = grad(L, this.params)
                    this.update_params(lr, grads)
                total = 0
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    pred = this(x)
                    total += this.loss(x, pred)
                print(total / images)
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W = this.W - lr * learnable_grads[0]
            this.U = this.U - lr * learnable_grads[1]
            this.b = this.b - lr * learnable_grads[2]
            this.Wo = this.Wo - lr * learnable_grads[3]
            this.bo = this.bo - lr * learnable_grads[4]

    print(DEVICE)
    W: ℝ[16,16] = rand_array(16, 16, 0.01)
    U: ℝ[784,16] = rand_array(784, 16, 0.02)
    b: ℝ[1,16] = zeros2d(1, 16)
    Wo: ℝ[16,784] = rand_array(16, 784, 0.05)
    bo: ℝ[1,784] = zeros2d(1, 784)
    deq: DEQ = DEQ(W, U, b, Wo, bo)

    images: ℝ = 50
    X: ℝ[50, 784] = load_mnist(images)      

    x0: ℝ[1,784] = [X[0]]
    recon_before: ℝ[1,784] = deq(x0)
    loss_before: ℝ = deq.loss(x0, recon_before)
    print(loss_before)

    epochs: ℕ = 20
    lr: ℝ = 0.001
    deq.train(X, epochs, lr, images)
    recon_after: ℝ[1,784] = deq(x0)
    loss_after: ℝ = deq.loss(x0, recon_after)
    print(loss_after)


Training plots
--------------

After running the code above (~90 minutes), you should see the average reconstruction loss decrease over epochs as the model learns to encode and decode the digits through its equilibrium state.

.. figure:: /_static/tutorial_files/deq/deq_train_plot.png
   :alt:
   :align: center
   :width: 750px


References
----------

.. [BaiDEQ2019] S. Bai, J. Z. Kolter, and V. Koltun,
    *Deep Equilibrium Models*.
    https://arxiv.org/abs/1909.01377

.. [ImplicitLayers] J. Z. Kolter, D. Duvenaud, and M. Johnson,
    *Deep Implicit Layers: Neural ODEs, Deep Equilibrium Models, and Beyond*.
    https://implicit-layers-tutorial.org/

.. [Wikipedia_Banach] Wikipedia,
    *Banach fixed-point theorem*.
    https://en.wikipedia.org/wiki/Banach_fixed-point_theorem