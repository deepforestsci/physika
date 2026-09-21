Deep Equilibrium Models
=======================

This tutorial introduces Deep Equilibrium Models (DEQs) and shows how to implement one in Physika.
The representational power of a network is its mathematical ability to capture, express, and approximate the complex patterns and functions hidden inside data. Broadly, deeper networks have more representational capacity, which is what lets a deep model represent something like an image.
Suppose we want the representational power of a very deep network, but without paying to store and backpropagate through every one of its layers.
A Deep Equilibrium Model gives us exactly this, because instead of stacking many distinct layers it applies one layer over and over until its output stops changing, and treats that settled (or steady state) value as the network's answer.
That steady state value is called an *equilibrium* (or *fixed point*), and finding it turns the forward pass of a neural network into a root-finding problem.

By the end of this tutorial you will understand what a fixed point is, why a weight-tied "infinite depth" network can be summarized by one, how to solve for that fixed point with a quasi-Newton root finder, and how Physika differentiates the loss back to the parameters through that fixed point for training.
You will then train a small DEQ that reconstructs handwritten digits from the MNIST dataset.
This tutorial is based on Bai, Kolter, and Koltun's *Deep Equilibrium Models* paper [BaiDEQ2019]_ and their Deep Implicit Layers tutorial [ImplicitLayers]_.


What are Deep Equilibrium Models?
---------------------------------

A conventional deep network computes a sequence of hidden states, one per layer:

.. math::
    h_1 = f_1(h_0, x), \quad h_2 = f_2(h_1, x), \quad \ldots, \quad h_T = f_T(h_{T-1}, x)

Each layer :math:`f_t` usually has its own parameters, and the memory needed for training grows with the number of layers :math:`T`, because every intermediate :math:`h_t` must be kept for the backward pass.

A Deep Equilibrium Model is built from two ideas that work together.
First, it ties the weights, so every layer is the same function :math:`f(\cdot, x, \theta)`.
Second, it asks what happens as the depth goes to infinity, that is apply same layer infinite times.
If repeatedly applying :math:`f` drives the hidden state toward a value that no longer changes, then that limiting value :math:`h^\star` satisfies

.. math::
    h^\star = f(h^\star, x, \theta).

A point that maps to itself under :math:`f` is called a fixed point.
Rather than run :math:`f` a fixed number of times, a DEQ directly *solves* for this fixed point, once the DEQ is able to do so we can say that the model has reached the equilibrium.
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

The **hidden state** :math:`h \in \mathbb{R}^{n}` is the internal representation the network refines, a vector of shape :math:`(n,)` (here :math:`n = 8`, larger n increases runtime significantly).

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

.. note::

   Here :math:`f` is a single fully connected layer for simplicity.
   If we want more depth inside :math:`f`, we can pull a layer out into a reusable
   ``linear`` (a weight-and-bias with a :math:`\tanh`, a commonly used layer for fully connected networks):

   .. code-block:: text

       def tanh(a: ℝ[p,q]): ℝ[p,q]:
           num: ℝ[p,q] = exp(a) - exp(-a)
           denom: ℝ[p,q] = exp(a) + exp(-a)
           return num / denom

       def linear(z: ℝ[1,n], W: ℝ[n,n], b: ℝ[1,n]): ℝ[1,n]:
           return tanh(z @ W + b)

   and stack it. A two-layer :math:`f` folds the input in once, then applies
   ``linear`` twice, each with its own weights:

   .. code-block:: text

       def f(h: ℝ[1,n], x: ℝ[1,d]): ℝ[1,n]:
           h1: ℝ[1,n] = linear(h + x @ U, W1, b1)   
           h2: ℝ[1,n] = linear(h1, W2, b2)
           return h2


Fixed Points and the Banach Fixed-Point Theorem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **fixed point** of a function :math:`g` is any input that the function returns unchanged,

.. math::
    g(h^\star) = h^\star.

(For instance, :math:`0` is a fixed point of :math:`\tanh`, since :math:`\tanh(0) = 0`.)
For a DEQ, the function is :math:`f(\cdot, x, \theta)` with :math:`x` and :math:`\theta` held fixed, and the equilibrium is a fixed point of that map.

For the equilibrium to be well defined, such a point must exist and be unique, existence gives the iteration a target to converge to, and uniqueness makes that target independent of where the solver starts.
The **Banach fixed-point theorem** answers both under one condition [Wikipedia_Banach]_.

**Contraction**: :math:`f` is said to be a contraction if it always brings pairs of points closer together by at least a constant factor there exists a Lipschitz constant :math:`L < 1` such that

.. math::
    \left\| f(a, x, \theta) - f(b, x, \theta) \right\| \le L \, \left\| a - b \right\| \qquad \text{for all } a, b,

where :math:`\|\cdot\|` denotes the Euclidean distance between two vectors.
Intuitively, applying a contraction shrinks distances, so it cannot spread points apart.

The Banach fixed-point theorem states that a contraction on a complete space has exactly one fixed point :math:`h^\star`, and that the simple iteration :math:`h_{k+1} = f(h_k, x, \theta)` converges to it from any starting point.
This repeated-application scheme is called Picard iteration, and its error shrinks geometrically as :math:`L^k`.

The theorem is what DEQ principally works on, as long as :math:`f` (the layer) is a contraction, the equilibrium exists and is unique.
Our layer makes this easy to arrange. The slope of :math:`\tanh` is :math:`\tanh'(z) = 1 - \tanh^2(z)`, which is largest at :math:`z = 0` where it equals :math:`1` and is smaller everywhere else, so :math:`\tanh` is *1-Lipschitz*: it never stretches a distance. We can see this by computing the slope directly, with the corresponding Physika snippet below:

.. code-block:: text

    t: ℝ[1,n] = tanh(z)
    slope: ℝ[1,n] = 1.0 - t * t        

Because :math:`\tanh` never stretches and :math:`W` only rescales, the layer satisfies :math:`\|f(a,x,\theta) - f(b,x,\theta)\| \le \|W\|\,\|a - b\|`, so keeping :math:`\|W\|` below :math:`1` makes :math:`f` a contraction.
This is why the weights are initialized small, which keeps the equilibrium unique and the solver well behaved. (The same :math:`1 - \tanh^2` slope returns below, where it builds the Jacobian.)

Solving a Linear System in Physika
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The forward solver we build next repeatedly needs to solve a linear system :math:`A x = b` for :math:`x`, so we set that tool up first.
Physika has no built-in linear solver, so we write a small ``linsolve`` using **Gaussian elimination**, used in this tutorial. 

The idea is to place the right-hand side next to the matrix, forming the augmented block :math:`[\,A \mid b\,]`, and then apply row operations that reduce the left block to the identity.
The right column is carried along and becomes the solution:

.. math::
    [\,A \mid b\,] \;\xrightarrow{\ \text{row ops}\ }\; [\,I \mid x\,].

Concretely, we sweep the columns one at a time. For column :math:`i` we take the diagonal entry as the pivot, divide that row by the pivot so the pivot becomes :math:`1`, and subtract the right multiple of the pivot row from every other row so the column is zero elsewhere.
A final back-substitution reads the solution off the reduced system.
Note that this returns the solution vector :math:`x` directly; it never forms :math:`A^{-1}`, which would be more work and less stable.

Two small points make the elimination fit the language cleanly.
First, ``eye`` builds the identity by filling a zero matrix and setting the diagonal entries to :math:`1` in a loop.
Second, each column sweep rebuilds the augmented matrix as a fresh array (``aug_next``) rather than writing into it in place, which keeps automatic differentiation happy when the solve is differentiated during training.
This is implemented in Physika, with the snippet below:

.. code-block:: text

    def eye(n: ℝ): ℝ[n, n]:
        I: ℝ[n, n] = for i:ℕ(n) → for j:ℕ(n) → j * 0.0
        for i:ℕ(n):
            I[i, i] = 1.0
        return I

    def zeros2d(n: ℝ, m: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → j * 0.0

    def zeros1d(k: ℝ): ℝ[k]:
        return for i:ℕ(k) → i * 0.0

    def linsolve(A: ℝ[8, 8], b: ℝ[8]): ℝ[8]:
        aug: ℝ[8, 9] = zeros2d(8, 9)
        for i:ℕ(8):
            for c:ℕ(8):
                aug[i, c] = A[i, c]
            aug[i, 8] = b[i]
        for i:ℕ(8):
            piv = zeros1d(9)
            for c:ℕ(9):
                piv[c] = aug[i, c]
            aug_next = zeros2d(8, 9)
            for r:ℕ(8):
                if r == i:
                    for c:ℕ(9):
                        aug_next[r, c] = piv[c] / piv[i]
                else:
                    fac = aug[r, i] / piv[i]
                    for c:ℕ(9):
                        aug_next[r, c] = aug[r, c] - fac * piv[c]
            aug = aug_next
        x: ℝ[8] = zeros1d(8)
        for i:ℕ(8):
            idx = 7 - i
            total = aug[idx, 8]
            for j:ℕ(idx + 1, 8):
                total = total - aug[idx, j] * x[j]
            x_next = zeros1d(8)
            for c:ℕ(8):
                if c == idx:
                    x_next[c] = total / aug[idx, idx]
                else:
                    x_next[c] = x[c]
            x = x_next
        return x

We use this in the DEQ forward pass in the next section, where each Newton step solves :math:`J \delta = g` for the update :math:`\delta`. Writing the solver in plain Physika costs nothing at training time, because Physika is fully differentiable and backpropagates straight through the elimination.

Solving for the Equilibrium (the Forward Pass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finding :math:`h^\star` is a root-finding problem, meaning we look for the point where a function equals zero. That function is the *residual*, the amount by which a point fails to be a fixed point:

.. math::
    g(h) = f(h, x, \theta) - h,

so that the equilibrium is exactly the value that makes the residual vanish, :math:`g(h^\star) = 0`.
A root of the residual is therefore an equilibrium of the layer, the settled hidden state the DEQ treats as its answer.
In Physika the residual is the expression it looks like, where ``h_star`` is the current iterate:

.. code-block:: text

    def zeros2d(n: ℝ, m: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → j * 0.0

    def f(h: ℝ[1,n], x: ℝ[1,d]): ℝ[1,n]:
        return tanh(h @ W + x @ U + b)

    h_star: ℝ[1,n] = zeros2d(1, n)        
    g = f(h_star, x) - h_star             

**Newton's method** solves :math:`g(h) = 0` by repeatedly replacing :math:`g` with its straight-line approximation.
Near the current iterate :math:`h_k`, the residual is well approximated by its first-order Taylor expansion,

.. math::
    g(h_k - \delta) \approx g(h_k) - J\,\delta,

where :math:`J = \partial g / \partial h` is the **residual Jacobian**, the matrix of partial derivatives of :math:`g` with respect to :math:`h`.
Newton picks the step :math:`\delta` that makes this linear approximation zero, that is it solves

.. math::
    J\,\delta = g(h_k), \qquad h_{k+1} = h_k - \delta.

The important thing to notice is that the first equation is a linear system :math:`J\delta = g`, so each Newton step solves a linear system rather than forming a matrix inverse.
Solving :math:`J\delta = g` is both cheaper and more numerically stable than building :math:`J^{-1}` and multiplying, and it is what the ``linsolve`` helper above does.

We still need :math:`J = \partial g/\partial h`. Since :math:`g = f - h`, differentiating the :math:`-h` term gives a :math:`-I`, so :math:`J = \partial f/\partial h - I`.
The layer Jacobian :math:`\partial f/\partial h` can be computed directly from an explicit formula. Differentiating :math:`f = \tanh(hW + xU + b)` brings down the slope :math:`\tanh'(z) = 1 - f^2` on each unit, times the linear weight, so :math:`\partial f/\partial h` is :math:`W` with each column scaled by :math:`1 - f^2`:

.. math::
    \frac{\partial f}{\partial h} = W \odot (1 - f^2), \qquad J = \frac{\partial f}{\partial h} - I.

That column scaling is the ``df_dh`` helper, implemented in Physika below:

.. code-block:: text

    def df_dh(W: ℝ[n,n], tanh_prime: ℝ[1,n]): ℝ[n,n]:
        return for r:ℕ(n) → for c:ℕ(n) → W[r, c] * tanh_prime[0, c]

With :math:`J` in hand, one Newton step is the linear solve followed by the update, which maps line for line onto the math (:math:`\delta` solves :math:`J\delta = g`, then :math:`h \leftarrow h - \delta`):
The Physika code below implements the Newton step, which is repeated in a loop until the residual is small enough, combining all the helpers defined above:

.. code-block:: text

    delta = linsolve(J, g[0])          
    h_star = h_star - [delta]          

The variant used in the code is the simplest quasi-Newton scheme, the **chord method** (modified Newton), which forms :math:`J` *once* at the starting point :math:`h_0 = 0` and reuses it for every step, rather than rebuilding it each iteration.
Freezing :math:`J` is what makes it quasi-Newton, and it is justified here because the layer is a contraction, so the equilibrium stays close to :math:`h_0` and one Jacobian is a good enough model for all the steps.
For comparison, the simplest solver of all is Picard iteration :math:`h_{k+1} = f(h_k, x, \theta)`, which needs no Jacobian at all but converges only linearly.

Differentiating Through the Equilibrium (the Backward Pass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model we need the gradient of the loss with respect to the parameters, which requires the gradient of the equilibrium :math:`h^\star` with respect to :math:`\theta`.
Differentiating the equilibrium condition itself gives us an exact solution, called implicit differentiation [BaiDEQ2019]_.
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
The :math:`-(\partial f/\partial h^\star - I)^{-1}` term is what an infinitely deep network contributes: in the scalar case it is the geometric series :math:`1 + f' + f'^2 + \cdots = (1 - f')^{-1}`, the summed influence of the layer applied over and over.

Differentiability in Physika
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Physika class compiles to a differentiable module, and ``grad()`` backpropagates through its methods with automatic differentiation.
We do not write a custom backward pass. Because the forward solver is an ordinary loop, Physika differentiates straight through the unrolled quasi-Newton iteration and into every parameter.


Methods for Solving the Fixed Point
-----------------------------------

The forward pass of a DEQ is only as good as the solver that produces :math:`h^\star`.
This is not an exhaustive list, but below are the solvers most commonly used, from simplest to most powerful.

1. Picard (Fixed-Point) Iteration
    The most direct solver simply iterates the layer, :math:`h_{k+1} = f(h_k, x, \theta)`.
    It requires nothing beyond evaluating :math:`f`, and by the Banach theorem it converges whenever :math:`f` is a contraction, but its convergence is only linear and it diverges if :math:`f` is not a contraction.

2. Newton's Method
    Newton's method [Wikipedia_Newton]_ uses the residual Jacobian :math:`J = \partial f/\partial h - I` to take much larger steps, solving :math:`J \delta = g(h_k)` and setting :math:`h_{k+1} = h_k - \delta`.
    Near the solution it converges quadratically, so it needs very few iterations, at the cost of forming and solving with the :math:`n \times n` Jacobian every step.

3. Quasi-Newton Methods
    Quasi-Newton methods keep Newton's fast convergence while avoiding a fresh exact Jacobian every step.
    The implementation below uses the simplest such scheme, the **chord method**, which forms the residual Jacobian once at the initial iterate and reuses it for every step.
    Production DEQs use stronger variants such as **Broyden's method** [Wikipedia_Broyden]_, which maintains a low-rank running approximation of the Jacobian, and **Anderson acceleration** [Wikipedia_Anderson]_, which forms each iterate as a least-squares-optimal mix of the last few [BaiDEQ2019]_.

Regardless of which solver is chosen, the backward pass is unchanged: the implicit-differentiation formula depends only on the converged :math:`h^\star`, so improving the solver never changes how gradients are computed, but it affects how fast the layer converges to the equilibrium.


Training a DEQ in Physika
-----------------------------

We now have every component needed to implement a simple DEQ model.
These are the layer :math:`f`, its Jacobian ``df_dh``, and the linear solve ``linsolve``.
It is an autoencoding DEQ, where an MNIST image :math:`x` drives the layer to an equilibrium hidden state :math:`h^\star`, and a linear decoder maps :math:`h^\star` back to a :math:`784`-dimensional reconstruction :math:`\hat{x}`, trained to match :math:`x`.

The ``equilibrium`` routine is where the pieces meet. It computes :math:`f_h = f(h_0, x)` and :math:`1 - f_h^2`, freezes the residual Jacobian :math:`J = \partial f/\partial h - I` once with ``df_dh(...) - eye(8)``, then runs the chord iteration: residual, linear solve, update.
The Physika snippets in this section contain the implementation of the equilibrium solver, calling of the solver, and the loss function, and how the training loop is implemented.

.. code-block:: text

    def equilibrium(x: ℝ[1,d]): ℝ[1,n]:
        num_solver_steps: ℕ = 3
        h_star: ℝ[1,n] = zeros2d(1, 8)
        f_h: ℝ[1,n] = f(h_star, x)
        tanh_prime: ℝ[1,n] = 1.0 - f_h * f_h
        J: ℝ[n,n] = df_dh(W, tanh_prime) - eye(8)
        for k:ℕ(num_solver_steps):
            g = f(h_star, x) - h_star
            delta = linsolve(J, g[0])
            h_star = h_star - [delta]
        return h_star

Three steps are enough because the contraction leaves :math:`h_0 = 0` already close to :math:`h^\star`, so the frozen-Jacobian iteration reaches the equilibrium in a handful of steps.

The call operator ``λ`` runs the solver and decodes the equilibrium into data space, :math:`\hat{x} = h^\star W_o + b_o`, and the loss is the squared reconstruction error :math:`\|\,x - \hat{x}\,\|^2` against the input image itself:

.. code-block:: text

    def λ(x: ℝ[1,d]) → ℝ[1,d]:
        h_star: ℝ[1,n] = equilibrium(x)
        return h_star @ Wo + bo

    def loss(target: ℝ[1,784], x_hat: ℝ[1,784]): ℝ:
        diff: ℝ[1,784] = target - x_hat
        return sum(diff * diff)


To train the model we need a DEQ class (shown as a snippet below), which we use to instantiate a model allowing us to refer to its methods (eg: ``this.loss``) and parameters (eg: ``this.W``) with ``this.`` representing the instance of the class.
When calling the class (``this(x)``), runs the ``λ`` method, so ``this(x)`` is equivalent to ``λ(x)``. The above sections are implemented as individual methods, for simplicity.
The Full Code section contains the complete Physika code, which can be run as-is.

.. code-block:: text

    class DEQ(W: ℝ[n,n], U: ℝ[d,n], b: ℝ[1,n], Wo: ℝ[n,d], bo: ℝ[1,d]):
        h_star: ℝ[1,n]
        def f(h: ℝ[1,n], x: ℝ[1,d]): ℝ[1,n]:
            return tanh(h @ W + x @ U + b)
        def equilibrium(x: ℝ[1,d]): ℝ[1,n]:
            num_solver_steps: ℕ = 3
            this.h_star = zeros2d(1, 8)
            f_h: ℝ[1,n] = this.f(this.h_star, x)
            tanh_prime: ℝ[1,n] = 1.0 - f_h * f_h
            J: ℝ[n,n] = df_dh(W, tanh_prime) - eye(8)
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
        def train(X: ℝ[50,784], epochs: ℕ, lr: ℝ, images: ℝ): ℝ[epochs]:
            loss: ℝ[epochs] = for i:ℕ(epochs) -> i*0
            for epoch:ℕ(epochs):
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    preds = this(x)
                    L = this.loss(x, preds)
                    learnable_grads = grad(L, this.learnable_params)
                    this.update_params(lr, learnable_grads)
                total = 0
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    pred = this(x)
                    total += this.loss(x, pred)
                epoch_loss = total/images
                loss[epoch] = epoch_loss
                print(epoch_loss)
            return loss
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W = this.W - lr * learnable_grads[0]
            this.U = this.U - lr * learnable_grads[1]
            this.b = this.b - lr * learnable_grads[2]
            this.Wo = this.Wo - lr * learnable_grads[3]
            this.bo = this.bo - lr * learnable_grads[4]



In the ``train`` method, the loss is computed for each image, the gradient of the loss with respect to the learnable parameters is computed with ``grad``, and the parameters are updated with a simple gradient descent step.
For convenient gradient computation ``this.learnable_params`` is a built-in that collects all the parameters of the class that are differentiable, so ``grad`` returns a list of gradients in the same order as the parameters.
The ``update_params`` method uses simple gradient descent, more sophisticated optimizers can be used as well.

So when training the model the entire training looks like the snippet below, shown is a simplified example on dummy data:

.. code-block:: text

    # initialize learnable parameters to zeros
    W: ℝ[8,8] = zeros2d(8, 8)
    U: ℝ[784,8] = zeros2d(784, 8)
    b: ℝ[1,8] = zeros2d(1, 8)
    Wo: ℝ[8,784] = zeros2d(8, 784)
    bo: ℝ[1,784] = zeros2d(1, 784)

    # train for 20 epochs with learning rate 0.001, on a single image (here all zeros)
    images: ℝ = 1
    X: ℝ[1, 784] = zeros2d(1, 784) 
    epochs: ℕ = 20
    lr: ℝ = 0.001
    # instantiate the DEQ class with all learnable parameters
    deq: DEQ = DEQ(W, U, b, Wo, bo)

    losses: ℝ[epochs] = deq.train(X, epochs, lr, images)

````

.. note::

    The DEQ is differentiable end to end with no hand-written backward. Gradients flow through the decoder, through the unrolled quasi-Newton solve (including ``linsolve``), and into :math:`W, U, b, W_o, b_o` automatically. Because the frozen Jacobian cancels at the fixed point, this recovers the exact implicit gradient.

Dataset
----------- 

MNIST is a dataset of handwritten digits, each a :math:`28 \times 28` grayscale image, when these images are flattened or expressed as a 1D vector we get a vector of size :math:`784`.
``load_mnist`` returns the first ``n`` MNIST digits as a ``ℝ[n, 784]`` array of flattened images. It is not a built-in; add this helper to ``physika/runtime.py``:

.. code-block:: python

    def load_mnist(n=1000):
        import torch
        from torchvision import datasets, transforms
        mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
        return torch.stack([mnist[i][0].view(784) for i in range(int(n))]).to(DEVICE)


Plotting Graphs
-----------------------------------
To plot the training curve, add the ``plot_deq_losses`` helper to ``physika/runtime.py``:

.. code-block:: python

    def plot_deq_losses(losses, before, after):
        import matplotlib.pyplot as plt
        losses = losses.cpu().detach().numpy()
        before = before.cpu().detach().numpy()
        after = after.cpu().detach().numpy()

        epochs = range(1, len(losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

        ax1.plot(epochs, losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("DEQ Training Curve\nLoss by epoch, 50 images of MNIST, 20 epochs")
        ax1.set_xticks(epochs)

        ax2.bar(["Before training", "After 20 epochs"], [float(before), float(after)], width=0.5)
        ax2.set_ylabel("Reconstruction loss")
        ax2.set_title("DEQ Reconstruction Loss\nImage 0, before vs. after 20 epochs of training")

        plt.tight_layout()
        plt.savefig("deq_train_plot.png", dpi=300, bbox_inches="tight")
        plt.show()

The section below combines all that we have covered as a single standalone ``.phyk`` file, and contains the whole DEQ model as a single class.

Full Code
---------

Here we load 50 images of MNIST, train the DEQ for 20 epochs, and plot the training curve and reconstruction loss before and after training.
We keep the learning rate as ``0.001``, and the number of solver steps as ``3``, and hidden size as ``8`` as the hyperparameters. 
The ``rand_array`` helper is used to initialize the learnable parameters with small random values.
The first line ``physika.seed(0)`` ensures that training runs can be reproduced exactly, without variations across runs. 
More details on this can be found in the `Sampling documentation <https://physika.readthedocs.io/en/latest/elf.html#random-sampling>`__.

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

    def linsolve(A: ℝ[8, 8], b: ℝ[8]): ℝ[8]:
        aug: ℝ[8, 9] = zeros2d(8, 9)
        for i:ℕ(8):
            for c:ℕ(8):
                aug[i, c] = A[i, c]
            aug[i, 8] = b[i]
        for i:ℕ(8):
            piv = zeros1d(9)
            for c:ℕ(9):
                piv[c] = aug[i, c]
            aug_next = zeros2d(8, 9)
            for r:ℕ(8):
                if r == i:
                    for c:ℕ(9):
                        aug_next[r, c] = piv[c] / piv[i]
                else:
                    fac = aug[r, i] / piv[i]
                    for c:ℕ(9):
                        aug_next[r, c] = aug[r, c] - fac * piv[c]
            aug = aug_next
        x: ℝ[8] = zeros1d(8)
        for i:ℕ(8):
            idx = 7 - i
            total = aug[idx, 8]
            for j:ℕ(idx + 1, 8):
                total = total - aug[idx, j] * x[j]
            x_next = zeros1d(8)
            for c:ℕ(8):
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
            this.h_star = zeros2d(1, 8)
            f_h: ℝ[1,n] = this.f(this.h_star, x)
            tanh_prime: ℝ[1,n] = 1.0 - f_h * f_h
            J: ℝ[n,n] = df_dh(W, tanh_prime) - eye(8)
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
        def train(X: ℝ[50,784], epochs: ℕ, lr: ℝ, images: ℝ): ℝ[epochs]:
            loss: ℝ[epochs] = for i:ℕ(epochs) -> i*0
            for epoch:ℕ(epochs):
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    preds = this(x)
                    L = this.loss(x, preds)
                    learnable_grads = grad(L, this.learnable_params)
                    this.update_params(lr, learnable_grads)
                total = 0
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    pred = this(x)
                    total += this.loss(x, pred)
                epoch_loss = total/images
                loss[epoch] = epoch_loss
                print(epoch_loss)
            return loss
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W = this.W - lr * learnable_grads[0]
            this.U = this.U - lr * learnable_grads[1]
            this.b = this.b - lr * learnable_grads[2]
            this.Wo = this.Wo - lr * learnable_grads[3]
            this.bo = this.bo - lr * learnable_grads[4]

    print(DEVICE)
    W: ℝ[8,8] = rand_array(8, 8, 0.01)
    U: ℝ[784,8] = rand_array(784, 8, 0.02)
    b: ℝ[1,8] = zeros2d(1, 8)
    Wo: ℝ[8,784] = rand_array(8, 784, 0.05)
    bo: ℝ[1,784] = zeros2d(1, 784)
    deq: DEQ = DEQ(W, U, b, Wo, bo)

    images: ℝ = 50
    # add load_mnist to physika/runtime.py to run on MNIST data
    X: ℝ[50, 784] = load_mnist(images) 

    x0: ℝ[1,784] = [X[0]]
    recon_before: ℝ[1,784] = deq(x0)
    loss_before: ℝ = deq.loss(x0, recon_before)
    print(loss_before)

    epochs: ℕ = 20
    lr: ℝ = 0.001
    losses: ℝ[epochs] = deq.train(X, epochs, lr, images)
    recon_after: ℝ[1,784] = deq(x0)
    loss_after: ℝ = deq.loss(x0, recon_after)
    print(loss_after)
    # add plot_deq_losses helper to physika/runtime.py to create and save plots
    plot_deq_losses(losses, loss_before, loss_after)


Training plots
--------------

After running the code above (~30 minutes), you should see the average reconstruction loss decrease over epochs as the model learns to encode and decode the digits through its equilibrium state.

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

.. [Wikipedia_Newton] Wikipedia,
    *Newton's method*.
    https://en.wikipedia.org/wiki/Newton%27s_method

.. [Wikipedia_Broyden] Wikipedia,
    *Broyden's method*.
    https://en.wikipedia.org/wiki/Broyden%27s_method

.. [Wikipedia_Anderson] Wikipedia,
    *Anderson acceleration*.
    https://en.wikipedia.org/wiki/Anderson_acceleration