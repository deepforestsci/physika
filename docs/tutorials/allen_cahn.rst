Parameter Learning of the Allen–Cahn Equation
===============================================

In this tutorial we solve both the *forward* and *inverse* problem for the
two-dimensional Allen–Cahn equation. The Allen–Cahn equation is a
reaction–diffusion PDE introduced by Sam Allen and John Cahn [AllenCahn1976]_ to model
how the boundaries between ordered regions of an alloy
move and coarsen. It is one of the canonical models of *phase separation*: a
field initialised close to an unstable state spontaneously organises into
sharp domains separated by thin interfaces. It is commonly used in material science
and fluid dynamics to model problems with complex moving interfaces.

We first solve the forward model, starting from a random field and show it
phase separate. We then solve the inverse problem: recovering the interface
parameter :math:`\varepsilon` from an observation of the field, by
differentiating through the entire solver.

The Equation
------------

The form of the Allen–Cahn equation we will use is:

.. math::

   \frac{\partial \eta}{\partial t}
   = \varepsilon^2 \nabla^2 \eta - (\eta^3 - \eta)

where:

- :math:`\eta(x, y, t)` is the order parameter, which is driven
  towards the two stable phases :math:`\eta = +1` and :math:`\eta = -1`
- :math:`\varepsilon` sets the width of the interface between the two phases —
  the parameter we will learn
- :math:`\varepsilon^2 \nabla^2 \eta` is a diffusion term that penalises sharp
  gradients
- :math:`-(\eta^3 - \eta)` is the reaction term. It is the negative derivative
  of the double-well potential :math:`W(\eta) = \tfrac{1}{4}(\eta^2 - 1)^2`, so
  it pushes :math:`\eta` away from the unstable state :math:`\eta = 0` and
  towards the wells at :math:`\pm 1`

The competition between these two terms produces phase separation: the non linear reaction
term amplifies fluctuations into :math:`\pm 1` domains, while the diffusion term
smooths the interfaces and drives them to coarsen over time.

We use a unit square domain :math:`[0, 1]^2` with **periodic** boundary
conditions.

The Numerical Method
--------------------

Allen–Cahn is *stiff*: the small parameter :math:`\varepsilon` multiplying the
Laplacian forces an explicit scheme to take tiny time steps. To avoid this we
use a **semi-implicit (IMEX)** scheme with a convex-splitting stabilization
:math:`S` [ShenYang2010]_.

Start with the PDE written as a time derivative equals a diffusion term plus a
reaction term:

.. math::

   \eta_t = \varepsilon^2 \nabla^2 \eta - f(\eta),
   \qquad f(\eta) = \eta^3 - \eta .

A fully explicit Euler step, with an implict diffusion term, and explicit non linear reaction term gives:

.. math::

   \frac{\eta_{n+1} - \eta_n}{\Delta t}
   = \varepsilon^2 \nabla^2 \eta_{n+1} - f(\eta_n).

This already avoids the explicit diffusion stability limit, but the explicit
nonlinear term can still make the update fragile at large time steps. We
therefore add a linear stabilization term :math:`S`. In discrete time this is
equivalent to damping the change from :math:`\eta_n` to :math:`\eta_{n+1}`:

.. math::

   \frac{\eta_{n+1} - \eta_n}{\Delta t}
   = \varepsilon^2 \nabla^2 \eta_{n+1}
   - f(\eta_n)
   - S(\eta_{n+1} - \eta_n).

.. math::

   (1 + \Delta t\,S - \Delta t\,\varepsilon^2 \nabla^2)\, \eta_{n+1}
   = (1 + \Delta t\,S)\, \eta_n - \Delta t\,(\eta_n^3 - \eta_n)

This is the stabilized semi-implicit equation used in the code. The important
practical effect is that :math:`S` makes the implicit operator more diagonally
dominant and damps the explicitly treated reaction term, so the scheme remains
stable at larger time steps.

**Discretizing the Laplacian.** On a periodic grid the Laplacian is a
five-point stencil,

.. math::

   [\nabla^2 \eta ]_{i,j} \approx
   \frac{\eta_{i-1,j} + \eta_{i+1,j} + \eta_{i,j-1} + \eta_{i,j+1}
   - 4\,\eta_{i,j}}{\Delta x^2},

which we implement with ``roll`` — shifting the field by one cell along each
axis with wrap-around gives the four neighbours for every point at once:

.. code-block:: text

    def neighbor_sum(eta: ℝ[m, n]): ℝ[m, n]:
        s: ℝ[m, n] = roll(eta, 1, 0) + roll(eta, -1, 0) + roll(eta, 1, 1) + roll(eta, -1, 1)
        return s

Substituting the stencil into the stabilized equation at grid cell
:math:`(i,j)` gives

.. math::

   (1 + \Delta t\,S)\eta^{n+1}_{i,j}
   - \Delta t\,\varepsilon^2
   \frac{
      \eta^{n+1}_{i-1,j}
      + \eta^{n+1}_{i+1,j}
      + \eta^{n+1}_{i,j-1}
      + \eta^{n+1}_{i,j+1}
      - 4\eta^{n+1}_{i,j}
   }{\Delta x^2}
   =
   (1 + \Delta t\,S)\eta^n_{i,j}
   - \Delta t((\eta^n_{i,j})^3 - \eta^n_{i,j}).

Now define

.. math::

   c = \frac{\Delta t\,\varepsilon^2}{\Delta x^2},
   \qquad
   \text{rhs}_{i,j}
   =
   (1 + \Delta t\,S)\eta^n_{i,j}
   - \Delta t((\eta^n_{i,j})^3 - \eta^n_{i,j}).

After collecting the center value :math:`\eta^{n+1}_{i,j}` on the left, the
linear equation for one cell is

.. math::

   (1 + \Delta t\,S + 4c)\eta^{n+1}_{i,j}
   - c\left(
      \eta^{n+1}_{i-1,j}
      + \eta^{n+1}_{i+1,j}
      + \eta^{n+1}_{i,j-1}
      + \eta^{n+1}_{i,j+1}
   \right)
   =
   \text{rhs}_{i,j}.

Solving this equation for the center value gives the Jacobi update used below:

.. math::

   \eta^{n+1}_{i,j}
   \leftarrow
   \frac{
      \text{rhs}_{i,j}
      + c\left(
         \eta^{n+1}_{i-1,j}
         + \eta^{n+1}_{i+1,j}
         + \eta^{n+1}_{i,j-1}
         + \eta^{n+1}_{i,j+1}
      \right)
   }{
      1 + \Delta t\,S + 4c
   }.


.. code-block:: text

    def semi_implicit_step(eta: ℝ[m, n], eps: ℝ): ℝ[m, n]:
        coeff: ℝ = dt * eps**2 / dx**2
        diag: ℝ = 1.0 + dt * stab + 4.0 * coeff
        rhs: ℝ[m, n] = (1.0 + dt * stab) * eta - dt * (eta**3 - eta)
        next_eta: ℝ[m, n] = eta
        for k:ℕ(0, jacobi_iters):
            next_eta = (rhs + coeff * neighbor_sum(next_eta)) / diag
        return next_eta

Stepping this forward in time gives the solver:

.. code-block:: text

    def solver(eps: ℝ): ℝ[m, n]:
        eta: ℝ[m, n] = ic
        for s:ℕ(0, num_steps):
            eta = semi_implicit_step(eta, eps)
        return eta

Set Up the Grid
---------------

We use a :math:`128 \times 128` periodic grid.

.. code-block:: text

    Nx: ℕ = 128
    dx: ℝ = 1.0 / Nx
    dt: ℝ = 2e-2
    stab: ℝ = 2.0           # convex-splitting stabilization (S)
    jacobi_iters: ℕ = 12    # Jacobi sweeps per implicit solve
    num_steps: ℕ = 80

The Random Initial Condition
----------------------------

Phase separation is seeded from noise. We initialise the field with small
fluctuations about the unstable state :math:`\eta = 0`, drawn once from a
uniform distribution on :math:`[-0.1, 0.1]`:

.. code-block:: text

    ic: ℝ[Nx, Nx] = for i:ℕ(Nx) → ε : ℝ[Nx] ~ 𝒰(-0.1, 0.1, Nx)

The Forward Problem: Phase Separation
-------------------------------------

Running the solver with a chosen :math:`\varepsilon` evolves the random field
forward in time. The reaction term amplifies the fluctuations into
:math:`\pm 1` domains, and the diffusion term then coarsens them — small
domains shrink and disappear while large ones grow.

.. code-block:: text

    true_eps: ℝ = 0.03
    true_values: ℝ[m, n] = solver(true_eps)

    #plot_field(ic)
    #plot_field(true_values)

.. note::
   ``plot_field`` is not a built-in Physika function. To visualise a field,
   add the following helper to ``physika/runtime.py``:

   .. code-block:: python

      def plot_field(field):
          import matplotlib.pyplot as plt
          import matplotlib.colors as mcolors

          data = field.detach().cpu().numpy()
          # Fixed [-1, 1] range (the two equilibrium phases); the asinh norm
          # expands contrast near eta = 0 so a small initial field is visible.
          norm = mcolors.AsinhNorm(linear_width=0.1, vmin=-1.0, vmax=1.0)
          plt.imshow(data, origin="lower", extent=(0, 1, 0, 1),
                     norm=norm, cmap="RdBu_r")
          plt.colorbar(label=r"$\eta$")
          plt.xlabel("x")
          plt.ylabel("y")
          plt.title("Allen–Cahn field")
          plt.show()

The figure below shows the field at several times. Starting from random noise, sharp
:math:`\pm 1` domains emerge and progressively coarsen — the hallmark of
Allen–Cahn dynamics.

.. figure:: _static/tutorial_files/allen_cahn_phase_separation.png
   :alt: Allen–Cahn phase separation from a random initial field
   :align: center
   :width: 750px


The Inverse Problem
-------------------

Now we turn the problem around. Suppose we observe the field ``true_values``
produced with the (unknown) true parameter :math:`\varepsilon = 0.03`, and we
want to recover :math:`\varepsilon`. We define a loss measuring the mean
squared error between the field predicted for a candidate :math:`\varepsilon`
and the observed field:

.. math::

   \mathcal{L}(\varepsilon)
   = \frac{1}{N} \sum_{i,j}
   ( \eta^{\text{pred}}_{i,j}(\varepsilon) - \eta^{\text{obs}}_{i,j} )^2

.. code-block:: text

    def calculate_loss(eps: ℝ): ℝ:
        pred: ℝ[m, n] = solver(eps)
        loss: ℝ = mean((pred - true_values)**2)
        return loss

.. note::
   The interface parameter :math:`\varepsilon` is only identifiable while the
   field is still evolving. Once the domains have fully saturated to
   :math:`\pm 1`, :math:`\varepsilon` only affects the sub-grid interface width
   and the loss becomes insensitive to it. We therefore observe at an
   intermediate time (``num_steps = 80``), before saturation.

Train with Gradient Descent
----------------------------

We start from an initial guess of :math:`\varepsilon = 0.06` and update it with
gradient descent. ``grad`` differentiates through the entire solver — every
time step and every Jacobi sweep — to obtain :math:`d\mathcal{L}/d\varepsilon`.

.. code-block:: text

    eps: ℝ = 0.06
    learning_rate: ℝ = 0.5
    epochs: ℕ = 40

    for epoch:ℕ(0, epochs):
        g = grad(calculate_loss, eps)
        eps = eps - learning_rate * g
        physika_print(eps)

Within about ten epochs the estimate converges to the true value
:math:`\varepsilon = 0.03` and the loss drops to zero:

.. figure:: _static/tutorial_files/allen_cahn_inverse_convergence.png
   :alt: Recovering the Allen–Cahn interface parameter by gradient descent
   :align: center
   :width: 750px

   Left: the recovered :math:`\varepsilon` converges to the true value.
   Right: the training loss.

Full Code
---------

.. code-block:: text

    def neighbor_sum(eta: ℝ[m, n]): ℝ[m, n]:
        s: ℝ[m, n] = roll(eta, 1, 0) + roll(eta, -1, 0) + roll(eta, 1, 1) + roll(eta, -1, 1)
        return s

    Nx: ℕ = 64
    dx: ℝ = 1.0 / Nx
    dt: ℝ = 2e-2
    stab: ℝ = 2.0
    jacobi_iters: ℕ = 12
    num_steps: ℕ = 80

    ic: ℝ[Nx, Nx] = for i:ℕ(Nx) → ε : ℝ[Nx] ~ 𝒰(-0.1, 0.1, Nx)

    def semi_implicit_step(eta: ℝ[m, n], eps: ℝ): ℝ[m, n]:
        coeff: ℝ = dt * eps**2 / dx**2
        diag: ℝ = 1.0 + dt * stab + 4.0 * coeff
        rhs: ℝ[m, n] = (1.0 + dt * stab) * eta - dt * (eta**3 - eta)
        next_eta: ℝ[m, n] = eta
        for k:ℕ(0, jacobi_iters):
            next_eta = (rhs + coeff * neighbor_sum(next_eta)) / diag
        return next_eta

    def solver(eps: ℝ): ℝ[m, n]:
        eta: ℝ[m, n] = ic
        for s:ℕ(0, num_steps):
            eta = semi_implicit_step(eta, eps)
        return eta

    true_eps: ℝ = 0.03
    true_values: ℝ[m, n] = solver(true_eps)

    #plot_field(ic)
    #plot_field(true_values)

    def calculate_loss(eps: ℝ): ℝ:
        pred: ℝ[m, n] = solver(eps)
        loss: ℝ = mean((pred - true_values)**2)
        return loss

    eps: ℝ = 0.06
    learning_rate: ℝ = 0.5
    epochs: ℕ = 40

    for epoch:ℕ(0, epochs):
        g = grad(calculate_loss, eps)
        eps = eps - learning_rate * g
        physika_print(eps)


References
----------

.. [AllenCahn1976] S. M. Allen and J. W. Cahn, *A microscopic theory for antiphase boundary
  motion and its application to antiphase domain coarsening*, Acta
  Metallurgica, 1979.
.. [ShenYang2010] Shen, J., & Yang, X. (2010). Numerical approximations of  Allen-Cahn and Cahn-Hilliard  equations. Discrete and Continuous Dynamical Systems, 28(4), 1669–1691. https://doi.org/10.3934/dcds.2010.28.1669

- `Allen–Cahn equation (Wikipedia) <https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation>`_
