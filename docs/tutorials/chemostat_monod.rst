Parameter Learning for a Chemostat (Monod Growth) Model
=======================================================

In this tutorial we learn the growth kinetics of a **chemostat** — a continuous
culture in which fresh medium is pumped in and culture is drawn off at the same
rate. It is the workhorse of quantitative microbiology and bioprocess
engineering, and its dynamics are governed by the **Monod growth law**, the
macroscopic relationship between nutrient concentration and growth rate that
host-resource models such as Weiße *et al.* derive from the underlying
gene-expression machinery. Here we take the law as given and recover its
constants from a measured culture.

The scaffolding is the two-state, full-trajectory-adjoint idiom of the
:doc:`/tutorials/sir` and :doc:`/tutorials/learn_parameter_lotka_volterra_ode`
tutorials, so only the model changes. What makes it worth reading is that it is
our first **microbial-growth / bioprocess** model, and it comes with a genuine
payoff: after fitting a single startup transient, the learned Monod law predicts
the entire steady-state **operating diagram** of the reactor — including the
critical dilution rate at which the culture *washes out*, a regime the optimiser
never saw during training.


The Equations
-------------

Let :math:`X` be the biomass concentration and :math:`S` the growth-limiting
substrate. Cells grow with a **Monod** specific rate
:math:`\mu(S) = \mu_{\max} S / (K_s + S)`, medium is diluted at rate :math:`D`
with feed concentration :math:`S_{\mathrm{in}}`, and substrate is consumed in
proportion to growth divided by the yield :math:`Y`:

.. math::

   \frac{dX}{dt} &= \big(\mu(S) - D\big)\, X \\
   \frac{dS}{dt} &= D\,(S_{\mathrm{in}} - S) - \mu(S)\, \frac{X}{Y}

The parameters we learn are :math:`\theta = [\mu_{\max}, K_s, Y]`, with true
values :math:`\mu_{\max} = 0.5`, :math:`K_s = 0.5`, :math:`Y = 0.5`. The
operating conditions :math:`D = 0.3` and :math:`S_{\mathrm{in}} = 10` are known.
The rate law is rational — only :math:`+\,-\,*\,/` — so it maps directly onto
Physika with no special functions.


The chemostat steady state
--------------------------

Setting both derivatives to zero, a growing culture settles where growth exactly
balances dilution, :math:`\mu(S^\ast) = D`. Solving the Monod law for
:math:`S^\ast`,

.. math::

   S^\ast = \frac{K_s\, D}{\mu_{\max} - D},
   \qquad X^\ast = Y\,(S_{\mathrm{in}} - S^\ast).

Two features make the chemostat special. First, the residual substrate
:math:`S^\ast` is set by the dilution rate and the kinetics **alone** — feeding a
richer medium raises the standing biomass :math:`X^\ast` but leaves
:math:`S^\ast` unchanged. Second, if the dilution rate is pushed too high the
cells cannot divide fast enough to keep up and are flushed out: the culture
**washes out** (:math:`X^\ast \to 0`) once :math:`D` exceeds the critical value

.. math::

   D_{\mathrm{crit}} = \frac{\mu_{\max}\, S_{\mathrm{in}}}{K_s + S_{\mathrm{in}}} .

At our true parameters :math:`D_{\mathrm{crit}} \approx 0.476`, comfortably above
the operating :math:`D = 0.3`, so the culture we simulate reaches a healthy
steady state.


Helper functions
----------------

We reuse the same dynamic-array helpers as the sibling tutorials
(``zero_1d_array`` / ``get_1d_array_length`` / ``append``); Physika arrays are
fixed-shape, so ``append`` allocates a new, one-longer array and copies into it:

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def append(x: ℝ[m], var: ℝ): ℝ[n]:
        new_length: ℝ = len(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        len_x: ℕ = get_1d_array_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results


Step 1: Define the ODE
----------------------

``f`` takes the two-dimensional state ``[X, S]`` and the parameters ``θ`` and
returns the derivatives. The dilution rate ``D`` and feed ``Sin`` are known
globals; both equations are written with the growth/feed term first so they have
no leading sign:

.. code-block:: text

    D: ℝ = 0.3
    Sin: ℝ = 10.0

    def f(state: ℝ[2], θ: ℝ[3]): ℝ[2]:
        X: ℝ = state[0]
        S: ℝ = state[1]
        mumax: ℝ = θ[0]
        Ks: ℝ = θ[1]
        Y: ℝ = θ[2]
        mu: ℝ = mumax * S / (Ks + S)
        dX: ℝ = (mu - D) * X
        dS: ℝ = D * (Sin - S) - mu * X / Y
        return [dX, dS]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge-Kutta method, identical to the
sibling tutorials, over the two-dimensional state:

.. math::

    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

.. code-block:: text

    def rk4_step(state: ℝ[2], θ: ℝ[3]): ℝ[2]:
        k1: ℝ[2] = f(state, θ)
        k2_state: ℝ[2] = state + 0.5 * dt * k1
        k2: ℝ[2] = f(k2_state, θ)
        k3_state: ℝ[2] = state + 0.5 * dt * k2
        k3: ℝ[2] = f(k3_state, θ)
        k4_state: ℝ[2] = state + dt * k3
        k4: ℝ[2] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


Step 3: Build the Trajectory Solver
-----------------------------------

We inoculate a small biomass into fresh medium (:math:`S = S_{\mathrm{in}}`) and
integrate 300 steps at ``dt = 0.1`` — a 30-unit window, about nine residence
times :math:`1/D`, long enough for the culture to reach its steady state —
collecting both trajectories:

.. code-block:: text

    dt: ℝ = 0.1
    timesteps: ℝ = 300

    def solver(θ: ℝ[3]): ℝ[2, m]:
        state: ℝ[2] = [0.05, 10.0]
        X_array: ℝ[1] = [0.05]
        S_array: ℝ[1] = [10.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            X = results[0]
            S = results[1]
            X_array = append(X_array, X)
            S_array = append(S_array, S)
            state = results
        return [X_array, S_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick parameters that give a healthy (non-washout) steady state and generate
the trajectories we will try to fit:

.. code-block:: text

    true_theta: ℝ[3] = [0.5, 0.5, 0.5]
    true_results: ℝ[2, m] = solver(true_theta)
    true_X: ℝ[m] = true_results[0]
    true_S: ℝ[m] = true_results[1]


Step 5: Adjoint Gradient from Both Observables
----------------------------------------------

We fit **both** the biomass and the substrate curves with a mean-squared-error
loss, and compute its gradient with the adjoint (reverse-mode) method. The
co-state is seeded at the terminal step and propagated backwards with a per-step
running cost; because both states are observed, the residual lives on both
components:

.. math::

    s_k = \Big[\,\tfrac{1}{m}\big(X_k - X_k^{\mathrm{true}}\big),\;
                \tfrac{1}{m}\big(S_k - S_k^{\mathrm{true}}\big)\,\Big]
          + s_{k+1}\, J_{\mathrm{state}}(y_k),

where the RK4 Jacobians come from ``grad`` and the parameter gradient accumulates
:math:`L \mathrel{+}= s\,J_\theta` along the sweep:

.. code-block:: text

    def adjoint_grad(θ: ℝ[3]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        X_array: ℝ[m] = states[0]
        S_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(X_array)
        s: ℝ[2] = [
            (X_array[m-1] - true_X[m-1]) / m,
            (S_array[m-1] - true_S[m-1]) / m
        ]
        L: ℝ[3] = zero_1d_array(3)
        for i:ℕ(m-1):
            idx = m - 2 - i
            X = X_array[idx]
            S = S_array[idx]
            state = [X, S]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [
                (X_array[idx] - true_X[idx]) / m,
                (S_array[idx] - true_S[idx]) / m
            ]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Adam
-----------------------

The three parameters are on comparable scales, but the yield enters through a
division (:math:`X/Y`), so as in the sibling tutorials we hand-roll a
bias-corrected Adam step for scale-free updates, starting from an off-target
guess:

.. code-block:: text

    θ: ℝ[3] = [0.7, 0.2, 0.7]
    learning_rate: ℝ = 0.03
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[3] = [0.0, 0.0, 0.0]
    v_adam: ℝ[3] = [0.0, 0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 2000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        θ = θ - learning_rate * mhat / (sqrt(vhat) + eps_adam)

    pred_results = solver(θ)

.. note::
   The committed ``tutorials/chemostat_monod.phyk`` sets ``epochs = 1`` so the
   test suite runs quickly. Raise it (e.g. ``2000``) to actually fit the model.


Step 7: Results
---------------

Fitting both curves, Adam drives the loss down by roughly thirteen orders of
magnitude (:math:`1.9\times10^{1} \to 2.7\times10^{-12}`) and recovers all three
constants **exactly** — :math:`\mu_{\max} = 0.500`, :math:`K_s = 0.500`,
:math:`Y = 0.500` (0.0% error) — from the off-target guess
:math:`[0.7, 0.2, 0.7]`. The learned Monod law is indistinguishable from the
true one, and evaluating it across dilution rates reproduces the reactor's whole
operating diagram: in particular it predicts washout at
:math:`D_{\mathrm{crit}} \approx 0.476`, even though every training point was
taken at the single operating rate :math:`D = 0.3`.

.. figure:: /_static/tutorial_files/output_chemostat_monod.png
   :alt: The Monod growth law recovered from a chemostat startup transient
   :align: center
   :width: 800px

   **A.** The measured startup transient — biomass rising, substrate falling — is
   fit exactly (learned dashed over the true solid curves). **B.** The recovered
   Monod growth law :math:`\mu(S)` matches the truth, with
   :math:`\mu_{\max}` as the saturating rate and :math:`K_s` the half-saturation
   substrate. **C.** Evaluated across dilution rates, the *learned* law predicts
   the steady-state biomass and the washout threshold — a regime never seen at
   the single training rate :math:`D = 0.3`.

A note on **what you must measure**. The half-saturation constant :math:`K_s`
only shapes the dynamics while :math:`S` is comparable to it, which happens
briefly near the end of the transient; the biomass curve alone therefore
constrains it poorly. Fitting biomass on its own recovers :math:`\mu_{\max}` and
:math:`Y` to about 1% but leaves :math:`K_s` loose — two fits from different
starts land at :math:`0.44` and :math:`0.55`, each reproducing the biomass curve
to :math:`10^{-5}`. Observing the substrate as well, as we do here, pins
:math:`K_s` down. It is the same lesson as the rest of the series in a
bioprocess setting: a good fit to one observable is not the same as a recovered
model, and the cure is to measure the quantity that actually carries the
information.

To visualise the fit, add a helper to ``physika/runtime.py`` as in the
FitzHugh-Nagumo tutorial and plot the trajectories of ``true_results`` against
``pred_results``.


Full Code
---------

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def append(x: ℝ[m], var: ℝ): ℝ[n]:
        new_length: ℝ = len(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        len_x: ℕ = get_1d_array_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results

    D: ℝ = 0.3
    Sin: ℝ = 10.0

    def f(state: ℝ[2], θ: ℝ[3]): ℝ[2]:
        X: ℝ = state[0]
        S: ℝ = state[1]
        mumax: ℝ = θ[0]
        Ks: ℝ = θ[1]
        Y: ℝ = θ[2]
        mu: ℝ = mumax * S / (Ks + S)
        dX: ℝ = (mu - D) * X
        dS: ℝ = D * (Sin - S) - mu * X / Y
        return [dX, dS]

    def rk4_step(state: ℝ[2], θ: ℝ[3]): ℝ[2]:
        k1: ℝ[2] = f(state, θ)
        k2_state: ℝ[2] = state + 0.5 * dt * k1
        k2: ℝ[2] = f(k2_state, θ)
        k3_state: ℝ[2] = state + 0.5 * dt * k2
        k3: ℝ[2] = f(k3_state, θ)
        k4_state: ℝ[2] = state + dt * k3
        k4: ℝ[2] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.1
    timesteps: ℝ = 300

    def solver(θ: ℝ[3]): ℝ[2, m]:
        state: ℝ[2] = [0.05, 10.0]
        X_array: ℝ[1] = [0.05]
        S_array: ℝ[1] = [10.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            X = results[0]
            S = results[1]
            X_array = append(X_array, X)
            S_array = append(S_array, S)
            state = results
        return [X_array, S_array]

    true_theta: ℝ[3] = [0.5, 0.5, 0.5]
    true_results: ℝ[2, m] = solver(true_theta)
    true_X: ℝ[m] = true_results[0]
    true_S: ℝ[m] = true_results[1]

    def adjoint_grad(θ: ℝ[3]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        X_array: ℝ[m] = states[0]
        S_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(X_array)
        s: ℝ[2] = [
            (X_array[m-1] - true_X[m-1]) / m,
            (S_array[m-1] - true_S[m-1]) / m
        ]
        L: ℝ[3] = zero_1d_array(3)
        for i:ℕ(m-1):
            idx = m - 2 - i
            X = X_array[idx]
            S = S_array[idx]
            state = [X, S]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [
                (X_array[idx] - true_X[idx]) / m,
                (S_array[idx] - true_S[idx]) / m
            ]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[3] = [0.7, 0.2, 0.7]
    learning_rate: ℝ = 0.03
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[3] = [0.0, 0.0, 0.0]
    v_adam: ℝ[3] = [0.0, 0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 2000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        θ = θ - learning_rate * mhat / (sqrt(vhat) + eps_adam)

    pred_results = solver(θ)


References
----------

- J. Monod, *The growth of bacterial cultures*, Annual Review of Microbiology 3, 371-394 (1949).
- A. Novick and L. Szilard, *Description of the chemostat*, Science 112(2920), 715-716 (1950).
- H. L. Smith and P. Waltman, *The Theory of the Chemostat* (Cambridge University Press, 1995).
