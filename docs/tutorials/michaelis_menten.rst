Parameter Learning for Michaelis–Menten Enzyme Kinetics
=======================================================

In this tutorial we learn the kinetic constants of an enzyme from measured
reaction progress curves. The model is the cornerstone of biochemical kinetics —
**Michaelis–Menten** — and the tutorial makes two points that recur throughout
computational cell modelling. First, it introduces the **quasi-steady-state
approximation (QSSA)**: the enzyme mechanism written in full is *stiff*, and the
QSSA is the standard reduction that removes the stiffness and makes the model
cheap to simulate and differentiate. Second, it is a clean lesson in
**identifiability**: fitting a single progress curve can reproduce the data almost
perfectly while getting the individual rate constants badly wrong — a mechanistic
cousin of the "low loss :math:`\neq` recovered parameters" caution in the
:doc:`/tutorials/hodgkin_huxley` tutorial. The scaffolding (RK4 stepper,
trajectory solver, full-trajectory adjoint, hand-rolled Adam) is identical to the
:doc:`/tutorials/repressilator`, :doc:`/tutorials/sir` and
:doc:`/tutorials/toggle_switch` tutorials; only the model changes.


The mechanism, and why it is stiff
----------------------------------

An enzyme :math:`E` binds substrate :math:`S` to form a complex :math:`C`, which
either falls apart again or turns over to release product :math:`P`:

.. math::

   E + S \;\underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}}\; C
   \;\xrightarrow{k_{\mathrm{cat}}}\; E + P .

As a full mass-action system this is four coupled ODEs. The trouble is that
binding and unbinding (:math:`k_1, k_{-1}`) are *far* faster than turnover
(:math:`k_{\mathrm{cat}}`), so the complex reaches its steady state almost
instantly while the substrate is consumed slowly. That separation of timescales
makes the system **stiff**: with our parameters the Jacobian's fastest and
slowest modes differ by a factor of :math:`\sim 1.3\times10^{4}`, and an explicit
Runge–Kutta solver is stable only for step sizes below
:math:`\approx 3.5\times10^{-5}` — at :math:`dt = 10^{-4}` it diverges outright.

The **quasi-steady-state approximation** removes this. Setting
:math:`dC/dt \approx 0` solves for the complex algebraically,
:math:`C^\ast = E_{\mathrm{tot}}\,S/(K_m + S)`, and collapses the mechanism into
the single Michaelis–Menten rate law

.. math::

   v(S) = \frac{V_{\max}\, S}{K_m + S}, \qquad
   V_{\max} = k_{\mathrm{cat}}\,E_{\mathrm{tot}}, \qquad
   K_m = \frac{k_{-1} + k_{\mathrm{cat}}}{k_1}.

The reduced description is **not stiff** (its dynamics carry no fast binding
mode), is stable at steps :math:`\sim 500\times` larger, and tracks the full
mechanism to about two percent. This is exactly the trade-off that makes larger
pathway and whole-cell models tractable, seen here on the smallest possible
system.


Setting up the learning problem
-------------------------------

We observe **product formation** in three separate experiments that start at
different substrate levels, :math:`S_0 = 1,\,10,\,40`, chosen to straddle
:math:`K_m`. So that experiments at very different scales weigh equally in the
loss, we track the **fractional conversion** :math:`f = P/S_0 \in [0, 1]`. With the
substrate conserved, :math:`S = S_0(1 - f)`, each experiment obeys

.. math::

   \frac{df}{dt} = \frac{V_{\max}\,(1 - f)}{K_m + S_0\,(1 - f)} .

We learn :math:`\theta = [k_{\mathrm{cat}},\, K_m]` with the total enzyme
:math:`E_{\mathrm{tot}} = 1` known, so :math:`V_{\max} = k_{\mathrm{cat}}`. True
values are :math:`k_{\mathrm{cat}} = 100`, :math:`K_m = 5`. The three experiments
are stacked into a single augmented state :math:`[f_1, f_2, f_3]` sharing
:math:`\theta`, exactly as the :doc:`/tutorials/toggle_switch` tutorial stacks two
initial conditions.


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

``f`` takes the three-dimensional state ``[f1, f2, f3]`` (the fractional
conversion in each experiment) and the parameters ``θ`` and returns the
derivatives. The three experiments start at fixed substrate levels
:math:`S_0 = 1, 10, 40`, declared as globals:

.. code-block:: text

    Etot: ℝ = 1.0
    S0_1: ℝ = 1.0
    S0_2: ℝ = 10.0
    S0_3: ℝ = 40.0

    def f(state: ℝ[3], θ: ℝ[2]): ℝ[3]:
        f1: ℝ = state[0]
        f2: ℝ = state[1]
        f3: ℝ = state[2]
        kcat: ℝ = θ[0]
        Km: ℝ = θ[1]
        Vmax: ℝ = kcat * Etot
        df1: ℝ = Vmax * (1.0 - f1) / (Km + S0_1 * (1.0 - f1))
        df2: ℝ = Vmax * (1.0 - f2) / (Km + S0_2 * (1.0 - f2))
        df3: ℝ = Vmax * (1.0 - f3) / (Km + S0_3 * (1.0 - f3))
        return [df1, df2, df3]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge–Kutta method — identical to the
sibling tutorials, over the three-dimensional augmented state:

.. math::

    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

.. code-block:: text

    def rk4_step(state: ℝ[3], θ: ℝ[2]): ℝ[3]:
        k1: ℝ[3] = f(state, θ)
        k2_state: ℝ[3] = state + 0.5 * dt * k1
        k2: ℝ[3] = f(k2_state, θ)
        k3_state: ℝ[3] = state + 0.5 * dt * k2
        k3: ℝ[3] = f(k3_state, θ)
        k4_state: ℝ[3] = state + dt * k3
        k4: ℝ[3] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


Step 3: Build the Trajectory Solver
-----------------------------------

We integrate forward over 200 steps at ``dt = 0.005``, long enough for all three
experiments to run essentially to completion, collecting the conversion curve of
each:

.. code-block:: text

    dt: ℝ = 0.005
    timesteps: ℝ = 200

    def solver(θ: ℝ[2]): ℝ[3, m]:
        state: ℝ[3] = [0.0, 0.0, 0.0]
        f1_array: ℝ[1] = [0.0]
        f2_array: ℝ[1] = [0.0]
        f3_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            f1 = results[0]
            f2 = results[1]
            f3 = results[2]
            f1_array = append(f1_array, f1)
            f2_array = append(f2_array, f2)
            f3_array = append(f3_array, f3)
            state = results
        return [f1_array, f2_array, f3_array]


Step 4: Generate Ground Truth Data
----------------------------------

We generate the three conversion curves at the true kinetic constants:

.. code-block:: text

    true_theta: ℝ[2] = [100.0, 5.0]
    true_results: ℝ[3, m] = solver(true_theta)
    true_f1: ℝ[m] = true_results[0]
    true_f2: ℝ[m] = true_results[1]
    true_f3: ℝ[m] = true_results[2]


Step 5: Adjoint Gradient
------------------------

All three conversion curves are measured, so we fit their combined
mean-squared-error loss

.. math::

    \mathcal{L}(\theta) = \frac{1}{2m} \sum_{k=0}^{m-1} \sum_{j=1}^{3}
        \left( f_{j,k} - f_{j,k}^{\mathrm{true}} \right)^2 ,

and compute its gradient with the adjoint (reverse-mode) method. The co-state is
seeded at the terminal step and propagated backwards with a per-step *running
cost*; here the residual lives on **every** component because all three
experiments are observed:

.. math::

    s_k = \tfrac{1}{m}\big(\mathbf{f}_k - \mathbf{f}_k^{\mathrm{true}}\big)
          + s_{k+1}\, J_{\mathrm{state}}(y_k),

where the RK4 Jacobians come from ``grad`` and the parameter gradient accumulates
:math:`L \mathrel{+}= s\,J_\theta` along the sweep:

.. code-block:: text

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[3, m] = solver(θ)
        f1_array: ℝ[m] = states[0]
        f2_array: ℝ[m] = states[1]
        f3_array: ℝ[m] = states[2]
        m: ℝ = get_1d_array_length(f1_array)
        s: ℝ[3] = [
            (f1_array[m-1] - true_f1[m-1]) / m,
            (f2_array[m-1] - true_f2[m-1]) / m,
            (f3_array[m-1] - true_f3[m-1]) / m
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            f1 = f1_array[idx]
            f2 = f2_array[idx]
            f3 = f3_array[idx]
            state = [f1, f2, f3]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            r1 = (f1_array[idx] - true_f1[idx]) / m
            r2 = (f2_array[idx] - true_f2[idx]) / m
            r3 = (f3_array[idx] - true_f3[idx]) / m
            residual = [r1, r2, r3]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Adam
-----------------------

The two constants sit on very different scales (:math:`k_{\mathrm{cat}} = 100`
versus :math:`K_m = 5`), so, as in the repressilator tutorial, we hand-roll a
bias-corrected Adam step for scale-free updates:

.. code-block:: text

    θ: ℝ[2] = [50.0, 20.0]
    learning_rate: ℝ = 0.2
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[2] = [0.0, 0.0]
    v_adam: ℝ[2] = [0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 6000

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
   The committed ``tutorials/michaelis_menten.phyk`` sets ``epochs = 1`` so the
   test suite runs quickly. Raise it (e.g. ``6000``) to actually fit the model.


Step 7: Results
---------------

Fitting all three conversion curves, Adam drives the loss down by roughly thirty
orders of magnitude and recovers both constants **exactly**
(:math:`k_{\mathrm{cat}}: 100.00`, :math:`K_m: 5.00`; true :math:`100`,
:math:`5`), matching every curve to machine precision (panel A).

.. figure:: /_static/tutorial_files/output_michaelis_menten.png
   :alt: Recovering enzyme kinetics from progress curves, and the low-substrate identifiability trap
   :align: center
   :width: 800px

   **A.** Fitting all three experiments (:math:`S_0 = 1, 10, 40`) recovers
   :math:`k_{\mathrm{cat}}` and :math:`K_m` exactly; the learned curves (dashed)
   sit on top of the data. **B.** Fitting **only** the low-substrate experiment
   (:math:`S_0 = 1 \ll K_m`) reproduces that curve almost perfectly (RMSE
   :math:`\approx 2\times10^{-3}`) yet returns :math:`k_{\mathrm{cat}}` and
   :math:`K_m` about :math:`40\%` too small. **C.** The reason: at low substrate
   only the *ratio* :math:`k_{\mathrm{cat}}/K_m` is constrained, so the estimate
   slides along that line; the true point is pinned only once the experiments
   straddle :math:`K_m`.


Why low substrate is not enough
-------------------------------

The contrast in panels B and C is the lesson. When :math:`S_0 \ll K_m` the rate
law linearises, :math:`v \approx (V_{\max}/K_m)\,S`, so the curve depends only on
the **specificity constant** :math:`k_{\mathrm{cat}}/K_m` — the two parameters are
*structurally non-identifiable* from that experiment. Fitting the
:math:`S_0 = 1` curve alone gives an excellent fit but
:math:`k_{\mathrm{cat}} \approx 63`, :math:`K_m \approx 3.0` (both :math:`\sim
40\%` low), and yet their ratio :math:`21.4 \approx 20` is spot on. Only when the
data include a saturating experiment (:math:`S_0 \gg K_m`, where the rate is
:math:`\approx V_{\max}` and pins :math:`k_{\mathrm{cat}}` on its own) do
:math:`k_{\mathrm{cat}}` and :math:`K_m` separate. This is the enzyme-kinetics
face of a theme running through the series — a good fit does not by itself certify
recovered parameters — and the fix is the familiar one: design experiments that
span the sensitive regime.

To visualise the fit, add a helper to ``physika/runtime.py`` as in the
FitzHugh–Nagumo tutorial and plot the trajectories of ``true_results`` against
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

    Etot: ℝ = 1.0
    S0_1: ℝ = 1.0
    S0_2: ℝ = 10.0
    S0_3: ℝ = 40.0

    def f(state: ℝ[3], θ: ℝ[2]): ℝ[3]:
        f1: ℝ = state[0]
        f2: ℝ = state[1]
        f3: ℝ = state[2]
        kcat: ℝ = θ[0]
        Km: ℝ = θ[1]
        Vmax: ℝ = kcat * Etot
        df1: ℝ = Vmax * (1.0 - f1) / (Km + S0_1 * (1.0 - f1))
        df2: ℝ = Vmax * (1.0 - f2) / (Km + S0_2 * (1.0 - f2))
        df3: ℝ = Vmax * (1.0 - f3) / (Km + S0_3 * (1.0 - f3))
        return [df1, df2, df3]

    def rk4_step(state: ℝ[3], θ: ℝ[2]): ℝ[3]:
        k1: ℝ[3] = f(state, θ)
        k2_state: ℝ[3] = state + 0.5 * dt * k1
        k2: ℝ[3] = f(k2_state, θ)
        k3_state: ℝ[3] = state + 0.5 * dt * k2
        k3: ℝ[3] = f(k3_state, θ)
        k4_state: ℝ[3] = state + dt * k3
        k4: ℝ[3] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.005
    timesteps: ℝ = 200

    def solver(θ: ℝ[2]): ℝ[3, m]:
        state: ℝ[3] = [0.0, 0.0, 0.0]
        f1_array: ℝ[1] = [0.0]
        f2_array: ℝ[1] = [0.0]
        f3_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            f1 = results[0]
            f2 = results[1]
            f3 = results[2]
            f1_array = append(f1_array, f1)
            f2_array = append(f2_array, f2)
            f3_array = append(f3_array, f3)
            state = results
        return [f1_array, f2_array, f3_array]

    true_theta: ℝ[2] = [100.0, 5.0]
    true_results: ℝ[3, m] = solver(true_theta)
    true_f1: ℝ[m] = true_results[0]
    true_f2: ℝ[m] = true_results[1]
    true_f3: ℝ[m] = true_results[2]

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[3, m] = solver(θ)
        f1_array: ℝ[m] = states[0]
        f2_array: ℝ[m] = states[1]
        f3_array: ℝ[m] = states[2]
        m: ℝ = get_1d_array_length(f1_array)
        s: ℝ[3] = [
            (f1_array[m-1] - true_f1[m-1]) / m,
            (f2_array[m-1] - true_f2[m-1]) / m,
            (f3_array[m-1] - true_f3[m-1]) / m
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            f1 = f1_array[idx]
            f2 = f2_array[idx]
            f3 = f3_array[idx]
            state = [f1, f2, f3]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            r1 = (f1_array[idx] - true_f1[idx]) / m
            r2 = (f2_array[idx] - true_f2[idx]) / m
            r3 = (f3_array[idx] - true_f3[idx]) / m
            residual = [r1, r2, r3]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[2] = [50.0, 20.0]
    learning_rate: ℝ = 0.2
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[2] = [0.0, 0.0]
    v_adam: ℝ[2] = [0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 6000

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

- L. Michaelis and M. L. Menten, *Die Kinetik der Invertinwirkung*, Biochemische Zeitschrift 49, 333–369 (1913).
- G. E. Briggs and J. B. S. Haldane, *A note on the kinetics of enzyme action*, Biochemical Journal 19(2), 338–339 (1925) — the quasi-steady-state derivation of the rate law.
- L. A. Segel and M. Slemrod, *The quasi-steady-state assumption: a case study in perturbation*, SIAM Review 31(3), 446–477 (1989).
- A. Cornish-Bowden, *Fundamentals of Enzyme Kinetics*, 4th ed. (Wiley-Blackwell, 2012).
