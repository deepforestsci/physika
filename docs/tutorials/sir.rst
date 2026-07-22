Parameter Learning for the SIR Epidemic Model
==============================================

In this tutorial we learn the parameters of the **SIR model** — the classic
compartmental model of an epidemic (Kermack & McKendrick, 1927). A population is
split into **S**\ usceptible, **I**\ nfected and **R**\ ecovered fractions, and
two rates govern the flow between them: a transmission rate :math:`\beta` and a
recovery rate :math:`\gamma`. It is a direct sibling of the
:doc:`/tutorials/fitzhugh_nagumo` tutorial (and of the Lotka–Volterra and
repressilator tutorials) — the scaffolding (RK4 stepper, trajectory solver,
full-trajectory adjoint) is identical, and only the model changes.

The idea worth reading carefully here is **partial observation**. In a real
outbreak you never measure who is susceptible or who has recovered — you measure
the number of active cases, i.e. :math:`I(t)`. So we deliberately fit against the
infected curve *alone*: the loss and its adjoint co-state touch only the ``I``
component. Remarkably, that single observed curve is enough to recover both rates
— and once you have them, the entire epidemic, including the unobserved ``S`` and
``R`` compartments, comes along for free.


The Equations
-------------

.. math::

   \frac{dS}{dt} = -\beta\, S\, I

   \frac{dI}{dt} = \beta\, S\, I - \gamma\, I

   \frac{dR}{dt} = \gamma\, I

The bilinear term :math:`\beta S I` is mass-action transmission: infections occur
when susceptibles meet infecteds. Infecteds then recover at rate :math:`\gamma`.
The population is conserved (:math:`S + I + R = 1`), so :math:`R` is fully
determined by :math:`S` and :math:`I`. The parameters we learn are
:math:`\theta = [\beta, \gamma]`, with true values :math:`\beta = 0.3`,
:math:`\gamma = 0.1` (a basic reproduction number :math:`R_0 = \beta/\gamma = 3`).


Fitting what you can measure
----------------------------

We only ever observe the infected curve :math:`I(t)`. This is enough to identify
both parameters, because the *shape* of that curve carries the information: its
early exponential growth rate is :math:`\beta - \gamma`, while the peak height
and the speed of the decline fix :math:`\beta` and :math:`\gamma` individually.

Concretely, we use a full-trajectory mean-squared-error loss **evaluated only on
the** :math:`I` **component**, and the adjoint co-state carries a residual on
that component alone — the ``S`` and ``R`` residuals are simply ``0.0``. This is
how a partial observation is expressed in the adjoint: unobserved states
contribute nothing to the loss, but they still propagate sensitivities backwards
through the dynamics, which is exactly why fitting ``I`` alone still constrains
:math:`\beta` and :math:`\gamma`.


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

``f`` takes the three-dimensional state ``[S, I, Rec]`` and the parameters ``θ``
and returns the derivatives. Note that ``R`` for Recovered is spelled ``Rec``
here: a bare ``R`` is reserved by Physika for the real type ``ℝ``, so it cannot
be used as a variable name. Because ``R`` does not appear on any right-hand side,
``f`` only needs to unpack ``S`` and ``I``:

.. code-block:: text

    def f(state: ℝ[3], θ: ℝ[2]): ℝ[3]:
        S: ℝ = state[0]
        I: ℝ = state[1]
        beta: ℝ = θ[0]
        gamma: ℝ = θ[1]
        dS: ℝ = 0.0 - beta * S * I
        dI: ℝ = beta * S * I - gamma * I
        dR: ℝ = gamma * I
        return [dS, dI, dR]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge–Kutta method — identical to the
sibling tutorials, over the three-dimensional state:

.. math::

    \begin{align*}
    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
    \end{align*}

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

We integrate forward over 200 steps from a mostly-susceptible population
(``S = 0.99``, ``I = 0.01``), collecting all three trajectories:

.. code-block:: text

    dt: ℝ = 0.5
    timesteps: ℝ = 200

    def solver(θ: ℝ[2]): ℝ[3, m]:
        state: ℝ[3] = [0.99, 0.01, 0.0]
        S_array: ℝ[1] = [0.99]
        I_array: ℝ[1] = [0.01]
        R_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            S = results[0]
            I = results[1]
            Rec = results[2]
            S_array = append(S_array, S)
            I_array = append(I_array, I)
            R_array = append(R_array, Rec)
            state = results
        return [S_array, I_array, R_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick true rates in the epidemic regime (:math:`R_0 = 3`) and generate the
trajectories we will try to fit:

.. code-block:: text

    true_theta: ℝ[2] = [0.3, 0.1]
    true_results: ℝ[3, m] = solver(true_theta)
    true_S: ℝ[m] = true_results[0]
    true_I: ℝ[m] = true_results[1]
    true_R: ℝ[m] = true_results[2]


Step 5: Adjoint Gradient from the Observed Curve
------------------------------------------------

We fit the **infected curve only**, with a mean-squared-error loss over the
:math:`I` samples,

.. math::

    \mathcal{L}(\theta) = \frac{1}{2m} \sum_{k=0}^{m-1}
        \left( I_k - I_k^{\mathrm{true}} \right)^2 ,

and compute its gradient with the adjoint (reverse-mode) method. The co-state is
seeded at the terminal step and propagated backwards with a per-step *running
cost* — but the residual lives **only on the** :math:`I` **component**; the
``S`` and ``R`` entries are ``0.0`` because those states are never observed:

.. math::

    s_k = \Big[\,0,\; \tfrac{1}{m}\big(I_k - I_k^{\mathrm{true}}\big),\; 0\,\Big]
          + s_{k+1}\, J_{\mathrm{state}}(y_k),

where the RK4 Jacobians come from ``grad`` and the parameter gradient accumulates
:math:`L \mathrel{+}= s\,J_\theta` along the sweep:

.. code-block:: text

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[3, m] = solver(θ)
        S_array: ℝ[m] = states[0]
        I_array: ℝ[m] = states[1]
        R_array: ℝ[m] = states[2]
        m: ℝ = get_1d_array_length(I_array)
        s: ℝ[3] = [
            0.0,
            (I_array[m-1] - true_I[m-1]) / m,
            0.0
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            S = S_array[idx]
            I = I_array[idx]
            Rec = R_array[idx]
            state = [S, I, Rec]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [0.0, (I_array[idx] - true_I[idx]) / m, 0.0]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Gradient Descent
-----------------------------------

The two rates are on comparable scales, so plain gradient descent suffices:

.. code-block:: text

    θ: ℝ[2] = [0.2, 0.2]
    learning_rate: ℝ = 0.5
    epochs: ℕ = 5000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g

    pred_results = solver(θ)

.. note::
   The committed ``tutorials/sir.phyk`` sets ``epochs = 1`` so the test suite
   runs quickly. Raise it (e.g. ``5000``) to actually fit the model.


Step 7: Results
---------------

Fitting the infected curve alone, gradient descent recovers both rates
essentially exactly (:math:`\beta: 0.300`, :math:`\gamma: 0.100`) and the loss
falls to machine precision:

.. figure:: /_static/tutorial_files/output_sir.png
   :alt: SIR full epidemic recovered from the infected curve alone
   :align: center
   :width: 800px

   Left: ground-truth (solid) vs learned (dashed) trajectories. Only the
   infected curve :math:`I(t)` (red) entered the loss, yet the unobserved
   susceptible :math:`S` and recovered :math:`R` compartments are recovered too.
   Right: the trajectory-MSE loss on :math:`I(t)` over training.

The take-away is that a well-chosen partial observation can be as informative as
the full state: because the unobserved compartments still shape the dynamics of
the one we *do* measure, the adjoint threads sensitivity back through them, and a
single measured curve pins down the whole model.

To visualise the fit, add a helper to ``physika/runtime.py`` as in the
FitzHugh–Nagumo tutorial and plot the compartments of ``true_results`` against
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

    def f(state: ℝ[3], θ: ℝ[2]): ℝ[3]:
        S: ℝ = state[0]
        I: ℝ = state[1]
        beta: ℝ = θ[0]
        gamma: ℝ = θ[1]
        dS: ℝ = 0.0 - beta * S * I
        dI: ℝ = beta * S * I - gamma * I
        dR: ℝ = gamma * I
        return [dS, dI, dR]

    def rk4_step(state: ℝ[3], θ: ℝ[2]): ℝ[3]:
        k1: ℝ[3] = f(state, θ)
        k2_state: ℝ[3] = state + 0.5 * dt * k1
        k2: ℝ[3] = f(k2_state, θ)
        k3_state: ℝ[3] = state + 0.5 * dt * k2
        k3: ℝ[3] = f(k3_state, θ)
        k4_state: ℝ[3] = state + dt * k3
        k4: ℝ[3] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.5
    timesteps: ℝ = 200

    def solver(θ: ℝ[2]): ℝ[3, m]:
        state: ℝ[3] = [0.99, 0.01, 0.0]
        S_array: ℝ[1] = [0.99]
        I_array: ℝ[1] = [0.01]
        R_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            S = results[0]
            I = results[1]
            Rec = results[2]
            S_array = append(S_array, S)
            I_array = append(I_array, I)
            R_array = append(R_array, Rec)
            state = results
        return [S_array, I_array, R_array]

    true_theta: ℝ[2] = [0.3, 0.1]
    true_results: ℝ[3, m] = solver(true_theta)
    true_S: ℝ[m] = true_results[0]
    true_I: ℝ[m] = true_results[1]
    true_R: ℝ[m] = true_results[2]

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[3, m] = solver(θ)
        S_array: ℝ[m] = states[0]
        I_array: ℝ[m] = states[1]
        R_array: ℝ[m] = states[2]
        m: ℝ = get_1d_array_length(I_array)
        s: ℝ[3] = [
            0.0,
            (I_array[m-1] - true_I[m-1]) / m,
            0.0
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            S = S_array[idx]
            I = I_array[idx]
            Rec = R_array[idx]
            state = [S, I, Rec]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [0.0, (I_array[idx] - true_I[idx]) / m, 0.0]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[2] = [0.2, 0.2]
    learning_rate: ℝ = 0.5
    epochs: ℕ = 5000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g

    pred_results = solver(θ)


References
----------

- W. O. Kermack and A. G. McKendrick, *A contribution to the mathematical theory of epidemics*, Proc. R. Soc. Lond. A 115, 700–721 (1927).
- `Compartmental models in epidemiology — Wikipedia <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_
