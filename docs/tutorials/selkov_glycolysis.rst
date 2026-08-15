Parameter Learning for the Selkov Glycolytic Oscillator
=======================================================

In this tutorial we learn the kinetic parameters of the **Selkov model** — the
classic two-variable caricature of *glycolytic oscillations* (Sel'kov, 1968).
Glycolysis, the pathway that breaks down sugar for energy, does not always run at
a steady rate: in yeast the concentrations of its intermediates rise and fall in
a sustained rhythm, driven by the allosteric enzyme phosphofructokinase, whose
product activates it. The Selkov model captures this with two metabolites and a
single feedback nonlinearity. It is a direct sibling of the
:doc:`/tutorials/repressilator` and :doc:`/tutorials/sir` tutorials — the
scaffolding (RK4 stepper, trajectory solver, full-trajectory adjoint, hand-rolled
Adam) is identical, and only the model changes.

Two things make it worth reading. First, it is our first **metabolic** model, and
the dynamics is a genuine **limit cycle** — a sustained oscillation the optimiser
must match in frequency, amplitude *and* phase, not a curve that settles to a
steady state. Second, as in the SIR tutorial the observation is **partial**: a
real experiment tracks a single metabolite, so we fit one measured curve alone.
Because the orbit is periodic, that single curve re-measures the same dynamics on
every cycle, and it turns out to be more than enough — both kinetic parameters
come back essentially exactly, and with them the *entire* oscillation, including
the metabolite we never measured.


The Equations
-------------

.. math::

   \frac{dx}{dt} &= a\,y + x^2 y - x \\
   \frac{dy}{dt} &= b - a\,y - x^2 y

Here :math:`x` is the activator (ADP) and :math:`y` the substrate (fructose-6-
phosphate). The nonlinear term :math:`x^2 y` is the allosteric feedback — a Hill
cooperativity of two, in which the product :math:`x` accelerates its own
production. Substrate is supplied at the constant rate :math:`b` and consumed by
the reaction; the activator is produced by the reaction and removed at unit rate.
The parameters we learn are :math:`\theta = [a, b]`, with true values
:math:`a = 0.1`, :math:`b = 0.6`.


Why it oscillates
-----------------

The model has a single steady state,

.. math::

   (x^\ast, y^\ast) = \left(b,\; \frac{b}{a + b^2}\right),

found by setting both derivatives to zero. Linearising there, the steady state is
*unstable* whenever the trace of the Jacobian is positive,

.. math::

   \tau = \frac{2b^2}{a + b^2} - b^2 - a - 1 > 0,

which at :math:`a = 0.1,\, b = 0.6` gives :math:`\tau \approx +0.1`. An unstable
steady state with bounded dynamics forces the trajectory onto a **limit cycle**
(Poincaré–Bendixson): the sustained glycolytic oscillation we fit below.


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

``f`` takes the two-dimensional state ``[x, y]`` and the parameters ``θ`` and
returns the derivatives. The activator equation is written with the removal term
last so it has no leading sign:

.. code-block:: text

    def f(state: ℝ[2], θ: ℝ[2]): ℝ[2]:
        x: ℝ = state[0]
        y: ℝ = state[1]
        a: ℝ = θ[0]
        b: ℝ = θ[1]
        dx: ℝ = a * y + x ** 2 * y - x
        dy: ℝ = b - a * y - x ** 2 * y
        return [dx, dy]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge–Kutta method — identical to the
sibling tutorials, over the two-dimensional state:

.. math::

    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

.. code-block:: text

    def rk4_step(state: ℝ[2], θ: ℝ[2]): ℝ[2]:
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

We integrate forward over 600 steps at ``dt = 0.05`` (a 30-unit window spanning a
few oscillation periods), collecting both metabolite trajectories:

.. code-block:: text

    dt: ℝ = 0.05
    timesteps: ℝ = 600

    def solver(θ: ℝ[2]): ℝ[2, m]:
        state: ℝ[2] = [1.0, 1.0]
        x_array: ℝ[1] = [1.0]
        y_array: ℝ[1] = [1.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            x = results[0]
            y = results[1]
            x_array = append(x_array, x)
            y_array = append(y_array, y)
            state = results
        return [x_array, y_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick parameters in the oscillatory regime and generate the trajectories we
will try to fit:

.. code-block:: text

    true_theta: ℝ[2] = [0.1, 0.6]
    true_results: ℝ[2, m] = solver(true_theta)
    true_x: ℝ[m] = true_results[0]
    true_y: ℝ[m] = true_results[1]


Step 5: Adjoint Gradient from the Observed Metabolite
-----------------------------------------------------

We fit the **activator curve** :math:`x(t)` **only**, with a mean-squared-error
loss over its samples,

.. math::

    \mathcal{L}(\theta) = \frac{1}{2m} \sum_{k=0}^{m-1}
        \left( x_k - x_k^{\mathrm{true}} \right)^2 ,

and compute its gradient with the adjoint (reverse-mode) method. The co-state is
seeded at the terminal step and propagated backwards with a per-step *running
cost* — but the residual lives **only on the** :math:`x` **component**; the
:math:`y` entry is ``0.0`` because that metabolite is never observed:

.. math::

    s_k = \Big[\,\tfrac{1}{m}\big(x_k - x_k^{\mathrm{true}}\big),\; 0\,\Big]
          + s_{k+1}\, J_{\mathrm{state}}(y_k),

where the RK4 Jacobians come from ``grad`` and the parameter gradient accumulates
:math:`L \mathrel{+}= s\,J_\theta` along the sweep:

.. code-block:: text

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        x_array: ℝ[m] = states[0]
        y_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(x_array)
        s: ℝ[2] = [
            (x_array[m-1] - true_x[m-1]) / m,
            0.0
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            x = x_array[idx]
            y = y_array[idx]
            state = [x, y]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [(x_array[idx] - true_x[idx]) / m, 0.0]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Adam
-----------------------

The two parameters are on comparable scales, but fitting a periodic orbit is a
stiffer optimisation problem than a monotone curve — a small parameter error
shifts the oscillation's phase and compounds over many cycles — so, as in the
repressilator tutorial, we hand-roll a bias-corrected Adam step for robust,
scale-free updates:

.. code-block:: text

    θ: ℝ[2] = [0.2, 0.45]
    learning_rate: ℝ = 0.01
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[2] = [0.0, 0.0]
    v_adam: ℝ[2] = [0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 3000

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
   The committed ``tutorials/selkov_glycolysis.phyk`` sets ``epochs = 1`` so the
   test suite runs quickly. Raise it (e.g. ``3000``) to actually fit the model.


Step 7: Results
---------------

Fitting the activator curve alone, Adam drives the loss down by nearly five
orders of magnitude and recovers both kinetic parameters essentially exactly
(:math:`a: 0.1001`, :math:`b: 0.6001`; true :math:`0.1`, :math:`0.6`). The fit to
the measured metabolite is visually perfect (RMSE :math:`\approx 1.1\times10^{-3}`)
— and because the parameters are now right, the **unobserved** metabolite
:math:`y(t)` comes back too, to RMSE :math:`\approx 1.9\times10^{-3}`, even though
it never entered the loss:

.. figure:: /_static/tutorial_files/output_selkov_glycolysis.png
   :alt: The glycolytic limit cycle recovered from a single measured metabolite
   :align: center
   :width: 800px

   **A.** The measured metabolite :math:`x(t)` is fit essentially perfectly.
   **B.** The *unobserved* metabolite :math:`y(t)`, which never entered the loss,
   is recovered too — the signature of an identifiable model. **C.** In the phase
   plane the trajectory is a closed **limit cycle**; starting from a wrong guess
   (grey) the optimiser recovers the true orbit (learned dashed over true solid).

This is the encouraging counterpart to the identifiability cautions elsewhere in
this series: here a single partial observation genuinely *does* pin the whole
model down. The reason is the same one that makes the SIR infected curve so
informative — the unobserved metabolite still shapes the dynamics of the one we
measure, so the adjoint threads sensitivity back through it — amplified by
periodicity: every cycle is another independent look at the same kinetics, and
the frequency, amplitude and waveform of a single oscillating metabolite tightly
constrain both rate constants.

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

    def f(state: ℝ[2], θ: ℝ[2]): ℝ[2]:
        x: ℝ = state[0]
        y: ℝ = state[1]
        a: ℝ = θ[0]
        b: ℝ = θ[1]
        dx: ℝ = a * y + x ** 2 * y - x
        dy: ℝ = b - a * y - x ** 2 * y
        return [dx, dy]

    def rk4_step(state: ℝ[2], θ: ℝ[2]): ℝ[2]:
        k1: ℝ[2] = f(state, θ)
        k2_state: ℝ[2] = state + 0.5 * dt * k1
        k2: ℝ[2] = f(k2_state, θ)
        k3_state: ℝ[2] = state + 0.5 * dt * k2
        k3: ℝ[2] = f(k3_state, θ)
        k4_state: ℝ[2] = state + dt * k3
        k4: ℝ[2] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.05
    timesteps: ℝ = 600

    def solver(θ: ℝ[2]): ℝ[2, m]:
        state: ℝ[2] = [1.0, 1.0]
        x_array: ℝ[1] = [1.0]
        y_array: ℝ[1] = [1.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            x = results[0]
            y = results[1]
            x_array = append(x_array, x)
            y_array = append(y_array, y)
            state = results
        return [x_array, y_array]

    true_theta: ℝ[2] = [0.1, 0.6]
    true_results: ℝ[2, m] = solver(true_theta)
    true_x: ℝ[m] = true_results[0]
    true_y: ℝ[m] = true_results[1]

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        x_array: ℝ[m] = states[0]
        y_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(x_array)
        s: ℝ[2] = [
            (x_array[m-1] - true_x[m-1]) / m,
            0.0
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            x = x_array[idx]
            y = y_array[idx]
            state = [x, y]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [(x_array[idx] - true_x[idx]) / m, 0.0]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[2] = [0.2, 0.45]
    learning_rate: ℝ = 0.01
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[2] = [0.0, 0.0]
    v_adam: ℝ[2] = [0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 3000

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

- E. E. Sel'kov, *Self-oscillations in glycolysis: a simple kinetic model*, European Journal of Biochemistry 4(1), 79–86 (1968).
- S. H. Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed. (Westview Press, 2015) — the Selkov model as a limit-cycle example.
- A. Goldbeter, *Biochemical Oscillations and Cellular Rhythms* (Cambridge University Press, 1996).
