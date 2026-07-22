Parameter Learning for the Genetic Toggle Switch
=================================================

In this tutorial we learn the parameters of the **genetic toggle switch** — the
synthetic bistable circuit of Gardner, Cantor & Collins (*Nature*, 2000). Two
genes mutually repress each other; with cooperative (Hill) repression the
circuit becomes **bistable**, settling into one of two stable states: "gene *u*
on, gene *v* off" or "gene *v* on, gene *u* off". It is a direct sibling of the
:doc:`/tutorials/fitzhugh_nagumo` tutorial (and of the repressilator) — the
scaffolding (RK4 stepper, trajectory solver, full-trajectory
adjoint) is identical, and here the parameters even share a scale, so plain
gradient descent fits them.

The idea worth reading carefully is not the optimiser but the **experimental
design**. A single experiment does not contain enough information to pin down
both parameters, no matter how long you train. The fix is to fit *two*
experiments at once, which we encode as a single augmented system. This is a
recurring theme in systems biology (and in inverse problems generally):
*identifiability is a property of the data you collect, not just of the
optimiser you run.*


The Equations
-------------

Writing :math:`u` and :math:`v` for the two repressor concentrations:

.. math::

   \frac{du}{dt} = \frac{a_1}{1 + v^{\,n}} - u

   \frac{dv}{dt} = \frac{a_2}{1 + u^{\,n}} - v

Each Hill term :math:`a_i/(1 + \cdot^{\,n})` is mutual repression: when one
protein is high, it shuts off production of the other. Here :math:`a_1, a_2`
are the maximal synthesis rates, and :math:`n = 2` is the Hill cooperativity
(fixed) — cooperativity is what makes the switch bistable rather than
graded. The parameters we learn are :math:`\theta = [a_1, a_2]`, with true
values :math:`a_1 = a_2 = 3`.


Why one experiment is not enough
--------------------------------

Suppose we run a single experiment whose initial condition falls into the
"``u`` high, ``v`` low" basin. The trajectory settles with :math:`v \approx 0`.
But :math:`a_2` only ever enters the dynamics through :math:`a_2/(1 + u^{\,n})`,
and with :math:`u` large that term is tiny — so the observed trajectory barely
depends on :math:`a_2`. Concretely, at a wrong guess the loss gradient
:math:`\partial \mathcal{L} / \partial a_2` is roughly **40× smaller** than
:math:`\partial \mathcal{L} / \partial a_1`. The parameter :math:`a_2` is not
strictly *unidentifiable* — with clean data and enough iterations it eventually
crawls to the right value — but it is badly **ill-conditioned**: it converges
far more slowly and is easily swamped by measurement noise.

The other basin ("``v`` high, ``u`` low") is the mirror image: there
:math:`a_2` is well-constrained and :math:`a_1` is the weak one. The remedy is
therefore to **fit both basins together**. We express the two experiments as a
single four-dimensional augmented state :math:`[u_1, v_1, u_2, v_2]`, where
:math:`(u_1, v_1)` starts in one basin and :math:`(u_2, v_2)` in the other. Both
copies share the same :math:`\theta`, so both :math:`a_1` and :math:`a_2` are
now well-constrained by the combined trajectory. Everything downstream — RK4,
solver, adjoint — is unchanged; only the state grows from 2-D to 4-D.


Helper functions
----------------

We reuse the same dynamic-array helpers as the Lotka–Volterra, FitzHugh–Nagumo
and repressilator tutorials (``zero_1d_array`` / ``get_1d_array_length`` /
``append``); Physika arrays are fixed-shape, so ``append`` allocates a new,
one-longer array and copies into it:

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

``f`` takes the four-dimensional augmented state ``[u1, v1, u2, v2]`` — the two
experiments side by side — and the parameters ``θ``, and returns the
derivatives. Note that ``(u1, v1)`` and ``(u2, v2)`` evolve independently but
under the *same* ``a1`` and ``a2``:

.. code-block:: text

    def f(state: ℝ[4], θ: ℝ[2]): ℝ[4]:
        u1: ℝ = state[0]
        v1: ℝ = state[1]
        u2: ℝ = state[2]
        v2: ℝ = state[3]
        a1: ℝ = θ[0]
        a2: ℝ = θ[1]
        du1: ℝ = a1 / (1.0 + v1 ** 2) - u1
        dv1: ℝ = a2 / (1.0 + u1 ** 2) - v1
        du2: ℝ = a1 / (1.0 + v2 ** 2) - u2
        dv2: ℝ = a2 / (1.0 + u2 ** 2) - v2
        return [du1, dv1, du2, dv2]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge–Kutta method — identical to the
sibling tutorials, now over a four-dimensional state:

.. math::

    \begin{align*}
    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
    \end{align*}

.. code-block:: text

    def rk4_step(state: ℝ[4], θ: ℝ[2]): ℝ[4]:
        k1: ℝ[4] = f(state, θ)
        k2_state: ℝ[4] = state + 0.5 * dt * k1
        k2: ℝ[4] = f(k2_state, θ)
        k3_state: ℝ[4] = state + 0.5 * dt * k2
        k3: ℝ[4] = f(k3_state, θ)
        k4_state: ℝ[4] = state + dt * k3
        k4: ℝ[4] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


Step 3: Build the Trajectory Solver
-----------------------------------

We integrate forward over 100 steps, collecting the trajectory of all four
variables. The two initial conditions ``[2.0, 0.1, 0.1, 2.0]`` place experiment
1 in the "``u`` high" basin and experiment 2 in the "``v`` high" basin:

.. code-block:: text

    dt: ℝ = 0.1
    timesteps: ℝ = 100

    def solver(θ: ℝ[2]): ℝ[4, m]:
        state: ℝ[4] = [2.0, 0.1, 0.1, 2.0]
        u1_array: ℝ[1] = [2.0]
        v1_array: ℝ[1] = [0.1]
        u2_array: ℝ[1] = [0.1]
        v2_array: ℝ[1] = [2.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            u1 = results[0]
            v1 = results[1]
            u2 = results[2]
            v2 = results[3]
            u1_array = append(u1_array, u1)
            v1_array = append(v1_array, v1)
            u2_array = append(u2_array, u2)
            v2_array = append(v2_array, v2)
            state = results
        return [u1_array, v1_array, u2_array, v2_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick the true (symmetric) parameters and generate the trajectories we will
try to fit:

.. code-block:: text

    true_theta: ℝ[2] = [3.0, 3.0]
    true_results: ℝ[4, m] = solver(true_theta)
    true_u1: ℝ[m] = true_results[0]
    true_v1: ℝ[m] = true_results[1]
    true_u2: ℝ[m] = true_results[2]
    true_v2: ℝ[m] = true_results[3]


Step 5: Adjoint Gradient with a Trajectory Loss
-----------------------------------------------

Exactly as in the sibling tutorials, we fit the **whole trajectory** with a
mean-squared-error loss over all :math:`m` samples,

.. math::

    \mathcal{L}(\theta) = \frac{1}{2m} \sum_{k=0}^{m-1}
        \left\| y_k - y_k^{\mathrm{true}} \right\|^2 ,

and compute its gradient with the adjoint (reverse-mode) method. The co-state
:math:`s_k = \partial \mathcal{L} / \partial y_k` is seeded at the terminal step
and propagated backwards with a per-step *running cost* added each step:

.. math::

    s_{m-1} = \frac{1}{m}\left(y_{m-1} - y_{m-1}^{\mathrm{true}}\right),
    \qquad
    s_k = \frac{1}{m}\left(y_k - y_k^{\mathrm{true}}\right) + s_{k+1}\,J_{\mathrm{state}}(y_k),

where the RK4 Jacobians come from ``grad``. The parameter gradient accumulates
:math:`L \mathrel{+}= s\,J_\theta` along the sweep — the same structure as the
repressilator, now with a four-component residual (two experiments stacked):

.. code-block:: text

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[4, m] = solver(θ)
        u1_array: ℝ[m] = states[0]
        v1_array: ℝ[m] = states[1]
        u2_array: ℝ[m] = states[2]
        v2_array: ℝ[m] = states[3]
        m: ℝ = get_1d_array_length(u1_array)
        s: ℝ[4] = [
            (u1_array[m-1] - true_u1[m-1]) / m,
            (v1_array[m-1] - true_v1[m-1]) / m,
            (u2_array[m-1] - true_u2[m-1]) / m,
            (v2_array[m-1] - true_v2[m-1]) / m
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            u1 = u1_array[idx]
            v1 = v1_array[idx]
            u2 = u2_array[idx]
            v2 = v2_array[idx]
            state = [u1, v1, u2, v2]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            r1 = (u1_array[idx] - true_u1[idx]) / m
            r2 = (v1_array[idx] - true_v1[idx]) / m
            r3 = (u2_array[idx] - true_u2[idx]) / m
            r4 = (v2_array[idx] - true_v2[idx]) / m
            residual = [r1, r2, r3, r4]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Gradient Descent
-----------------------------------

Because :math:`a_1` and :math:`a_2` share a scale, and because the two-basin
design has made both well-conditioned, plain gradient descent is enough — no
adaptive optimiser needed:

.. code-block:: text

    θ: ℝ[2] = [2.0, 2.0]
    learning_rate: ℝ = 1.0
    epochs: ℕ = 2000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g

    pred_results = solver(θ)

.. note::
   The committed ``tutorials/toggle_switch.phyk`` sets ``epochs = 1`` so the
   test suite runs quickly. Raise it (e.g. ``2000``) to actually fit the model.


Step 7: Results
---------------

Fitting both basins together, gradient descent recovers both parameters to
better than 0.01% (:math:`a_1: 3.000`, :math:`a_2: 3.000`) and the
trajectory-MSE loss falls to machine precision:

.. figure:: /_static/tutorial_files/output_toggle_switch.png
   :alt: Toggle-switch two-basin fit and alpha2 identifiability
   :align: center
   :width: 800px

   Left: phase portrait in the :math:`(u, v)` plane. The two experiments start
   from opposite initial conditions (open circles) and settle into the two
   different stable states (stars) — experiment 1 to "``u`` high, ``v`` low"
   and experiment 2 to the mirror; ground truth is solid, the learned fit
   (dashed) lies on top. Right: recovery of :math:`a_2` versus epoch. Fitting a
   single experiment leaves :math:`a_2` crawling for hundreds of epochs;
   fitting both experiments locks it in almost immediately.

The right panel is the whole point. With one experiment, :math:`a_1` is
recovered quickly but :math:`a_2` — whose gradient is ~40× weaker in that basin
— barely moves for a long time. With two experiments, both parameters are
well-conditioned and converge together. The lesson generalises well beyond this
model: when a parameter is poorly identifiable, the highest-leverage fix is
often a better-designed experiment rather than a better optimiser.

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

    def f(state: ℝ[4], θ: ℝ[2]): ℝ[4]:
        u1: ℝ = state[0]
        v1: ℝ = state[1]
        u2: ℝ = state[2]
        v2: ℝ = state[3]
        a1: ℝ = θ[0]
        a2: ℝ = θ[1]
        du1: ℝ = a1 / (1.0 + v1 ** 2) - u1
        dv1: ℝ = a2 / (1.0 + u1 ** 2) - v1
        du2: ℝ = a1 / (1.0 + v2 ** 2) - u2
        dv2: ℝ = a2 / (1.0 + u2 ** 2) - v2
        return [du1, dv1, du2, dv2]

    def rk4_step(state: ℝ[4], θ: ℝ[2]): ℝ[4]:
        k1: ℝ[4] = f(state, θ)
        k2_state: ℝ[4] = state + 0.5 * dt * k1
        k2: ℝ[4] = f(k2_state, θ)
        k3_state: ℝ[4] = state + 0.5 * dt * k2
        k3: ℝ[4] = f(k3_state, θ)
        k4_state: ℝ[4] = state + dt * k3
        k4: ℝ[4] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.1
    timesteps: ℝ = 100

    def solver(θ: ℝ[2]): ℝ[4, m]:
        state: ℝ[4] = [2.0, 0.1, 0.1, 2.0]
        u1_array: ℝ[1] = [2.0]
        v1_array: ℝ[1] = [0.1]
        u2_array: ℝ[1] = [0.1]
        v2_array: ℝ[1] = [2.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            u1 = results[0]
            v1 = results[1]
            u2 = results[2]
            v2 = results[3]
            u1_array = append(u1_array, u1)
            v1_array = append(v1_array, v1)
            u2_array = append(u2_array, u2)
            v2_array = append(v2_array, v2)
            state = results
        return [u1_array, v1_array, u2_array, v2_array]

    true_theta: ℝ[2] = [3.0, 3.0]
    true_results: ℝ[4, m] = solver(true_theta)
    true_u1: ℝ[m] = true_results[0]
    true_v1: ℝ[m] = true_results[1]
    true_u2: ℝ[m] = true_results[2]
    true_v2: ℝ[m] = true_results[3]

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[4, m] = solver(θ)
        u1_array: ℝ[m] = states[0]
        v1_array: ℝ[m] = states[1]
        u2_array: ℝ[m] = states[2]
        v2_array: ℝ[m] = states[3]
        m: ℝ = get_1d_array_length(u1_array)
        s: ℝ[4] = [
            (u1_array[m-1] - true_u1[m-1]) / m,
            (v1_array[m-1] - true_v1[m-1]) / m,
            (u2_array[m-1] - true_u2[m-1]) / m,
            (v2_array[m-1] - true_v2[m-1]) / m
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            u1 = u1_array[idx]
            v1 = v1_array[idx]
            u2 = u2_array[idx]
            v2 = v2_array[idx]
            state = [u1, v1, u2, v2]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            r1 = (u1_array[idx] - true_u1[idx]) / m
            r2 = (v1_array[idx] - true_v1[idx]) / m
            r3 = (u2_array[idx] - true_u2[idx]) / m
            r4 = (v2_array[idx] - true_v2[idx]) / m
            residual = [r1, r2, r3, r4]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[2] = [2.0, 2.0]
    learning_rate: ℝ = 1.0
    epochs: ℕ = 2000

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g

    pred_results = solver(θ)


References
----------

- T. S. Gardner, C. R. Cantor and J. J. Collins, *Construction of a genetic toggle switch in Escherichia coli*, Nature 403, 339–342 (2000).
- `Multistability — Wikipedia <https://en.wikipedia.org/wiki/Multistability>`_
