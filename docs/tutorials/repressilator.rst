Parameter Learning for the Repressilator
=========================================

In this tutorial we learn the parameters of the **repressilator** — the
synthetic three-gene oscillator of Elowitz & Leibler (*Nature*, 2000). Three
repressor genes are wired in a ring: each protein represses the transcription of
the next gene, and this cyclic negative feedback produces **sustained
oscillations**. It is a direct sibling of the
:doc:`/tutorials/fitzhugh_nagumo` tutorial: the scaffolding (RK4 stepper,
trajectory solver, full-trajectory adjoint) is identical, and only the
right-hand side ``f``, the six-dimensional state, and the *optimiser* change.

The one change worth reading carefully is the optimiser. FitzHugh–Nagumo fits
with plain gradient descent; here the parameters span very different scales
(:math:`a \approx 40` versus :math:`a_0, \beta \approx 1`), so a single learning
rate cannot serve them all. We therefore hand-roll a **bias-corrected Adam**
step — scale-free updates expressed with the same Physika primitives — and
recover all three parameters essentially exactly.


The Equations
-------------

Writing :math:`m_i` for the mRNA and :math:`p_i` for the protein of gene
:math:`i`, with the repressor of gene :math:`i` being protein :math:`j` (the
ring is :math:`1 \leftarrow 3 \leftarrow 2 \leftarrow 1`):

.. math::

   \frac{dm_i}{dt} = a_0 + \frac{a}{1 + p_j^{\,n}} - m_i

   \frac{dp_i}{dt} = \beta\,(m_i - p_i)

The Hill term :math:`a/(1 + p_j^n)` is the repression: when the repressor
:math:`p_j` is high, transcription of gene :math:`i` is shut off. Here
:math:`a` is the maximal transcription rate, :math:`a_0` a small leak,
:math:`\beta` the protein-to-mRNA timescale ratio, and :math:`n = 2` the Hill
cooperativity (fixed). The parameters we learn are
:math:`\theta = [a, a_0, \beta]`.


Helper functions
----------------

We reuse the same dynamic-array helpers as the Lotka–Volterra and FitzHugh–
Nagumo tutorials (``zero_1d_array`` / ``get_1d_array_length`` / ``append``);
Physika arrays are fixed-shape, so ``append`` allocates a new, one-longer array
and copies into it:

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

``f`` takes the six-dimensional state ``[m1, m2, m3, p1, p2, p3]`` and the
parameters ``θ`` and returns the derivatives. Note the cyclic coupling: gene 1
is repressed by ``p3``, gene 2 by ``p1``, gene 3 by ``p2``:

.. code-block:: text

    def f(state: ℝ[6], θ: ℝ[3]): ℝ[6]:
        m1: ℝ = state[0]
        m2: ℝ = state[1]
        m3: ℝ = state[2]
        p1: ℝ = state[3]
        p2: ℝ = state[4]
        p3: ℝ = state[5]
        a: ℝ = θ[0]
        a0: ℝ = θ[1]
        beta: ℝ = θ[2]
        dm1: ℝ = a0 + a / (1.0 + p3 ** 2) - m1
        dm2: ℝ = a0 + a / (1.0 + p1 ** 2) - m2
        dm3: ℝ = a0 + a / (1.0 + p2 ** 2) - m3
        dp1: ℝ = beta * (m1 - p1)
        dp2: ℝ = beta * (m2 - p2)
        dp3: ℝ = beta * (m3 - p3)
        return [dm1, dm2, dm3, dp1, dp2, dp3]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge–Kutta method — identical to the
FitzHugh–Nagumo stepper, now over a six-dimensional state:

.. math::

    \begin{align*}
    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
    \end{align*}

.. code-block:: text

    def rk4_step(state: ℝ[6], θ: ℝ[3]): ℝ[6]:
        k1: ℝ[6] = f(state, θ)
        k2_state: ℝ[6] = state + 0.5 * dt * k1
        k2: ℝ[6] = f(k2_state, θ)
        k3_state: ℝ[6] = state + 0.5 * dt * k2
        k3: ℝ[6] = f(k3_state, θ)
        k4_state: ℝ[6] = state + dt * k3
        k4: ℝ[6] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


Step 3: Build the Trajectory Solver
-----------------------------------

We integrate forward over 200 steps, collecting the trajectory of all six
variables:

.. code-block:: text

    dt: ℝ = 0.1
    timesteps: ℝ = 200

    def solver(θ: ℝ[3]): ℝ[6, m]:
        state: ℝ[6] = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        m1_array: ℝ[1] = [0.0]
        m2_array: ℝ[1] = [0.0]
        m3_array: ℝ[1] = [0.0]
        p1_array: ℝ[1] = [1.0]
        p2_array: ℝ[1] = [2.0]
        p3_array: ℝ[1] = [3.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            m1 = results[0]
            m2 = results[1]
            m3 = results[2]
            p1 = results[3]
            p2 = results[4]
            p3 = results[5]
            m1_array = append(m1_array, m1)
            m2_array = append(m2_array, m2)
            m3_array = append(m3_array, m3)
            p1_array = append(p1_array, p1)
            p2_array = append(p2_array, p2)
            p3_array = append(p3_array, p3)
            state = results
        return [m1_array, m2_array, m3_array, p1_array, p2_array, p3_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick true parameters in the oscillating regime and generate the trajectories
we will try to fit:

.. code-block:: text

    true_theta: ℝ[3] = [40.0, 1.0, 1.0]
    true_results: ℝ[6, m] = solver(true_theta)
    true_m1: ℝ[m] = true_results[0]
    true_m2: ℝ[m] = true_results[1]
    true_m3: ℝ[m] = true_results[2]
    true_p1: ℝ[m] = true_results[3]
    true_p2: ℝ[m] = true_results[4]
    true_p3: ℝ[m] = true_results[5]


Step 5: Adjoint Gradient with a Trajectory Loss
-----------------------------------------------

Exactly as in FitzHugh–Nagumo, we fit the **whole trajectory** with a
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
:math:`L \mathrel{+}= s\,J_\theta` along the sweep — the same six lines as
FitzHugh–Nagumo, now with a six-component residual:

.. code-block:: text

    def adjoint_grad(θ: ℝ[3]): ℝ[n]:
        states: ℝ[6, m] = solver(θ)
        m1_array: ℝ[m] = states[0]
        m2_array: ℝ[m] = states[1]
        m3_array: ℝ[m] = states[2]
        p1_array: ℝ[m] = states[3]
        p2_array: ℝ[m] = states[4]
        p3_array: ℝ[m] = states[5]
        m: ℝ = get_1d_array_length(m1_array)
        s: ℝ[6] = [
            (m1_array[m-1] - true_m1[m-1]) / m,
            (m2_array[m-1] - true_m2[m-1]) / m,
            (m3_array[m-1] - true_m3[m-1]) / m,
            (p1_array[m-1] - true_p1[m-1]) / m,
            (p2_array[m-1] - true_p2[m-1]) / m,
            (p3_array[m-1] - true_p3[m-1]) / m
        ]
        L: ℝ[3] = zero_1d_array(3)
        for i:ℕ(m-1):
            idx = m - 2 - i
            m1 = m1_array[idx]
            m2 = m2_array[idx]
            m3 = m3_array[idx]
            p1 = p1_array[idx]
            p2 = p2_array[idx]
            p3 = p3_array[idx]
            state = [m1, m2, m3, p1, p2, p3]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            r1 = (m1_array[idx] - true_m1[idx]) / m
            r2 = (m2_array[idx] - true_m2[idx]) / m
            r3 = (m3_array[idx] - true_m3[idx]) / m
            r4 = (p1_array[idx] - true_p1[idx]) / m
            r5 = (p2_array[idx] - true_p2[idx]) / m
            r6 = (p3_array[idx] - true_p3[idx]) / m
            residual = [r1, r2, r3, r4, r5, r6]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Adam
-----------------------

Here is the one real departure from FitzHugh–Nagumo. The transcription rate
:math:`a \approx 40` is far larger than :math:`a_0` and :math:`\beta \approx 1`,
and its gradient is correspondingly *smaller* (the Hill term dilutes it). Under
plain gradient descent a single learning rate either crawls on :math:`a` or
diverges on :math:`\beta`. The standard fix is a **per-parameter adaptive**
step, so we hand-roll **Adam** — Physika expresses it with the same vector
arithmetic and ``sqrt`` we already use:

.. math::

    m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g, \qquad
    v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g^2,

.. math::

    \hat m_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad
    \hat v_t = \frac{v_t}{1-\beta_2^{\,t}}, \qquad
    \theta \mathrel{-}= \text{lr}\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.

Dividing by :math:`\sqrt{\hat v_t}` makes every parameter's step the same order
of magnitude regardless of its gradient scale — exactly what a
disparate-scale problem needs:

.. code-block:: text

    θ: ℝ[3] = [30.0, 1.5, 0.7]
    learning_rate: ℝ = 0.05
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[3] = [0.0, 0.0, 0.0]
    v_adam: ℝ[3] = [0.0, 0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 800

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
   The committed ``tutorials/repressilator.phyk`` sets ``epochs = 1`` so the
   test suite runs quickly. Raise it (e.g. ``800``) to actually fit the model.


Step 7: Results
---------------

After training, the learned trajectory sits exactly on top of the ground truth
and the loss falls by more than six orders of magnitude:

.. figure:: /_static/tutorial_files/output_repressilator.png
   :alt: Repressilator true vs learned trajectories and loss convergence
   :align: center
   :width: 800px

   Left: ground-truth (solid) vs learned (dashed) protein trajectories
   ``p1``, ``p2``, ``p3``; the grey line is the initial guess. Right:
   trajectory-MSE loss over training.

All three parameters are recovered to within 0.05% —
:math:`a: 39.98`, :math:`a_0: 1.00`, :math:`\beta: 1.00` — because the sustained
oscillation makes the whole trajectory sensitive to every parameter. This is
the pay-off of the adaptive optimiser: the same problem stalls under plain
gradient descent but is solved cleanly by the hand-rolled Adam step.

To visualise the fit, add a helper to ``physika/runtime.py`` as in the
FitzHugh–Nagumo tutorial and plot the protein trajectories of ``true_results``
against ``pred_results``.


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

    def f(state: ℝ[6], θ: ℝ[3]): ℝ[6]:
        m1: ℝ = state[0]
        m2: ℝ = state[1]
        m3: ℝ = state[2]
        p1: ℝ = state[3]
        p2: ℝ = state[4]
        p3: ℝ = state[5]
        a: ℝ = θ[0]
        a0: ℝ = θ[1]
        beta: ℝ = θ[2]
        dm1: ℝ = a0 + a / (1.0 + p3 ** 2) - m1
        dm2: ℝ = a0 + a / (1.0 + p1 ** 2) - m2
        dm3: ℝ = a0 + a / (1.0 + p2 ** 2) - m3
        dp1: ℝ = beta * (m1 - p1)
        dp2: ℝ = beta * (m2 - p2)
        dp3: ℝ = beta * (m3 - p3)
        return [dm1, dm2, dm3, dp1, dp2, dp3]

    def rk4_step(state: ℝ[6], θ: ℝ[3]): ℝ[6]:
        k1: ℝ[6] = f(state, θ)
        k2_state: ℝ[6] = state + 0.5 * dt * k1
        k2: ℝ[6] = f(k2_state, θ)
        k3_state: ℝ[6] = state + 0.5 * dt * k2
        k3: ℝ[6] = f(k3_state, θ)
        k4_state: ℝ[6] = state + dt * k3
        k4: ℝ[6] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.1
    timesteps: ℝ = 200

    def solver(θ: ℝ[3]): ℝ[6, m]:
        state: ℝ[6] = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        m1_array: ℝ[1] = [0.0]
        m2_array: ℝ[1] = [0.0]
        m3_array: ℝ[1] = [0.0]
        p1_array: ℝ[1] = [1.0]
        p2_array: ℝ[1] = [2.0]
        p3_array: ℝ[1] = [3.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            m1 = results[0]
            m2 = results[1]
            m3 = results[2]
            p1 = results[3]
            p2 = results[4]
            p3 = results[5]
            m1_array = append(m1_array, m1)
            m2_array = append(m2_array, m2)
            m3_array = append(m3_array, m3)
            p1_array = append(p1_array, p1)
            p2_array = append(p2_array, p2)
            p3_array = append(p3_array, p3)
            state = results
        return [m1_array, m2_array, m3_array, p1_array, p2_array, p3_array]

    true_theta: ℝ[3] = [40.0, 1.0, 1.0]
    true_results: ℝ[6, m] = solver(true_theta)
    true_m1: ℝ[m] = true_results[0]
    true_m2: ℝ[m] = true_results[1]
    true_m3: ℝ[m] = true_results[2]
    true_p1: ℝ[m] = true_results[3]
    true_p2: ℝ[m] = true_results[4]
    true_p3: ℝ[m] = true_results[5]

    def adjoint_grad(θ: ℝ[3]): ℝ[n]:
        states: ℝ[6, m] = solver(θ)
        m1_array: ℝ[m] = states[0]
        m2_array: ℝ[m] = states[1]
        m3_array: ℝ[m] = states[2]
        p1_array: ℝ[m] = states[3]
        p2_array: ℝ[m] = states[4]
        p3_array: ℝ[m] = states[5]
        m: ℝ = get_1d_array_length(m1_array)
        s: ℝ[6] = [
            (m1_array[m-1] - true_m1[m-1]) / m,
            (m2_array[m-1] - true_m2[m-1]) / m,
            (m3_array[m-1] - true_m3[m-1]) / m,
            (p1_array[m-1] - true_p1[m-1]) / m,
            (p2_array[m-1] - true_p2[m-1]) / m,
            (p3_array[m-1] - true_p3[m-1]) / m
        ]
        L: ℝ[3] = zero_1d_array(3)
        for i:ℕ(m-1):
            idx = m - 2 - i
            m1 = m1_array[idx]
            m2 = m2_array[idx]
            m3 = m3_array[idx]
            p1 = p1_array[idx]
            p2 = p2_array[idx]
            p3 = p3_array[idx]
            state = [m1, m2, m3, p1, p2, p3]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            r1 = (m1_array[idx] - true_m1[idx]) / m
            r2 = (m2_array[idx] - true_m2[idx]) / m
            r3 = (m3_array[idx] - true_m3[idx]) / m
            r4 = (p1_array[idx] - true_p1[idx]) / m
            r5 = (p2_array[idx] - true_p2[idx]) / m
            r6 = (p3_array[idx] - true_p3[idx]) / m
            residual = [r1, r2, r3, r4, r5, r6]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[3] = [30.0, 1.5, 0.7]
    learning_rate: ℝ = 0.05
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[3] = [0.0, 0.0, 0.0]
    v_adam: ℝ[3] = [0.0, 0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 800

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

- M. B. Elowitz and S. Leibler, *A synthetic oscillatory network of transcriptional regulators*, Nature 403, 335–338 (2000).
- `Repressilator — Wikipedia <https://en.wikipedia.org/wiki/Repressilator>`_
