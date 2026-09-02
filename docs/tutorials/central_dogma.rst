Parameter Learning for the Central Dogma of Molecular Biology
=============================================================

In this tutorial we learn the kinetic parameters of the **central dogma** — the
production line that runs in every cell, in which DNA is transcribed into
messenger RNA and mRNA is translated into protein. A minimal two-stage model
tracks the mRNA level :math:`m(t)` and the protein level :math:`p(t)` of a single
gene, governed by a transcription rate :math:`k_{tx}`, a translation rate
:math:`k_{tl}`, and first-order decay rates :math:`\gamma_m, \gamma_p`. It is a
direct sibling of the :doc:`/tutorials/sir` and :doc:`/tutorials/repressilator`
tutorials — the scaffolding (RK4 stepper, trajectory solver, full-trajectory
adjoint, hand-rolled Adam) is identical, and only the model changes.

The idea worth reading carefully here is **structural identifiability**, and it
is a cautionary one. As in the SIR tutorial we fit a **partial observation**: a
fluorescent reporter measures *protein*, not mRNA, so we fit the :math:`p(t)`
curve alone. But this time a perfect fit is *not* enough. The protein data pin
down the two decay rates and the **product** :math:`k_{tx} k_{tl}` exactly — yet
they cannot separate transcription from translation, no matter how clean the
measurement. The synthesis rates are individually unrecoverable from protein
alone; to identify them you must also measure mRNA. This is precisely the lesson
the `DREAM8 <https://doi.org/10.1371/journal.pcbi.1004096>`_ whole-cell parameter
challenge met at genome scale — teams drove the *fit* error down by many orders
of magnitude while the *parameter* error barely moved — and it is the same
phenomenon as the conductance compensation in the
:doc:`/tutorials/hodgkin_huxley` tutorial: **a low loss is not a recovered
model**.


The Equations
-------------

.. math::

   \frac{dm}{dt} &= k_{tx} - \gamma_m\, m \\
   \frac{dp}{dt} &= k_{tl}\, m - \gamma_p\, p

Transcription produces mRNA at a constant rate :math:`k_{tx}`; each mRNA is
translated into protein at rate :math:`k_{tl}`; both species are removed by
first-order decay (degradation plus growth dilution) at rates :math:`\gamma_m`
and :math:`\gamma_p`. The parameters we learn are
:math:`\theta = [k_{tx}, k_{tl}, \gamma_m, \gamma_p]`, with true values
:math:`k_{tx} = 4`, :math:`k_{tl} = 2`, :math:`\gamma_m = 1`,
:math:`\gamma_p = 0.25`. Starting from an empty cell (:math:`m = p = 0`), mRNA
rises with time constant :math:`1/\gamma_m` and protein follows on the slower
:math:`1/\gamma_p`.


Why protein alone cannot separate transcription from translation
----------------------------------------------------------------

The mRNA equation is linear and independent of protein, so it can be solved in
closed form:

.. math::

   m(t) = \frac{k_{tx}}{\gamma_m}\left(1 - e^{-\gamma_m t}\right).

Substituting this into the protein equation, the term that drives protein is

.. math::

   k_{tl}\, m(t) = \frac{k_{tl}\, k_{tx}}{\gamma_m}\left(1 - e^{-\gamma_m t}\right).

Read that carefully: :math:`k_{tx}` and :math:`k_{tl}` enter the **entire**
protein trajectory only through the *product* :math:`k_{tx} k_{tl}`. Two cells
with very different transcription and translation rates but the same product
produce **identical** protein curves for all time. No protein measurement —
however dense or noise-free — can tell them apart. This is a *structural*
non-identifiability, not a numerical one: only the product
:math:`k_{tx} k_{tl}` and the two decay rates (fixed by the observed timescales)
are recoverable. At steady state the same fact reads
:math:`p_\infty = k_{tx} k_{tl} / (\gamma_m \gamma_p)`.

The escape is experimental, not algorithmic: **measure mRNA too**. The mRNA
plateau :math:`m_\infty = k_{tx}/\gamma_m` fixes :math:`k_{tx}` directly, and the
product then gives :math:`k_{tl}`. We demonstrate both cases below.


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

``f`` takes the two-dimensional state ``[mr, pr]`` (mRNA and protein) and the
parameters ``θ`` and returns the derivatives. The state is named ``mr``/``pr``
rather than ``m``/``p`` because a bare ``m`` is used as the trajectory-length
symbol in the array types below:

.. code-block:: text

    def f(state: ℝ[2], θ: ℝ[4]): ℝ[2]:
        mr: ℝ = state[0]
        pr: ℝ = state[1]
        ktx: ℝ = θ[0]
        ktl: ℝ = θ[1]
        gm: ℝ = θ[2]
        gp: ℝ = θ[3]
        dmr: ℝ = ktx - gm * mr
        dpr: ℝ = ktl * mr - gp * pr
        return [dmr, dpr]


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

    def rk4_step(state: ℝ[2], θ: ℝ[4]): ℝ[2]:
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

We integrate forward over 250 steps from an empty cell (``mr = 0``, ``pr = 0``),
collecting both trajectories:

.. code-block:: text

    dt: ℝ = 0.1
    timesteps: ℝ = 250

    def solver(θ: ℝ[4]): ℝ[2, m]:
        state: ℝ[2] = [0.0, 0.0]
        mr_array: ℝ[1] = [0.0]
        pr_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            mr = results[0]
            pr = results[1]
            mr_array = append(mr_array, mr)
            pr_array = append(pr_array, pr)
            state = results
        return [mr_array, pr_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick biologically-ordered rates (fast mRNA turnover, slow protein turnover)
and generate the trajectories we will try to fit:

.. code-block:: text

    true_theta: ℝ[4] = [4.0, 2.0, 1.0, 0.25]
    true_results: ℝ[2, m] = solver(true_theta)
    true_mr: ℝ[m] = true_results[0]
    true_pr: ℝ[m] = true_results[1]


Step 5: Adjoint Gradient from the Observed Curve
------------------------------------------------

We fit the **protein curve only**, with a mean-squared-error loss over the
:math:`p` samples,

.. math::

    \mathcal{L}(\theta) = \frac{1}{2m} \sum_{k=0}^{m-1}
        \left( p_k - p_k^{\mathrm{true}} \right)^2 ,

and compute its gradient with the adjoint (reverse-mode) method. The co-state is
seeded at the terminal step and propagated backwards with a per-step *running
cost* — but the residual lives **only on the** :math:`p` **component**; the mRNA
entry is ``0.0`` because that state is never observed:

.. math::

    s_k = \Big[\,0,\; \tfrac{1}{m}\big(p_k - p_k^{\mathrm{true}}\big)\,\Big]
          + s_{k+1}\, J_{\mathrm{state}}(y_k),

where the RK4 Jacobians come from ``grad`` and the parameter gradient accumulates
:math:`L \mathrel{+}= s\,J_\theta` along the sweep:

.. code-block:: text

    def adjoint_grad(θ: ℝ[4]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        mr_array: ℝ[m] = states[0]
        pr_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(pr_array)
        s: ℝ[2] = [
            0.0,
            (pr_array[m-1] - true_pr[m-1]) / m
        ]
        L: ℝ[4] = zero_1d_array(4)
        for i:ℕ(m-1):
            idx = m - 2 - i
            mr = mr_array[idx]
            pr = pr_array[idx]
            state = [mr, pr]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [0.0, (pr_array[idx] - true_pr[idx]) / m]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Adam
-----------------------

The transcription rate (:math:`\sim 4`) and the protein decay rate
(:math:`\sim 0.25`) differ by more than an order of magnitude, so — as in the
repressilator tutorial — plain gradient descent is ill-conditioned and we
hand-roll a bias-corrected Adam step, whose per-coordinate scaling makes the
update scale-free:

.. code-block:: text

    θ: ℝ[4] = [3.0, 3.0, 0.7, 0.4]
    learning_rate: ℝ = 0.02
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[4] = [0.0, 0.0, 0.0, 0.0]
    v_adam: ℝ[4] = [0.0, 0.0, 0.0, 0.0]
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
   The committed ``tutorials/central_dogma.phyk`` sets ``epochs = 1`` so the test
   suite runs quickly. Raise it (e.g. ``3000``) to actually fit the model.


Step 7: Results
---------------

Fitting the protein curve alone, Adam drives the loss down by roughly
eight orders of magnitude and the protein fit is visually perfect
(RMSE :math:`\approx 1.2\times10^{-4}`). And yet the two synthesis rates are
**not** recovered:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * - parameter
     - true
     - protein only
     - protein + mRNA
     - identified by protein?
   * - :math:`k_{tx}` (transcription)
     - 4.00
     - 2.83
     - 3.94
     - no
   * - :math:`k_{tl}` (translation)
     - 2.00
     - 2.83
     - 2.01
     - no
   * - :math:`\gamma_m` (mRNA decay)
     - 1.00
     - 1.00
     - 0.98
     - yes
   * - :math:`\gamma_p` (protein decay)
     - 0.25
     - 0.25
     - 0.25
     - yes
   * - :math:`k_{tx}\,k_{tl}` (product)
     - 8.00
     - 8.00
     - 7.92
     - yes

The decay rates and the product :math:`k_{tx} k_{tl}` come back essentially
exactly, but :math:`k_{tx}` and :math:`k_{tl}` land far from their true values —
and starting the optimiser elsewhere makes them land somewhere *else* on the same
hyperbola :math:`k_{tx} k_{tl} = 8`. The moment we also feed the optimiser the
mRNA curve (the *protein + mRNA* column), all four parameters snap to their true
values.

.. figure:: /_static/tutorial_files/output_central_dogma.png
   :alt: A perfect protein fit does not recover the transcription and translation rates
   :align: center
   :width: 800px

   **A.** The measured protein curve is fit essentially perfectly. **B.** The
   *same* fitted model gets the unobserved mRNA level wrong by 29%
   — the tell-tale sign of an unidentified model. **C.** The transcription and
   translation rates are structurally non-identifiable from protein: two
   different starts (A, B) converge to different points on the same product
   hyperbola :math:`k_{tx} k_{tl} = 8`, while fitting protein **and** mRNA
   (green) recovers the true point.

The take-away is the mirror image of the SIR tutorial. There, a single
well-chosen partial observation was enough to recover the whole model; here, an
equally perfect fit to a single observable leaves half the parameters
undetermined, because they enter the observed dynamics only as a product. Which
parameters a measurement can identify is a property of the *model and the
observable together* — and checking it (by probing whether different fits give
the same parameters, or the same predictions on an *unmeasured* quantity) is as
important as achieving a low loss. This is the identifiability problem that makes
whole-cell model calibration hard, exactly as DREAM8 reported.

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

    def f(state: ℝ[2], θ: ℝ[4]): ℝ[2]:
        mr: ℝ = state[0]
        pr: ℝ = state[1]
        ktx: ℝ = θ[0]
        ktl: ℝ = θ[1]
        gm: ℝ = θ[2]
        gp: ℝ = θ[3]
        dmr: ℝ = ktx - gm * mr
        dpr: ℝ = ktl * mr - gp * pr
        return [dmr, dpr]

    def rk4_step(state: ℝ[2], θ: ℝ[4]): ℝ[2]:
        k1: ℝ[2] = f(state, θ)
        k2_state: ℝ[2] = state + 0.5 * dt * k1
        k2: ℝ[2] = f(k2_state, θ)
        k3_state: ℝ[2] = state + 0.5 * dt * k2
        k3: ℝ[2] = f(k3_state, θ)
        k4_state: ℝ[2] = state + dt * k3
        k4: ℝ[2] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.1
    timesteps: ℝ = 250

    def solver(θ: ℝ[4]): ℝ[2, m]:
        state: ℝ[2] = [0.0, 0.0]
        mr_array: ℝ[1] = [0.0]
        pr_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            mr = results[0]
            pr = results[1]
            mr_array = append(mr_array, mr)
            pr_array = append(pr_array, pr)
            state = results
        return [mr_array, pr_array]

    true_theta: ℝ[4] = [4.0, 2.0, 1.0, 0.25]
    true_results: ℝ[2, m] = solver(true_theta)
    true_mr: ℝ[m] = true_results[0]
    true_pr: ℝ[m] = true_results[1]

    def adjoint_grad(θ: ℝ[4]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        mr_array: ℝ[m] = states[0]
        pr_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(pr_array)
        s: ℝ[2] = [
            0.0,
            (pr_array[m-1] - true_pr[m-1]) / m
        ]
        L: ℝ[4] = zero_1d_array(4)
        for i:ℕ(m-1):
            idx = m - 2 - i
            mr = mr_array[idx]
            pr = pr_array[idx]
            state = [mr, pr]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [0.0, (pr_array[idx] - true_pr[idx]) / m]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[4] = [3.0, 3.0, 0.7, 0.4]
    learning_rate: ℝ = 0.02
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[4] = [0.0, 0.0, 0.0, 0.0]
    v_adam: ℝ[4] = [0.0, 0.0, 0.0, 0.0]
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

- B. Alberts et al., *Molecular Biology of the Cell*, 6th ed. (Garland Science, 2014) — the central dogma and gene expression kinetics.
- U. Alon, *An Introduction to Systems Biology*, 2nd ed. (CRC Press, 2019) — mRNA/protein dynamics and steady state.
- J. R. Karr et al., *Summary of the DREAM8 Parameter Estimation Challenge: Toward Parameter Identification for Whole-Cell Models*, PLoS Comput. Biol. 11(5):e1004096 (2015).
