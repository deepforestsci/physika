Parameter Learning for the Izhikevich Neuron: the Saltation Gradient
====================================================================

Every other ODE in this series is **smooth**, and smoothness is exactly why
plain gradient descent recovers their parameters. The **Izhikevich** spiking
neuron (Izhikevich, 2003) is different: it is a *hybrid* system with a discrete
**spike reset**, and that discontinuity is what makes robot contact dynamics,
integrate-and-fire circuits and every other reset-driven model hard to learn.
This tutorial is a focused study of that difficulty and of the tool that fixes
the *gradient* — the **saltation** (event-aware) derivative — together with an
honest account of what it does *not* fix.

It is a natural sequel to the :doc:`/tutorials/fitzhugh_nagumo` and
:doc:`/tutorials/hodgkin_huxley` tutorials, the smooth spiking models in this
series. The Izhikevich neuron reproduces the same firing patterns with far less
arithmetic — a quadratic voltage equation plus a hard reset — which is what makes
it the standard testbed for differentiating through spikes.

Two lessons run through the tutorial:

1. A fixed-step integrator that applies the reset **at the step boundary**
   quantises each spike time to the ``dt`` grid, turning the loss into a
   staircase whose exact gradient is locally correct but useless for descent.
   The **saltation** construction — locate the crossing *inside* the step, reset
   there, integrate the remainder — removes the staircase, and Physika
   differentiates straight through it with ``grad``. From a good start, log-space
   Adam then recovers the parameter to machine precision.
2. It is still not enough in general. The spike-train loss is **globally
   non-convex** — one basin per spike count — so recovery is start-dependent.
   Differentiability is *necessary but not sufficient*.


The Equations
-------------

.. math::

   \frac{dv}{dt} &= 0.04\,v^2 + 5\,v + 140 - u + I \\
   \frac{du}{dt} &= a\,(b\,v - u) \\
   \text{if } v \ge 30:\quad & v \leftarrow c,\; u \leftarrow u + d

Here :math:`v` is the membrane potential (mV) and :math:`u` a slow recovery
current. The quadratic :math:`0.04 v^2` is the regenerative spike upstroke; left
alone it diverges to :math:`+\infty`, so the **reset** fires the instant
:math:`v` reaches :math:`+30`, snapping :math:`v` back to :math:`c` and bumping
:math:`u` by :math:`d`. We use the regular-spiking parameters
:math:`b = 0.2,\ c = -65,\ d = 8` and a constant drive :math:`I = 10` (which
produces three spikes in the window), and we learn the single recovery rate
:math:`a` with true value :math:`a = 0.02`.


Why the reset breaks naive learning
-----------------------------------

Integrate with a fixed step and check the guard :math:`v \ge 30` only at step
boundaries, and the reset is applied on the *next* grid time after the true
crossing. As a parameter is varied, the true crossing slides continuously, but
the grid time at which the reset lands jumps in whole ``dt`` increments. Each
jump shifts the entire downstream trajectory by one sample, so the
trajectory-MSE loss is a **staircase** in the parameter (Panel A, red). Its exact
gradient faithfully reports the slope of whichever tread you are standing on —
which says nothing about where the true parameter is.

In continuous time the spike times :math:`t_k(\theta)` are smooth functions of
the parameters (implicit function theorem applied to the guard), so the true
trajectory is piecewise-smooth and a *useful* gradient exists. The correction
that accounts for the moving event time is the **saltation matrix**

.. math::

   \Xi = R_x + \frac{\big(f^{+} - R_x\,f^{-}\big)\,n^{\top}}{n^{\top} f^{-}},

with guard :math:`g(v) = v - 30` so :math:`n = \nabla g = [1, 0]`, reset
:math:`R(v,u) = (c,\, u + d)` so :math:`R_x = \left[\begin{smallmatrix} 0 & 0 \\
0 & 1 \end{smallmatrix}\right]`, and :math:`f^{-}, f^{+}` the vector field just
before and after the jump. Rather than assemble :math:`\Xi` by hand, we obtain it
automatically by making the **event time itself differentiable**: integrate up to
the crossing, apply the reset there, then integrate the remainder of the step.
Autograd through that construction *is* the saltation-corrected sensitivity.


Helper functions
----------------

We reuse the dynamic-array helpers from the sibling tutorials
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
        new_length: ℝ = get_1d_array_length(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        len_x: ℕ = get_1d_array_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results


Step 1: The boundary-check solver
---------------------------------

Izhikevich's own C code advances :math:`v` in two half-steps for stability, then
:math:`u` once. The baseline solver checks the guard *after* the full step and
resets on the grid — this is the integrator that produces the staircase loss:

.. code-block:: text

    b: ℝ = 0.2
    c: ℝ = -65.0
    d: ℝ = 8.0
    dt: ℝ = 0.5
    timesteps: ℕ = 200
    I: ℝ = 10.0

    def solver_hard(a: ℝ): ℝ[m]:
        v: ℝ = -65.0
        u: ℝ = b * v
        v_array: ℝ[1] = [-65.0]
        for i:ℕ(timesteps):
            v += 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            v += 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            u += dt * (a * (b * v - u))
            if v >= 30.0:
                v = c
                u = u + d
            v_array = append(v_array, v)
        return v_array


Step 2: The saltation solver
----------------------------

The saltation solver differs only in what happens when a step *crosses* the
threshold. We compute the crossing fraction :math:`\phi = (30 - v)/(v_{\text{full}}
- v)`, integrate for :math:`\phi\,dt` up to the event, apply the reset there, and
integrate the remaining :math:`(1-\phi)\,dt`. Because :math:`\phi` is a smooth
function of the state (and hence of :math:`a`), differentiating through this is
the saltation-corrected gradient — no jump matrix assembled by hand.

.. note::
   The intermediate quantities are declared **before** the loop and only
   reassigned inside it. A new typed declaration (``name: ℝ = ...``) placed
   inside a loop body does not lower correctly today, so hoisting them is both the
   idiom and a small workaround.

.. code-block:: text

    def solver_salt(a: ℝ): ℝ[m]:
        v: ℝ = -65.0
        u: ℝ = b * v
        vfull: ℝ = 0.0
        ufull: ℝ = 0.0
        frac: ℝ = 0.0
        hf: ℝ = 0.0
        vc: ℝ = 0.0
        u_cross: ℝ = 0.0
        hr: ℝ = 0.0
        ur: ℝ = 0.0
        vv: ℝ = 0.0
        v_array: ℝ[1] = [-65.0]
        for i:ℕ(timesteps):
            vfull = v + 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            vfull = vfull + 0.5 * dt * (0.04 * vfull * vfull + 5.0 * vfull + 140.0 - u + I)
            ufull = u + dt * (a * (b * vfull - u))
            if vfull >= 30.0:
                frac = (30.0 - v) / (vfull - v)
                hf = frac * dt
                vc = v + 0.5 * hf * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
                vc = vc + 0.5 * hf * (0.04 * vc * vc + 5.0 * vc + 140.0 - u + I)
                u_cross = u + hf * (a * (b * vc - u))
                hr = (1.0 - frac) * dt
                ur = u_cross + d
                vv = c + 0.5 * hr * (0.04 * c * c + 5.0 * c + 140.0 - ur + I)
                vv = vv + 0.5 * hr * (0.04 * vv * vv + 5.0 * vv + 140.0 - ur + I)
                u = ur + hr * (a * (b * vv - ur))
                v = vv
            else:
                v = vfull
                u = ufull
            v_array = append(v_array, v)
        return v_array


Step 3: Generate the measured spike train
-----------------------------------------

The "data" is the saltation-accurate voltage trace at the true parameter — the
spike train an experiment would record:

.. code-block:: text

    true_a: ℝ = 0.02
    v_data: ℝ[m] = solver_salt(true_a)


Step 4: Two gradients of the same loss
--------------------------------------

Both fits minimise the trajectory MSE against the measured train,

.. math::

    \mathcal{L}(a) = \frac{1}{m} \sum_{k=0}^{m-1}
        \left( v_k(a) - v_k^{\mathrm{data}} \right)^2 ,

and we ask Physika for its gradient with ``grad``. Unlike the smooth siblings,
which hand-roll a reverse-mode adjoint, here we differentiate straight through
the reset — that is the whole point:

.. code-block:: text

    def loss_hard(a: ℝ): ℝ:
        v_pred: ℝ[n] = solver_hard(a)
        L: ℝ = 0.0
        m: ℝ = get_1d_array_length(v_pred)
        for i:ℕ(m):
            L += (v_pred[i] - v_data[i]) ** 2
        return L/m

    def loss_salt(a: ℝ): ℝ:
        v_pred: ℝ[n] = solver_salt(a)
        L: ℝ = 0.0
        m: ℝ = get_1d_array_length(v_pred)
        for i:ℕ(m):
            L += (v_pred[i] - v_data[i]) ** 2
        return L/m

    a_probe: ℝ = 0.025
    grad(loss_hard, a_probe)
    grad(loss_salt, a_probe)

At :math:`a = 0.025` the boundary-check gradient is the slope of a staircase
tread; the saltation gradient (:math:`\approx 1.93\times10^{5}` in double
precision) matches an :math:`h\to 0` finite difference to full precision. Panel A
shows why: the red loss steps, the blue loss is smooth.


Step 5: Recover ``a`` with log-space Adam
-----------------------------------------

We optimise :math:`a` in log space (scale-free and positive) driven by the
saltation gradient, with the chain rule
:math:`\partial \mathcal{L}/\partial(\log a) = a\,\partial\mathcal{L}/\partial a`:

.. code-block:: text

    a_log: ℝ = log(0.040)
    learning_rate: ℝ = 0.004
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 0.0
    epochs: ℕ = 1

    for i:ℕ(epochs):
        a_lin = exp(a_log)
        g = grad(loss_salt, a_lin)
        g_log = g * a_lin
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g_log
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g_log * g_log)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        a_log = a_log - learning_rate * mhat / (sqrt(vhat) + eps_adam)

    a_final: ℝ = exp(a_log)

.. note::
   The committed ``tutorials/izhikevich.phyk`` sets ``epochs = 1`` so the test
   suite runs quickly. Raise it (e.g. ``800``) to actually fit the parameter.


Step 6: Results
---------------

Starting from :math:`a_0 = 0.040`, the saltation gradient drives the loss down by
about twelve orders of magnitude and recovers :math:`a = 0.020000` (true
:math:`0.020`) to six digits — loss :math:`\approx 1.1\times10^{-10}` in the
single precision Physika uses, and :math:`\approx 10^{-24}` in double. The
event-aware gradient does exactly its job.

.. figure:: /_static/tutorial_files/output_izhikevich.png
   :alt: The saltation gradient recovers the Izhikevich parameter, but only from a good start
   :align: center
   :width: 800px

   **A.** Near :math:`a = 0.025` the boundary-check loss (red) is a **staircase**
   — the gradient of a tread carries no useful information — while the saltation
   loss (blue) is **smooth**. (The curves are offset only because they are
   different integrators fit to the same data; the shape is the point.)
   **B.** From a good start the saltation gradient recovers :math:`a` to machine
   precision. **C.** Zoom out and the loss is **globally non-convex**: a scan
   over :math:`a` has a sharp isolated minimum at the truth (star, :math:`L=0`)
   surrounded by many basins, one per spike count (colour). A start at
   :math:`a_0=0.040` funnels into the true basin; a *closer* start at
   :math:`a_0=0.030` gets stuck.

The second panel is the encouraging half; the third is the caution. Making the
reset differentiable removes a real pathology — the staircase — and is genuinely
the right primitive for hybrid systems, so much so that it is exactly what
trajectory optimisation and control for **robot contact dynamics** need, where
the search is initialised near a feasible trajectory and never sees the global
wall. But an exact gradient is not a global optimiser. Fitting a spike train from
an arbitrary start is a non-convex problem with one basin per spike count; it
needs multi-start, continuation, or a loss that does not depend on exact spike
alignment. Differentiability is necessary, not sufficient — the same lesson the
:doc:`/tutorials/hodgkin_huxley` tutorial reaches for conductances, sharpened
here by a discontinuity.

To visualise the fit, add a helper to ``physika/runtime.py`` as in the
FitzHugh–Nagumo tutorial and plot ``v_data`` against ``solver_salt(a_final)``.


Full Code
---------

.. code-block:: text

    # --- array helpers (same idiom as the sibling ODE tutorials) ---
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
        new_length: ℝ = get_1d_array_length(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        len_x: ℕ = get_1d_array_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results

    # --- fixed Izhikevich regular-spiking (RS) constants; only `a` is learnable ---
    b: ℝ = 0.2
    c: ℝ = -65.0
    d: ℝ = 8.0
    dt: ℝ = 0.5
    timesteps: ℕ = 200
    I: ℝ = 10.0          # suprathreshold drive -> repetitive firing (3 spikes)

    # --- baseline: guard checked at step boundaries -> spike time on the dt grid ---
    def solver_hard(a: ℝ): ℝ[m]:
        v: ℝ = -65.0
        u: ℝ = b * v
        v_array: ℝ[1] = [-65.0]
        for i:ℕ(timesteps):
            v += 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            v += 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            u += dt * (a * (b * v - u))
            if v >= 30.0:
                v = c
                u = u + d
            v_array = append(v_array, v)
        return v_array

    # --- saltation: locate the crossing inside the step, reset there, finish the step ---
    def solver_salt(a: ℝ): ℝ[m]:
        v: ℝ = -65.0
        u: ℝ = b * v
        vfull: ℝ = 0.0
        ufull: ℝ = 0.0
        frac: ℝ = 0.0
        hf: ℝ = 0.0
        vc: ℝ = 0.0
        u_cross: ℝ = 0.0
        hr: ℝ = 0.0
        ur: ℝ = 0.0
        vv: ℝ = 0.0
        v_array: ℝ[1] = [-65.0]
        for i:ℕ(timesteps):
            vfull = v + 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            vfull = vfull + 0.5 * dt * (0.04 * vfull * vfull + 5.0 * vfull + 140.0 - u + I)
            ufull = u + dt * (a * (b * vfull - u))
            if vfull >= 30.0:
                frac = (30.0 - v) / (vfull - v)
                hf = frac * dt
                vc = v + 0.5 * hf * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
                vc = vc + 0.5 * hf * (0.04 * vc * vc + 5.0 * vc + 140.0 - u + I)
                u_cross = u + hf * (a * (b * vc - u))
                hr = (1.0 - frac) * dt
                ur = u_cross + d
                vv = c + 0.5 * hr * (0.04 * c * c + 5.0 * c + 140.0 - ur + I)
                vv = vv + 0.5 * hr * (0.04 * vv * vv + 5.0 * vv + 140.0 - ur + I)
                u = ur + hr * (a * (b * vv - ur))
                v = vv
            else:
                v = vfull
                u = ufull
            v_array = append(v_array, v)
        return v_array

    # --- the "measured" spike train: the saltation-accurate trajectory at the true a ---
    true_a: ℝ = 0.02
    v_data: ℝ[m] = solver_salt(true_a)

    def loss_hard(a: ℝ): ℝ:
        v_pred: ℝ[n] = solver_hard(a)
        L: ℝ = 0.0
        m: ℝ = get_1d_array_length(v_pred)
        for i:ℕ(m):
            L += (v_pred[i] - v_data[i]) ** 2
        return L/m

    def loss_salt(a: ℝ): ℝ:
        v_pred: ℝ[n] = solver_salt(a)
        L: ℝ = 0.0
        m: ℝ = get_1d_array_length(v_pred)
        for i:ℕ(m):
            L += (v_pred[i] - v_data[i]) ** 2
        return L/m

    # --- Lesson 1: the boundary-check gradient vs the saltation gradient at a=0.025 ---
    a_probe: ℝ = 0.025
    grad(loss_hard, a_probe)
    grad(loss_salt, a_probe)

    # --- Lesson 1 (cont.): recover `a` with log-space Adam on the saltation gradient ---
    a_log: ℝ = log(0.040)
    learning_rate: ℝ = 0.004
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 0.0
    epochs: ℕ = 1

    for i:ℕ(epochs):
        a_lin = exp(a_log)
        g = grad(loss_salt, a_lin)
        g_log = g * a_lin
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g_log
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g_log * g_log)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        a_log = a_log - learning_rate * mhat / (sqrt(vhat) + eps_adam)

    a_final: ℝ = exp(a_log)
    print(a_final)


References
----------

- E. M. Izhikevich, *Simple model of spiking neurons*, IEEE Transactions on Neural Networks 14(6), 1569–1572 (2003).
- E. M. Izhikevich, *Dynamical Systems in Neuroscience* (MIT Press, 2007).
- I. A. Hiskens and M. A. Pai, *Trajectory sensitivity analysis of hybrid systems*, IEEE Transactions on Circuits and Systems I, 47(2), 204–220 (2000) — the saltation matrix for state jumps.
