Parameter Learning for a Goldbeter-Koshland Ultrasensitive Switch
=================================================================

In this tutorial we learn the kinetic constants of a **Goldbeter-Koshland
covalent-modification cycle** — the elementary building block of cellular
signalling. A single target protein is toggled between an unmodified and a
modified form by two opposing enzymes, a *kinase* and a *phosphatase* (think of
a phosphorylation site switched on and off). Each enzyme obeys Michaelis-Menten
kinetics, yet the pair together can behave as a near-digital **switch**: this is
the famous *zero-order ultrasensitivity* of Goldbeter and Koshland (1981), and
it is how graded signals are sharpened into all-or-none decisions at every tier
of a MAPK cascade.

The scaffolding is the same as its systems-biology siblings — the RK4 stepper,
the trajectory solver, the adjoint sweep and the hand-rolled Adam of the
:doc:`/tutorials/toggle_switch` and :doc:`/tutorials/repressilator` tutorials —
so only the model and the loss change. Two things make it worth reading. First,
it is our first **signalling** model, and its defining feature is the
**ultrasensitive dose-response**: the steady-state modified fraction is a graded
curve when the enzymes are unsaturated but a sharp switch when they are
saturated, with an effective Hill coefficient that grows as the Michaelis
constants shrink. Second, it carries a clean **structural-identifiability**
lesson. We fit the *steady-state* response — the object an experimentalist
actually measures — and a single operating point turns out to fit perfectly yet
return the wrong constants: one steady-state reading is one equation in two
unknowns. Only a dose-response spanning the switch recovers both constants, an
echo of the specificity-constant caution in the Michaelis-Menten tutorial and
the conductance compensation of :doc:`/tutorials/hodgkin_huxley`.


The Equations
-------------

Let :math:`w = W^\ast/W_{\mathrm{tot}}` be the modified fraction of the target
protein (so :math:`1 - w` is the unmodified fraction, and we set
:math:`W_{\mathrm{tot}} = 1`). The kinase, with maximal rate :math:`V_k`, acts on
the unmodified pool; the phosphatase, with maximal rate :math:`V_p`, acts on the
modified pool; each is Michaelis-Menten with constant :math:`K_1`, :math:`K_2`:

.. math::

   \frac{dw}{dt} = \underbrace{V_k \,\frac{1 - w}{K_1 + (1 - w)}}_{\text{modification}}
                 - \underbrace{V_p \,\frac{w}{K_2 + w}}_{\text{demodification}} .

The input signal is the **activity ratio** :math:`u = V_k/V_p` (we fix
:math:`V_p = 1` and vary :math:`V_k`). The parameters we learn are the two
Michaelis constants :math:`\theta = [K_1, K_2]`, with true values
:math:`K_1 = K_2 = 0.1` — both small compared with :math:`W_{\mathrm{tot}} = 1`,
so the enzymes run near saturation and the cycle is ultrasensitive.

The rate law uses only :math:`+\,-\,*\,/`, so it is exactly representable in
Physika with no need for the exponentials that a Hill function would require.


Zero-order ultrasensitivity
----------------------------

Setting :math:`dw/dt = 0` gives the steady-state balance

.. math::

   u \,\frac{1 - w^\ast}{K_1 + (1 - w^\ast)} = \frac{w^\ast}{K_2 + w^\ast},

a quadratic in :math:`w^\ast` whose solution is the **Goldbeter-Koshland
function** :math:`w^\ast(u; K_1, K_2)`. Its shape depends entirely on the size of
the Michaelis constants:

- **Unsaturated** (:math:`K_1, K_2 \gg 1`): each enzyme works far below
  saturation, its rate is roughly proportional to its substrate, and the response
  is *graded* — a hyperbola that rises gently with :math:`u`.
- **Saturated / zero-order** (:math:`K_1, K_2 \ll 1`): each enzyme runs near its
  maximal rate for almost any nonzero substrate, so the two fluxes are nearly
  constant and cross abruptly as :math:`u` passes :math:`1`. The steady state
  then flips from :math:`w^\ast \approx 0` to :math:`w^\ast \approx 1` over a tiny
  change in :math:`u` — a **switch**.

The steepness is captured by an **effective Hill coefficient**
:math:`n_H = \ln 81 / \ln(u_{90}/u_{10})`, the fold-change in input needed to
drive the output from 10% to 90%. It grows sharply as the constants shrink:

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - Michaelis constant :math:`K_1 = K_2`
     - effective :math:`n_H`
   * - :math:`1.0` (unsaturated)
     - :math:`1.33`
   * - :math:`0.3`
     - :math:`2.00`
   * - :math:`0.1` (our model)
     - :math:`3.74`
   * - :math:`0.03`
     - :math:`9.57`
   * - :math:`0.01` (saturated)
     - :math:`26.08`

Remarkably, this ultrasensitivity needs no cooperative binding — a single
monomeric site suffices. It is the *saturation* of the enzymes, set by a small
:math:`K`, that sharpens the switch. Recovering :math:`K_1` and :math:`K_2` is
therefore the same as recovering *how switch-like* the cycle is.


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


Step 1: Define the ODE — seven doses at once
--------------------------------------------

A dose-response is a *set* of experiments: the same cycle driven at several
kinase activities. We run seven of them in parallel by stacking their modified
fractions into one augmented state :math:`[w_1, \dots, w_7]` that shares
:math:`\theta`, with the activities :math:`u_1, \dots, u_7 = 0.7, \dots, 1.3`
straddling the switch midpoint :math:`u = 1`. ``f`` returns the seven
derivatives, each with its own activity but the same :math:`K_1, K_2`:

.. code-block:: text

    Vp: ℝ = 1.0
    u1: ℝ = 0.7
    u2: ℝ = 0.8
    u3: ℝ = 0.9
    u4: ℝ = 1.0
    u5: ℝ = 1.1
    u6: ℝ = 1.2
    u7: ℝ = 1.3

    def f(state: ℝ[7], θ: ℝ[2]): ℝ[7]:
        w1: ℝ = state[0]
        w2: ℝ = state[1]
        w3: ℝ = state[2]
        w4: ℝ = state[3]
        w5: ℝ = state[4]
        w6: ℝ = state[5]
        w7: ℝ = state[6]
        K1: ℝ = θ[0]
        K2: ℝ = θ[1]
        dw1: ℝ = u1 * (1.0 - w1) / (K1 + (1.0 - w1)) - Vp * w1 / (K2 + w1)
        dw2: ℝ = u2 * (1.0 - w2) / (K1 + (1.0 - w2)) - Vp * w2 / (K2 + w2)
        dw3: ℝ = u3 * (1.0 - w3) / (K1 + (1.0 - w3)) - Vp * w3 / (K2 + w3)
        dw4: ℝ = u4 * (1.0 - w4) / (K1 + (1.0 - w4)) - Vp * w4 / (K2 + w4)
        dw5: ℝ = u5 * (1.0 - w5) / (K1 + (1.0 - w5)) - Vp * w5 / (K2 + w5)
        dw6: ℝ = u6 * (1.0 - w6) / (K1 + (1.0 - w6)) - Vp * w6 / (K2 + w6)
        dw7: ℝ = u7 * (1.0 - w7) / (K1 + (1.0 - w7)) - Vp * w7 / (K2 + w7)
        return [dw1, dw2, dw3, dw4, dw5, dw6, dw7]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge-Kutta method, identical to the
sibling tutorials, over the seven-dimensional state:

.. math::

    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

.. code-block:: text

    def rk4_step(state: ℝ[7], θ: ℝ[2]): ℝ[7]:
        k1: ℝ[7] = f(state, θ)
        k2_state: ℝ[7] = state + 0.5 * dt * k1
        k2: ℝ[7] = f(k2_state, θ)
        k3_state: ℝ[7] = state + 0.5 * dt * k2
        k3: ℝ[7] = f(k3_state, θ)
        k4_state: ℝ[7] = state + dt * k3
        k4: ℝ[7] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


Step 3: Build the Trajectory Solver
-----------------------------------

Each cycle starts fully unmodified and relaxes to its steady state. We integrate
200 steps at ``dt = 0.1`` — a 20-unit window, long enough for every dose to
settle — collecting all seven curves:

.. code-block:: text

    dt: ℝ = 0.1
    timesteps: ℝ = 200

    def solver(θ: ℝ[2]): ℝ[7, m]:
        state: ℝ[7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        w1_array: ℝ[1] = [0.0]
        w2_array: ℝ[1] = [0.0]
        w3_array: ℝ[1] = [0.0]
        w4_array: ℝ[1] = [0.0]
        w5_array: ℝ[1] = [0.0]
        w6_array: ℝ[1] = [0.0]
        w7_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            w1 = results[0]
            w2 = results[1]
            w3 = results[2]
            w4 = results[3]
            w5 = results[4]
            w6 = results[5]
            w7 = results[6]
            w1_array = append(w1_array, w1)
            w2_array = append(w2_array, w2)
            w3_array = append(w3_array, w3)
            w4_array = append(w4_array, w4)
            w5_array = append(w5_array, w5)
            w6_array = append(w6_array, w6)
            w7_array = append(w7_array, w7)
            state = results
        return [w1_array, w2_array, w3_array, w4_array, w5_array, w6_array, w7_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick both Michaelis constants small (:math:`0.1 \ll W_{\mathrm{tot}}`), so the
cycle is ultrasensitive, and generate the seven trajectories we will try to fit:

.. code-block:: text

    true_theta: ℝ[2] = [0.1, 0.1]
    true_results: ℝ[7, m] = solver(true_theta)
    true_w1: ℝ[m] = true_results[0]
    true_w2: ℝ[m] = true_results[1]
    true_w3: ℝ[m] = true_results[2]
    true_w4: ℝ[m] = true_results[3]
    true_w5: ℝ[m] = true_results[4]
    true_w6: ℝ[m] = true_results[5]
    true_w7: ℝ[m] = true_results[6]


Step 5: Adjoint Gradient of a Steady-State Loss
-----------------------------------------------

A dose-response experiment measures the *settled* response, so we fit the
**steady-state endpoints only** — a terminal loss over the seven doses,

.. math::

    \mathcal{L}(\theta) = \frac{1}{2\,n_{\mathrm{exp}}} \sum_{i=1}^{n_{\mathrm{exp}}}
        \left( w_i(T) - w_i^{\mathrm{true}}(T) \right)^2 ,
    \qquad n_{\mathrm{exp}} = 7 .

Because the residual lives only at the final step, the adjoint is even simpler
than the running-cost sweep of the sibling tutorials: the co-state is **seeded
once** from the terminal residual and then propagated straight back through the
solver with no further source term,

.. math::

    s_{T} = \frac{1}{n_{\mathrm{exp}}}\big(w(T) - w^{\mathrm{true}}(T)\big),
    \qquad s_{k} = s_{k+1}\, J_{\mathrm{state}}(w_k),

while the parameter gradient accumulates :math:`L \mathrel{+}= s\,J_\theta` along
the sweep. The RK4 Jacobians come from ``grad``:

.. code-block:: text

    n_exp: ℝ = 7.0

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[7, m] = solver(θ)
        w1_array: ℝ[m] = states[0]
        w2_array: ℝ[m] = states[1]
        w3_array: ℝ[m] = states[2]
        w4_array: ℝ[m] = states[3]
        w5_array: ℝ[m] = states[4]
        w6_array: ℝ[m] = states[5]
        w7_array: ℝ[m] = states[6]
        m: ℝ = get_1d_array_length(w1_array)
        s: ℝ[7] = [
            (w1_array[m-1] - true_w1[m-1]) / n_exp,
            (w2_array[m-1] - true_w2[m-1]) / n_exp,
            (w3_array[m-1] - true_w3[m-1]) / n_exp,
            (w4_array[m-1] - true_w4[m-1]) / n_exp,
            (w5_array[m-1] - true_w5[m-1]) / n_exp,
            (w6_array[m-1] - true_w6[m-1]) / n_exp,
            (w7_array[m-1] - true_w7[m-1]) / n_exp
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            w1 = w1_array[idx]
            w2 = w2_array[idx]
            w3 = w3_array[idx]
            w4 = w4_array[idx]
            w5 = w5_array[idx]
            w6 = w6_array[idx]
            w7 = w7_array[idx]
            state = [w1, w2, w3, w4, w5, w6, w7]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            s = s @ J_state
        return L


Step 6: Train with Adam
-----------------------

We start from a deliberately **asymmetric** guess, :math:`[K_1, K_2] = [0.02,
0.5]`, so that recovering the symmetric truth :math:`[0.1, 0.1]` shows both
constants are pinned *separately*, not merely as a ratio. As in the sibling
tutorials we hand-roll a bias-corrected Adam step for scale-free updates:

.. code-block:: text

    θ: ℝ[2] = [0.02, 0.5]
    learning_rate: ℝ = 0.02
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
   The committed ``tutorials/goldbeter_koshland.phyk`` sets ``epochs = 1`` so the
   test suite runs quickly. Raise it (e.g. ``3000``) to actually fit the model.


Step 7: Results
---------------

Fitting the full dose-response, Adam drives the terminal loss down by roughly
fourteen orders of magnitude (:math:`2.4\times10^{-1} \to 2\times10^{-15}`) and
recovers **both** Michaelis constants exactly — :math:`K_1 = 0.100`,
:math:`K_2 = 0.100` (0.0% error) — even though it started from the lopsided guess
:math:`[0.02, 0.5]`. The seven steady-state readings, taken together, pin the
switch's sharpness with no ambiguity.

The instructive failure is to fit **one dose at a time**. A single steady-state
reading is a single number, and :math:`w^\ast(u; K_1, K_2)` is one equation in
two unknowns: the pairs :math:`(K_1, K_2)` consistent with it form an entire
*level curve*. Optimising against one dose therefore drives the loss to **exactly
zero** while landing at the **wrong** constants — and at a *different* wrong pair
from every starting guess. Two runs on the dose :math:`u = 1.3` illustrate it:
both reach loss :math:`0` yet settle at :math:`(0.23, 0.47)` and :math:`(0.18,
0.32)`, two distinct points on the same curve, each reproducing the measured
fraction :math:`w^\ast = 0.785` perfectly. (The symmetric midpoint :math:`u = 1`
is worse still: there :math:`w^\ast = 1/2` for *every* :math:`K`, so it says
nothing at all about the switch.)

.. figure:: /_static/tutorial_files/output_goldbeter_koshland.png
   :alt: Zero-order ultrasensitivity and recovery of the switch constants
   :align: center
   :width: 800px

   **A.** Zero-order ultrasensitivity: the steady-state modified fraction
   :math:`w^\ast(V_k/V_p)` is graded when the enzymes are unsaturated
   (:math:`K = 1`, :math:`n_H \approx 1.3`) but a sharp switch when they are
   saturated (:math:`K = 0.01`, :math:`n_H \approx 26`). **B.** Fitting the whole
   dose-response (seven measured points) recovers both constants exactly from an
   asymmetric start. **C.** Why one dose is not enough: a single steady-state
   reading fixes only a *level curve* in the :math:`(K_1, K_2)` plane (two
   one-dose fits, loss :math:`= 0`, sit at different points on the yellow curve);
   pairing a second dose gives a second curve, and the two intersect only at the
   true :math:`(0.1, 0.1)`.

The moral matches the rest of the series: a low loss is not the same as a
recovered model. Here the fix is pure experimental design — to measure how sharp
a switch is, you must sample *across* the switch, not at a single operating
point. Because the ultrasensitive response is exactly what makes the cycle a
useful signalling element, the dose-response is both the phenomenon worth
studying and the data that identifies it.

To visualise the fit, add a helper to ``physika/runtime.py`` as in the
FitzHugh-Nagumo tutorial and plot the steady states of ``true_results`` against
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

    Vp: ℝ = 1.0
    u1: ℝ = 0.7
    u2: ℝ = 0.8
    u3: ℝ = 0.9
    u4: ℝ = 1.0
    u5: ℝ = 1.1
    u6: ℝ = 1.2
    u7: ℝ = 1.3

    def f(state: ℝ[7], θ: ℝ[2]): ℝ[7]:
        w1: ℝ = state[0]
        w2: ℝ = state[1]
        w3: ℝ = state[2]
        w4: ℝ = state[3]
        w5: ℝ = state[4]
        w6: ℝ = state[5]
        w7: ℝ = state[6]
        K1: ℝ = θ[0]
        K2: ℝ = θ[1]
        dw1: ℝ = u1 * (1.0 - w1) / (K1 + (1.0 - w1)) - Vp * w1 / (K2 + w1)
        dw2: ℝ = u2 * (1.0 - w2) / (K1 + (1.0 - w2)) - Vp * w2 / (K2 + w2)
        dw3: ℝ = u3 * (1.0 - w3) / (K1 + (1.0 - w3)) - Vp * w3 / (K2 + w3)
        dw4: ℝ = u4 * (1.0 - w4) / (K1 + (1.0 - w4)) - Vp * w4 / (K2 + w4)
        dw5: ℝ = u5 * (1.0 - w5) / (K1 + (1.0 - w5)) - Vp * w5 / (K2 + w5)
        dw6: ℝ = u6 * (1.0 - w6) / (K1 + (1.0 - w6)) - Vp * w6 / (K2 + w6)
        dw7: ℝ = u7 * (1.0 - w7) / (K1 + (1.0 - w7)) - Vp * w7 / (K2 + w7)
        return [dw1, dw2, dw3, dw4, dw5, dw6, dw7]

    def rk4_step(state: ℝ[7], θ: ℝ[2]): ℝ[7]:
        k1: ℝ[7] = f(state, θ)
        k2_state: ℝ[7] = state + 0.5 * dt * k1
        k2: ℝ[7] = f(k2_state, θ)
        k3_state: ℝ[7] = state + 0.5 * dt * k2
        k3: ℝ[7] = f(k3_state, θ)
        k4_state: ℝ[7] = state + dt * k3
        k4: ℝ[7] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.1
    timesteps: ℝ = 200

    def solver(θ: ℝ[2]): ℝ[7, m]:
        state: ℝ[7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        w1_array: ℝ[1] = [0.0]
        w2_array: ℝ[1] = [0.0]
        w3_array: ℝ[1] = [0.0]
        w4_array: ℝ[1] = [0.0]
        w5_array: ℝ[1] = [0.0]
        w6_array: ℝ[1] = [0.0]
        w7_array: ℝ[1] = [0.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            w1 = results[0]
            w2 = results[1]
            w3 = results[2]
            w4 = results[3]
            w5 = results[4]
            w6 = results[5]
            w7 = results[6]
            w1_array = append(w1_array, w1)
            w2_array = append(w2_array, w2)
            w3_array = append(w3_array, w3)
            w4_array = append(w4_array, w4)
            w5_array = append(w5_array, w5)
            w6_array = append(w6_array, w6)
            w7_array = append(w7_array, w7)
            state = results
        return [w1_array, w2_array, w3_array, w4_array, w5_array, w6_array, w7_array]

    true_theta: ℝ[2] = [0.1, 0.1]
    true_results: ℝ[7, m] = solver(true_theta)
    true_w1: ℝ[m] = true_results[0]
    true_w2: ℝ[m] = true_results[1]
    true_w3: ℝ[m] = true_results[2]
    true_w4: ℝ[m] = true_results[3]
    true_w5: ℝ[m] = true_results[4]
    true_w6: ℝ[m] = true_results[5]
    true_w7: ℝ[m] = true_results[6]

    n_exp: ℝ = 7.0

    def adjoint_grad(θ: ℝ[2]): ℝ[n]:
        states: ℝ[7, m] = solver(θ)
        w1_array: ℝ[m] = states[0]
        w2_array: ℝ[m] = states[1]
        w3_array: ℝ[m] = states[2]
        w4_array: ℝ[m] = states[3]
        w5_array: ℝ[m] = states[4]
        w6_array: ℝ[m] = states[5]
        w7_array: ℝ[m] = states[6]
        m: ℝ = get_1d_array_length(w1_array)
        s: ℝ[7] = [
            (w1_array[m-1] - true_w1[m-1]) / n_exp,
            (w2_array[m-1] - true_w2[m-1]) / n_exp,
            (w3_array[m-1] - true_w3[m-1]) / n_exp,
            (w4_array[m-1] - true_w4[m-1]) / n_exp,
            (w5_array[m-1] - true_w5[m-1]) / n_exp,
            (w6_array[m-1] - true_w6[m-1]) / n_exp,
            (w7_array[m-1] - true_w7[m-1]) / n_exp
        ]
        L: ℝ[2] = zero_1d_array(2)
        for i:ℕ(m-1):
            idx = m - 2 - i
            w1 = w1_array[idx]
            w2 = w2_array[idx]
            w3 = w3_array[idx]
            w4 = w4_array[idx]
            w5 = w5_array[idx]
            w6 = w6_array[idx]
            w7 = w7_array[idx]
            state = [w1, w2, w3, w4, w5, w6, w7]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            s = s @ J_state
        return L

    θ: ℝ[2] = [0.02, 0.5]
    learning_rate: ℝ = 0.02
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

- A. Goldbeter and D. E. Koshland Jr., *An amplified sensitivity arising from covalent modification in biological systems*, Proc. Natl. Acad. Sci. USA 78(11), 6840-6844 (1981).
- J. E. Ferrell Jr. and S. H. Ha, *Ultrasensitivity part I: Michaelian responses and zero-order ultrasensitivity*, Trends in Biochemical Sciences 39(10), 496-503 (2014).
- A. Goldbeter, *Biochemical Oscillations and Cellular Rhythms* (Cambridge University Press, 1996).
