Parameter Learning for ODE: FitzHugh–Nagumo
============================================

In this tutorial we learn the parameters of the **FitzHugh–Nagumo** model, a
smooth two-dimensional reduction of the Hodgkin–Huxley equations that is the
standard minimal model of an excitable neuron. It is a close cousin of the
:doc:`/tutorials/learn_parameter_lotka_volterra_ode` tutorial: the scaffolding
(RK4 stepper, trajectory solver, adjoint gradient) is the same, and only the
right-hand side ``f``, the loss, and one term in the adjoint change.

The one change worth reading carefully is the loss. Lotka–Volterra fits the
**final** state; here we fit the **whole trajectory**, which is the right choice
for neuronal dynamics (a single terminal sample is not enough to identify the
parameters). This adds a per-step *running cost* term to the adjoint, derived
below.


The Equations
-------------

The FitzHugh–Nagumo system couples a fast voltage-like variable :math:`v` to a
slow recovery variable :math:`w`:

.. math::

   \frac{dv}{dt} = v - \frac{v^3}{3} - w + I

   \frac{dw}{dt} = \varepsilon\,(v + a - b\,w)

Here :math:`v` is the membrane potential, :math:`w` the recovery current, and
:math:`\theta = [a, b, \varepsilon, I]` are the parameters we want to learn.
:math:`\varepsilon \ll 1` makes :math:`w` evolve slowly relative to :math:`v`,
which is what gives the model its excitable, spike-like behaviour.


Helper functions
----------------

We reuse the same dynamic-array helpers as the Lotka–Volterra tutorial. Physika
arrays are fixed-shape, so ``append`` allocates a new, one-longer array and
copies into it:

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

``f`` takes the current state ``[v, w]`` and parameters ``θ`` and returns the
derivatives ``[dv/dt, dw/dt]``:

.. code-block:: text

    def f(state: ℝ[2], θ: ℝ[4]): ℝ[2]:
        v: ℝ = state[0]
        w: ℝ = state[1]
        a: ℝ = θ[0]
        b: ℝ = θ[1]
        eps: ℝ = θ[2]
        Iext: ℝ = θ[3]
        dv: ℝ = v - (v ** 3) / 3.0 - w + Iext
        dw: ℝ = eps * (v + a - b * w)
        return [dv, dw]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge–Kutta method, which evaluates
the derivative at four points per step:

.. math::

    \begin{align*}
    k_1 &= f(y_n, \theta) \\
    k_2 &= f\left(y_n + \tfrac{h}{2} k_1, \theta\right) \\
    k_3 &= f\left(y_n + \tfrac{h}{2} k_2, \theta\right) \\
    k_4 &= f(y_n + h \, k_3, \theta) \\
    y_{n+1} &= y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
    \end{align*}

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

We integrate forward from the initial condition :math:`v_0 = -1,\ w_0 = 1` over
200 steps, collecting the full trajectory of both variables:

.. code-block:: text

    dt: ℝ = 0.1
    timesteps: ℝ = 200

    def solver(θ: ℝ[4]): ℝ[2, m]:
        state: ℝ[2] = [-1.0, 1.0]
        v_array: ℝ[1] = [-1.0]
        w_array: ℝ[1] = [1.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            v = results[0]
            w = results[1]
            v_array = append(v_array, v)
            w_array = append(w_array, w)
            state = results
        return [v_array, w_array]


Step 4: Generate Ground Truth Data
----------------------------------

We pick true parameters and generate the trajectories we will try to fit:

.. code-block:: text

    true_theta: ℝ[4] = [0.7, 0.8, 0.08, 0.5]
    true_results: ℝ[2, m] = solver(true_theta)
    true_v: ℝ[m] = true_results[0]
    true_w: ℝ[m] = true_results[1]


Step 5: Adjoint Gradient with a Trajectory Loss
-----------------------------------------------

As in the Lotka–Volterra tutorial we use the adjoint (reverse-mode) method: it
computes the gradient with respect to every parameter in a single backward
sweep over the trajectory. The difference is the loss.

Forward Pass
~~~~~~~~~~~~

The trajectory is produced by iterating the RK4 step, :math:`y_{n+1} =
\mathrm{RK4}(y_n, \theta)`, via ``solver(θ)``.

Loss and Co-state
~~~~~~~~~~~~~~~~~~

Instead of comparing only the final state, we compare the entire trajectory
with a mean-squared-error loss over all :math:`m` samples:

.. math::

    \mathcal{L}(\theta) = \frac{1}{2m} \sum_{k=0}^{m-1}
        \left\| y_k - y_k^{\mathrm{true}} \right\|^2

The adjoint (co-state) :math:`s_k = \partial \mathcal{L} / \partial y_k` now
receives a contribution at **every** step, not just the last. Its terminal
value is

.. math::

    s_{m-1} = \frac{1}{m}\left(y_{m-1} - y_{m-1}^{\mathrm{true}}\right),

and it is propagated backwards with a *running cost* term added each step:

.. math::

    s_k = \underbrace{\frac{1}{m}\left(y_k - y_k^{\mathrm{true}}\right)}_{\text{running cost}}
          + \; s_{k+1} \, J_{\mathrm{state}}(y_k).

Backward Pass
~~~~~~~~~~~~~

At each step we linearise the RK4 map with two Jacobians obtained from
``grad`` — one with respect to the state and one with respect to the
parameters:

.. math::

    J_{\mathrm{state}} = \frac{\partial\,\mathrm{RK4}(y_k, \theta)}{\partial y_k},
    \qquad
    J_{\theta} = \frac{\partial\,\mathrm{RK4}(y_k, \theta)}{\partial \theta},

and accumulate the parameter gradient while updating the co-state:

.. math::

    L \mathrel{+}= s \, J_{\theta}, \qquad
    s \mathrel{=} \tfrac{1}{m}\!\left(y_k - y_k^{\mathrm{true}}\right) + s \, J_{\mathrm{state}}.

The only lines that differ from Lotka–Volterra are the ``1/m`` normalisation of
the terminal co-state and the two ``residual`` lines inside the loop:

.. code-block:: text

    def adjoint_grad(θ: ℝ[4]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        v_array: ℝ[m] = states[0]
        w_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(v_array)
        s: ℝ[2] = [
            (v_array[m-1] - true_v[m-1]) / m,
            (w_array[m-1] - true_w[m-1]) / m
        ]
        L: ℝ[4] = zero_1d_array(4)
        for i:ℕ(m-1):
            idx = m - 2 - i
            v = v_array[idx]
            w = w_array[idx]
            state = [v, w]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            rv = (v_array[idx] - true_v[idx]) / m
            rw = (w_array[idx] - true_w[idx]) / m
            residual = [rv, rw]
            s = residual + (s @ J_state)
        return L


Step 6: Train with Gradient Descent
-----------------------------------

We start from an initial guess and take plain gradient-descent steps. FitzHugh–
Nagumo is more stiffly conditioned than Lotka–Volterra, so a smaller learning
rate is used:

.. code-block:: text

    θ: ℝ[4] = [0.6, 0.7, 0.10, 0.6]
    learning_rate: ℝ = 0.003
    epochs: ℕ = 800

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g

    pred_results = solver(θ)

.. note::
   The committed ``tutorials/fitzhugh_nagumo.phyk`` sets ``epochs = 1`` so the
   test suite runs quickly. Raise it (e.g. ``800``) to actually fit the model.


Step 7: Results
---------------

After training, the recovered trajectory matches the ground truth and the loss
falls by roughly three orders of magnitude:

.. figure:: /_static/tutorial_files/output_fitzhugh_nagumo.png
   :alt: FitzHugh-Nagumo true vs learned trajectory and loss convergence
   :align: center
   :width: 800px

   Left: ground-truth (solid) vs learned (dashed) trajectories for ``v`` and
   ``w``; the grey line is the initial guess. Right: trajectory-MSE loss over
   training.

The fast parameters :math:`\varepsilon` and :math:`I` are recovered almost
exactly, while :math:`a` and :math:`b` are recovered only approximately: they
enter through the slow recovery variable :math:`w`, which barely evolves over
this horizon, so the data constrain them weakly. This *identifiability* gap is a
useful thing to see in a real inverse problem — lengthening the horizon or
adding a second trajectory would tighten it.

To visualise the fit, add a helper to ``physika/runtime.py`` (Physika has no
built-in ``plot_trajectories``):

.. code-block:: python

    def plot_trajectories(true_results, pred_results):
        import matplotlib.pyplot as plt

        plt.plot(true_results[0, :], color="tab:blue", label="v (True)")
        plt.plot(pred_results[0, :], color="tab:blue", linestyle="--", label="v (Predicted)")
        plt.plot(true_results[1, :], color="tab:red", label="w (True)")
        plt.plot(pred_results[1, :], color="tab:red", linestyle="--", label="w (Predicted)")

        plt.xlabel("Time Step")
        plt.ylabel("State")
        plt.title("FitzHugh-Nagumo: True vs Predicted")
        plt.legend()
        plt.show()


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
        v: ℝ = state[0]
        w: ℝ = state[1]
        a: ℝ = θ[0]
        b: ℝ = θ[1]
        eps: ℝ = θ[2]
        Iext: ℝ = θ[3]
        dv: ℝ = v - (v ** 3) / 3.0 - w + Iext
        dw: ℝ = eps * (v + a - b * w)
        return [dv, dw]

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
    timesteps: ℝ = 200

    def solver(θ: ℝ[4]): ℝ[2, m]:
        state: ℝ[2] = [-1.0, 1.0]
        v_array: ℝ[1] = [-1.0]
        w_array: ℝ[1] = [1.0]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            v = results[0]
            w = results[1]
            v_array = append(v_array, v)
            w_array = append(w_array, w)
            state = results
        return [v_array, w_array]

    true_theta: ℝ[4] = [0.7, 0.8, 0.08, 0.5]
    true_results: ℝ[2, m] = solver(true_theta)
    true_v: ℝ[m] = true_results[0]
    true_w: ℝ[m] = true_results[1]

    def adjoint_grad(θ: ℝ[4]): ℝ[n]:
        states: ℝ[2, m] = solver(θ)
        v_array: ℝ[m] = states[0]
        w_array: ℝ[m] = states[1]
        m: ℝ = get_1d_array_length(v_array)
        s: ℝ[2] = [
            (v_array[m-1] - true_v[m-1]) / m,
            (w_array[m-1] - true_w[m-1]) / m
        ]
        L: ℝ[4] = zero_1d_array(4)
        for i:ℕ(m-1):
            idx = m - 2 - i
            v = v_array[idx]
            w = w_array[idx]
            state = [v, w]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            rv = (v_array[idx] - true_v[idx]) / m
            rw = (w_array[idx] - true_w[idx]) / m
            residual = [rv, rw]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[4] = [0.6, 0.7, 0.10, 0.6]
    learning_rate: ℝ = 0.003
    epochs: ℕ = 800

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        θ = θ - learning_rate * g

    pred_results = solver(θ)
    #plot_trajectories(true_results, pred_results)


References
----------

- `FitzHugh–Nagumo model — Wikipedia <https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model>`_
- R. FitzHugh, *Impulses and physiological states in theoretical models of nerve membrane*, Biophysical Journal, 1961.
- J. Nagumo, S. Arimoto, S. Yoshizawa, *An active pulse transmission line simulating nerve axon*, Proc. IRE, 1962.
