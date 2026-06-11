Parameter Learning of the Schrödinger Equation
===============================================

In this tutorial we will learn how to estimate the parameter of the
time-dependent Schrödinger equation using gradient descent in Physika.
The Schrödinger equation is a partial differential equation that governs
the evolution of the wave function of a non-relativistic quantum-mechanical
system.
A non-relativistic quantum mechanical system is one where the particle moves
at speeds much slower than the speed of light, so relativistic effects can
be safely ignored. In this regime the Schrödinger equation is the right tool
- it describes everyday quantum phenomena like electrons in atoms, particles
trapped in a well, and tunnelling through a barrier.
Here we simulate a Gaussian wave packet interacting with a potential
barrier and recover the barrier height from obser.ved wavefunction data.

The Equation
------------

The time-dependent Schrödinger equation is:

.. math::

    \begin{align*}
    i\hbar \frac{\partial \psi(x,t)}{\partial t}
    =
    \hat{H}\psi(x,t)
    \end{align*}

where:

- :math:`\psi(x, t)` is the wavefunction - a complex-valued field whose
  squared modulus :math:`|\psi|^2` gives the probability density of finding
  the particle at position :math:`x` at time :math:`t`
- :math:`\hbar` is the reduced Planck constant
- :math:`\hat{H}` is the Hamiltonian operator - the total energy of the system

The Hamiltonian
---------------

The Hamiltonian splits into kinetic and potential energy:

.. math::

    \begin{align*}
    \hat{H} &= \hat{T} + \hat{V} = -\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} + V(x)
    \end{align*}

Substituting into the Schrödinger equation:

.. math::

    \begin{align*}
    i\hbar \frac{\partial \psi}{\partial t} &= -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x)\psi
    \end{align*}

Rearranging for :math:`\frac{\partial \psi}{\partial t}` — the quantity we
need for time-stepping:

.. math::

    \begin{align*}
    \frac{\partial \psi}{\partial t} = -\frac{i}{\hbar} \left( -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x)\psi \right)
    \end{align*}

This is the RHS of the Schrödinger equation - the rate of change of the
wavefunction at each point in space.

Helper functions
----------------

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        dx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * dx
        return x
    
    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def get_2d_array_num_rows(x: ℂ[m, n]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def zero_complex_2d_array(rows: ℝ, cols: ℝ): ℂ[m, n]:
        results: ℂ[rows, cols] = for i:N(rows) -> for j:N(cols) -> j * 1j
        return results
    
    def append_row(x: ℂ[m, n], row: ℂ[n]): ℂ[k, n]:
        rows: ℝ = get_2d_array_num_rows(x)
        cols: ℝ = get_1d_array_length(x[0])
        new_rows: ℝ = rows + 1
        new_array: ℝ[new_rows, cols] = zero_complex_2d_array(new_rows, cols)
        for i:ℕ(0, rows):
            for j:ℕ(0, cols):
                new_array[i, j] = x[i, j]
        for j:ℕ(0, cols):
            new_array[rows, j] = row[j]
        return new_array

Grid and Physical Constants
-----------------------------

.. code-block:: text

    Nx: ℕ = 1024
    x: ℝ[Nx] = linspace(-200, 200, Nx)
    dx: ℝ = 0.3910

    hbar: ℝ = 1.0
    mass: ℝ = 1.0

The timestep is chosen using the CFL (Courant–Friedrichs–Lewy) stability
condition. The CFL condition is a constraint on the timestep :math:`\Delta t`
relative to the spatial spacing :math:`\Delta x` - if the timestep is too
large, numerical errors grow unboundedly and the simulation blows up. For
the Schrödinger equation the condition takes the form:

.. math::

    \begin{align*}
    \Delta t &= \alpha \cdot \frac{m \Delta x^2}{\hbar}
    \end{align*}

where :math:`\alpha = 0.2` is the CFL factor. Keeping :math:`\alpha` well
below 1 ensures the wavefunction evolves stably without numerical blow-up:

.. code-block:: text

    cfl_factor : ℝ = 0.2
    dt: ℝ = cfl_factor * (mass * dx**2) / hbar
    t_final: ℝ = 100.0
    Nt: ℕ = 3271


The Initial Condition — Gaussian Wave Packet
---------------------------------------------

We initialize the wavefunction as a Gaussian wave packet:

.. math::

    \begin{align*}
    \psi(x, 0) &= \frac{1}{\sqrt{\sigma\sqrt{\pi}}} \exp(ik_0 x) \exp\!\left(-\frac{(x - x_0)^2}{2\sigma^2}\right)
    \end{align*}

This is a product of three parts:

- :math:`\frac{1}{\sqrt{\sigma\sqrt{\pi}}}` :- normalization factor ensuring
  :math:`\int |\psi|^2 dx = 1`, so the total probability of finding the
  particle somewhere is 1
- :math:`\exp(ik_0 x)` :- a plane wave with wavenumber :math:`k_0`, giving
  the packet a mean momentum :math:`p = \hbar k_0` and making it travel in
  the positive :math:`x` direction
- :math:`\exp\!\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)` :- a Gaussian
  envelope centred at :math:`x_0` with width :math:`\sigma`, localising the
  particle in space

.. code-block:: text

    x0: ℝ = -50.0    # initial position
    k0: ℝ = 2.0      # wavenumber (controls momentum)
    sigma: ℝ = 10.0     # width of the wave packet

    psi0: ℂ[Nx] = (1 / sigma*sqrt(3.14))**0.5 * exp(1j * k0 * x) * exp(-((x - x0)**2) / (2 * sigma**2))


Discretizing the RHS
---------------------

We discretize space into :math:`N_x` points with uniform spacing
:math:`\Delta x`. The wavefunction becomes a vector
:math:`\psi_i = \psi(x_i, t)`, and the continuous spatial derivative
:math:`\frac{\partial^2 \psi}{\partial x^2}` is replaced by a finite
difference stencil that only uses the values at neighbouring grid points.
This turns the PDE into a system of ODEs - one per grid point


**Step 1 — Discretize the second spatial derivative**

Using centered finite differences:

.. math::

    \begin{align*}
    \frac{\partial^2 \psi}{\partial x^2} \bigg|_i &\approx \frac{\psi_{i-1} - 2\psi_i + \psi_{i+1}}{\Delta x^2}
    \end{align*}

**Step 2 — Apply the kinetic energy operator**

Substituting the finite difference approximation into the kinetic term:

.. math::

    \begin{align*}
    \hat{T}\psi_i &= -\frac{\hbar^2}{2m} \cdot \frac{\psi_{i-1} - 2\psi_i + \psi_{i+1}}{\Delta x^2}
    \end{align*}

**Step 3 — Apply the potential energy operator**

The potential term is a pointwise multiplication:

.. math::

    \begin{align*}
    \hat{V}\psi_i &= V_i \cdot \psi_i
    \end{align*}

**Step 4 — Assemble the full discretized Hamiltonian**

Combining kinetic and potential terms:

.. math::

    \begin{align*}
    \hat{H}\psi_i &= -\frac{\hbar^2}{2m} \cdot \frac{\psi_{i-1} - 2\psi_i + \psi_{i+1}}{\Delta x^2} + V_i \psi_i
    \end{align*}

**Step 5 — Compute the full RHS**

Dividing by :math:`i\hbar`:

.. math::

    \begin{align*}
    \frac{\partial \psi_i}{\partial t} &= -\frac{i}{\hbar} \left( -\frac{\hbar^2}{2m} \cdot \frac{\psi_{i-1} - 2\psi_i + \psi_{i+1}}{\Delta x^2} + V_i \psi_i \right)
    \end{align*}

This maps directly to ``schrodinger_rhs``. The ``roll`` operation shifts the
array by one index to access :math:`\psi_{i-1}` and :math:`\psi_{i+1}` for
all spatial points simultaneously:

.. code-block:: text

    def schrodinger_rhs(psi: ℂ[m], V: ℝ[n], dx: ℝ, hbar: ℝ, mass: ℝ): ℂ[o]:
        psi_xx: ℂ[Nx] = (roll(psi, -1) - 2*psi + roll(psi, 1)) / (dx**2)
        H_psi: ℂ[Nx] = -(hbar**2 / (2*mass)) * psi_xx + V * psi
        result: ℂ[Nx] = -1j / hbar * H_psi
        return result



Potential Barrier
-----------------

We place a rectangular potential barrier of height :math:`V_0` centred at
:math:`x = 0`:

.. math::

    \begin{align*}
    V(x) &= \begin{cases} V_0 & \text{if } |x| < 15 \\ 0 & \text{otherwise} \end{cases}
    \end{align*}

When the wave packet hits this barrier, part of it is reflected and part
tunnels through. The ratio of transmitted to reflected probability depends
sensitively on :math:`V_0`, which is why it is a learnable parameter.

.. code-block:: text

    def make_potential(V_value: ℝ): ℝ[m]:
        x: ℝ[Nx] = linspace(-200, 200, Nx)
        V: ℝ[Nx] = zero_1d_array(Nx)
        for i:ℕ(0, Nx):
            if abs(x[i]) < 15:
                V[i] = V_value
        return V



The RK4 Solver
---------------

Because :math:`\psi` is complex-valued, the RK4 solver operates on complex
arrays. The structure is identical to the wave equation solver — only the
RHS function changes:

.. code-block:: text

    def RK4_step(psi: ℂ[m], dt: ℝ, V: ℝ[n], dx: ℝ, hbar: ℝ, mass: ℝ): ℂ[o]:
        k1: ℂ[Nx] = schrodinger_rhs(psi, V, dx, hbar, mass)
        k2: ℂ[Nx] = schrodinger_rhs(psi + 0.5 * dt * k1, V, dx, hbar, mass)
        k3: ℂ[Nx] = schrodinger_rhs(psi + 0.5 * dt * k2, V, dx, hbar, mass)
        k4: ℂ[Nx] = schrodinger_rhs(psi + dt * k3, V, dx, hbar, mass)
        psi_next: ℂ[Nx] = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return psi_next


The full solver runs the wave packet forward in time, storing a snapshot
every 5 steps to build the history:

.. code-block:: text

    def solver(V: ℝ[m]): ℝ[m]:
        x: ℝ[Nx] = linspace(-200, 200, Nx)
        psi0: ℂ[Nx] = ((1 / sigma*sqrt(3.14)) ** 0.5 * exp(1j * k0 * x) * exp(-((x - x0) ** 2) / (2 * sigma**2))) 
        psi: ℂ[Nx] = psi0
        history: ℝ[1] = [0]
        counter: ℕ = 0
        for i:ℕ(0, Nt):
            psi = RK4_step(psi, dt, V, dx, hbar, mass)
            counter = counter + 1
            if counter == 5:
                history = append_row(history, psi)
                counter = 0
        return history


.. note::
    The current implementation of ``append_row`` can become a performance bottleneck for long simulations. To speed up
    training, replace ``append_row`` with ``append_row_runtime`` inside the ``solver``
    function and add the following implementation to ``physika/runtime.py``.
    

    .. code-block:: python

        def append_row_runtime(arr, val):
            # val is an array — stack as new row
            if isinstance(arr, torch.Tensor) and arr.dim() == 1 and arr.shape[0] != val.shape[0]:
                # first call — arr is placeholder [0], start fresh
                return val.unsqueeze(0)
            elif isinstance(arr, torch.Tensor) and arr.dim() == 2:
                return torch.cat([arr, val.unsqueeze(0)], dim=0)
            else:
                return val.unsqueeze(0)


Generate Ground Truth Data
---------------------------

We fix the true barrier height at :math:`V_0 = 1.8` and run the solver to
produce the ground truth wavefunction history:

.. code-block:: text

    V: ℝ[Nx] = make_potential(1.8)
    true_values: ℂ[m, n] = solver(V)

    create_plot(true_values, psi0, x, V)


.. note::
    ``create_plot`` is not a built-in Physika function. Add the
    following helper function to ``physika/runtime.py``:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import animation

        def update(frame, true_values, psi, line_prob, line_re, line_im):
            """
            Evolves the wave function for several time steps per frame to smooth the animation,
            and updates the probability density and real/imaginary parts of the wave function.
            """
            psi = true_values[frame]

            # print((psi.real**2+psi.imag**2).sum())
            line_prob.set_ydata(np.abs(psi) ** 2)
            line_re.set_ydata(np.real(psi))
            line_im.set_ydata(np.imag(psi))
            return line_prob, line_re, line_im


        def initialize_plot(psi0, x, V):
            """
            Sets up the figure and initial plots for the animation.
            """
            fig, (ax_prob, ax_reim) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            V_scale_prob = np.max(np.abs(psi0)) ** 2 * 1.5
            V_scale_reim = np.max(np.abs(psi0)) * 1.5

            ax_prob.plot(x, V / 1.8 * V_scale_prob, "k--", lw=1.5, label="Potential")
            (line_prob,) = ax_prob.plot(
                x, np.abs(psi0) ** 2, "b-", lw=2, label=r"$|\psi(x,t)|^2$"
            )
            ax_prob.set_ylabel(r"$|\psi(x,t)|^2$")
            ax_prob.set_title("Quantum Tunneling")
            ax_prob.legend(loc="upper right")

            ax_reim.plot(x, V / 1.8 * V_scale_reim, "k--", lw=1.5, label="Potential")
            (line_re,) = ax_reim.plot(x, np.real(psi0), "b-", lw=2, label=r"Re{$\psi$}")
            (line_im,) = ax_reim.plot(x, np.imag(psi0), "r-", lw=2, label=r"Im{$\psi$}")

            return fig, line_prob, line_re, line_im



        def create_plot(true_values, psi0, x, V):
            true_values = true_values.detach().cpu().numpy()
            psi0 = psi0.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            fig, line_prob, line_re, line_im = initialize_plot(psi0, x, V)

            # Create animation for the live display.
            ani = animation.FuncAnimation(
                fig,
                update,
                fargs=(true_values, psi0, line_prob, line_re, line_im),
                frames= len(true_values),
                interval=30,
                blit=False,
            )

            plt.tight_layout()
            plt.show()

.. figure:: /_static/tutorial_files/true_values_plot.gif
   :alt: 
   :align: center
   :width: 700px

   ground truth animation




Intial guess
----------------

.. code-block:: text

    guess_barrier_height: ℝ = 6.0
    guess_V: ℝ[Nx] = make_potential(guess_barrier_height)
    guess_values: ℂ[m, n] = solver(guess_V)
    create_plot(guess_values, psi0, x, guess_V)


.. figure:: /_static/tutorial_files/initial_guess_plot.gif
   :alt: 
   :align: center
   :width: 700px

   Initial guess animation

Define the Loss
----------------

Since :math:`\psi` is complex, we cannot square the difference directly.
Instead we take the absolute value first, which gives the magnitude of the
complex difference at each point, then square it:

.. math::

    \begin{align*}
    \mathcal{L}(V_0) &= \frac{1}{N} \sum_{i,t} \left| \psi_i^{\text{pred}}(t) - \psi_i^{\text{true}}(t) \right|^2
    \end{align*}

.. code-block:: text

    def calculate_loss(barrier_height: ℝ): ℝ:
        V_current: ℝ[Nx] = make_potential(barrier_height)
        pred: ℂ[m, n] = solver(V_current)
        loss: ℝ = mean(abs(pred - true_values)**2)
        return loss

The Adam Optimizer
-------------------

We use Adam instead of plain gradient descent because the loss landscape
for the barrier height is nonlinear and benefits from adaptive learning rates:

.. math::

    \begin{align*}
    m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
    \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
    \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
    \theta_t &= \theta_{t-1} - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    \end{align*}


.. code-block:: text

    def adam(bh: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999V: ℝ[Nx] = make_potential(1.8)
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        bh_new: ℝ = bh - lr * m_hat / (sqrt(v_hat) + eps)
        return [bh_new, m_new, v_new, t + 1.0]

Training Loop
--------------

Starting from an initial guess of :math:`V_0 = 6.0`, gradient descent with
Adam recovers the true barrier height of :math:`1.8`:

.. code-block:: text

    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.1

    epochs: N = 40

    for i:ℕ(epochs):
        physika_print(i)
        g = grad(calculate_loss, new_barrier_height)
        result = adam(new_barrier_height, g, m_adam, v_adam, t_adam, lr)
        new_barrier_height  = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]

    pred_V: ℝ[Nx] = make_potential(guess_barrier_height)
    pred_results: ℂ[m, n] = solver(pred_V)
    create_plot(pred_results, psi0, x, pred_V)

``grad()`` differentiates through the entire solver — through all RK4 steps,
through the complex arithmetic of ``schrodinger_rhs``, and through
``make_potential`` — treating ``barrier_height`` as the single learnable
scalar.


.. figure:: /_static/tutorial_files/pred_results_plot.gif
   :alt: 
   :align: center
   :width: 700px

   final prediction animation

Full code
--------------

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        dx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * dx
        return x
    
    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def get_2d_array_num_rows(x: ℂ[m, n]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def zero_complex_2d_array(rows: ℝ, cols: ℝ): ℂ[m, n]:
        results: ℂ[rows, cols] = for i:N(rows) -> for j:N(cols) -> j * 1j
        return results
    
    def append_row(x: ℂ[m, n], row: ℂ[n]): ℂ[k, n]:
        rows: ℝ = get_2d_array_num_rows(x)
        cols: ℝ = get_1d_array_length(x[0])
        new_rows: ℝ = rows + 1
        new_array: ℝ[new_rows, cols] = zero_complex_2d_array(new_rows, cols)
        for i:ℕ(0, rows):
            for j:ℕ(0, cols):
                new_array[i, j] = x[i, j]
        for j:ℕ(0, cols):
            new_array[rows, j] = row[j]
        return new_array

    Nx: ℕ = 1024
    Nt: ℕ = 3271
    x: ℝ[Nx] = linspace(-200, 200, Nx)
    dx: ℝ = 0.3910

    hbar: ℝ = 1.0
    mass: ℝ = 1.0

    cfl_factor: ℝ = 0.2
    dt: ℝ = cfl_factor * (mass * dx**2) / hbar
    t_final: ℝ = 100.0

    x0: ℝ = -50.0    # initial position
    k0: ℝ = 2.0      # wavenumber (controls momentum)
    sigma: ℝ = 10.0     # width of the wave packet

    psi0: ℂ[Nx] = (1 / sigma*sqrt(3.14))**0.5 * exp(1j * k0 * x) * exp(-((x - x0)**2) / (2 * sigma**2))


    def schrodinger_rhs(psi: ℂ[m], V: ℝ[n], dx: ℝ, hbar: ℝ, mass: ℝ): ℂ[o]:
        psi_xx: ℂ[Nx] = (roll(psi, -1) - 2*psi + roll(psi, 1)) / (dx**2)
        H_psi: ℂ[Nx] = -(hbar**2 / (2*mass)) * psi_xx + V * psi
        result: ℂ[Nx] = -1j / hbar * H_psi
        return result


    def make_potential(V_value: ℝ): ℝ[m]:
        V: ℝ[Nx] = zero_1d_array(Nx)
        x: ℝ[Nx] = linspace(-200, 200, Nx)
        for i:ℕ(0, Nx):
            if abs(x[i]) < 15:
                V[i] = V_value
        return V


    def RK4_step(psi: ℂ[m], dt: ℝ, V: ℝ[n], dx: ℝ, hbar: ℝ, mass: ℝ): ℂ[o]:
        k1: ℂ[Nx] = schrodinger_rhs(psi, V, dx, hbar, mass)
        k2: ℂ[Nx] = schrodinger_rhs(psi + 0.5 * dt * k1, V, dx, hbar, mass)
        k3: ℂ[Nx] = schrodinger_rhs(psi + 0.5 * dt * k2, V, dx, hbar, mass)
        k4: ℂ[Nx] = schrodinger_rhs(psi + dt * k3, V, dx, hbar, mass)
        psi_next: ℂ[Nx] = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return psi_next


    def solver(V: ℝ[m]): ℂ[m, n]:
        x: ℝ[Nx] = linspace(-200, 200, Nx)
        psi0: ℂ[Nx] = ((1 / sigma*sqrt(3.14)) ** 0.5 * exp(1j * k0 * x) * exp(-((x - x0) ** 2) / (2 * sigma**2)))
        history: ℂ[1, Nx] = [psi0]
        counter: ℕ = 0
        psi = psi0
        for i:ℕ(0, Nt):
            psi = RK4_step(psi, dt, V, dx, hbar, mass)
            counter = counter + 1
            if counter == 5:
                history = append_row(history, psi)
                counter = 0
        return history



    V: ℝ[Nx] = make_potential(1.8)
    true_values: ℂ[m, n] = solver(V)
    create_plot(true_values, psi0, x, V)

    guess_barrier_height: ℝ = 6.0
    guess_V: ℝ[Nx] = make_potential(guess_barrier_height)
    guess_values: ℂ[m, n] = solver(guess_V)
    create_plot(guess_values, psi0, x, guess_V)


    def calculate_loss(barrier_height: ℝ): ℝ:
        V_current: ℝ[Nx] = make_potential(barrier_height)
        pred: ℂ[m, n] = solver(V_current)
        loss: ℝ = mean(abs(pred - true_values)**2)
        return loss


    def adam(bh: ℝ, g: ℝ, m: ℝ, v: ℝ, t: ℝ, lr: ℝ) : ℝ[4]:
        beta1: ℝ = 0.9
        beta2: ℝ = 0.999
        eps: ℝ = 1e-8
        m_new: ℝ = beta1 * m + (1.0 - beta1) * g
        v_new: ℝ = beta2 * v + (1.0 - beta2) * g**2
        m_hat: ℝ = m_new / (1.0 - beta1**t)
        v_hat: ℝ = v_new / (1.0 - beta2**t)
        bh_new: ℝ = bh - lr * m_hat / (sqrt(v_hat) + eps)
        return [bh_new, m_new, v_new, t + 1.0]

    m_adam: ℝ = 0.0
    v_adam: ℝ = 0.0
    t_adam: ℝ = 1.0
    lr: ℝ = 0.1

    epochs: ℕ = 40

    for i:ℕ(epochs):
        physika_print(i)
        g = grad(calculate_loss, guess_barrier_height)
        result  = adam(guess_barrier_height, g, m_adam, v_adam, t_adam, lr)
        guess_barrier_height  = result[0]
        m_adam = result[1]
        v_adam = result[2]
        t_adam = result[3]

    pred_V: ℝ[Nx] = make_potential(guess_barrier_height)
    pred_results: ℂ[m, n] = solver(pred_V)
    create_plot(pred_results, psi0, x, pred_V)



References
----------

- `Schrödinger equation - Wikipedia <https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation>`_
- `Quantum tunneling simulation code <https://www.astro.utoronto.ca/~mhvk/AST1410/python/quantum_tunneling.py>`_
- `Schrödinger FDTD - SciPy Cookbook <https://scipy-cookbook.readthedocs.io/items/SchrodingerFDTD.html>`_
- `Schrödinger FDTD PDF Notes <https://scipy-cookbook.readthedocs.io/_static/items/attachments/SchrodingerFDTD/Schrodinger%5FFDTD.pdf>`_
- `Wave Functions and Operators - Fiveable Computational Chemistry <https://fiveable.me/computational-chemistry/unit-3/wave-functions-operators/study-guide/kLX80x0GqycgGNHp>`_
