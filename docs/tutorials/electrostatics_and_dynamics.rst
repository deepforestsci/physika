Electrostatics and Dynamics
===========================

In this tutorial we will explore application of physika for conducting simple
simulations of electrical systems.

Coulomb's Law
-------------

Coulomb's inverse-square law or Coulomb's law describes the amount of force
between two electrically charged particles at rest. This electric force is
conventionally called the electrostatic force or Coulomb force.

The law states that the magnitude, or absolute value, of the attractive or
repulsive electrostatic force between two point charges is directly
proportional to the product of the magnitudes of their charges and inversely
proportional to the square of the distance between them. [coulomb]_

Electrostatic Force
~~~~~~~~~~~~~~~~~~~

The force exerted on a point charge :math:`q_2` at position
:math:`\mathbf{x}_2` by a point charge :math:`q_1` at position
:math:`\mathbf{x}_1` is

.. math::

    \mathbf{F}_{12} = \frac{1}{4 \pi \varepsilon_0}
    \frac{q_1 q_2}{\lVert \mathbf{x}_2 - \mathbf{x}_1 \rVert^2}
    \, 
    \frac{\mathbf{x}_2 - \mathbf{x}_1}
    {\lVert \mathbf{x}_2 - \mathbf{x}_1 \rVert}

where :math:`\varepsilon_0` is the vacuum permittivity.

.. code:: text

    def F(q1: ℝ, q2: ℝ, x1: ℝ[3], x2: ℝ[3]): ℝ[3]:
        return (1 / (4 * π * ε0)) * (q1 * q2 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

    coulomb_f = F(1, 1, [0, 0, 0], [1, 1, 1])
    print(coulomb_f)

output:

.. code:: text

    [6918773248.0, 6918773248.0, 6918773248.0] ∈ ℝ[3]

Electric Field
~~~~~~~~~~~~~~

The electric field produced by a point charge :math:`q_1` at position
:math:`\mathbf{x}_1`, evaluated at a point :math:`\mathbf{x}_2`, is:

.. math::

    \mathbf{E}(\mathbf{x}_2) = \frac{\mathbf{F}_{12}}{q_2}
    = \frac{1}{4 \pi \varepsilon_0}
    \frac{q_1}{\lVert \mathbf{x}_2 - \mathbf{x}_1 \rVert^2}
    \, \frac{\mathbf{x}_2 - \mathbf{x}_1}{\lVert \mathbf{x}_2 - \mathbf{x}_1 \rVert}

.. code:: text

    def E(q1: ℝ, x1: ℝ[3], x2: ℝ[3]): ℝ[3]:
        return (1 / (4 * π * ε0)) * (q1 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

    coulomb_e = E(1, [0, 0, 0], [1, 1, 1])
    print(coulomb_e)

Output::

    [3459386624.0, 3459386624.0, 3459386624.0] ∈ ℝ[3]

Simulating an RC Circuit
------------------------

An RC (resistor-capacitor) circuit consists of a resistor and a capacitor
connected to a voltage source. Simulating an RC circuit helps us understand
how a capacitor charges and discharges over a time period.

Governing Differential Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a capacitor of capacitance :math:`C` discharging through a resistor of
resistance :math:`R`, the current leaving the capacitor is
:math:`I = -C \, \frac{dV}{dt}`, and Ohm's law gives the current through the
resistor as :math:`I = V / R`. Setting the two equal gives

.. math::

    R C \, \frac{dV}{dt} + V = 0
    \quad \Longrightarrow \quad
    \frac{dV}{dt} = -\frac{V}{R C},
    \qquad V(0) = V_0

where :math:`\tau = R C` is the time constant of the circuit.

.. code:: text

    def dV_dt(V: ℝ, Capa: ℝ, Ress: ℝ): ℝ:
        return -V / (Ress * Capa)

Time Integration with RK4
~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate the voltage forward in time we use the fourth-order
Runge-Kutta method (RK4). For an ODE :math:`\frac{dV}{dt} = f(V)` and a time
step :math:`\Delta t`, one step evaluates four slopes

.. math::

    k_1 &= f(V_n) \\
    k_2 &= f\left(V_n + \tfrac{\Delta t}{2} k_1\right) \\
    k_3 &= f\left(V_n + \tfrac{\Delta t}{2} k_2\right) \\
    k_4 &= f\left(V_n + \Delta t \, k_3\right)

and combines them as a weighted average:

.. math::

    V_{n+1} = V_n + \frac{\Delta t}{6} \left( k_1 + 2 k_2 + 2 k_3 + k_4 \right)

.. code:: text

    def rk4_step(V: ℝ, Capa: ℝ, Ress: ℝ, dt: ℝ): ℝ:
        k1: ℝ = dV_dt(V, Capa, Ress)
        k2: ℝ = dV_dt(V + 0.5 * dt * k1, Capa, Ress)
        k3: ℝ = dV_dt(V + 0.5 * dt * k2, Capa, Ress)
        k4: ℝ = dV_dt(V + dt * k3, Capa, Ress)
        return V + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~

``voltage_at`` applies ``rk4_step``, starting from the initial
voltage across all the given timesteps. Then we use ``current_at``,
:math:`I = V / R`, to turn the final voltage into current.

.. code:: text

    def voltage_at(V: ℝ, Capa: ℝ, Ress: ℝ, dt: ℝ, steps: ℕ): ℝ:
        for i: ℕ(steps):
            V = rk4_step(V, Capa, Ress, dt)
        return V

    def current_at(V: ℝ, Capa: ℝ, Ress: ℝ, dt: ℝ, steps: ℕ): ℝ:
        V: ℝ = voltage_at(V, Capa, Ress, dt, steps)
        return V / Ress

Here, we take a 1 F capacitor charged to 10 V and discharge it through a 10 Ω
resistor, so the time consant is :math:`\tau = RC = 10` s. With
:math:`\Delta t = 0.01` s and 200 steps the simulation takes :math:`t = 2` s,
which is :math:`0.2\tau`.

.. code:: text

    Capa, Ress, V0: ℝ = 1.0, 10.0, 10.0
    dt: ℝ = 0.01
    steps: ℕ = 200

    V_final: ℝ = voltage_at(V0, Capa, Ress, dt, steps)
    print(V_final)

    I_final: ℝ = current_at(V0, Capa, Ress, dt, steps)
    print(I_final)

Output::

    8.18730753077985 ∈ ℝ
    0.818730753077985 ∈ ℝ

.. note::

   Time Constant: :math:`\tau = RC` tells us how quickly the capacitor
   discarges. It is not the total time required for discharge.

Comparing with the Analytical Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RC equation has an exact solution, which gives us a check on the numerical
result:

.. math::

    V(t) = V_0 \, e^{-t / RC}

.. code:: text

    V_exact: ℝ = V0 * exp(-dt * steps / (Ress * Capa))
    print(V_exact)

Output::

    8.187307357788086 ∈ ℝ

The simulated and exact voltages agree to seven significant figures.

Helper Functions
----------------

3D Distance
~~~~~~~~~~~

The Euclidean distance between two points
:math:`\mathbf{p}_1, \mathbf{p}_2 \in \mathbb{R}^3` is

.. math::

    \lVert \mathbf{p}_1 - \mathbf{p}_2 \rVert
    = \sqrt{\sum_{i=0}^{2} \left( p_{1,i} - p_{2,i} \right)^2}

.. code:: text

    def dist_3d(p1: ℝ[3], p2: ℝ[3]): ℝ:
        total: ℝ = 0
        for i: ℕ(3):
            total += (p1[i] - p2[i])**2
        return total**(1/2)

Future Work
-----------

A possible future direction for this work could be using graphs to represent
the circuit. This way much more complicated system could be represented. A task
of this size will also require a graph traversal method which can solve for
current flow and instantaneous voltages at different vertex along the path of
current in the said graph. For this to happen we will have to implement a
complete circuit simulation toolkit.

Full Code
---------

.. literalinclude:: ../../tutorials/electrostatics_and_dynamics.phyk
   :language: text

References
----------

.. [coulomb] Coulomb (1785). "Premier mémoire sur l'électricité et le
   magnétisme" [First dissertation on electricity and magnetism]. Histoire de
   l'Académie Royale des Sciences [History of the Royal Academy of Sciences]
   (in French). pp. 569–577.
