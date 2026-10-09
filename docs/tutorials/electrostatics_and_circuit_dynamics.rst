Electrostatics and Circuit Dynamics
===================================

In this tutorial we will explore application of physika for conducting simple
simulations of electrical systems. **Electrostatics** studies charges that are
stationary and the forces/fields they produce. It is needed to understand
how capacitors, energy sources and it is the basis of many electrical and
electronic devices. While **Circuit Dynamics** refers to studying how electric
charges move and change over time due to the influence of electric and magnetic
fields.

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

.. code:: text

    def F(q1: ℝ, q2: ℝ, x1: ℝ[3], x2: ℝ[3]): ℝ[3]:
        return (1 / (4 * π * ε0)) * (q1 * q2 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

    coulomb_f: ℝ[3] = F(2, 2, [0, 0, 0], [1, 1, 1])
    print(coulomb_f)

output:

.. code:: text

    [6918773248.0, 6918773248.0, 6918773248.0] ∈ ℝ[3]

.. note::

   **dist_3d** is a function for finding euclidian norm. Implementation
   details in helper section.
   
   :math:`\varepsilon_0` is vacuum permittivity.

   :math:`\pi` is an irrational mathematical constant.

Electric Field
~~~~~~~~~~~~~~

The electric field produced by a point charge :math:`q_1` at position
:math:`\mathbf{x}_1`, evaluated at a point :math:`\mathbf{x}_2`, is:

.. math::

    \mathbf{E}(\mathbf{x}_2)
    = \frac{1}{4 \pi \varepsilon_0}
    \frac{q_1}{\lVert \mathbf{x}_2 - \mathbf{x}_1 \rVert^2}
    \, \frac{\mathbf{x}_2 - \mathbf{x}_1}{\lVert \mathbf{x}_2 - \mathbf{x}_1 \rVert}

where :math:`\varepsilon_0` is the vacuum permittivity.

.. code:: text

    def E(q1: ℝ, x1: ℝ[3], x2: ℝ[3]): ℝ[3]:
        return (1 / (4 * π * ε0)) * (q1 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

    coulomb_e: ℝ[3] = E(2, [0, 0, 0], [1, 1, 1])
    print(coulomb_e)

Output:

.. code:: text

    [3459386624.0, 3459386624.0, 3459386624.0] ∈ ℝ[3]

Sum of Electric field on a point due to multiple charges
--------------------------------------------------------

Consider a collection of :math:`N` particles each of which has charge
:math:`q_i` and is located at location :math:`x_i`. These particles create an
electric field :math:`E` which extends throughout the system. We can measure
its combined effect using:

.. math::

   \vec{E}_{\mathrm{total}}(\vec{x}_2)
   =
   \sum_{i=1}^{n}
   \frac{q_i}{4\pi\varepsilon_0}
   \frac{\vec{x}_2-\vec{x}_i}
   {\|\vec{x}_2-\vec{x}_i\|^3}

Where:

- :math:`n` is the number of source charges.
- :math:`q_i` is the i-th source charge.
- :math:`x_i` is the position of the i-th source charge.
- :math:`x_2` is the observation point.
- :math:`E_total` is the net electric field, obtained by summing the contributions from all n charges.

.. code:: text

    def E_n(n: ℝ, q: ℝ[n], x: ℝ[n, 3], x2: ℝ[3]): ℝ[3]:
        total: ℝ[3] = [0.0, 0.0, 0.0]
        for i: ℕ(n):
            total += E(q[i], x[i], x2)
        return total

    n: ℝ = 10
    q: ℝ[3] = sample_normal1D(n)
    x: ℝ[3, 3] = sample_normal3D(n)

    coulomb_e_n: ℝ[3] = E_n(n, q, x, [1, 1, 1])
    print(coulomb_e_n)





Change in Electric Field w.r.t Position
---------------------------------------

The derivative of Electric field w.r.t. positon describes how the electric field
changes spacially as the observation point moves. As the electric field is a
vector, its derivative w.r.t. 3-dimensional position vector is a 3x3 jacobian
matrix. Its entries represent the rate of change of electric field with respect
to each of the directional componenets.

.. math::

   \nabla \vec{E} =
   \begin{bmatrix}
   \frac{\partial E_x}{\partial x} & \frac{\partial E_x}{\partial y} & \frac{\partial E_x}{\partial z} \\
   \frac{\partial E_y}{\partial x} & \frac{\partial E_y}{\partial y} & \frac{\partial E_y}{\partial z} \\
   \frac{\partial E_z}{\partial x} & \frac{\partial E_z}{\partial y} & \frac{\partial E_z}{\partial z}
   \end{bmatrix}

Where, :math:`E_x`, :math:`E_y` and :math:`E_z` are electric field componenets in the respective directions.

.. code:: text

    # Charge and positions
    q1: ℝ = 2.0
    x1: ℝ[3] = [0.0, 0.0, 0.0]
    x2: ℝ[3] = [1.0, 1.0, 1.0]

    # Electric field
    coulomb_e: ℝ[3] = E(q1, x1, x2)

    # Jacobian of the electric field with respect to x2
    dE_dx2: ℝ[3] = grad(E(q1, x1, x2), x2)

    print(coulomb_e)
    print(dE_dx2)

Output:

.. code:: text

    [[0.0, -3459386880.0, -3459386880.0], [-3459386880.0, 0.0, -3459386880.0], [-3459386880.0, -3459386880.0, 0.0]] ∈ ℝ[3,3]


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

.. code:: text

    π, ε0: ℝ = 3.14159, 8.854e-12

    # p1: Position 1 in 3d space
    # p2: Position 2 in 3d space
    # Eucledian Distance for 3d points
    def dist_3d(p1: ℝ[3], p2: ℝ[3]): ℝ:
        total: ℝ = 0
        for i: ℕ(3):
            total += (p1[i] - p2[i])**2
        return total**(1/2)

    # q1: Charge at position x1
    # q2: Charge at position x2
    # x1: Position of the first charge
    # x2: Position where the force is evaluated
    # Coulomb force
    def F(q1: ℝ, q2: ℝ, x1: ℝ[3], x2: ℝ[3]): ℝ[3]:
        return (1 / (4 * π * ε0)) * (q1 * q2 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

    coulomb_f: ℝ[3] = F(2, 2, [0, 0, 0], [1, 1, 1])
    print(coulomb_f)

    # q1: Charge at position x1
    # x1: Position of the first charge
    # x2: Position where the field/force is evaluated
    # Electric Field
    def E(q1: ℝ, x1: ℝ[3], x2: ℝ[3]): ℝ[3]:
        return (1 / (4 * π * ε0)) * (q1 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

    coulomb_e: ℝ[3] = E(2, [0, 0, 0], [1, 1, 1])
    print(coulomb_e)

    # Charge and positions
    q1: ℝ = 2.0
    x1: ℝ[3] = [0.0, 0.0, 0.0]
    x2: ℝ[3] = [1.0, 1.0, 1.0]

    # Electric field
    coulomb_e: ℝ[3] = E(q1, x1, x2)

    # Jacobian of the electric field with respect to x2
    dE_dx2: ℝ[3] = grad(E(q1, x1, x2), x2)

    print(coulomb_e)
    print(dE_dx2)

    # n: Number of point charges in the system
    # q: Charge of each of the n points
    # x: Positions of each of the n points
    # x2: Position where total electric field is evaluated
    # Electric Field due to many point charges
    def E_n(n: ℝ, q: ℝ[n], x: ℝ[n, 3], x2: ℝ[3]): ℝ[3]:
        total: ℝ[3] = [0.0, 0.0, 0.0]
        for i: ℕ(n):
            total += E(q[i], x[i], x2)
        return total

    # Random sampling inside functions
    def sample_normal1D(x: ℝ): ℝ[m]:
        t ~ 𝒩(0, 1, x)
        return t

    def sample_normal3D(x: ℝ): ℝ[p, m]:
        t: ℝ[x, 3] = for i : ℕ(x) → ε : ℝ[3] ~ Normal(0, 1, 3)
        return t

    n: ℝ = 10
    q : ℝ[3] = sample_normal1D(n)
    x : ℝ[3, 3] = sample_normal3D(n)

    coulomb_e_n: ℝ[3] = E_n(n, q, x, [1, 1, 1])
    print(coulomb_e_n)

    # ----------
    # RC Circuit
    # ----------

    # Capa: Capacitance C (named Capa because R is an alias for ℝ)
    # Ress: Resistance R (named Ress for the same reason)
    def dV_dt(V: ℝ, Capa: ℝ, Ress: ℝ): ℝ:
        return -V / (Ress * Capa)

    def rk4_step(V: ℝ, Capa: ℝ, Ress: ℝ, dt: ℝ): ℝ:
        k1: ℝ = dV_dt(V, Capa, Ress)
        k2: ℝ = dV_dt(V + 0.5 * dt * k1, Capa, Ress)
        k3: ℝ = dV_dt(V + 0.5 * dt * k2, Capa, Ress)
        k4: ℝ = dV_dt(V + dt * k3, Capa, Ress)
        return V + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def voltage_at(V: ℝ, Capa: ℝ, Ress: ℝ, dt: ℝ, steps: ℕ): ℝ:
        for i: ℕ(steps):
            V = rk4_step(V, Capa, Ress, dt)
        return V

    def current_at(V: ℝ, Capa: ℝ, Ress: ℝ, dt: ℝ, steps: ℕ): ℝ:
        V: ℝ = voltage_at(V, Capa, Ress, dt, steps)
        return V / Ress

    Capa, Ress, V0: ℝ = 1.0, 10.0, 10.0
    Δt: ℝ = 0.01
    steps: ℕ = 200

    V_final: ℝ = voltage_at(V0, Capa, Ress, Δt, steps)
    print(V_final)

    I_final: ℝ = current_at(V0, Capa, Ress, Δt, steps)
    print(I_final)

    # Analytical solution:
    V_exact: ℝ = V0 * exp(-Δt * steps / (Ress * Capa))
    print(V_exact)

References
----------

.. [coulomb] Coulomb (1785). "Premier mémoire sur l'électricité et le
   magnétisme" [First dissertation on electricity and magnetism]. Histoire de
   l'Académie Royale des Sciences [History of the Royal Academy of Sciences]
   (in French). pp. 569–577.
