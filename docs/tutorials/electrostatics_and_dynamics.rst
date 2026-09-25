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

where :math:`\varepsilon_0` is the vacuum permittivity and
:math:`\hat{\mathbf{r}}_{12}` is the unit vector pointing from
:math:`q_1` to :math:`q_2`. Like charges (:math:`q_1 q_2 > 0`) repel and
unlike charges (:math:`q_1 q_2 < 0`) attract.

.. code:: text

    def F(q1: R, q2: R, x1: R[3], x2: R[3]): R[3]:
        return (1 / (4 * π * ε0)) * (q1 * q2 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

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

    def E(q1: R, x1: R[3], x2: R[3]): R[3]:
        return (1 / (4 * π * ε0)) * (q1 / dist_3d(x1, x2)**2) * ((x2 - x1) / dist_3d(x1, x2))

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

    def dV_dt(V: R, Capa: R, Ress: R): R:
        return -V / (Ress * Capa)

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

    def dist_3d(p1: R[3], p2: R[3]): R:
        total: R = 0
        for i: N(3):
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

Coulomb (1785). "Premier mémoire sur l'électricité et le magnétisme"
[First dissertation on electricity and magnetism]. Histoire de l'Académie Royale des Sciences [History of the Royal Academy of Sciences] (in French). pp. 569–577.
