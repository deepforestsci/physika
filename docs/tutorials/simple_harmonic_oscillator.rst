============================================================
Harmonic Oscillator — Differentiable Time Evolution
============================================================

The harmonic oscillator is a system that, when displaced from its equilibrium position, experiences a restoring force F proportional to the displacement x.
The equation can be derived by substituting Hooke's Law into Newton's Second Law ``F = ma``.


.. math::

    \begin{align*}
    \vec{F} &= -k \vec{x}
    \end{align*}

where :math:`\vec{F}` is force, :math:`k` is the spring constant. and :math:`\vec{x}` is displacement.


Deriving the General Solution
------------------------------

Substituting Hooke's Law into Newton's Second Law :math:`F = ma` gives:

.. math::

    \begin{align*}
    m\ddot{x} &= -kx \\[0.5em]
    \ddot{x} &= -\frac{k}{m}x
    \end{align*}

This is a second-order linear ODE. We define the angular frequency :math:`\omega`
to simplify notation:

.. math::

    \begin{align*}
    \omega &= \sqrt{\frac{k}{m}}
    \end{align*}

So the equation becomes :math:`\ddot{x} = -\omega^2 x`. The general solution
to this ODE is a superposition of sine and cosine with frequency :math:`\omega`:

.. math::

    \begin{align*}
    x(t) &= A \cos(\omega t) + B \sin(\omega t)
    \end{align*}

where :math:`A` and :math:`B` are constants determined by initial conditions.
Taking the derivative to get velocity:

.. math::

    \begin{align*}
    \dot{x}(t) &= -A\omega \sin(\omega t) + B\omega \cos(\omega t)
    \end{align*}

Evaluating both at :math:`t = 0` and applying :math:`x(0) = x_0`,
:math:`\dot{x}(0) = v_0`:

.. math::

    \begin{align*}
    x(0) &= A = x_0 \\
    \dot{x}(0) &= B\omega = v_0 \implies B = \frac{v_0}{\omega}
    \end{align*}

Substituting back gives the complete closed-form solution:

.. math::

    \begin{align*}
    x(t) &= x_0 \cos(\omega t) + \frac{v_0}{\omega} \sin(\omega t)
    \end{align*}



Helper functions
----------------

.. code-block:: text

    def get_1d_array_length(x: ℝ[m]): ℝ:
    total: ℝ = 0
    temp: ℝ = 0
    for i:
        temp = x[i]
        total += 1
    return total

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i:ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols: ℝ): R[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results


Gaussian Elimination
----------------------------


.. code-block:: text

    def solve(A: ℝ[m, m], b: ℝ[m]): ℝ[m]:
        n: ℝ = get_1d_array_length(b)
        Aug: ℝ[m, m] = zero_2d_array(n, n+1)
        result: ℝ[m] = zero_1d_array(n)
        # create augmented matrix
        for i:ℕ(n):
            for j:ℕ(n):
                Aug[i, j] = A[i, j]
            Aug[i, n] = b[i]
        # convert into upper triangular matrix
        for pivot:ℕ(n):
            for i:ℕ(pivot+1, n):
                factor = Aug[i, pivot] / Aug[pivot, pivot]
                for j:ℕ(pivot, n+1):
                    Aug[i, j] = Aug[i, j] - factor * Aug[pivot, j]
        # back substitution
        for i:ℕ(n):
            ri = n - 1 - i
            result[ri] = Aug[ri, n]
            for j:ℕ(ri+1, n):
                result[ri] = result[ri] - Aug[ri, j] * result[j]
            result[ri] = result[ri] / Aug[ri, ri]
        return result

The algorithm proceeds in three phases:

- **Augmented matrix** — the coefficient matrix :math:`A` and the right-hand
  side vector :math:`b` are merged into a single matrix called as Augmented matrix.

- **Forward elimination** — for each pivot row, all entries below the diagonal
  are zeroed out by subtracting a scaled multiple of the pivot row from every
  row below it, producing an upper triangular matrix.

- **Back substitution** — After we get the upper triangular matrix, we first calculate 
  the value of the last variable. Then plug this value to find the value of next variable. 
  Then plug these two values to find the next variables..


In the specific case of the Simple Harmonic Oscillator, the system of equations we are solving is:

.. math::
    \begin{bmatrix} 
    1 & 0 \\ 
    0 & \omega 
    \end{bmatrix}
    \begin{bmatrix} 
    a \\ 
    b 
    \end{bmatrix}
    =
    \begin{bmatrix} 
    x_0 \\ 
    v_0 
    \end{bmatrix}

Since the coefficient matrix is already diagonal, we don't strictly need ``solve()`` here, as the variables :math:`a` and :math:`b` 
are already decoupled. It is not needed in this case, but we are providing it as a pedagogical example for the general case.


Time Evolution
--------------

The time evolution operator :math:`U` encodes the full solution directly,
using ``solve()`` to determine the coefficients from initial conditions:

.. code-block:: text

    def U(k: ℝ, m: ℝ, t: ℝ, x0: ℝ, v0: ℝ): ℝ:
        omega: ℝ = (k / m) ** 0.5
        A: ℝ[2, 2] = [[1.0, 0.0], [0.0, omega]]
        B: ℝ[2] = [x0, v0]
        coeffs: ℝ[2] = solve(A, B)
        a: ℝ = coeffs[0]
        b: ℝ = coeffs[1]
        return a * cos(omega * t) + b * sin(omega * t)


Physical Constants and Initial Conditions
-----------------------------------------

.. code-block:: text

    k: ℝ = 1.0    # spring constant
    m: ℝ = 1.0    # mass

    x0: ℝ = 1.0   # initial displacement
    v0: ℝ = 0.0   # initial velocity

    U(k, m, 0.0 , x0, v0) # t = 0 * π
    U(k, m, 1.5708, x0, v0) # t = π / 2
    U(k, m, 3.1416, x0, v0) # t = π
    U(k, m, 4.7124, x0, v0) # t = 3 * π / 2
    U(k, m, 6.2832, x0, v0) # t = 2 * π

Visualize trajectory
--------------------

.. code-block:: text

    animate(U, k, m, x0, v0, 0.0, 31.1416)  # evolve over 10π

.. note::

   ``animate()`` is available as a built-in function in ``runtime.py``.


.. figure:: /_static/tutorial_files/output_harmonic_oscillator.gif
   :alt: Predicted trajecotory vs ground truth
   :align: center
   :width: 700px

Velocity via Differentiation
-----------------------------

Since :math:`U` is a differentiable function of :math:`t`, the velocity
is simply its derivative:


.. code-block:: text

    t0: ℝ = 0.0
    grad(U(k, m, t0, x0, v0), t0)

    t1: ℝ = 1.5708                          
    grad(U(k, m, t1, x0, v0), t1)

    t2: ℝ = 3.1416               
    grad(U(k, m, t2, x0, v0), t2)


References
----------

- `Harmonic Oscillator — Wikipedia <https://en.wikipedia.org/wiki/Harmonic_oscillator>`_
- `Gauss Elimination Method — Vedantu <https://www.vedantu.com/maths/gauss-elimination-method>`_
- `Gaussian Elimination — CP Algorithms <https://cp-algorithms.com/linear_algebra/linear-system-gauss.html>`_