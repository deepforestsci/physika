Solving the 1D Helmholtz Equation: From Finite Differences to Parameter Learning
=================================================================================

In this tutorial, we will solve one-dimensional Helmholtz equation, which is defined as: [Wikipedia_helmholtz]_

.. math:: 

    \frac{d^2 u}{dx^2} + k^2 u = 0

We are going to solve this using boundary value problem method, where we discretize the continuous ODE into discrete space,
and then define the boundaries of the discrete space. [Wikipedia_boundary_value]_  [Jensen_Linear_Algebra]_

We can use the central difference formula to discretize :math:`\frac{d^2 u}{dx^2}` into :math:`\frac{u_{j-1} - 2u_j + u_{j+1}}{\Delta x^2}`, 
so the discretized version of one-dimensional Helmholtz equation becomes: [Niemeyer_Finite_Difference]_

.. math::

    \frac{u_{j+1} - 2u_j + u_{j-1}}{\Delta x^2} + k^2 u_j = 0 


In Physika, we can define this by:

.. code-block:: text

    def central_difference(u: ℝ[m], j: ℕ): ℝ:
        return (u[j-1] - 2 * u[j] + u[j+1]) / (Δx**2)

    def helmholtz_equation(u: ℝ[m], j: ℕ, k: ℝ): ℝ:
        return central_difference(u, j) + k**2 * u[j]



Problem setup
----------------

We will solve the equation using domain size of :math:`x = [0, 1]`, using :math:`n = 4` equally spaced intervals, and
:math:`k = 2`.
For the boundary values, we are using :math:`u(0) = 0` and :math:`u(1) = 1`

Therefore, our discrete space :math:`X` gets divides from 0 to 1 into 4 nodes which, gives us 5 discrete grid points:


+--------------------------+------+------+------+------+------+
| **n-value**              | 0    | 1    | 2    | 3    | 4    |
+--------------------------+------+------+------+------+------+
| **node** :math:`x_j`     | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
+--------------------------+------+------+------+------+------+

Through boundary conditions we already know first and last values of :math:`n = 0, 4`, so we need to find values of interior nodes, which are
:math:`n = 1, 2, 3`. We can also calculate value of :math:`\Delta x` with :math:`\frac{1}{4}` which becomes :math:`0.25`.

This tutorial is divided into 2 main sections. In the first section, we will solve this equation numerically step by step, and 
then in second section, we will learn the parameter :math:`k` using differentiable gaussian solver with Physika code.


Section 1: Solve numerically
------------------------------



We start by substituting the central difference approximation into the
one-dimensional Helmholtz equation:

.. math::

    \frac{u_{j-1} - 2u_j + u_{j+1}}{\Delta x^2}
    + k^2 u_j = 0

For our problem, :math:`\Delta x = 0.25 = \frac{1}{4}` and :math:`k = 2`.
Therefore:

.. math::

    \frac{u_{j-1} - 2u_j + u_{j+1}}
         {\left(0.25\right)^2}
    + 2^2 u_j = 0

simplifying the equation gives:

.. math::

    16\left(u_{j-1} - 2u_j + u_{j+1}\right) + 4u_j = 0


expanding the terms:

.. math::

    16u_{j-1} - 32u_j + 16u_{j+1} + 4u_j = 0

combining the terms involving :math:`u_j`:

.. math::

    16u_{j-1} - 28u_j + 16u_{j+1} = 0 \label{eq:helmholtz_solver_discrete}

Now, we need to find values of :math:`u_j = [0, 1, 2, 3, 4]`, from which we already know boundary value as :math:`u_{j0} = 0`
and :math:`u_{j4} = 1`.
Therefore, we need to find values of :math:`u_j = 1, 2, 3`. we can start by simply putting values of required :math:`j`
into equation :math:`\ref{eq:helmholtz_solver_discrete}`


First interior node :math:`u_j = 1`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


For the first interior node, we set :math:`j = 1`:

.. math::

    16u_{1-1} - 28u_1 + 16u_{1+1} = 0

which simplifies to:

.. math::

    16u_0 - 28u_1 + 16u_2 = 0

since the boundary value :math:`u_0 = 0`, we obtain:

.. math::

    -28u_1 + 16u_2 = 0


Second interior node :math:`u_j = 2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the second interior node, we set :math:`j = 2`:

.. math::

    16u_{2-1} - 28u_2 + 16u_{2+1} = 0

which simplifies to:

.. math::

    16u_1 - 28u_2 + 16u_3 = 0


Third interior node :math:`u_j = 3`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the third interior node, we set :math:`j = 3`:

.. math::

    16u_{3-1} - 28u_3 + 16u_{3+1} = 0

which simplifies to:

.. math::

    16u_2 - 28u_3 + 16u_4 = 0

since the boundary value :math:`u_4 = 1`, we obtain:

.. math::

    16u_2 - 28u_3 = -16


Derive final system of equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The complete system of linear equations, including both the boundary
and interior nodes, is:

.. math::

    u_0 &= 0 \\
    -28u_1 + 16u_2 &= 0 \\
    16u_1 - 28u_2 + 16u_3 &= 0 \\
    16u_2 - 28u_3 &= -16 \\
    u_4 &= 1



Converting this system of linear equations into matrix form gives us:

.. math::

    \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 \\
        0 & -28 & 16 & 0 & 0 \\
        0 & 16 & -28 & 16 & 0 \\
        0 & 0 & 16 & -28 & 0 \\
        0 & 0 & 0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        u_0 \\
        u_1 \\
        u_2 \\
        u_3 \\
        u_4
    \end{bmatrix}
    =
    \begin{bmatrix}
        0 \\
        0 \\
        0 \\
        -16 \\
        1
    \end{bmatrix}



Now we can use :doc:`Gaussian elimination <linear_solve_gaussian_elimination>` method from Physika to solve this, which gives us:

.. math::

    u =
    \begin{bmatrix}
        0.0 \\
        0.5378 \\
        0.9412 \\
        1.1092 \\
        1.0
    \end{bmatrix}


we can visualize this results, by simply putting this values along with x-axi values:

We can visualize these results by simply putting the computed values
along with their corresponding :math:`x`-axis values:

+--------------------------+----------+----------+----------+----------+----------+
| **n-value**              | 0        | 1        | 2        | 3        | 4        |
+--------------------------+----------+----------+----------+----------+----------+
| **node** :math:`x_j`     | 0.00     | 0.25     | 0.50     | 0.75     | 1.00     |
+--------------------------+----------+----------+----------+----------+----------+
| **value** :math:`u_j`    | 0.0000   | 0.5378   | 0.9412   | 1.1092   | 1.0000   |
+--------------------------+----------+----------+----------+----------+----------+

.. figure:: /_static/tutorial_files/helmholtz/helmholtz_results.png
   :alt: Numerical solution of the 1D Helmholtz equation
   :align: center
   :width: 500px

   Numerical solution of the 1D Helmholtz equation

To visualize above plot, use the below python script:

.. code-block:: python

    import matplotlib.pyplot as plt

    u = [0.0, 0.5378, 0.9412, 1.1092, 1.0]
    x = [0.0, 0.25, 0.5, 0.75, 1.0]

    plt.plot(x, u, marker="o")
    plt.title("1D Helmholtz Equation")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True)

    plt.show()


Section 2: learn parameter :math:`k`
-------------------------------------

In this sub-section, we are going to learn the parameter :math:`k` from the Helmholtz equation equation using differentiable
Gaussian solver and SGD (Stochastic gradient descent) training loop defined in Physika.
We are using same setup of parameters used in Section 1 while solving numerically.

.. code-block:: text

    # helper function
    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        Δx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * Δx
        return x

    x0, x1, n: ℝ = 0, 1, 10
    Δx: ℝ = (x1 - x0) / n
    u_x0, u_x1: ℝ = 0, 1

    k: ℝ = 2

    X = linspace(0, 1, n+1)

In first section, where we solve helmholtz equation numerically, we used ``n=4`` for simplicity. But here we are using ``n=10``.
We can also use ``n=20`` or ``n=50`` to get a smoother curve on predicted trajectory, but currenly Physika doesn't have support to detach tensors
which is required in training loop. Therefore, we are using ``n=4`` which also gives expected results.

The helmholtz_equation is:

.. code-block:: text

    def central_difference(u: ℝ[m], j: ℕ): ℝ:
        return (u[j-1] - 2 * u[j] + u[j+1]) / (Δx**2)

    def helmholtz_equation(u: ℝ[m], j: ℕ, k: ℝ): ℝ:
        return central_difference(u, j) + k**2 * u[j]

For each interior node, we need to obtain the three coefficients
corresponding to :math:`u_{j-1}`, :math:`u_j`, and :math:`u_{j+1}`.
We can obtain these coefficients by evaluating the equation using
unit vectors:

Define the solver
^^^^^^^^^^^^^^^^^


.. code-block:: text

    def get_row_coeffs(j: ℕ, k: ℝ): ℝ[3]:
        e_left: ℝ[m] = zero_1d_array(n+1)
        e_left[j-1] = 1
        a: ℝ = helmholtz_equation(e_left, j, k)

        e_center: ℝ[m] = zero_1d_array(n+1)
        e_center[j] = 1
        b: ℝ = helmholtz_equation(e_center, j, k)

        e_right: ℝ[m] = zero_1d_array(n+1)
        e_right[j+1] = 1
        c: ℝ = helmholtz_equation(e_right, j, k)

        return [a, b, c]


We can use these coefficients to assemble the matrix :math:`A` and
right-hand side :math:`b` of our linear system:

.. code-block:: text

    def assemble_matrix(k: ℝ, n: ℕ): list:
        n_size: ℝ = n + 1
        A: ℝ[n_size, n_size] = zero_2d_array(n_size, n_size)
        b: ℝ[n_size] = zero_1d_array(n_size)
        for j:ℕ(1, n):
            c1, c2, c3 = get_row_coeffs(j, k)
            if j-1 == 0:
                b[j] = b[j] - c1 * u_x0
            else:
                A[j, j-1] = c1
            A[j, j] = c2
            if j+1 == n + 0:
                b[j] = b[j] - c3 * u_x1
            else:
                A[j, j+1] = c3
        # --------------------------------
        # Apply boundary conditions
        # --------------------------------
        A[0, 0] = 1
        b[0] = u_x0
        A[n, n] = 1
        b[n] = u_x1
        results: list = [A, b]
        return results


After this, we can define a ``solver`` function, which will use matrix :math:`A` and :math:`b` from ``assemble_matrx`` function,
and pass it to ``gaussian_solver`` function which will return all the :math:`u` values:
 
.. code-block:: text

    # importing gaussian_solve function from `tutorials/`, in future gaussian_solve will
    # get imported from stdlib
    from tutorials.linear_solve_gaussian_elimination import gaussian_solve, get_2d_array_num_cols, get_2d_array_num_rows, get_1d_array_length

    def solver(k: ℝ, n: ℕ): R[n]:
        results: list = assemble_matrix(k, n)
        A = results[0]
        b = results[1]
        u: ℝ[n] = gaussian_solve(A, b)
        return u
    
We use the known value :math:`k = 2` to generate our true
solution:

.. code-block:: text

    true_u = solver(k, n)

Define the loss
^^^^^^^^^^^^^^^

We are using the mean squared error (MSE) as our loss function:

.. math::

    L =
    \frac{1}{N}
    \sum_{i=0}^{N-1}
    \left(u_i^{\mathrm{true}} - u_i^{\mathrm{pred}}\right)^2

In Physika, we can define this as:

.. code-block:: text

    def mse_loss(true_u: ℝ[m], pred_u: ℝ[m]): ℝ:
        total_len: ℝ = get_1d_array_length(pred_u)
        square_diff: ℝ[5] = (true_u - pred_u) ** 2
        total: ℝ = 0
        for i:N(total_len):
            total = total + square_diff[i]
        return total / total_len

Training loop
^^^^^^^^^^^^^

We start with random guess for parameter :math:`k` as ``guess_k = 2.6``:

.. code-block:: text

    guess_k: ℝ = 2.6

At every epoch, we solve the Helmholtz equation using our current
guess for :math:`guess_k`, calculate the loss, differentiate the loss with
respect to :math:`guess_k`, and update :math:`guess_k`:

.. code-block:: text

    # dummy initial value for losses array
    losses: ℝ[1] = [100]
    epochs: ℕ = 300
    lr: ℝ = 0.01

    for i:ℕ(epochs):
        print(i)
        pred_u = solver(guess_k, n)
        loss = mse_loss(true_u, pred_u)
        losses = append(losses, loss)
        grad = grad(loss, guess_k)
        guess_k = guess_k - lr * grad
        print(guess_k)

    print(guess_k)

After training for 300 epochs, the :math:`guess_k` value should be closer to our original :math:`k` value.
Here is what the loss curve and predicted trajectory after training looks like:

.. code-block:: text

    pred_traj = solver(guess_k, n)
    plot_loss(losses)
    plot_results(X, pred_traj)    


.. figure:: /_static/tutorial_files/helmholtz/loss_curve.png
   :alt: 
   :align: center
   :width: 500px

   Loss curve after training

.. figure:: /_static/tutorial_files/helmholtz/pred_traj.png
   :alt: 
   :align: center
   :width: 500px

   Predicted trajectory after training

.. note::
        add ``plot_loss`` and ``plot_results`` function in ``physika/runtime.py`` file.

    .. code-block:: python

        def plot_loss(losses):
            import matplotlib.pyplot as plt
            plt.plot(losses[1:].detach().numpy())
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)

            plt.show()
        
        def plot_results(X, pred_traj):
            plt.plot(X, pred_traj.detach().numpy())
            import matplotlib.pyplot as plt
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("1D Helmholtz Equation")
            plt.show()


Full code
----------

.. code-block:: text

    # --------------------------------
    # Helper functions
    # --------------------------------

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols: ℝ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results

    def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
        x: ℝ[n] = zero_1d_array(n)
        Δx: ℝ = (end - start) / (n - 1)
        for i:ℕ(0, n):
            x[i] = start + i * Δx
        return x
    
    def append(x: ℝ[1], var: ℝ): ℝ[n]:
        new_length: ℝ = len(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        len_x: ℕ = get_1d_array_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results

    # -------------------------------------
    # Define domain
    # -------------------------------------


    x0, x1, n: ℝ = 0, 1, 4
    Δx: ℝ = (x1 - x0) / n
    u_x0, u_x1: ℝ = 0, 1

    k: ℝ = 2

    X = linspace(0, 1, n+1)

    # -------------------------------------
    # Discretize the equation
    # -------------------------------------

    def central_difference(u: ℝ[m], j: ℕ): ℝ:
        return (u[j-1] - 2 * u[j] + u[j+1]) / (Δx**2)

    def helmholtz_equation(u: ℝ[m], j: ℕ, k: ℝ): ℝ:
        return central_difference(u, j) + k**2 * u[j]

    # -------------------------------------
    # Helper functions to get 
    # coefficients from linear equation
    # -------------------------------------


    def get_row_coeffs(j: ℕ, k: ℝ): ℝ[3]:
        e_left: ℝ[m] = zero_1d_array(n+1)
        e_left[j-1] = 1
        a: ℝ = helmholtz_equation(e_left, j, k)
        e_center: ℝ[m] = zero_1d_array(n+1)
        e_center[j] = 1
        b: ℝ = helmholtz_equation(e_center, j, k)
        e_right: ℝ[m] = zero_1d_array(n+1)
        e_right[j+1] = 1
        c: ℝ = helmholtz_equation(e_right, j, k)
        return [a, b, c]


    def assemble_matrix(k: ℝ, n: ℕ): list:
        n_size: ℝ = n + 1
        A: ℝ[n_size, n_size] = zero_2d_array(n_size, n_size)
        b: ℝ[n_size] = zero_1d_array(n_size)
        for j:ℕ(1, n):
            c1, c2, c3 = get_row_coeffs(j, k)
            if j-1 == 0:
                b[j] = b[j] - c1 * u_x0
            else:
                A[j, j-1] = c1
            A[j, j] = c2
            if j+1 == n + 0:
                b[j] = b[j] - c3 * u_x1
            else:
                A[j, j+1] = c3
        # --------------------------------
        # Apply boundary conditions
        # --------------------------------
        A[0, 0] = 1
        b[0] = u_x0
        A[n, n] = 1
        b[n] = u_x1
        results: list = [A, b]
        return results

    # -------------------------------------
    # Define solver
    # -------------------------------------

    # importing gaussian_solve function from `tutorials/`, in future gaussian_solve will
    # get imported from stdlib
    from tutorials.linear_solve_gaussian_elimination import gaussian_solve, get_2d_array_num_cols, get_2d_array_num_rows, get_1d_array_length

    def solver(k: ℝ, n: ℕ): R[n]:
        results: list = assemble_matrix(k, n)
        A = results[0]
        b = results[1]
        u: ℝ[n] = gaussian_solve(A, b)
        return u

    true_u = solver(k, n)

    # -------------------------------------
    # loss function (MSE)
    # -------------------------------------


    def mse_loss(true_u: ℝ[m], pred_u: ℝ[m]): ℝ:
        total_len: ℝ = get_1d_array_length(pred_u)
        square_diff: ℝ[5] = (true_u - pred_u) ** 2
        total: ℝ = 0
        for i:N(total_len):
            total = total + square_diff[i]
        return total / total_len

    # -------------------------------------
    # Training loop
    # -------------------------------------

    # dummy initial value for losses array
    losses: ℝ[1] = [100]
    guess_k: ℝ = 2.6

    epochs: ℕ = 300
    lr: ℝ = 0.01

    for i:ℕ(epochs):
        print(i)
        pred_u = solver(guess_k, n)
        loss = mse_loss(true_u, pred_u)
        losses = append(losses, loss)
        grad = grad(loss, guess_k)
        guess_k = guess_k - lr * grad
        print(guess_k)


    print(guess_k)
    plot_loss(losses)

References
----------

.. [Wikipedia_helmholtz] Wikipedia contributors, *Helmholtz equation*, Wikipedia.
   https://en.wikipedia.org/wiki/Helmholtz_equation

.. [Wikipedia_boundary_value] Wikipedia contributors, *Boundary value problem*, Wikipedia.
   https://en.wikipedia.org/wiki/Boundary_value_problem

.. [Jensen_Linear_Algebra] Paul A. Jensen, *Linear Algebra: An Introduction to Data Science*, University of Illinois at Urbana-Champaign.
   https://courses.physics.illinois.edu/bioe210/sp2019/LADS_Part1.pdf

.. [Niemeyer_Finite_Difference] Kyle Niemeyer, *Finite difference method*, Mechanical Engineering Methods.
   https://kyleniemeyer.github.io/ME373-book/content/bvps/finite-difference.html