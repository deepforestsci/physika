Linear solve using Gaussian Elimination
============================================

In this tutorial we will learn how to solve linear equations using Linear solve method, particularly
Gaussian Elimination method [Wikipedia_GaussianElim]_ .

The Equation
------------

Following is the linear equations which we are going to solve:

.. math::
 
   \begin{aligned}
   x + 2y + z &= 8 \\
   3x + y - z &= 2 \\
   2x - y + z &= 3
   \end{aligned}
   \label{linear_equation_form}

First we will write this equations in form of ``Ax = B`` which is as follow:

.. math::

    \underset{A}{
    \begin{bmatrix}
    1 & 2 & 1 \\
    3 & 1 & -1 \\
    2 & -1 & 1
    \end{bmatrix}}
    \underset{x}{
    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}}
    =
    \underset{B}{
    \begin{bmatrix}
    8 \\
    2 \\
    3
    \end{bmatrix}}
    \label{ax_equation}

In physika we define this matrices such as:

.. code-block:: text

    A: ℝ[3,3] = [
        [1, 2, 1],
        [3, 1, -1],
        [2, -1, 1]
    ]
    b: ℝ[3] = [8, 2, 3]



Gaussian elimination method
------------------------------

Gaussian elimination is a method for solving a system of linear equations which are in form of :math:`Ax = b`. It works by
eliminating variables, row by row, until the system (matrix) is reduced to a form that can be solved directly which is defined as
Upper triangular matrix :math:`U`. In this tutorial, we will understand the method step by step with code.

The method can be divided into three main steps:

* **Build the augmented matrix** - Combine the coefficient matrix :math:`A` and right-hand side vector :math:`b` into single Augmented matrix, this allows same row operations applied to both
  matrices at once.
* **Perform Forward elimination** - Use row operations to eliminate variables and transform the matrix into upper triangular matrix.
* **Back substitution to find values** - Start with the last equation and solve upwards to find values of all variables.

Before starting lets create a function as:

.. code-block:: text

    def gaussian_solve(A: ℝ[m, n], b: ℝ[n]): ℝ[m]:
        ...

- ``A: ℝ[m, n]`` - represents ``A`` matrix
- ``b: ℝ[n]`` - represents ``b`` matrix
- ``ℝ[m]`` - represents return type, which will be solution vector for x, y, z values
  
Step 1 - Augmented matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the row operations need to be applied to both the :math:`A` and :math:`b` matrix, we combine them into 
a single **Augmented matrix**, In Physika, this can be written as:


.. code-block:: text

    a_row: ℝ = get_2d_array_num_rows(A)
    a_col: ℝ = get_2d_array_num_cols(A)

    new_col: ℝ = a_col + 1
    aug: ℝ[a_row, new_col] = zeros(a_row, new_col)
    for i:ℕ(a_row):
        aug[i, :a_col] = A[i, :]
        aug[i, a_col] = b[i]

Here, the :math:`A` matrix is of size ``3x3`` and :math:`b` is of size ``1x3``, the augmented matrix will have shape of ``3x4``
so we loop through number of rows of :math:`A` which is 3, and add each row from :math:`A` ``aug[i, :a_col] = A[i, :]``
and :math:`b` ``aug[i, a_col] = b[i]`` together into the ``aug`` matrix row.

After that augmented matrix looks like this:

.. math::

    \left[\begin{array}{ccc|c}
    1 &  2 &  1 & 8 \\
    3 & 1 & -1 & 2 \\
    2 & -1 &  1 & 3
    \end{array}\right]


Step 2 - Forward elimination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of the Forward elimination is to transform the augmented matrix into an Upper triangular matrix.
We do this by going through each column and removing the values below the diagonal.

This process gets furher divided into 3 sub-section, which are:

- **Partial pivoting** - Choose the pivot as largest absolute value in the current column.
- **swap rows using buffer** - Swap the pivot row with current row.
- **Elimination** - Use pivot value to eliminate entries below it.

We will go through each of this in detail, To give some context, here is the physika code
that performs Forward elimination:


.. code-block:: text

    # -------------------------
    # Forward elimination
    # -------------------------
    for i:ℕ(a_row):
        # -------------------------
        # Partial pivoting
        # -------------------------
        max_row = i
        for k:ℕ(i + 1, a_row):
            if abs(aug[k, i]) > abs(aug[max_row, i]):
                max_row = k
        # ----------------------------------------
        # Swap max_row with the current i-th row
        # ----------------------------------------
        if max_row != i:
            lo = min(i, max_row)
            hi = max(i, max_row)
            rows_above = aug[:lo, :]
            pivot_row = aug[hi:hi+1, :]
            rows_between = aug[lo+1:hi, :]
            current_row = aug[lo:lo+1, :]
            rows_below = aug[hi+1:, :]
            aug = concat(
                rows_above,
                pivot_row,
                rows_between,
                current_row,
                rows_below,
            )
        # -------------------------
        # Forward elimination
        # -------------------------
        pivot_row = aug[i:i+1, :]
        pivot_value = aug[i, i]
        rows_below = aug[i+1:, :]
        elimination_factors = rows_below[:, i:i+1] / pivot_value
        eliminated_rows = rows_below - elimination_factors * pivot_row
        aug = concat(aug[:i+1, :], eliminated_rows)

The outer for loop will loop through each row of the augmented matrix ``a_row`` which value is 3.

Now we will go step by step in first iteration of outer loop.

2.1 Partial pivoting
********************


For each step, we start with the diagonal element as the pivot. We then
look at the values below it in the same column and choose the one with the
largest absolute value. This is called **partial pivoting**. [Chasnov_PartialPivot]_ 

.. code-block:: text

    # -------------------------
    # Partial pivoting
    # -------------------------
    max_row = i
    for k:ℕ(i + 1, a_row):
        if abs(aug[k, i]) > abs(aug[max_row, i]):
            max_row = k

For the first iteration the pivot is at first value of first row,
column which is denoted by red box in below matrix:

.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\boxed{1}} & 2 & 1 & 8 \\
    3 & 1 & -1 & 2 \\
    2 & -1 & 1 & 3
    \end{array}\right]


The ``k`` loop here starts from row number 1, and pluck out each number from first column:

.. math::

    \begin{bmatrix}
    \color{red}{\boxed{1}} \\
    3 \\
    2
    \end{bmatrix}

and then sequentially (column wise) we compare each values with each other and update the ``max_row`` value.
For example:

.. math::

   \begin{array}{cccc}
   \text{Step} & \text{Comparison} & \text{Decision} & max\_row \\[1.5ex]
   \hline \\[-1ex]
   \text{Init} & - & - & \mathbf{0} \\[1.5ex]
   k = 1 & |3| > |1| & \color{green}{\checkmark \text{ Update}} & \mathbf{1} \\[1.5ex]
   k = 2 & |2| > |3| & \color{gray}{\times \text{ Skip}} & \mathbf{1}
   \end{array}

Now after this value of ``max_row`` gets updated to 1 which is second row.


2.2 swap rows into buffer
***************************

After partial pivoting, ``max_row`` contains the row that has the largest pivot value.
If ``max_row`` is different from the current row ``i``, we need to swap these two rows. 
Since ``max_row`` value got updated to 1, which is the second row we will swap the first and second row.

Before row swap:

.. math::

    \left[\begin{array}{ccc|c}
    1 & 2 & 1 & 8 \\
    3 & 1 & -1 & 2 \\
    2 & -1 & 1 & 3
    \end{array}\right]
    \begin{array}{l}
    \left.\begin{array}{c} ~ \\ ~ \end{array}\right\} \text{Swap} \\
    ~
    \end{array}

After row swap:

.. math::

    \left[\begin{array}{ccc|c}
    \color{green}{\mathbf{3}} & \color{green}{\mathbf{1}} & \color{green}{\mathbf{-1}} & \color{green}{\mathbf{2}} \\
    1 & 2 & 1 & 8 \\
    2 & -1 & 1 & 3
    \end{array}\right]


also after swapping the pivot value also gets updated now which is 3:


.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\boxed{3}} & 1 & -1 & 2 \\
    1 & 2 & 1 & 8 \\
    2 & -1 & 1 & 3
    \end{array}\right]


Now to do this with Physika code, we split the augmented matrix into different sections around
the two rows being swapped. ``pivot_row`` contains the selected ``max_row``, while ``current_row`` contains
the current i-th row. The remaining rows are kept in their original order. We then concatenate these sections
with the two rows exchanged.

.. code-block:: text

    # ----------------------------------------
    # Swap max_row with the current i-th row
    # ----------------------------------------
    if max_row != i:
        lo = min(i, max_row)
        hi = max(i, max_row)
        rows_above = aug[:lo, :]
        pivot_row = aug[hi:hi+1, :]
        rows_between = aug[lo+1:hi, :]
        current_row = aug[lo:lo+1, :]
        rows_below = aug[hi+1:, :]
        aug = concat(
            rows_above,
            pivot_row,
            rows_between,
            current_row,
            rows_below,
        )

Here, ``rows_above`` and ``rows_below`` remain unchanged, while ``pivot_row`` and ``current_row`` are placed in each other's positions.


2.3 Elimination
***************************

Once the row-swapping is done, we move to the Elimination section where we transform our augmented matrix into upper-triangular form.
For this first iteration we will eliminate all the entries below the pivot value to zeros.

.. code-block:: text

    # -------------------------
    # Forward elimination
    # -------------------------
    pivot_row = aug[i:i+1, :]
    pivot_value = aug[i, i]
    rows_below = aug[i+1:, :]
    elimination_factors = rows_below[:, i:i+1] / pivot_value
    eliminated_rows = rows_below - elimination_factors * pivot_row
    aug = concat(aug[:i+1, :], eliminated_rows)


Here, ``pivot_row`` contains the current pivot row, while rows_below contains all rows below it. We then calculate
an ``elimination_factor`` for each row based on its value in the pivot column.
These factors are then used to subtract a multiple of the pivot row from each row below, making the entries below the pivot equal to zero.
Finally, the unchanged rows above and including the pivot are concatenated with the eliminated_rows to form the updated augmented matrix.
After repeating this process for each pivot, the augmented matrix is transformed into upper-triangular form.

Lets do this numerically to understand better:
From our previous row-swapping step, the augmented matrix is:

.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\mathbf{3}} & 1 & -1 & 2 \\
    1 & 2 & 1 & 8 \\
    2 & -1 & 1 & 3
    \end{array}\right]

The goal of this step is to make all entries below the pivot :math:`\color{red}{\mathbf{3}}` equal to zero.

For the first row below the pivot, the elimination factor is:

.. math::

    \frac{1}{3}

We subtract :math:`\frac{1}{3}` times the first row from the second row:

.. math::

    \begin{array}{rcccl}
    \text{source row:} & [1, & 2, & 1, & 8] \\
    -\left(\frac{1}{3} \times \text{pivot row}\right): &
    -[1, & \frac{1}{3}, & -\frac{1}{3}, & \frac{2}{3}] \\[1ex]
    \hline \\[-1.5ex]
    \text{new row:} & [\mathbf{0}, & \mathbf{\frac{5}{3}}, &
    \mathbf{\frac{4}{3}}, & \mathbf{\frac{22}{3}}]
    \end{array}

For the second row below the pivot, the elimination factor is:

.. math::

    \frac{2}{3}

We subtract :math:`\frac{2}{3}` times the first row from the third row:

.. math::

    \begin{array}{rcccl}
    \text{source row:} & [2, & -1, & 1, & 3] \\
    -\left(\frac{2}{3} \times \text{pivot row}\right): &
    -[2, & \frac{2}{3}, & -\frac{2}{3}, & \frac{4}{3}] \\[1ex]
    \hline \\[-1.5ex]
    \text{new row:} & [\mathbf{0}, & \mathbf{-\frac{5}{3}}, &
    \mathbf{\frac{5}{3}}, & \mathbf{\frac{5}{3}}]
    \end{array}

Therefore, after the first elimination step, the augmented matrix becomes:

.. math::

    \left[\begin{array}{ccc|c}
    \color{red}{\mathbf{3}} & 1 & -1 & 2 \\[1ex]
    \color{green}{\mathbf{0}} & \frac{5}{3} & \frac{4}{3} & \frac{22}{3} \\[1ex]
    \color{green}{\mathbf{0}} & -\frac{5}{3} & \frac{5}{3} & \frac{5}{3}
    \end{array}\right]

This completes the first iteration of the outer loop. We then move to the next pivot and repeat the same process, eliminating all entries below it.

After all iterations, the augmented matrix is transformed into upper-triangular form:

.. math::

    \left[\begin{array}{ccc|c}
    \mathbf{3} & 1 & -1 & 2 \\[1ex]
    0 & \mathbf{\frac{5}{3}} & \frac{4}{3} & \frac{22}{3} \\[1ex]
    0 & 0 & \mathbf{3} & 9
    \end{array}\right]


Step 3 - Back substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Once the forward elimination transforms augmented matrix into upper-triangular matrix, we use Back substitution to find values
of x, y, and z.
To understand why this is called as "Back substitution", lets take a look at our final upper-triangular matrix

.. math::

    \left[\begin{array}{ccc|c}
    \mathbf{3} & 1 & -1 & 2 \\[1ex]
    0 & \mathbf{\frac{5}{3}} & \frac{4}{3} & \frac{22}{3} \\[1ex]
    0 & 0 & \mathbf{3} & 9
    \end{array}\right]

now lets convert each row into system of linear equations, just like at beginning of the tutorial we converted linear equations into matrix form,
here we convert matrix form into linear equations:


.. math::

   \begin{aligned}
   3x + y - z &= 2 \\[1ex]
   \frac{5}{3}y + \frac{4}{3}z &= \frac{22}{3} \\[1ex]
   3z &= 9
   \end{aligned}

Notice how the third equation contains only one variable :math:`z` which we can easily solve and find value of :math:`z`:


Solve for :math:`z` (Row 3)
****************************************

From the third equation:

.. math::

   3z = 9

.. math::

   z = \frac{9}{3} = \mathbf{3}

Solve for :math:`y` (Row 2)
****************************************

Substitute :math:`z = 3` into the second equation:

.. math::

   \frac53y + \frac43(3) = \frac{22}3

.. math::

   \frac53y + 4 = \frac{22}3

.. math::

   \frac53y = \frac{22}3 - \frac{12}3 = \frac{10}3

.. math::

   y = \frac{10}3 \times \frac35 = \mathbf2


Solve for :math:`x` (Row 1)
****************************************

Substitute :math:`y = 2` and :math:`z = 3` into the first equation:

.. math::

   3x + 2 - 3 = 2

.. math::

   3x - 1 = 2

.. math::

   3x = 3

.. math::

   x = \frac33 = \mathbf1


Therefore, the final solution vector is

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z
    \end{bmatrix}
    =
    \begin{bmatrix}
    1\\
    2\\
    3
    \end{bmatrix}

We can do this in Physika code by using below code:

.. code-block:: text

    # -------------------------
    # Back substitution
    # -------------------------
    x = zeros(0)
    for i:N(a_row):
        idx = a_col - 1 - i
        total = aug[idx, a_col]
        for j:N(idx+1, a_row):
            total = total - aug[idx, j] * x[j - idx - 1]
        val = total / aug[idx, idx]
        val = reshape(val, 1)
        x = concat(val, x)
    return x


Differentiability
------------------


This implementation is fully differentiable. See the :doc:`1D Helmholtz tutorial <linear_solve_helmholtz_equation>` for a
complete example using this solver.


Efficiency on Larger Matrices
------------------------------

To evaluate the computational efficiency of this implementation of gaussian elimination, we measure its execution time for matrices of increasing sizes.
The following table shows the time taken to solve linear systems of different dimensions (approx time).

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Matrix Size
     - Time (s)
   * - 200 × 200
     - 0.34
   * - 500 × 500
     - 2.30
   * - 1000 × 1000
     - 11.00
   * - 2000 × 2000
     - 56.00



Full code
---------

.. code-block:: text

    # ----------------------------
    # Helper functions
    # ----------------------------

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols:ℝ ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:N(rows) -> for j:N(cols) -> j*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def get_2d_array_num_rows(x: ℝ[m, n]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def get_2d_array_num_cols(x: ℝ[m, n]): ℝ:
        return get_1d_array_length(x[0])

    def arange(n: ℕ): ℝ[n]:
        arr: ℝ[n] = for i: ℕ(n) → i
        return arr

    def max(x: ℝ, y: ℝ): ℝ:
        if x>y:
            return x
        else:
            return y

    def min(x: ℝ, y: ℝ): ℝ:
        if x<y:
            return x
        else:
            return y


    # ----------------------------
    # Gaussial solve function
    # ----------------------------


    def gaussian_solve(A: R[m, n], b: R[n]): R[n]:
        a_row: ℝ = get_2d_array_num_rows(A)
        a_col: ℝ = get_2d_array_num_cols(A)
        # -------------------------
        # Create augmented matrix
        # -------------------------
        new_col: ℝ = a_col + 1
        aug: ℝ[a_row, new_col] = zeros(a_row, new_col)
        for i:ℕ(a_row):
            aug[i, :a_col] = A[i, :]
            aug[i, a_col] = b[i]
        # -------------------------
        # Forward elimination
        # -------------------------
        for i:ℕ(a_row):
            # -------------------------
            # Partial pivoting
            # -------------------------
            max_row = i
            for k:ℕ(i + 1, a_row):
                if abs(aug[k, i]) > abs(aug[max_row, i]):
                    max_row = k
            # ----------------------------------------
            # Swap max_row with the current i-th row
            # ----------------------------------------
            if max_row != i:
                lo = min(i, max_row)
                hi = max(i, max_row)
                rows_above = aug[:lo, :]
                pivot_row = aug[hi:hi+1, :]
                rows_between = aug[lo+1:hi, :]
                current_row = aug[lo:lo+1, :]
                rows_below = aug[hi+1:, :]
                aug = concat(
                    rows_above,
                    pivot_row,
                    rows_between,
                    current_row,
                    rows_below,
                )
            # -------------------------
            # Forward elimination
            # -------------------------
            pivot_row = aug[i:i+1, :]
            pivot_value = aug[i, i]
            rows_below = aug[i+1:, :]
            elimination_factors = rows_below[:, i:i+1] / pivot_value
            eliminated_rows = rows_below - elimination_factors * pivot_row
            aug = concat(aug[:i+1, :], eliminated_rows)
        # -------------------------
        # Back substitution
        # -------------------------
        x = zeros(0)
        for i:N(a_row):
            idx = a_col - 1 - i
            total = aug[idx, a_col]
            for j:N(idx+1, a_row):
                total = total - aug[idx, j] * x[j - idx - 1]
            val = total / aug[idx, idx]
            val = reshape(val, 1)
            x = concat(val, x)
        return x


    A: ℝ[3,3] = [
        [1, 2, 1],
        [3, 1, -1],
        [2, -1, 1]
    ]
    b: ℝ[3] = [8, 2, 3]

    gaussian_solve(A, b)


References
----------

.. [Wikipedia_GaussianElim] Wikipedia contributors, *Gaussian elimination*, Wikipedia,
  The Free Encyclopedia. https://en.wikipedia.org/wiki/Gaussian_elimination
.. [Chasnov_PartialPivot] J. R. Chasnov, *Partial Pivoting*, in Numerical Methods,
  LibreTexts, Hong Kong University of Science and Technology.
  https://math.libretexts.org/Bookshelves/Applied_Mathematics/Numerical_Methods_(Chasnov)/03%3A_System_of_Equations/3.03%3A_Partial_Pivoting