LU Decomposition method in Physika
===================================

In this tutorial we will implement the LU decomposition method in Physika. It is also recommended to read the :doc:`Gaussian elimination tutorial <linear_solve_gaussian_elimination>`,
as the concepts of pivoting and row operations explained in that tutorial will help make the LU decomposition method easier to understand.


What is LU decomposition?
---------------------------


LU decomposition, also known as LU factorization method which factors a square matrix :math:`A` into the product of two simpler matrices,
a lower triangular matrix :math:`L` and  and an upper triangular matrix :math:`U`. [WikipediaLU]_
For this tutorial, the Doolittle algorithm will be used to perform LU decomposition. This method provides an alternative way to factor :math:A without
going through the cumbersome steps of Gaussian elimination, which will be explained in the sections below.
The matrix :math:`A` is decomposed into a lower triangular matrix :math:L and an upper triangular matrix :math:`U`. Together, these matrices give the following equation: [GraphoeLU]_

.. math::

    LU = PA \label{lu_equation}


where,

- ``L`` is lower triangular matrix where diagonal entries are 1.
- ``U`` is upper triangular matrix contains the pivot rows after elimination.
- ``P`` is permutation matrix (initialized as an identity matrix,which will record row swaps performed for partial pivoting)
- ``A`` is the square matrix which we are going to solve to find ``L`` and ``U``.


This tutorial further gets divided into two main sections. In first section we will solve example matrix :math:`A` numerically to learn how 
LU decomposition method works and in second section we will implement the method in Physika.


Section 1: Solve numerically
-----------------------------


In this section we will solve numerically to understand the concept of LU decomposition, Lets consider the following matrix :math:`A` as:

.. math::

   A = \begin{bmatrix}
   -1.0 & 0.0 & 3.0 \\
   2.0 & 1.0 & 3.0 \\
   1.0 & 1.0 & 2.0
   \end{bmatrix}

In Physika we can define this matrix as:

.. code-block:: text

    A: ℝ[3, 3] = [
        [-1.0, 0.0, 3.0],
        [2.0, 1.0, 3.0],
        [1.0, 1.0, 2.0]
    ]

now lets put :math:`A` in equation :math:`\eqref{lu_equation}` which gives us:

.. math::

    L U = P \begin{bmatrix}
    -1.0 & 0.0 & 3.0 \\
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}

Since we already know that :math:`P` is an Identity matrix, lets substitute matrix :math:`P`, which will gives us:

.. math::

    L U = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    -1.0 & 0.0 & 3.0 \\
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}

Now the goal is to find :math:`L` and :math:`U` matrices.

So lets start by finding values of matrix :math:`U` through elimination method, we did same thing in gaussian elimination tutorial also, but 
for this case we are going to save the multipliers which we use for row operations.

Calculate :math:`U` matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Since the goals is to use :math:`A` and convert it into Upper triangular matrix :math:`U`, We will make zeros under the pivot row which is diagonal row of matrix :math:`A` (highlighted by red boxes)


.. math::

    A = \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{-1.0} & 0.0 & 3.0 \\
    2.0 & \bbox[2pt, border: 1.5pt solid red]{1.0} & 3.0 \\
    1.0 & 1.0 & \bbox[2pt, border: 1.5pt solid red]{2.0}
    \end{bmatrix}

We start with column 0, particularly from values under pivot value of that column, so for first column pivot value is `-1.0` (highlighted by red box).

.. math::

    \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{-1.0} \\
    2.0 \\
    1.0
    \end{bmatrix}

Here you can see that absolute value of ``A[0, 1]`` which is `2` is higher than absolute value of ``A[0, 0]`` which is `-1.0` (pivot value), we have to swap f  irst row with second row which
will give us:

.. math::

   A = \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{2.0} & 1.0 & 3.0 \\
   -1.0 & \bbox[2pt, border: 1.5pt solid red]{0.0} & 3.0 \\
   1.0 & 1.0 & \bbox[2pt, border: 1.5pt solid red]{2.0}
   \end{bmatrix}

Also remember to do the same row swapping in matrix :math:`P`, which will gives us:


.. math::

    L U = \begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{2.0} & 1.0 & 3.0 \\
    -1.0 & 0.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}


Now, lets make values under pivot value (marked as red box) in column 1 zeros, for that we have to perform row operations on Row-2 and Row-3

.. math::

    R_2 \leftarrow R_2 + 0.5 R_1 \label{first_row_operation}


.. math::

    R_3 \leftarrow R_3 - 0.5 R_1 \label{second_row_operation}

After this matrix :math:`U` becomes:

.. math::

   U = \begin{bmatrix}
   \bbox[2pt, border: 1.5pt solid red]{2.0} & 1.0 & 3.0 \\
   \color{green}{\mathbf{0.0}} & \bbox[2pt, border: 1.5pt solid red]{0.5} & 0.5 \\
   \color{green}{\mathbf{0.0}} & 0.5 & \bbox[2pt, border: 1.5pt solid red]{4.5}
   \end{bmatrix}


Now lets make zero under second pivot value which is at ``A[1, 1]`` which is in second row as 0.5, for that we will perform following row operation:

.. math::

    R_3 \leftarrow R_3 - 1 R_2 \label{third_row_operation}

After this, we finally get our matrix :math:`U` which is:

.. math::

   U = \begin{bmatrix}
   2.0 & 1.0 & 3.0 \\
   \color{green}{\mathbf{0.0}} & 0.5 & 0.5 \\
   \color{green}{\mathbf{0.0}} & \color{green}{\mathbf{0.0}} & 4.0
   \end{bmatrix}

Lets put the matrix :math:`U`, :math:`P` and :math:`A` into equation :math:`\eqref{lu_equation}`

.. math::

   L
   \begin{bmatrix}
   2.0 & 1.0 & 3.0 \\
   \color{green}{\mathbf{0.0}} & 0.5 & 0.5 \\
   \color{green}{\mathbf{0.0}} & \color{green}{\mathbf{0.0}} & 4.0
   \end{bmatrix}
   =
   \begin{bmatrix}
   0 & 1 & 0 \\
   1 & 0 & 0 \\
   0 & 0 & 1
   \end{bmatrix}
   \begin{bmatrix}
   2.0 & 1.0 & 3.0 \\
   -1.0 & 0.0 & 3.0 \\
   1.0 & 1.0 & 2.0
   \end{bmatrix}


Calculate :math:`L` matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At beginning the :math:`L` matrix is initialized as Lower Triangular matrix with all one's in diagonal row, which gives us:

.. math::

   L = \begin{bmatrix}
   1.0 & 0.0 & 0.0 \\
   l_{21} & 1.0 & 0.0 \\
   l_{31} & l_{32} & 1.0
   \end{bmatrix}

Now, the unknown entries :math:`l_{21}`, :math:`l_{31}`, and :math:`l_{32}` are simply the exact multipliers we used during the row operations in equations :math:`\eqref{first_row_operation}`
, :math:`\eqref{second_row_operation}` and :math:`\eqref{third_row_operation}` in elimination step which are:

.. math::

   \begin{aligned}
   R_2 &\leftarrow R_2 + \bbox[2pt, border: 1.5pt solid red]{0.5} R_1 \\
   R_3 &\leftarrow R_3 - \bbox[2pt, border: 1.5pt solid red]{0.5} R_1 \\
   R_3 &\leftarrow R_3 - \bbox[2pt, border: 1.5pt solid red]{1.0} R_2
   \end{aligned}

In above equation, multipliers are denoted by red boxes which are values of our :math:`L` matrix:

.. math::

   L = \begin{bmatrix}
   1.0 & 0.0 & 0.0 \\
   \color{green}{\mathbf{-0.5}} & 1.0 & 0.0 \\
   \color{green}{\mathbf{0.5}} & \color{green}{\mathbf{1.0}} & 1.0
   \end{bmatrix}

Final solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now lets put :math:`L` in equation :math:`\eqref{lu_equation}` which gives us:

.. math::

    \begin{bmatrix}
    1.0 & 0.0 & 0.0 \\
    0.5 & 1.0 & 0.0 \\
    -0.5 & 1.0 & 1.0
    \end{bmatrix}
    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    0.0 & 0.5 & 0.5 \\
    0.0 & 0.0 & 4.0
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    -1.0 & 0.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}


Now we can also check correctness of our answer by multiplying matrix :math:`L` with 


.. math::

    L \cdot U = P \cdot A


.. math::

    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0 \\
    -1.0 & 0.0 & 3.0
    \end{bmatrix}
    =
    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0 \\
    -1.0 & 0.0 & 3.0
    \end{bmatrix}


Section 2: Solve through Physika code
--------------------------------------

Before starting lets create a function as:

.. code-block:: text

    def lu_decomposition(A: ℝ[n, n]): ℝ[3, n, n]:
    ...

- ``A: ℝ[n, n]`` - represents ``A`` matrix.
- ``ℝ[3, n, n]`` - represents return type of function, which will return matrix :math:`P`, :math:`L` and :math:`U`.


.. code-block:: text
    
    n_size: ℝ = get_2d_array_num_rows(A)
    
    L: ℝ[n_size, n_size] = zeros(n_size, n_size)
    U: ℝ[n_size, n_size] = zeros(n_size, n_size)
    P: ℝ[n_size, n_size] = zeros(n_size, n_size)
    # Create P as Identity matrix
    for i:ℕ(n_size):
        P[i, i] = 1.0
    # Buffers for each matrix used in row swapping
    buf_a: ℝ[n_size] = zeros(n_size)
    buf_l: ℝ[n_size] = zeros(n_size)
    buf_p: ℝ[n_size] = zeros(n_size)


- ``n_size`` - is total number of rows to loop.
- ``L``, ``U``, ``P`` - represent the Lower triangular matrix, Upper triangular matrix, and Permutation matrix, respectively.
- ``buf_a``, ``buf_l``, ``buf_p`` - represent temporary row buffers used to perform row swaps on matrices :math:`A`, :math:`L`, and :math:`P` during partial pivoting.\

.. code-block:: text

    for j:ℕ(n_size):
        # Find pivot 
        max_row = j
        for i:ℕ(j+1, n_size):
            if abs(A[i, j]) > abs(A[max_row, j]):
                max_row = i
        # swap row if max_row got changed
        if max_row != j:
            # swap A
            for k:ℕ(n_size):
                buf_a[k] = A[j, k]
            for k:ℕ(n_size):
                A[j, k] = A[max_row, k]
            for k:ℕ(n_size):
                A[max_row, k] = buf_a[k]
            # swap L
            for k:ℕ(j):
                buf_l[k] = L[j, k]
            for k:ℕ(j):
                L[j, k] = L[max_row, k]
            for k:ℕ(j):
                L[max_row, k] = buf_l[k]
            # swap P
            for k:ℕ(n_size):
                buf_p[k] = P[j, k]
            for k:ℕ(n_size):
                P[j, k] = P[max_row, k]
            for k:ℕ(n_size):
                P[max_row, k] = buf_p[k]

This is similar to what we did in Gaussian elimination tutorial, so In this outer loop we iterate column :math:`j` ``(j = 0, 1, ..., n_size-1)`` to perform
partial pivoting, we first scan down column :math:`j` to find the row with maximum absolute value, if we find a larger pivot value we swap the rows in matrix :math:`P` and :math:`A` .

.. code-block:: text

    for i:ℕ(j + 1):
        partial = 0.0
        for k:ℕ(i):
            partial = partial + U[k, j] * L[i, k]
        U[i, j] = A[i, j] - partial


Next, in the same outer loop code after partial pivoting we start to build our Upper triangular matrix :math:`U`, same approach we did in gaussian elimination:

.. math::

    U_{ij} = A_{ij} - \sum_{k=0}^{i-1} L_{ik}\,U_{kj}, \qquad 0 \le i \le j


.. code-block:: text

    for i:ℕ(j, n_size):
            partial = 0.0
            for k:ℕ(j):
                partial = partial + U[k, j] * L[i, k]
            L[i, j] = (A[i, j] - partial) / U[j, j]

Here we build the Lower triangular matrix :math:`L`, by storing multipliers values:

.. math::

    L_{ij} = \frac{A_{ij} - \displaystyle\sum_{k=0}^{j-1} L_{ik}\,U_{kj}}{U_{jj}}, \qquad j \le i \le n-1


Here is the full ``lu_decomposition`` function:

.. code-block:: text

    def lu_decomposition(A: ℝ[n, n]): ℝ[3, n, n]:
    n_size: ℝ = get_2d_array_num_rows(A)
    
    L: ℝ[n_size, n_size] = zeros(n_size, n_size)
    U: ℝ[n_size, n_size] = zeros(n_size, n_size)
    P: ℝ[n_size, n_size] = zeros(n_size, n_size)
    # Create P as Identity matrix
    for i:ℕ(n_size):
        P[i, i] = 1.0
    # Buffers for each matrix used in row swapping
    buf_a: ℝ[n_size] = zeros(n_size)
    buf_l: ℝ[n_size] = zeros(n_size)
    buf_p: ℝ[n_size] = zeros(n_size)
    
    
    for j:ℕ(n_size):
        # Find pivot 
        max_row = j
        for i:ℕ(j+1, n_size):
            if abs(A[i, j]) > abs(A[max_row, j]):
                max_row = i
        # swap row if max_row got changed
        if max_row != j:
            # swap A
            for k:ℕ(n_size):
                buf_a[k] = A[j, k]
            for k:ℕ(n_size):
                A[j, k] = A[max_row, k]
            for k:ℕ(n_size):
                A[max_row, k] = buf_a[k]
            # swap L
            for k:ℕ(j):
                buf_l[k] = L[j, k]
            for k:ℕ(j):
                L[j, k] = L[max_row, k]
            for k:ℕ(j):
                L[max_row, k] = buf_l[k]
            # swap P
            for k:ℕ(n_size):
                buf_p[k] = P[j, k]
            for k:ℕ(n_size):
                P[j, k] = P[max_row, k]
            for k:ℕ(n_size):
                P[max_row, k] = buf_p[k]
        for i:ℕ(j + 1):
            partial = 0.0
            for k:ℕ(i):
                partial = partial + U[k, j] * L[i, k]
            U[i, j] = A[i, j] - partial
        
        for i:ℕ(j, n_size):
            partial = 0.0
            for k:ℕ(j):
                partial = partial + U[k, j] * L[i, k]
            L[i, j] = (A[i, j] - partial) / U[j, j]
    
    return [P, L, U]


We can test this function with:

.. code-block:: text

    A: ℝ[3, 3] = [
        [-1.0, 0.0, 3.0],
        [2.0, 1.0, 3.0],
        [1.0, 1.0, 2.0]
    ]


    A_original: ℝ[3, 3] = for i:ℕ(3) -> for j:ℕ(3) -> A[i, j]

    results = lu_decomposition(A)
    P_matrix: ℝ[3, 3] = results[0]
    L_matrix: ℝ[3, 3] = results[1]
    U_matrix: ℝ[3, 3] = results[2]

    print(P_matrix)
    print(L_matrix)
    print(U_matrix)

    LU: ℝ[3, 3] = L_matrix @ U_matrix
    PA: ℝ[3, 3] = P_matrix @ A_original

    print(LU)
    print(PA)




Full code
-------------

.. code-block:: text

    # Helper function
    def get_2d_array_num_rows(x: ℝ[m, n]): ℕ:
        total: ℕ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total


    def lu_decomposition(A: ℝ[n, n]): ℝ[3, n, n]:
        n_size: ℝ = get_2d_array_num_rows(A)
        
        L: ℝ[n_size, n_size] = zeros(n_size, n_size)
        U: ℝ[n_size, n_size] = zeros(n_size, n_size)
        P: ℝ[n_size, n_size] = zeros(n_size, n_size)
        # Create P as Identity matrix
        for i:ℕ(n_size):
            P[i, i] = 1.0
        # Buffers for each matrix used in row swapping
        buf_a: ℝ[n_size] = zeros(n_size)
        buf_l: ℝ[n_size] = zeros(n_size)
        buf_p: ℝ[n_size] = zeros(n_size)
        
        
        for j:ℕ(n_size):
            # Find pivot 
            max_row = j
            for i:ℕ(j+1, n_size):
                if abs(A[i, j]) > abs(A[max_row, j]):
                    max_row = i
            # swap row if max_row got changed
            if max_row != j:
                # swap A
                for k:ℕ(n_size):
                    buf_a[k] = A[j, k]
                for k:ℕ(n_size):
                    A[j, k] = A[max_row, k]
                for k:ℕ(n_size):
                    A[max_row, k] = buf_a[k]
                # swap L
                for k:ℕ(j):
                    buf_l[k] = L[j, k]
                for k:ℕ(j):
                    L[j, k] = L[max_row, k]
                for k:ℕ(j):
                    L[max_row, k] = buf_l[k]
                # swap P
                for k:ℕ(n_size):
                    buf_p[k] = P[j, k]
                for k:ℕ(n_size):
                    P[j, k] = P[max_row, k]
                for k:ℕ(n_size):
                    P[max_row, k] = buf_p[k]
            for i:ℕ(j + 1):
                partial = 0.0
                for k:ℕ(i):
                    partial = partial + U[k, j] * L[i, k]
                U[i, j] = A[i, j] - partial
            
            for i:ℕ(j, n_size):
                partial = 0.0
                for k:ℕ(j):
                    partial = partial + U[k, j] * L[i, k]
                L[i, j] = (A[i, j] - partial) / U[j, j]
        
        return [P, L, U]


    A: ℝ[3, 3] = [
        [-1.0, 0.0, 3.0],
        [2.0, 1.0, 3.0],
        [1.0, 1.0, 2.0]
    ]


    A_original: ℝ[3, 3] = for i:ℕ(3) -> for j:ℕ(3) -> A[i, j]

    results = lu_decomposition(A)
    P_matrix: ℝ[3, 3] = results[0]
    L_matrix: ℝ[3, 3] = results[1]
    U_matrix: ℝ[3, 3] = results[2]

    print(P_matrix)
    print(L_matrix)
    print(U_matrix)

    LU: ℝ[3, 3] = L_matrix @ U_matrix
    PA: ℝ[3, 3] = P_matrix @ A_original

    print(LU)
    print(PA)


References
----------

.. [WikipediaLU] Wikipedia contributors, *LU decomposition*, Wikipedia, The Free Encyclopedia. https://en.wikipedia.org/wiki/LU_decomposition
.. [GraphoeLU] Graphoe, *LU Factorization - Numerical Methods for Linear Systems*, Graphoe Resources. https://graphoe.com/resources/numerical-methods/linear-system/lu-factorization