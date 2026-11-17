Matrix inverse using Gauss-Jordan Elimination
==============================================

In this tutorial we will invert a square matrix using Gauss-Jordan elimination
[Wikipedia_GaussJordan]_ using 2 approaches:

- the naive pivoting rule that only swaps rows when a pivot is exactly zero
- the partial pivoting [Wikipedia_PartialPivot]_, which always swaps the
  largest available pivot into place.

The two algorithms that this tutorial follows are stated in
[Graves_GJEProject]_.

The method
----------

Gauss-Jordan elimination reduces a matrix to *reduced* row-echelon form, where
every pivot is 1 and is the only nonzero entry in its column. Applied to the
augmented matrix :math:`[A \mid I]`, the row operations that turn the left block
into the identity turn the right block into :math:`A^{-1}`:

.. math::

   \left[\, A \mid I \,\right]
   \;\xrightarrow{\ \text{row operations}\ }\;
   \left[\, I \mid A^{-1} \,\right]

This works because every row operation is a left-multiplication by some
invertible matrix. If the product of all of them is :math:`P`, then reducing the
left block means :math:`PA = I`, so :math:`P = A^{-1}`, and the same :math:`P`
has been applied to the right block all along.

The reduction proceeds one column at a time. At each step we hold a current row
index :math:`r` and column index :math:`c`, and we:

* **Select a pivot** : choose a row :math:`p` at or below :math:`r` whose entry
  in column :math:`c` is suitable, then :math:`R_r \leftrightarrow R_p`.
* **Scale the pivot row** : :math:`R_r \to \tfrac{1}{\alpha} R_r` for the pivot
  :math:`\alpha = A_{r,c}`, which makes the pivot 1.
* **Eliminate** : :math:`R_i \to R_i - A_{i,c} R_r` for every :math:`i \neq r`,
  which makes column :math:`c` zero everywhere but row :math:`r`.
* **Step down and right** : :math:`r \to r + 1` and :math:`c \to c + 1`.

The pivot selection step is the only place where the two algorithms differ.

Building the augmented matrix
------------------------------

``augment`` writes :math:`M` into the left half and the identity into the right
half. The nested ``if`` is the identity's Kronecker delta
[Wikipedia_KroneckerDelta]_: column ``j`` of the
right half holds a 1 only on row ``j - k``.

.. code-block:: text

    def augment(M: ℝ[x, y], k: ℕ): ℝ[a, b]:
        aug: ℝ[a, b] = zeros(k, 2*k)
        for i: ℕ(k):
            for j: ℕ(2*k):
                if j < k:
                    aug[i, j] = M[i, j]
                else:
                    if j-k == i:
                        aug[i, j] = 1.0
        return aug

Row operations as matrices
---------------------------

Rather than writing into rows of ``aug``, both algorithms build a small matrix
for each row operation and multiply. Two helpers cover everything needed.

``swap_matrix`` is the identity with rows ``ra`` and ``rb`` exchanged. When
``ra == rb`` the four assignments collapse back to ``P[ra, ra] = 1.0``, so no
branch is needed for the no-swap case.

.. code-block:: text

    def swap_matrix(ra: ℝ, rb: ℝ, k: ℕ): ℝ[a, a]:
        P: ℝ[a, a] = zeros(k, k)
        for i: ℕ(k):
            P[i, i] = 1.0
        P[ra, ra] = 0.0
        P[rb, rb] = 0.0
        P[ra, rb] = 1.0
        P[rb, ra] = 1.0
        return P

``elim_matrix`` performs the scale and the elimination in a single product. It
is the identity with the pivot's diagonal entry replaced by :math:`1/\text{piv}`
and the rest of column :math:`r` holding the elimination factors:

.. math::

   E_{r,r} = \frac{1}{\text{piv}}, \qquad
   E_{i,r} = -\frac{\text{aug}_{i,c}}{\text{piv}} \quad (i \neq r)

Multiplying by :math:`E` then carries out both row operations at once.

.. math::

   R_r \to \frac{1}{\text{piv}} R_r, \qquad
   R_i \to R_i - \frac{\text{aug}_{i,c}}{\text{piv}} R_r \quad (i \neq r)

.. code-block:: text

    def elim_matrix(aug: ℝ[x, y], r: ℕ, c: ℕ, k: ℕ): ℝ[a, a]:
        piv: ℝ = aug[r, c]
        E: ℝ[a, a] = zeros(k, k)
        for i: ℕ(k):
            E[i, i] = 1.0
        E[r, r] = 1.0/piv
        for i: ℕ(k):
            if i != r:
                E[i, r] = 0.0 - aug[i, c]/piv
        return E

Writing the row operations this way is what makes both algorithms
differentiable.

Naive pivoting
--------------

The first strategy swaps a row in only when the pivot position holds a zero, and
takes the least row below :math:`r` with a nonzero entry in that column.
If no nonzero entry exists anywhere in the column, the matrix is rank-deficient
in that column: ``r`` stays put while ``c`` advances, which is the loop's way of
moving to the next column without consuming a pivot row.

.. code-block:: text

    def gje_naive(M: ℝ[x, y], k: ℕ): ℝ[a, b]:
        aug: ℝ[a, b] = augment(M, k)
        r: ℕ = 0
        for c: ℕ(k):
            if r < k:
                found = 0.0
                cand = r
                if aug[r, c] != 0:
                    found = 1.0
                for i: ℕ(r+1, k):
                    if found < 0.5:
                        if aug[i, c] != 0:
                            cand = i
                            found = 1.0
                if found > 0.5:
                    Pm = swap_matrix(r, cand, k)
                    aug = Pm @ aug
                    Em = elim_matrix(aug, r, c, k)
                    aug = Em @ aug
                    r = r + 1
        return aug

Swamping: why naive pivoting is not enough
-------------------------------------------

Consider the following matrix:

.. math::

   A = \begin{bmatrix} 10^{-20} & 1 \\ 1 & 2 \end{bmatrix}

The pivot :math:`10^{-20}` is nonzero, so naive pivoting never swaps. It divides
by :math:`10^{-20}`, producing entries of order :math:`10^{20}`, and the
subsequent subtraction of :math:`2 - 10^{20}` rounds to :math:`-10^{20}` (the
original 2 is annihilated entirely). This failure mode is called *swamping*, and
nothing about it is visible in the output: no zero pivot, no NaN, no warning.
[Higham2002]_ analyses this absorption and the growth factors behind it in
detail.

Partial pivoting
----------------

The fix is to choose the pivot row to be the one with the largest absolute
value in the subcolumn, rather than merely a nonzero one.

.. code-block:: text

    def gje_pp(M: ℝ[x, y], k: ℕ, ptol: ℝ): ℝ[a, b]:
        aug: ℝ[a, b] = augment(M, k)
        r: ℕ = 0
        for c: ℕ(k):
            if r < k:
                p = r
                pmax = abs(aug[r, c])
                for i: ℕ(r+1, k):
                    if abs(aug[i, c]) > pmax:
                        pmax = abs(aug[i, c])
                        p = i
                if pmax > ptol:
                    Pm = swap_matrix(r, p, k)
                    aug = Pm @ aug
                    Em = elim_matrix(aug, r, c, k)
                    aug = Em @ aug
                    r = r + 1
        return aug

Detecting a singular matrix
----------------------------

The reduction succeeds exactly when the left block ends up equal to
:math:`I_k`, so the test is a comparison against the identity over the *whole*
block [MaLi2017]_. Checking only the diagonal is not sufficient as the matrix
:math:`\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`, reduced without any
swapping, leaves a left block of
:math:`\begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}`, whose diagonal sums to
:math:`k` even though the block is not the identity.

.. code-block:: text

    def identity_defect(aug: ℝ[x, y], k: ℕ): ℝ:
        total: ℝ = 0.0
        for i: ℕ(k):
            for j: ℕ(k):
                if i == j:
                    total += abs(aug[i, j] - 1.0)
                else:
                    total += abs(aug[i, j])
        return total

When the defect is within tolerance the right half is returned as the inverse.
Otherwise the result is a matrix of NaN.

.. code-block:: text

    def extract_inv(aug: ℝ[x, y], k: ℕ): ℝ[a, b]:
        out: ℝ[a, b] = zeros(k, k)
        out[:, :] = aug[:, k:2*k]
        return out

    def nan_matrix(k: ℕ): ℝ[k, k]:
        return for i: ℕ(k) → for j: ℕ(k) → sqrt(0.0 - 1.0)

    def inverse_naive(M: ℝ[x, y], k: ℕ, tol: ℝ): ℝ[a, b]:
        red: ℝ[u, v] = gje_naive(M, k)
        if identity_defect(red, k) < tol:
            return extract_inv(red, k)
        else:
            return nan_matrix(k)

    def inverse_pp(M: ℝ[x, y], k: ℕ, tol: ℝ, ptol: ℝ): ℝ[a, b]:
        red: ℝ[u, v] = gje_pp(M, k, ptol)
        if identity_defect(red, k) < tol:
            return extract_inv(red, k)
        else:
            return nan_matrix(k)

Examples
--------

Four matrices exercise the interesting cases. Each is inverted by both
strategies and by the built-in ``⁻¹`` operator for comparison. Physika computes
in float32, so the values below are the printed results rather than exact
decimals, and the matrix rows are split across lines for readability.

.. code-block:: text

    n: ℕ = 3
    tol: ℝ = 0.001
    ptol: ℝ = 1.0e-6

A well conditioned matrix, where all three agree.

.. code-block:: text

    A: ℝ[n, n] = [[5.0, 7.0, 9.0], [4.0, 3.0, 8.0], [7.0, 5.0, 6.0]]
    inverse_naive(A, n, tol)
    inverse_pp(A, n, tol, ptol)
    A⁻¹

Output::

    naive:    [[-0.20952379703521729, 0.028571395203471184, 0.27619048953056335],
               [0.3047619163990021, -0.3142857253551483, -0.038095228374004364],
               [-0.009523831307888031, 0.22857145965099335, -0.12380953133106232]]
    pp:       [[-0.20952384173870087, 0.028571441769599915, 0.27619048953056335],
               [0.3047619163990021, -0.3142857551574707, -0.03809523582458496],
               [-0.009523805230855942, 0.22857142984867096, -0.12380953133106232]]
    built-in: [[-0.20952382683753967, 0.028571423143148422, 0.27619048953056335],
               [0.3047619163990021, -0.3142856955528259, -0.03809525445103645],
               [-0.009523811750113964, 0.22857141494750977, -0.12380952388048172]]

A singular matrix, whose rows are linearly dependent. Both strategies detect it
and return NaN. The built-in operator is not applied here, because it raises an
error on a singular matrix instead of returning a value.

.. code-block:: text

    B: ℝ[n, n] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    inverse_naive(B, n, tol)
    inverse_pp(B, n, tol, ptol)

Output::

    naive:    [[nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]
    pp:       [[nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]

A zero in the pivot position. Naive pivoting handles this by construction, since
a zero pivot is exactly the case it swaps for, so all three agree exactly.

.. code-block:: text

    n2: ℕ = 2
    P: ℝ[n2, n2] = [[0.0, 1.0], [1.0, 0.0]]
    inverse_naive(P, n2, tol)
    inverse_pp(P, n2, tol, ptol)
    P⁻¹

Output::

    naive:    [[0.0, 1.0], [1.0, 0.0]]
    pp:       [[0.0, 1.0], [1.0, 0.0]]
    built-in: [[0.0, 1.0], [1.0, 0.0]]

The swamping case, and the only one where the strategies disagree.

.. code-block:: text

    S: ℝ[n2, n2] = [[1.0e-20, 1.0], [1.0, 2.0]]
    inverse_naive(S, n2, tol)
    inverse_pp(S, n2, tol, ptol)
    S⁻¹

Output::

    naive:    [[0.0, 1.0], [1.0, -9.999999682655225e-21]]
    pp:       [[-2.0, 1.0], [1.0, -9.999999682655225e-21]]
    built-in: [[-2.0, 1.0], [1.0, -9.999999682655225e-21]]

The ``-2.0`` has been lost entirely by the naive strategy, while partial
pivoting matches the built-in inverse exactly. Note that the zero pivot in ``P``
is *not* where naive pivoting fails. It takes a pivot that is small but nonzero
to expose the difference.

Differentiability
-----------------

Both reductions can be differentiated. For example:

.. code-block:: text

    def total_naive(M: ℝ[x, y], k: ℕ, tol: ℝ): ℝ:
        return sum(inverse_naive(M, k, tol))

    def total_pp(M: ℝ[x, y], k: ℕ, tol: ℝ, ptol: ℝ): ℝ:
        return sum(inverse_pp(M, k, tol, ptol))

    ∇(total_naive(A, n, tol), A)
    ∇(total_pp(A, n, tol, ptol), A)

Output::

    naive:  [[-0.008163264021277428, 0.004081629682332277, -0.008163265883922577],
             [0.0054421634413301945, -0.002721085213124752, 0.005442182533442974],
             [-0.01088434737175703, 0.005442176479846239, -0.010884357616305351]]

    pp:     [[-0.008163266815245152, 0.004081634804606438, -0.008163264952600002],
             [0.005442180670797825, -0.0027210917323827744, 0.0054421816021203995],
             [-0.010884357616305351, 0.005442179273813963, -0.010884356684982777]]

References
----------

.. [Graves_GJEProject] S. J. Graves, *Project: Gauss-Jordan Elimination*,
  section 5.2 of AAM: Algorithms for Applied Mathematics. Source of
  Algorithm 5.2.1 (naive pivoting) and Algorithm 5.2.5 (partial pivoting).
  https://sj-graves.github.io/algorithms-book/sec-matapps-gje_proj.html
.. [Wikipedia_GaussJordan] Wikipedia contributors, *Gauss-Jordan elimination*,
  Wikipedia, The Free Encyclopedia.
  https://en.wikipedia.org/wiki/Gaussian_elimination#Gauss%E2%80%93Jordan_elimination
.. [Wikipedia_PartialPivot] Wikipedia contributors, *Pivot element*, Wikipedia,
  The Free Encyclopedia. https://en.wikipedia.org/wiki/Pivot_element
.. [Wikipedia_KroneckerDelta] Wikipedia contributors, *Kronecker delta*,
  Wikipedia, The Free Encyclopedia.
  https://en.wikipedia.org/wiki/Kronecker_delta
.. [Higham2002] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
  2nd edition, SIAM, 2002. Chapter 2 (Floating-Point Arithmetic) and Chapter 9
  (LU Factorization) analyse floating-point absorption and the growth factors
  that arise during Gaussian elimination.
.. [MaLi2017] J. Ma and Y. Li, *Gauss-Jordan elimination method for computing all
  types of generalized inverses related to the {1}-inverse*, Journal of
  Computational and Applied Mathematics, 321 (2017) 26-43.
  https://doi.org/10.1016/j.cam.2017.02.010
