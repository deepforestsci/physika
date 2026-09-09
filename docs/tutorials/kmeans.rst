K-Means Clustering
==================

This tutorial implements **K-Means clustering** (Lloyd's algorithm) in Physika.
Unlike most of the other tutorials it is a *pure forward* computation: the
assignment step is an :math:`\operatorname{arg\,min}`, which is piecewise
constant, so K-Means has no useful gradient and there are no ``grad(...)`` calls.
What it does exercise is ordinary Physika programming — functions, ``for``
loops, ``for``-expressions, ``if``/``else`` and shape checking.

We fix the number of clusters at :math:`k = 2` so the assignment can be written
as a single comparison. The generalisation to :math:`k > 2` is an
:math:`\operatorname{arg\,min}` loop over the centroids.


The Algorithm
-------------

Given :math:`n` points :math:`x_i \in \mathbb{R}^2` and :math:`k` centroids
:math:`c_j`, Lloyd's algorithm alternates two steps until the centroids stop
moving:

.. math::

   \text{(assign)} \quad & \ell_i \;=\; \operatorname*{arg\,min}_{j}\;
        \lVert x_i - c_j \rVert^2 \\[4pt]
   \text{(update)} \quad & c_j \;=\;
        \frac{1}{|\{i : \ell_i = j\}|} \sum_{i : \ell_i = j} x_i

The objective it (locally) minimises is the within-cluster sum of squares,
:math:`\sum_i \lVert x_i - c_{\ell_i} \rVert^2`, which decreases at every step,
so the iteration converges.


Step 1: Squared distance
------------------------

Working with the *squared* distance avoids a ``sqrt`` and does not change which
centroid is nearest:

.. code-block:: text

    def sq_dist(a: ℝ[2], b: ℝ[2]): ℝ:
        e0: ℝ = a[0] - b[0]
        e1: ℝ = a[1] - b[1]
        return e0 * e0 + e1 * e1


Step 2: Assignment step
-----------------------

With :math:`k = 2` the ``arg min`` is just "which of the two is closer": compute
both squared distances and compare them.

.. code-block:: text

    def nearest_of_two(point: ℝ[2], C: ℝ[2, 2]): ℝ:
        d0: ℝ = sq_dist(point, C[0])
        d1: ℝ = sq_dist(point, C[1])
        label: ℝ = 0.0
        if d1 < d0:
            label = 1.0
        return label

    def assign_labels(X: ℝ[n, 2], C: ℝ[2, 2]): ℝ[n]:
        return for i:ℕ(n) -> nearest_of_two(X[i], C)

The ``for``-expression ``for i:ℕ(n) -> nearest_of_two(X[i], C)`` builds the
length-``n`` label vector in one line — one call per row of ``X``.

.. note::
   The comparison is deliberately kept between two scalars returned by
   ``sq_dist``. Comparing a value read straight out of a tensor at a loop-varying
   index (``if X[i] < ...``) is not something the type checker can currently
   resolve; routing the value through a function whose return type is ``ℝ``
   sidesteps that.


Step 3: Update step
-------------------

Each new centroid is the mean of the points that currently carry its label. We
accumulate the coordinate sums and a count in a single pass, then divide. If a
cluster is momentarily empty we keep its previous centroid rather than divide by
zero.

.. code-block:: text

    def cluster_mean(X: ℝ[n, 2], labels: ℝ[n], target: ℝ, fallback: ℝ[2]): ℝ[2]:
        s0: ℝ = 0.0
        s1: ℝ = 0.0
        cnt: ℝ = 0.0
        for i:ℕ(n):
            if labels[i] == target:
                s0 += X[i, 0]
                s1 += X[i, 1]
                cnt += 1.0
        if cnt > 0.0:
            return [s0 / cnt, s1 / cnt]
        else:
            return fallback

    def update_centroids(X: ℝ[n, 2], labels: ℝ[n], C_old: ℝ[2, 2]): ℝ[2, 2]:
        c0: ℝ[2] = cluster_mean(X, labels, 0.0, C_old[0])
        c1: ℝ[2] = cluster_mean(X, labels, 1.0, C_old[1])
        return [c0, c1]


Step 4: The Lloyd iteration
---------------------------

A toy dataset of two well-separated blobs, centroids seeded from two data points,
and a fixed number of sweeps:

.. code-block:: text

    n: ℝ = 8
    iters: ℕ = 10

    X: ℝ[8, 2] = [
        [1.0, 1.0], [1.5, 2.0], [1.0, 2.5], [2.0, 1.5],
        [8.0, 8.0], [8.5, 9.0], [9.0, 8.5], [7.5, 9.5]
    ]

    C: ℝ[2, 2] = [[1.0, 1.0], [8.0, 8.0]]

    labels: ℝ[8] = for i:ℕ(n) -> i * 0.0
    for step:ℕ(iters):
        labels = assign_labels(X, C)
        C = update_centroids(X, labels, C)

    print(labels)
    print(C)


Results
-------

The two blobs are cleanly separated, so the iteration converges after a single
sweep and then stays put::

    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] ∈ ℝ[8]
    [[1.375, 1.75], [8.25, 8.75]] ∈ ℝ[2,2]

The first four points land in cluster ``0`` and the last four in cluster ``1``,
and each centroid sits exactly at its blob's mean.


Extending to ``k`` clusters
---------------------------

For general :math:`k`, replace ``nearest_of_two`` with an ``arg min`` loop that
keeps a running best distance and best index over ``C``, and replace the two
explicit ``cluster_mean`` calls with a loop over the ``k`` clusters. The
structure of the Lloyd iteration is unchanged.


References
----------

- S. Lloyd, *Least squares quantization in PCM*, IEEE Trans. Inf. Theory 28, 129–137 (1982).
- `k-means clustering — Wikipedia <https://en.wikipedia.org/wiki/K-means_clustering>`_
