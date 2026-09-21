Circular Fingerprints
=====================

In this tutorial we introduce a way of representing molecules called,
`molecular fingerprint`. While there are many ways to represent molecules
`molecular fingerprint` is simple and easy to use in many applications.

.. note::

    This implementation`s results differs slightly from `rdkit` because `rdkit`
    takes care of many edge cases, and some repeating atoms. Including so much
    details in a educational content will make it difficult for new readers to
    get started so a simplified approach has been taken here.

What is a Fingerprint?
----------------------

For applying allmost all of the computational techniques on atomic systems we
need to have a mathematical representation of them. This creates a task of how
to do so, with this also comes the question what information to include in it.
Does representing H2O `Water` according to its atomic number [1, 8, 1] does it
becomes useful for the Chem Statistician. In a world where we don't know what
this array represent we will think that index 1 is 8 times index 0, something
like this we will think, but does Oxygen Equals 8 Hydrogens, it does not and we
know this. But how will we tell the machine about it.

To overcome these problems molecular fingerprints try to convert these data
into forms which are inter related and also easy to use for algorithms. One
such fingerprinting technique is `Extended Connectivity Fingerprint` or `ECFP`.
They are also sometimes called as `Circular Fingerprints`. It starts from very
basic infomations like bonds, charges etc and created a feature out of them
a hashing function. After that as we want it to be more dense with more
information we will increase the algorithm's radius by 1, now it will include
the information of the atom as well as its immediate neighbour. Now if we
increase it's radius again by 1 then it will include information about it's
neighbour's neighbour also.

.. note::

   Atomic Number:

Molecular Graph Representation
------------------------------

For representing the molecules we have used a Undirected Graph, which contains
the informations like **atomic_number** and **formal_charge** of each of the
atoms in the molecule.

An undirected graph is a graph in which traversal can happen in both direction
to and fro a node. This makes them suitable for representing relationships in
atomic/molecular systems.

.. code-block:: text

    class Molecule():
        adjacency: ℝ[n, n]
        atomic_num: ℝ[n]
        formal_charge: ℝ[n]
        def num_atoms() → ℝ:
            return get_2d_array_num_rows(this.adjacency) * 1.0
        def has_edge(u: ℝ, v: ℝ) → ℝ:
            m: ℝ[n, n] = this.adjacency
            r: ℝ[n] = m[u]
            return r[v]
        def neighbors(u: ℝ) → ℝ[n]:
            m: ℝ[n, n] = this.adjacency
            return m[u]
        def add_weighted_edge(u: ℝ, v: ℝ, w: ℝ):
            m: ℝ[n, n] = this.adjacency
            k: R = get_2d_array_num_rows(m)
            new_adj: ℝ[n, n] = for a : ℕ(k) → for b : ℕ(k) → m[a, b]
            new_adj[u, v] = w
            new_adj[v, u] = w
            this.adjacency = new_adj
        def add_edge(u: ℝ, v: ℝ):
            this.add_weighted_edge(u, v, 1.0)

    def new_molecule(atomic_num: ℝ[n], formal_charge: ℝ[n]): Molecule:
        n_atoms: ℕ = len(atomic_num)
        z: ℝ[n_atoms, n_atoms] = for a : ℕ(n_atoms) → for b : ℕ(n_atoms) → (a + b) * 0.0
        g: Molecule = Molecule()
        g.adjacency = z
        g.atomic_num = atomic_num
        g.formal_charge = formal_charge
        return g

.. note::

   Formal Charge: 

Limitations of ECFP
-------------------

- It is generally generated from a 2d molecular graph. This causes it to loose
  3d data such as 3d distances, bond angles etc.
- It has limited use case in representing long range interactions. It only
  describes the molecule within a specified radius.
- Sometimes 2 different atoms/substructures might produce same hash value. This
  will result in hash collision when they might be completly unique atoms.

Other Fingerprinting Techniques
-------------------------------

Helper Functions
----------------

Modulo
~~~~~~

In computing and mathematics, the modulo operation returns the reminder of a
division.

.. math::

    & r \gets s,
    \qquad
    d \gets m\,2^{\mathrm{BITS}-1}

    & \qquad\text{for } k=0,\ldots,\mathrm{BITS}-1:

    & \qquad\qquad
    r \gets r-d \quad \text{if } r \geq d

    & \qquad\qquad
    d \gets \frac{d}{2}

    & \qquad
    \mathrm{result} \gets r

.. code-block:: text

  def modulo(s: R, m: R): R:
      r: R = s
      d: R = m * (2.0 ** (BITS - 1))
      for k:ℕ(BITS):
          if r >= d:
              r = r - d
          d = d / 2.0
      return r

Floor
~~~~~

Floor function is a function that takes a real number `x` as input and returns
the greatest integer less than or equal to `x`.

.. math::

    & r \gets |x|,
    \qquad
    n \gets 0,
    \qquad
    p \gets 2^{\mathrm{BITS}-1}

    & \qquad\text{for } k=0,\ldots,\mathrm{BITS}-1:

    & \qquad\qquad
    (r,\; n) \gets (r-p,\; n+p) \quad \text{if } r \geq p

    & \qquad\qquad
    p \gets \frac{p}{2}

    & \qquad
    \mathrm{result} \gets
    \begin{cases}
      n & \text{if } x \geq 0 \cr
      -n-1 & \text{if } x < 0 \text{ and } r > 0 \cr
      -n & \text{if } x < 0 \text{ and } r = 0
    \end{cases}

.. code-block:: text

  def floor(x: R): R:
      a: R = x
      if x < 0.0:
          a = 0.0 - x
      r: R = a
      n: R = 0.0
      p: R = 2.0 ** (BITS - 1)
      for k:ℕ(BITS):
          if r >= p:
              r = r - p
              n = n + p
          p = p / 2.0
      result: R = n
      if x < 0.0:
          if r > 0.0:
              result = 0.0 - n - 1.0
          else:
              result = 0.0 - n
      return result

Bubble Sort
~~~~~~~~~~~

Bubble Sort is the simplest sorting algorithm that works by repeatedly swapping
the adjacent elements if they are in the wrong order.

.. math::

    & k \gets |x|,
    \qquad
    y_a \gets x_a \quad \text{for } a=0,\ldots,k-1

    & \qquad\text{for } i=0,\ldots,k-1:

    & \qquad\qquad\text{for } j=0,\ldots,k-2:

    & \qquad\qquad\qquad
    (y_j,\; y_{j+1}) \gets (y_{j+1},\; y_j) \quad \text{if } y_j > y_{j+1}

    & \qquad
    \mathrm{result} \gets y

.. code:: text

   def bubble_sort(xs: ℝ[n]): ℝ[n]:
       k: ℕ = len(xs)
       ys: ℝ[k] = for a : ℕ(k) → xs[a]
       for i : ℕ(k):
           for j : ℕ(k - 1):
               if ys[j] > ys[j + 1]:
                   t: ℝ = ys[j] + 0.0
                   ys[j] = ys[j + 1]
                   ys[j + 1] = t
       return ys

get_sum_of_1d_array
~~~~~~~~~~~~~~~~~~~

``get_sum_of_1d_array`` function computes the sum of all elements in a
one-dimensional array:

.. math::

    s = \sum_{i=1}^{m} x_i

It performs the reduction explicitly using a loop:

.. code-block:: text

    def get_sum_of_1d_array(x: ℝ[m]): ℝ:
        total: ℝ = 0
        for i:
            total += x[i]
        return total

.. note::
   Reduction: A common programming pattern that collapses (or "reduces") a
   collection of values into a single value, such as a sum, by repeatedly
   combining elements. Summing an array's elements in a loop, as
   ``get_sum_of_1d_array`` does, is a simple example.

Full Code
---------

References
----------

.. [PattersonHennessy] David A. Patterson and John L. Hennessy.
   *Computer Organization and Design: The Hardware/Software Interface*.
   Chapter 3, "Arithmetic for Computers." Morgan Kaufmann.

.. [ErcegovacLang2004] Milos D. Ercegovac and Tomas Lang.
   *Digital Arithmetic*. Morgan Kaufmann, 2004.
   Chapter 5, "Division."

.. [DivisionAlgorithm] "Division algorithm."
   Wikipedia, The Free Encyclopedia.
   https://en.wikipedia.org/wiki/Division_algorithm#Restoring_division

