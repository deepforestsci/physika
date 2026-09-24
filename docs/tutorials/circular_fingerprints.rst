Circular Fingerprints
=====================

In this tutorial, we introduce **molecular fingerprint**, a efficient way of
representing molecules. Although there are many ways to represent molecules,
a molecular fingerprint is simple and versatile making them useful for
molecular similarity search, virtual screening, drug discovery, toxicity
prediction etc.

.. note::

    This implementation's results differ slightly from `rdkit` because `rdkit`
    handles many edge cases and repeating atoms. Including so much
    details in educational content will make it difficult for new readers to
    get started, so a simplified approach has been taken here.

What is a Fingerprint?
----------------------

For applying almost all of the computational techniques on atomic systems, we
need to have a mathematical representation of them. This creates a task of how
to do so; with this also comes the question of what information to include in
it. Does representing H2O Water according to its atomic number [1, 8, 1] make
it useful for the Chem Statistician? In a world where we don’t know what this
array represents, we will think that index 1 is 8 times index 0, but does
oxygen equal 8 hydrogens? It does not, and we know this. But how will we tell
the machine about it? To overcome these problems, molecular fingerprints try to
convert these data into forms which are interrelated and also easy to use for
algorithms. One such fingerprinting technique is Extended Connectivity
Fingerprint or ECFP. [RogersHahn2010]_

Morgan Algorithm
----------------

The Morgan algorithm is a graph-based molecular representation method that
assigns numerical identifiers to atoms based on their neighbouring structural
environments. It was first described as a way of giving every molecule a
unique machine-readable description. [Morgan1965]_ It starts from very
basic information like bonds, charges, etc and creates a feature out of them
using a hashing function. After that, as we want it to be more dense with more
information, we will increase the algorithm’s radius by 1; now it will include
the information of the atom as well as its immediate neighbour. Now if we
increase its radius again by 1, then it will include information about its
neighbour’s neighbour also.

.. math::

    & z \gets \mathrm{atomic\_num}(g),
    \qquad
    k \gets |g|,
    \qquad
    \mathrm{fp}_b \gets 0 \quad \text{for } b=0,\ldots,N_{\mathrm{BITS}}-1

    & \mathrm{id}^{(0)} \gets \mathrm{initial\_ids}(g)

    & \qquad\text{for } a=0,\ldots,k-1:

    & \qquad\qquad
    \mathrm{fp}_{\,\mathrm{id}^{(0)}_a \bmod N_{\mathrm{BITS}}} \gets 1
    \quad \text{if } z_a > 1

    & \qquad\text{for } r=0,\ldots,R-1:

    & \qquad\qquad
    \mathrm{id}^{(r+1)} \gets \mathrm{update\_ids}\left(g,\; \mathrm{id}^{(r)},\; r+1\right)

    & \qquad\qquad\text{for } a=0,\ldots,k-1:

    & \qquad\qquad\qquad
    \mathrm{fp}_{\,\mathrm{id}^{(r+1)}_a \bmod N_{\mathrm{BITS}}} \gets 1
    \quad \text{if } z_a > 1

    & \qquad
    \mathrm{result} \gets \mathrm{fp}

    & \mathrm{ECFP}(g,\; D) = \mathrm{fingerprint}\left(g,\; R = \tfrac{D}{2}\right)

.. code:: text

    def fingerprint(g: Molecule, radius: ℝ): ℝ[N_BITS]:
        z: ℝ[n] = g.atomic_num
        k: ℝ = get_2d_array_num_rows(g.adjacency)
        fp: ℝ[N_BITS] = for b : ℕ(N_BITS) → b * 0.0
        ids: ℝ[n] = initial_ids(g)
        for a : ℕ(k):
            if z[a] > 1.0:
                fp[modulo(ids[a], N_BITS)] = 1.0
        for r : ℕ(radius):
            ids = update_ids(g, ids, r + 1.0)
            for a : ℕ(k):
                if z[a] > 1.0:
                    fp[modulo(ids[a], N_BITS)] = 1.0
        return fp

    def ecfp(g: Molecule, diameter: ℝ): ℝ[N_BITS]:
        return fingerprint(g, diameter / 2.0)

Atom Invariants
---------------

Before looking at any neighbours, each atom is described by five numbers,
called its invariants. These properties/numbers does not depends on the
ordering of the atoms:

1. Atomic number
2. Degree, the number of bonded atoms
3. Formal charge
4. Number of attached hydrogens
5. Aromatic, :math:`1` if the atom has an aromatic (order 1.5) bond,
   otherwise :math:`0`

.. code:: text

    def degree(g: Molecule, u: ℕ): ℝ:
        m: ℝ[n, n] = g.adjacency
        k: ℝ = get_2d_array_num_rows(m)
        d: ℝ = 0
        for v : ℕ(k):
            if m[u, v] > 0.0:
                d += 1
        return d

    def hydrogens(g: Molecule, u: ℕ): ℝ:
        m: ℝ[n, n] = g.adjacency
        z: ℝ[n] = g.atomic_num
        k: ℝ = get_2d_array_num_rows(m)
        h: ℝ = 0
        for v : ℕ(k):
            if m[u, v] > 0.0:
                if z[v] == 1.0:
                    h += 1
        return h

    def aromatic(g: Molecule, u: ℕ): ℝ:
        m: ℝ[n, n] = g.adjacency
        k: ℝ = get_2d_array_num_rows(m)
        r: ℝ = 0
        for v : ℕ(k):
            if m[u, v] == 1.5:
                r = 1
        return r

    def invariants(g: Molecule): ℝ[5, n]:
        m: ℝ[n, n] = g.adjacency
        z: ℝ[n] = g.atomic_num
        c: ℝ[n] = g.formal_charge
        k: ℝ = get_2d_array_num_rows(m)
        inv: ℝ[5, k] = for i : ℕ(5) → for a : ℕ(k) → a * 0.0
        inv[0, :] = z
        inv[2, :] = c
        for a : ℕ(k):
            inv[1, a] = degree(g, a)
            inv[3, a] = hydrogens(g, a)
            inv[4, a] = aromatic(g, a)
        return inv

The radius :math:`0` identifier of an atom is the hash of its five invariants.
The hash starts from :math:`h = 17` and adds one value at a time using
:math:`h \gets (31\,h + x) \bmod M` with :math:`M = 65521`:

.. math::

    \mathrm{id}^{(0)}_u = \mathrm{hash}\bigl(z_u,\; \deg_u,\; c_u,\;
    \mathrm{H}_u,\; \mathrm{arom}_u\bigr)

.. code:: text

    def hash_list(xs: ℝ[n]): ℝ:
        h: ℝ = 17.0
        for i : ℕ(len(xs)):
            h = modulo(h * 31.0 + xs[i], M)
        return h

    def initial_ids(g: Molecule): ℝ[n]:
        inv: ℝ[5, n] = invariants(g)
        k: ℝ = get_2d_array_num_rows(g.adjacency)
        new_ids: ℝ[k] = for a : ℕ(k) → a * 0.0
        for a : ℕ(k):
            new_ids[a] = hash_list(inv[:, a])
        return new_ids

Identity Update Procedure
-------------------------

Every atom gets a number, its identifier, that describes its surroundings. At
radius :math:`0` it describes only the atom itself. Each step up in radius adds
one more layer of neighbours.

To update the identifier of an atom :math:`u`:

1. For each neighbour :math:`v`, make a key from the bond order
   :math:`m_{uv}` and the neighbour's current identifier.
2. Sort the keys, so the order in which atoms were numbered does not matter.
3. Hash the radius :math:`r`, the atom's own identifier and the sorted keys
   together. The result is the new identifier.

.. math::

    \mathrm{id}^{(r)}_u = \mathrm{hash}\Bigl(r,\;
    \mathrm{id}^{(r-1)}_u,\;
    \mathrm{sort}\bigl\{\, 2\, m_{uv} M + \mathrm{id}^{(r-1)}_v
    \;:\; v \text{ bonded to } u \,\bigr\}\Bigr)

The hash adds one value at a time, starting from :math:`h = 17`, using
:math:`h \gets (31\,h + x) \bmod M` with :math:`M = 65521`.

.. code:: text

    def hash_step(h: ℝ, x: ℝ): ℝ:
        return modulo(h * 31.0 + x, M)

    def update_ids(g: Molecule, ids: ℝ[n], r: ℝ): ℝ[n]:
        m: ℝ[n, n] = g.adjacency
        k: ℝ = get_2d_array_num_rows(m)
        new_ids: ℝ[k] = for a : ℕ(k) → a * 0.0
        keys: ℝ[k] = for a : ℕ(k) → a * 0.0
        for u : ℕ(k):
            for v : ℕ(k):
                keys[v] = 0.0
                if m[u, v] > 0.0:
                    keys[v] = 2.0 * m[u, v] * M + ids[v]
            keys = bubble_sort(keys)
            h: ℝ = 17.0
            h = hash_step(h, r)
            h = hash_step(h, ids[u])
            for i : ℕ(k):
                if keys[i] > 0.0:
                    h = hash_step(h, keys[i])
            new_ids[u] = h
        return new_ids

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
            k: ℝ = get_2d_array_num_rows(m)
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

   Formal Charge: It is the hypothetical charge assigned to an atom in a
   molecule when we assume bond electrons are shared equally between atoms.

Example
-------

A molecule is built in three steps: create it from its atomic numbers and
formal charges, add the bonds, then hash it into a fingerprint. Benzene
:math:`\mathrm{C_6H_6}` is six aromatic carbons in a ring, each carrying one
hydrogen:

.. code:: text

    C6H6_atomic_num: ℝ[12] = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]
    C6H6_formal_charge: ℝ[12] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    C6H6: Molecule = new_molecule(C6H6_atomic_num, C6H6_formal_charge)

    C6H6.add_weighted_edge(0.0, 1.0, 1.5)
    C6H6.add_weighted_edge(1.0, 2.0, 1.5)
    C6H6.add_weighted_edge(2.0, 3.0, 1.5)
    C6H6.add_weighted_edge(3.0, 4.0, 1.5)
    C6H6.add_weighted_edge(4.0, 5.0, 1.5)
    C6H6.add_weighted_edge(5.0, 0.0, 1.5)
    C6H6.add_edge(0.0, 6.0)
    C6H6.add_edge(1.0, 7.0)
    C6H6.add_edge(2.0, 8.0)
    C6H6.add_edge(3.0, 9.0)
    C6H6.add_edge(4.0, 10.0)
    C6H6.add_edge(5.0, 11.0)

Toluene :math:`\mathrm{C_7H_8}` is the same ring with one hydrogen replaced by
a methyl group, and methane :math:`\mathrm{CH_4}` is a single carbon with four
hydrogens. With all three built, ``ecfp`` folds each one into a bit vector and
``tanimoto`` compares them:

.. code:: text

    CH4_ecfp4: ℝ[N_BITS] = ecfp(CH4, 4)
    C6H6_ecfp4: ℝ[N_BITS] = ecfp(C6H6, 4)
    C7H8_ecfp4: ℝ[N_BITS] = ecfp(C7H8, 4)

    get_sum_of_1d_array(CH4_ecfp4)
    get_sum_of_1d_array(C6H6_ecfp4)
    get_sum_of_1d_array(C7H8_ecfp4)

    tanimoto(C6H6_ecfp4, C6H6_ecfp4)
    tanimoto(C6H6_ecfp4, C7H8_ecfp4)
    tanimoto(CH4_ecfp4, C6H6_ecfp4)
    tanimoto(CH4_ecfp4, C7H8_ecfp4)

Output::

   3.0 ∈ ℝ
   3.0 ∈ ℝ
   12.0 ∈ ℝ
   1.0 ∈ ℝ
   0.25 ∈ ℝ
   0.0 ∈ ℝ
   0.0 ∈ ℝ

A molecule compared with itself gives :math:`1`. Toluene shares 3 of its bits
with benzene, Methane shares nothing with either ring, giving :math:`0`.

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

Molecular fingerprinting includes a wide variety of techniques for representing
chemical structures as machine readable feature vectors. In addition to **ECFP**
several other approches are used in cheminformatics. Functional Class
Fingerprints (FCFP) extends the circular fingerprints by utilising their
chemical functionality rather than just identities. PubChem fingerprints use
predefined sets of patterns to generate fixed length binary representation,
making them useful for rapid large scale screening.

Helper Functions
----------------

Modulo
~~~~~~

In computing and mathematics, the modulo operation returns the reminder of a
division.

The implementation below follows restoring division, the shift-and-subtract
method used in hardware dividers [PattersonHennessy]_ [ErcegovacLang2004]_
[DivisionAlgorithm]_. It starts with the largest multiple
:math:`m\,2^{\mathrm{BITS}-1}` of the divisor and halves it on each step,
subtracting it whenever it still fits. What remains at the end is the
remainder.

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

  def modulo(s: ℝ, m: ℝ): ℝ:
      r: ℝ = s
      d: ℝ = m * (2.0 ** (BITS - 1))
      for k:ℕ(BITS):
          if r >= d:
              r = r - d
          d = d / 2.0
      return r

Bubble Sort
~~~~~~~~~~~

Bubble Sort is the simplest sorting algorithm that works by repeatedly swapping
the adjacent elements if they are in the wrong order.

In ``update_ids`` each atom's neighbour are fed one by one into
``hash_step``, and the result depends on the order they arrive in. This order
can be arbitrary, as same molecule can be written with a different atom order. Due to this
we could get a different fingerprint. Sorting the keys first gives a fixed order, so atoms with the
same environment always get the same identifier.

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

Tanimoto Similarity
~~~~~~~~~~~~~~~~~~~

The Tanimoto similarity , also known as the Jaccard index
, of two binary fingerprints :math:`a` and :math:`b` is
the number of bits set in both divided by the number of bits set in either.
[Tanimoto1958]_ [Jaccard1901]_
It is :math:`1` for identical fingerprints and :math:`0` when no bits are
shared.

.. math::

    T(a, b) = \frac{\sum_{i} a_i b_i}
                   {\sum_{i} a_i + \sum_{i} b_i - \sum_{i} a_i b_i}

.. code:: text

    def tanimoto(a: ℝ[n], b: ℝ[n]): ℝ:
        both: ℝ = get_sum_of_1d_array(a * b)
        return both / (get_sum_of_1d_array(a) + get_sum_of_1d_array(b) - both)

Full Code
---------

.. literalinclude:: ../../tutorials/circular_fingerprints.phyk
   :language: text

References
----------

.. [RogersHahn2010] David Rogers and Mathew Hahn.
   "Extended-Connectivity Fingerprints."
   *Journal of Chemical Information and Modeling*, 2010.

.. [Morgan1965] H. L. Morgan.
   "The Generation of a Unique Machine Description for Chemical Structures -
   A Technique Developed at Chemical Abstracts Service."
   *Journal of Chemical Documentation*, 1965.

.. [Tanimoto1958] T. T. Tanimoto.
   *An Elementary Mathematical Theory of Classification and Prediction*.
   IBM Internal Report, 1958.

.. [Jaccard1901] Paul Jaccard.
   "Etude comparative de la distribution florale dans une portion des Alpes
   et des Jura."
   *Bulletin de la Societe Vaudoise des Sciences Naturelles*, 1901.

.. [PattersonHennessy] David A. Patterson and John L. Hennessy.
   *Computer Organization and Design: The Hardware/Software Interface*.
   Chapter 3, "Arithmetic for Computers." Morgan Kaufmann.

.. [ErcegovacLang2004] Milos D. Ercegovac and Tomas Lang.
   *Digital Arithmetic*. Morgan Kaufmann, 2004.
   Chapter 5, "Division."

.. [DivisionAlgorithm] "Division algorithm."
   Wikipedia, The Free Encyclopedia.
   https://en.wikipedia.org/wiki/Division_algorithm#Restoring_division

