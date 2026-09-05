An Atom / Atoms container
===================================

In this tutorial we define an ``Atoms`` class that represents a molecule or a
crystal, together with an ``Atom`` class that provides a lightweight handle on
a single atom within it. The design follows the `Atomic Simulation Environment
<https://wiki.fysik.dtu.dk/ase/>`_ (ASE), whose ``ase.atoms.Atoms`` and
``ase.atom.Atom`` are the conventional means of representing and passing atomic
structures in computational chemistry and materials science. The ``Atom`` class
built here carries the full set of per-atom attributes that ``ase.atom.Atom``
exposes.

Design
------

``ase.atom.Atom`` exposes eleven per-atom attributes: ``number``, ``symbol``,
``mass``, ``position``, ``x``, ``y``, ``z``, ``momentum``, ``tag``, ``magmom``,
and ``charge`` (plus ``index``, its position in the parent). Physika has no
string type, so ``symbol`` is omitted and ``number`` stands in for it; every
other attribute is reproduced.

``Atoms`` therefore holds one array per attribute, plus the cell:

===============  =================================================
``Atoms`` array  ``ase.atom.Atom`` attribute
===============  =================================================
``numbers``      ``number`` (also ``symbol`` in ASE)
``positions``    ``position``, and ``x`` / ``y`` / ``z``
``masses``       ``mass``
``tags``         ``tag``
``momenta``      ``momentum``
``magmoms``      ``magmom``
``charges``      ``charge``
``cell``         --- (a ``3x3`` structure-level quantity, not per-atom)
===============  =================================================

``Atom`` is a pair consisting of a parent ``Atoms`` and an index. It stores no
data of its own; each of its methods reads one entry out of the corresponding
parent array. ``atoms.atom(idx)`` returns such a handle, in the same way that
``atoms[i]`` does in ASE.

One object owns the flat arrays, and a second provides a view of a single
element. Physika arrays cannot contain class instances, so ``Atoms`` does not
retain a collection of ``Atom`` objects; ``atom(idx)`` reconstructs one on each
call, which is indistinguishable from indexing.

Several characteristics of Physika determine how the code is written:

- the identifiers ``i`` and ``j`` denote the imaginary unit, so loop and
  argument indices are named ``a``, ``b``, and ``idx``;
- a blank line is permitted only between top-level items, so method bodies
  contain none;
- a comment may not appear between a ``class`` header and its first ``def``, so
  each class is documented in the comment block immediately preceding it;
- a typed declaration (``p: ℝ[3] = ...``) inside a loop body is parsed as a
  tuple unpack, so a loop body uses plain assignment or inlines the value;
- a nested ``for`` statement within a method requires an explicit ``ℕ`` range,
  so the affected methods begin by binding ``k: ℕ = len(...)`` and iterate over
  ``ℕ(k)``;
- the body of a ``for ... → e`` comprehension must reference the loop variable,
  which is the purpose of the ``* 0.0`` terms in the zero-tensor initialisers.

The Atoms class
---------------

``Atoms`` receives the seven per-atom arrays and the cell as constructor
parameters. The dimension variable ``n`` binds every per-atom array to a
common atom count, which the type checker verifies at each construction and
call. The accessors each return one row of the relevant array; every other
method is defined in terms of them.

.. code-block:: text

   class Atoms(numbers: ℝ[n], positions: ℝ[n, 3], masses: ℝ[n], tags: ℝ[n], momenta: ℝ[n, 3], magmoms: ℝ[n], charges: ℝ[n], cell: ℝ[3, 3]):
       def count() → ℝ:
           return len(this.numbers) * 1.0
       def atomic_number(a: ℝ) → ℝ:
           nums: ℝ[n] = this.numbers
           return nums[a]
       def mass(a: ℝ) → ℝ:
           m: ℝ[n] = this.masses
           return m[a]
       def position(a: ℝ) → ℝ[3]:
           pos: ℝ[n, 3] = this.positions
           return pos[a]
       def momentum(a: ℝ) → ℝ[3]:
           mom: ℝ[n, 3] = this.momenta
           return mom[a]
       def tag(a: ℝ) → ℝ:
           t: ℝ[n] = this.tags
           return t[a]
       def magmom(a: ℝ) → ℝ:
           mm: ℝ[n] = this.magmoms
           return mm[a]
       def charge(a: ℝ) → ℝ:
           q: ℝ[n] = this.charges
           return q[a]

``count`` returns the number of atoms, corresponding to ``len(atoms)`` in ASE.

Aggregate quantities
~~~~~~~~~~~~~~~~~~~~~~

``total_mass`` and ``total_charge`` are direct summations. ``kinetic_energy``
sums :math:`|\mathbf{p}_a|^2 / (2 m_a)` over the atoms, using the momenta and
masses arrays. ``center_of_mass`` computes the mass-weighted mean position
:math:`\mathbf{R} = \frac{1}{M}\sum_a m_a \mathbf{r}_a`:

.. code-block:: text

       def total_mass() → ℝ:
           m: ℝ[n] = this.masses
           return sum(m)
       def total_charge() → ℝ:
           q: ℝ[n] = this.charges
           return sum(q)
       def kinetic_energy() → ℝ:
           mom: ℝ[n, 3] = this.momenta
           mass: ℝ[n] = this.masses
           k: ℕ = len(mass)
           total: ℝ = 0.0
           for a : ℕ(k):
               total += sum(mom[a] * mom[a]) / (2.0 * mass[a])
           return total
       def center_of_mass() → ℝ[3]:
           pos: ℝ[n, 3] = this.positions
           mass: ℝ[n] = this.masses
           k: ℕ = len(pos)
           acc: ℝ[3] = for c : ℕ(3) → c * 0.0
           for a : ℕ(k):
               acc = acc + pos[a] * mass[a]
           return acc * (1.0 / sum(mass))

The initialiser ``acc: ℝ[3] = for c : ℕ(3) → c * 0.0`` produces a zero vector
of fixed length: it is a for-expression whose body, ``c * 0.0``, refers to the
loop variable - a body that does not is rejected - and evaluates to zero. In
``kinetic_energy`` the per-atom momentum is inlined as ``mom[a]`` rather than
bound to a typed local, since a typed declaration inside a loop body is parsed
as a tuple unpack.

Distances
~~~~~~~~~

``distance`` returns the Euclidean norm of the difference between two position
rows. ``all_distances`` populates an ``n x n`` matrix with a nested loop; both
loops carry an explicit ``ℕ(k)`` range, as nested ``for`` statements in a
method require.

.. code-block:: text

       def distance(a: ℝ, b: ℝ) → ℝ:
           pos: ℝ[n, 3] = this.positions
           d: ℝ[3] = pos[a] - pos[b]
           return sqrt(sum(d * d))
       def all_distances() → ℝ[n, n]:
           pos: ℝ[n, 3] = this.positions
           k: ℕ = len(pos)
           result: ℝ[n, n] = for a : ℕ(k) → for b : ℕ(k) → (a + b) * 0.0
           for a : ℕ(k):
               for b : ℕ(k):
                   d: ℝ[3] = pos[a] - pos[b]
                   result[a, b] = sqrt(sum(d * d))
           return result

The unit cell
~~~~~~~~~~~~~~

``count_species`` counts the atoms of a given atomic number. ``volume`` returns
the absolute value of the determinant of the ``3x3`` cell, expressed as a
cofactor expansion along the first row so that it remains valid for a
non-orthogonal cell:

.. code-block:: text

       def count_species(z: ℝ) → ℝ:
           nums: ℝ[n] = this.numbers
           k: ℕ = len(nums)
           total: ℝ = 0.0
           for a : ℕ(k):
               if nums[a] == z:
                   total += 1.0
           return total
       def volume() → ℝ:
           c: ℝ[3, 3] = this.cell
           m0: ℝ = c[1, 1] * c[2, 2] - c[1, 2] * c[2, 1]
           m1: ℝ = c[1, 0] * c[2, 2] - c[1, 2] * c[2, 0]
           m2: ℝ = c[1, 0] * c[2, 1] - c[1, 1] * c[2, 0]
           det: ℝ = c[0, 0] * m0 - c[0, 1] * m1 + c[0, 2] * m2
           return abs(det)

Non-mutating transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASE's ``translate`` modifies the structure in place. Physika values are
immutable, so ``translate`` constructs and returns a new ``Atoms`` from a
shifted copy of the positions, carrying the other six arrays and the cell
through unchanged; ``centered`` is defined in terms of it and relocates the
centre of mass to the origin.

.. code-block:: text

       def translate(shift: ℝ[3]) → Atoms:
           pos: ℝ[n, 3] = this.positions
           k: ℕ = len(pos)
           moved: ℝ[n, 3] = for a : ℕ(k) → for c : ℕ(3) → pos[a, c] + shift[c]
           return Atoms(this.numbers, moved, this.masses, this.tags, this.momenta, this.magmoms, this.charges, this.cell)
       def centered() → Atoms:
           com: ℝ[3] = this.center_of_mass()
           return this.translate(com * -1.0)

A method may return an instance of its own class and may invoke another method
on ``this``; ``centered`` does both.

The Atom class
--------------

``Atom`` is declared before ``Atoms`` and refers to ``Atoms`` in its
constructor as a forward reference - the ordering used for ``Site`` and
``System`` in ``lattice.phyk``. It has one method per ``ase.atom.Atom``
attribute (``symbol`` excepted), and a ``distance_to`` helper that is not part
of ASE's ``Atom``. None of the methods access an array directly; each delegates
through ``this.atoms``:

.. code-block:: text

   class Atom(atoms: Atoms, index: ℝ):
       def idx() → ℝ:
           return this.index
       def number() → ℝ:
           return this.atoms.atomic_number(this.index)
       def mass() → ℝ:
           return this.atoms.mass(this.index)
       def position() → ℝ[3]:
           return this.atoms.position(this.index)
       def x() → ℝ:
           p: ℝ[3] = this.atoms.position(this.index)
           return p[0]
       def y() → ℝ:
           p: ℝ[3] = this.atoms.position(this.index)
           return p[1]
       def z() → ℝ:
           p: ℝ[3] = this.atoms.position(this.index)
           return p[2]
       def momentum() → ℝ[3]:
           return this.atoms.momentum(this.index)
       def tag() → ℝ:
           return this.atoms.tag(this.index)
       def magmom() → ℝ:
           return this.atoms.magmom(this.index)
       def charge() → ℝ:
           return this.atoms.charge(this.index)
       def distance_to(b: ℝ) → ℝ:
           return this.atoms.distance(this.index, b)

A single method on ``Atoms`` completes the relationship:

.. code-block:: text

       def atom(idx: ℝ) → Atom:
           return Atom(this, idx)

Example: a water molecule
-------------------------

The water molecule is defined with oxygen at the origin and the two O–H bonds
of length 0.9584 Å separated by 104.45°, within a 10 Å box. Atom 0 is oxygen
(:math:`Z = 8`) and atoms 1 and 2 are hydrogen (:math:`Z = 1`). The atoms carry
TIP3P partial charges, opposing momenta along ``x``, and tags marking the two
hydrogens; the magnetic moments are zero.

.. code-block:: text

   w_numbers: ℝ[3] = [8.0, 1.0, 1.0]
   w_positions: ℝ[3, 3] = [
       [ 0.00000, 0.00000, 0.0],
       [ 0.95840, 0.00000, 0.0],
       [-0.23990, 0.92790, 0.0]
   ]
   w_masses: ℝ[3] = [15.999, 1.008, 1.008]
   w_tags: ℝ[3] = [0.0, 1.0, 1.0]
   w_momenta: ℝ[3, 3] = [
       [0.00, 0.0, 0.0],
       [0.20, 0.0, 0.0],
       [-0.20, 0.0, 0.0]
   ]
   w_magmoms: ℝ[3] = [0.0, 0.0, 0.0]
   w_charges: ℝ[3] = [-0.834, 0.417, 0.417]
   w_cell: ℝ[3, 3] = [
       [10.0, 0.0, 0.0],
       [0.0, 10.0, 0.0],
       [0.0, 0.0, 10.0]
   ]
   water = Atoms(w_numbers, w_positions, w_masses, w_tags, w_momenta, w_magmoms, w_charges, w_cell)

   water.count()
   water.total_mass()
   water.total_charge()
   water.kinetic_energy()
   water.center_of_mass()
   water.distance(0.0, 1.0)
   water.all_distances()

Output::

   3.0 ∈ ℝ
   18.014999389648438 ∈ ℝ
   0.0 ∈ ℝ
   0.0396825410425663 ∈ ℝ
   [0.04020250216126442, 0.05191913619637489, 0.0] ∈ ℝ[3]
   0.9584000110626221 ∈ ℝ
   [[0.0, 0.9584000110626221, 0.958410382270813], [0.9584000110626221, 0.0, 1.5155597925186157], [0.958410382270813, 1.5155597925186157, 0.0]] ∈ ℝ[3,3]

The partial charges sum to zero, and the kinetic energy is that of the two
hydrogens, :math:`2 \times 0.20^2 / (2 \times 1.008)`. The bond length
``distance(0, 1)`` is 0.9584 Å, and the distance matrix contains this value in
both O–H positions and 1.516 Å for the hydrogen–hydrogen separation.

A single ``Atom`` may now be extracted and every ``ase.atom.Atom`` attribute
read off it; each call is forwarded to ``water``:

.. code-block:: text

   o = water.atom(0.0)
   o.idx()
   o.number()
   o.mass()
   o.position()
   o.charge()
   h1 = water.atom(1.0)
   h1.number()
   h1.x()
   h1.momentum()
   h1.tag()
   h1.magmom()
   h1.charge()
   h1.distance_to(2.0)

Output::

   0.0 ∈ ℝ
   8.0 ∈ ℝ
   15.99899959564209 ∈ ℝ
   [0.0, 0.0, 0.0] ∈ ℝ[3]
   -0.8339999914169312 ∈ ℝ
   1.0 ∈ ℝ
   0.9584000110626221 ∈ ℝ
   [0.20000000298023224, 0.0, 0.0] ∈ ℝ[3]
   1.0 ∈ ℝ
   0.0 ∈ ℝ
   0.4169999957084656 ∈ ℝ
   1.5155597925186157 ∈ ℝ

``o.charge()`` returns the oxygen partial charge and ``h1.momentum()`` the
hydrogen momentum, both read straight from ``water``'s arrays through the
handle. ``h1.distance_to(2.0)`` gives the hydrogen–hydrogen distance, matching
the value reported by the matrix.

Translating the molecule displaces its centre of mass by the same vector, and
``centered`` returns it to the origin:

.. code-block:: text

   shift: ℝ[3] = [1.0, 2.0, 3.0]
   moved = water.translate(shift)
   moved.center_of_mass()
   centred = water.centered()
   centred.center_of_mass()

Output::

   [1.0402023792266846, 2.0519192218780518, 3.0] ∈ ℝ[3]
   [0.0, -3.3086120510006367e-09, 0.0] ∈ ℝ[3]

The molecule() helper
---------------------

Supplying seven arrays for every structure is verbose. A helper function drops
a set of atoms into a cubic box and zero-fills the four dynamical arrays
(tags, momenta, magmoms, charges), which is the state ``ase.build.molecule``
returns for a named species. The zero arrays are built with for-expressions
whose bodies reference the loop variable.

.. code-block:: text

   def molecule(numbers: ℝ[p], positions: ℝ[p, 3], masses: ℝ[p], box: ℝ): Atoms:
       k: ℕ = len(numbers)
       zeros1: ℝ[p] = for a : ℕ(k) → numbers[a] * 0.0
       zeros3: ℝ[p, 3] = for a : ℕ(k) → for c : ℕ(3) → positions[a, c] * 0.0
       cell: ℝ[3, 3] = [
           [box, 0.0, 0.0],
           [0.0, box, 0.0],
           [0.0, 0.0, box]
       ]
       return Atoms(numbers, positions, masses, zeros1, zeros3, zeros1, zeros1, cell)

A carbon monoxide molecule built this way has zero charge, tag, and kinetic
energy:

.. code-block:: text

   co_numbers: ℝ[2] = [6.0, 8.0]
   co_positions: ℝ[2, 3] = [
       [0.0, 0.0, 0.0],
       [1.128, 0.0, 0.0]
   ]
   co_masses: ℝ[2] = [12.011, 15.999]
   co = molecule(co_numbers, co_positions, co_masses, 8.0)
   co.atom(0.0).charge()
   co.atom(1.0).tag()
   co.kinetic_energy()

Output::

   0.0 ∈ ℝ
   0.0 ∈ ℝ
   0.0 ∈ ℝ

Example: a periodic crystal
---------------------------

The same class represents a periodic solid. The following is the two-atom
primitive cell of rock-salt sodium chloride (:math:`a = 5.64` Å). The lattice
vectors are the face-centred set :math:`\tfrac{a}{2}\{(0,1,1),(1,0,1),(1,1,0)\}`,
so the cell is non-orthogonal and ``volume`` evaluates the full determinant.
Sodium (:math:`Z = 11`) is placed at the origin and chlorine (:math:`Z = 17`)
at the body centre, with the :math:`\pm 1` formal charges of the ionic solid.

.. code-block:: text

   nacl_numbers: ℝ[2] = [11.0, 17.0]
   nacl_positions: ℝ[2, 3] = [
       [0.00, 0.00, 0.00],
       [2.82, 2.82, 2.82]
   ]
   nacl_masses: ℝ[2] = [22.990, 35.450]
   nacl_tags: ℝ[2] = [0.0, 0.0]
   nacl_momenta: ℝ[2, 3] = [
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0]
   ]
   nacl_magmoms: ℝ[2] = [0.0, 0.0]
   nacl_charges: ℝ[2] = [1.0, -1.0]
   nacl_cell: ℝ[3, 3] = [
       [0.00, 2.82, 2.82],
       [2.82, 0.00, 2.82],
       [2.82, 2.82, 0.00]
   ]
   nacl = Atoms(nacl_numbers, nacl_positions, nacl_masses, nacl_tags, nacl_momenta, nacl_magmoms, nacl_charges, nacl_cell)
   nacl.count_species(11.0)
   nacl.count_species(17.0)
   nacl.total_charge()
   nacl.distance(0.0, 1.0)
   nacl.volume()

Output::

   1.0 ∈ ℝ
   1.0 ∈ ℝ
   0.0 ∈ ℝ
   4.884383201599121 ∈ ℝ
   44.851531982421875 ∈ ℝ

The cell contains one sodium and one chlorine atom, the formal charges cancel,
the sodium–chlorine separation is :math:`2.82\sqrt{3} \approx 4.88` Å, and the
primitive-cell volume is :math:`a^3/4 \approx 44.85` Å³, in agreement with the
determinant.

Running the program
-------------------

.. code-block:: bash

   physika tutorials/atoms.phyk

The ``--print-code`` flag prints the generated PyTorch. ``Atoms`` and ``Atom``
each compile to an ``nn.Module`` subclass, the accessors compile to tensor
indexing, and ``atom(idx)`` compiles to a constructor call.

Further reading
---------------

- ``examples/lattice.phyk`` applies the same division between an owning object
  and a single-element view to a kinetic Monte Carlo lattice.
- The Classes section of :doc:`/examples` describes method calls, fields of
  class type, and differentiation through methods.
- :doc:`/language` documents the loop constructs and the tensor type used
  throughout this program.
