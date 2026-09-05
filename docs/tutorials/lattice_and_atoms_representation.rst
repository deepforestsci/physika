Lattice and Atoms representation
================================

In this tutorial we will represent atoms in a lattice using physika.
The design follows the `Atomic Simulation Environment
<https://wiki.fysik.dtu.dk/ase/>`_ (ASE), whose ``ase.atoms.Atoms`` and
``ase.atom.Atom`` are the conventional means of representing and passing atomic
structures in computational chemistry and materials science.

Design
------

``Atom`` exposes eleven per-atom attributes: ``number``, ``symbol``,
``mass``, ``position``, and ``charge`` (plus ``index``, its position in the
parent). Physika has no string type, so ``symbol`` is omitted and ``number``
stands in for it.

``Atoms`` therefore holds one array per attribute, plus the cell:

===============  =================================================
``Atoms`` array  attribute
===============  =================================================
``numbers``      Atomic number of the atoms (also ``symbol`` in ASE)
``positions``    Position of the atom in lattice
``masses``       Atomic mass of the atoms
``cell``         Cell level structure containing all the atoms
===============  =================================================

``Atom`` is a pair consisting of a parent ``Atoms`` and an index. It stores no
data of its own; each of its methods reads one entry out of the corresponding
parent array. ``atoms.atom(idx)`` returns such a handle.

One object owns the flat arrays, and a second provides a view of a single
element. Physika arrays cannot contain class instances, so ``Atoms`` does not
retain a collection of ``Atom`` objects; ``atom(idx)`` reconstructs one on each
call, which is indistinguishable from indexing.

The Atoms class
---------------

``Atoms`` receives the per-atom arrays and the cell as constructor
parameters. The dimension variable ``n`` binds every per-atom array to a
common atom count, which the type checker verifies at each construction and
call. The accessors each return one row of the relevant array; every other
method is defined in terms of them.
