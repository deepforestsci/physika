Density Functional Theory with SimpleDFT
========================================

This tutorial walks through SimpleDFT, a plane-wave Density Functional Theory (DFT)
code written in Physika. We begin with a brief overview of the DFT formulation used 
by SimpleDFT, and then see how it is implemented in Physika. At the end we put everything 
back together and run a full self-consistent calculation for a hydrogen atom, a helium atom, and a 
hydrogen molecule.


The implementation is based on SimpleDFT.jl [SimpleDFTjl]_, an intentionally small plane-wave DFT 
code. We use its formulation as a foundation for building DFT capabilities in Physika, 
keeping the implementation close to the underlying equations while taking advantage of Physika's 
type system and automatic differentiation.


Before looking at SimpleDFT itself, let’s first introduce the DFT formulation
that it is built on.

Density Functional Theory
-------------------------

Density Functional Theory (DFT) is a quantum-mechanical framework for calculating
the ground-state properties of electrons in atoms, molecules, and materials. Its
central idea is to describe the electronic system using the electron density rather
than the full many-electron wavefunction.


From the many-electron problem to DFT
-------------------------------------

A molecule is a quantum system of positively charged nuclei and electrons interacting
through Coulomb forces. Because the nuclei are much heavier than the electrons, the
Born-Oppenheimer approximation allows us to treat the nuclear positions as fixed while
solving the electronic problem [BornOppenheimer1927]_.

For :math:`N` electrons, the electronic Schrödinger equation [Schrodinger1926]_ is described by a
many-electron wavefunction

.. math::

   \Psi(\mathbf r_1,\mathbf r_2,\ldots,\mathbf r_N),

which depends on the positions of all :math:`N` electrons, and therefore on
:math:`3N` spatial coordinates. As the number of electrons increases, the
number of coordinates needed to represent the wavefunction grows as well.
For example, a grid with 100 points along each spatial direction would
require :math:`100^{3N}` values to represent the wavefunction, which becomes
enormous even for a modest number of electrons.

DFT instead uses the **electron density**

.. math::

   n(\mathbf r) =
   \text{number of electrons per unit volume at } \mathbf r,

which depends only on the three spatial coordinates, regardless of the number of
electrons.

Why the density is enough
-------------------------

Why can the density replace the full many-electron wavefunction? The
Hohenberg-Kohn theorems [HK1964]_ provide the theoretical foundation for this.

The first Hohenberg-Kohn theorem tells us why the electron density contains
enough information to describe the ground state. For a fixed number of
electrons and a fixed electron-electron interaction, the ground-state
electron density determines the external potential :math:`v_{\mathrm{ext}}`
produced by the nuclei.

To see why this is important, recall that the Hamiltonian is the operator that
contains the different contributions to the energy of a quantum-mechanical
system. For the electrons, it can be written as

.. math::

   \hat H = \hat T + \hat V_{\mathrm{ee}} + \hat V_{\mathrm{ext}},

where :math:`\hat T` is the kinetic energy of the electrons,
:math:`\hat V_{\mathrm{ee}}` is their mutual Coulomb repulsion, and
:math:`\hat V_{\mathrm{ext}}` is their interaction with the nuclei.

For a fixed number of electrons and a fixed electron-electron interaction,
:math:`\hat T` and :math:`\hat V_{\mathrm{ee}}` are fixed. The arrangement of
the nuclei enters through :math:`v_{\mathrm{ext}}(\mathbf r)`. Therefore, if
the ground-state density determines this external potential, it determines the
Hamiltonian and, consequently, all ground-state properties of the system.

The second Hohenberg-Kohn theorem provides a way to find this ground state. It
states that the total energy can be written as a functional of the density,
:math:`E[n]`, and that the true ground-state density :math:`n_0` minimizes this
functional:

.. math::

   E_0 = E[n_0] = \min_n E[n].

In principle, this reduces the many-electron problem to a minimization over
the three-dimensional density. The difficulty is that the exact functional
:math:`E[n]` is not known. In particular, the kinetic energy of the interacting
electrons cannot be written in a simple explicit form as a functional of
:math:`n`.

Kohn and Sham [KS1965]_ introduced a practical way to handle this difficulty.
They constructed an auxiliary system of non-interacting electrons with the
same ground-state density as the real interacting system. Since these
electrons do not interact with each other, their kinetic energy can be
expressed in terms of single-particle orbitals :math:`\psi_i`:

.. math::

   T_s =
   -\frac{1}{2}\sum_i f_i
   \left\langle \psi_i \middle| \nabla^2 \middle| \psi_i \right\rangle,

where :math:`f_i` is the occupation of orbital :math:`i`.

The electron density is reconstructed from these orbitals as

.. math::

   n(\mathbf r) = \sum_i f_i |\psi_i(\mathbf r)|^2.

The Kohn-Sham system therefore gives us a convenient way to calculate the
kinetic energy and the electron density. To recover the energy of the real
interacting system, we must then account for the electron-nucleus interaction,
the classical electron-electron repulsion, the nucleus-nucleus interaction,
and the remaining many-electron effects. These remaining effects are included
in the exchange-correlation energy :math:`E_{\mathrm{xc}}[n]`.

The Kohn-Sham energy
--------------------

The Kohn-Sham formulation [KS1965]_ allows the total energy to be written as a functional 
of the electron density :math:`n`:

.. math::

   E[n] =
   T_s[n]
   + E_{\mathrm{en}}[n]
   + E_{\mathrm{H}}[n]
   + E_{\mathrm{xc}}[n]
   + E_{\mathrm{ii}}.

The five terms have direct physical interpretations:

- :math:`T_s`: kinetic energy of the non-interacting Kohn-Sham electrons. It
  describes the energy associated with the spatial variation of the
  Kohn-Sham orbitals.

- :math:`E_{\mathrm{en}}`: electron-nucleus electrostatic interaction.
  
- :math:`E_{\mathrm{H}}`: classical electrostatic repulsion between the
  electrons.

- :math:`E_{\mathrm{xc}}`: exchange-correlation energy. It accounts for
  quantum-mechanical exchange and correlation effects, as well as the
  difference between the true interacting kinetic energy and :math:`T_s`.

- :math:`E_{\mathrm{ii}}`: electrostatic repulsion between the nuclei.

The exchange-correlation term is the only part whose exact density dependence
is unknown. Approximating this term is therefore central to practical DFT.

This energy expression is the central quantity that we will implement in
Physika. The next section introduces **SimpleDFT**, the particular plane-wave
DFT formulation used in this tutorial.

What SimpleDFT is
------------------

SimpleDFT [SimpleDFT]_ [SimpleDFTjl]_ is a deliberately minimal plane-wave DFT
code. It follows the DFT++ formulation of Ismail-Beigi and Arias
[IsmailBeigiArias2000]_ [AriasDFT]_, which expresses DFT as a small algebra of
operators acting on coefficient vectors. The result is an implementation that
closely follows the mathematical formulation, making the connection between
the equations and the code explicit.

SimpleDFT focuses on a small, well-defined subset of DFT:

* restricted (spin-paired) Kohn-Sham DFT,
* LDA functionals, without density gradients,
* an all-electron Coulomb potential, without pseudopotentials,
* gamma-point calculations only,
* steepest-descent minimization,
* random starting orbitals.

**The goal of this tutorial is therefore not to provide a production DFT code, but a compact
implementation in which the quantities from the theory remain explicit in the
source.**

Porting SimpleDFT to Physika adds two important features:

* **Mathematical types:** Physika's types reflect the objects being represented. For example, a plane-wave coefficient vector can be declared as `ℂ[p]` and a real-space field as `ℝ[k]`, with their shapes checked before execution.
* **Automatic differentiation:** Physika provides ``grad(f, x)``, which computes the gradient of an expression ``f`` with respect to a variable ``x``. Here we use it to get the gradient of the total energy with respect to the orbital coefficients, rather than working that derivative out by hand.

From the DFT equations to code
------------------------------

The implementation follows the same sequence as the DFT calculation itself.
We first define the atomic positions, simulation cell, and reciprocal-space
grid. These provide the representation needed to construct the plane-wave
operators. The operators are then used to transform the orbitals, construct
the electron density, and obtain the Hartree and exchange-correlation
potentials. These quantities enter the different contributions to the total
energy, which is ultimately minimized in the self-consistent field
calculation.

In Physika, these pieces are kept separate so that each part of the
mathematical formulation has a corresponding implementation. The main
components are:

.. list-table::
   :header-rows: 1
   :widths: 32 24 44

   * - DFT component
     - Physika file
     - Key classes / functions
   * - Atomic structure and reciprocal-space grid
     - ``dft_atoms``
     - ``Atoms``, ``volume``, ``gx``, ``gy``, ``gz``, ``g2``, ``g2c``,
       ``active``, ``sf``
   * - DFT++ operators and Fourier transforms
     - ``dft_operators``
     - ``op_O``, ``op_L``, ``op_Linv``, ``op_J``, ``op_I``,
       ``op_Jdag``, ``op_Idag``
   * - Orbital orthogonalization and density
     - ``dft_density``
     - ``orth``, ``get_n_total``
   * - Hartree potential
     - ``dft_density``
     - ``get_phi``
   * - Nuclear potential
     - ``dft_potentials``
     - ``coulomb``
   * - Exchange-correlation functionals
     - ``dft_xc``
     - ``lda_x``, ``lda_c_chachiyo``
   * - Energy evaluation
     - ``dft_energies``
     - ``get_Ekin``, ``get_Ecoul``, ``get_Exc``, ``get_Een``,
       ``get_Eewald``, ``get_E``
   * - SCF state and energy minimization
     - ``dft_scf``
     - ``SCF``, ``init_W``, ``energy_of_W``, ``sd``, ``runSCF``

The examples corresponding to these components are under ``examples/`` and
can be run directly. For example:

.. code-block:: bash

   physika examples/dft_atoms.phyk

Each example is self-contained, so the individual pieces can also be
tested independently. We start with the first step in the calculation:
defining the simulation cell and the real and reciprocal-space grids.


Simulation setup: ``dft_atoms``
-------------------------------

Everything starts with the simulation cell. SimpleDFT places the system in a cubic cell of side :math:`a`
with periodic boundary conditions and represents fields on a uniform real-space grid. For the isolated systems 
considered here, we choose a sufficiently large cell to reduce interactions between periodic images.
Periodic boundary conditions make plane waves a natural basis. For a cubic cell of side :math:`a`, 
the corresponding reciprocal-lattice vectors are:

.. math::

    \mathbf G = \frac{2\pi}{a}(n_1,n_2,n_3),
    \qquad n_i \in \mathbb Z.

The real-space grid provides a finite representation of fields such as the
electron density and electrostatic potentials. The FFT transforms these fields
between their real-space values and their representation on the full
reciprocal-space grid. For the Kohn-Sham orbitals, we retain only the
reciprocal-space components whose kinetic energy is below a chosen cutoff
:math:`E_{\mathrm{cut}}`:

.. math::

    \frac{1}{2}|\mathbf G|^2 \leq E_{\mathrm{cut}}
    \qquad\Longleftrightarrow\qquad
    |\mathbf G|^2 \leq 2E_{\mathrm{cut}}.

The plane waves satisfying this condition form the active basis. Orbitals are
stored on this smaller set, while densities and potentials are represented on
the full grid.

In Physika, the system is represented by the ``Atoms`` class. It stores the
simulation parameters needed to construct the real and reciprocal-space
grids, the active plane-wave basis, and the nuclear structure:

.. code-block:: text

    class Atoms:
        a: ℝ              # cubic cell side
        ecut: ℝ           # kinetic-energy cutoff
        s1: ℕ             # grid points, axis 1
        s2: ℕ             # grid points, axis 2
        s3: ℕ             # grid points, axis 3
        Natoms: ℕ         # atom count
        px: ℝ[Natoms]     # nuclear positions, x
        py: ℝ[Natoms]     # nuclear positions, y
        pz: ℝ[Natoms]     # nuclear positions, z
        Nstate: ℕ         # occupied states
        Z_nuc: ℝ[Natoms]  # nuclear charge per atom
        f: ℝ[Nstate]      # occupation number per state

The methods of the ``Atoms`` class fall into four categories:

- **Cell geometry.** ``volume()`` returns the cell volume.

- **Real-space grid.** ``coord_x()``, ``coord_y()``, and ``coord_z()``
  return the uniformly spaced grid points along each axis.

- **Reciprocal-space grid.** ``gx()``, ``gy()``, and ``gz()`` give the
  reciprocal-space coordinates, while ``g2()`` gives :math:`|\mathbf{G}|^2`.
  ``fold_freq()`` converts FFT indices to signed frequencies.

- **Plane-wave cutoff.** ``active()`` selects the reciprocal-space points
  satisfying :math:`|\mathbf{G}|^2 \le 2E_{\mathrm{cut}}`, while ``g2c()``
  gives :math:`|\mathbf{G}|^2` for this active set.

The ``sf()`` method constructs the nuclear structure factor,

.. math::

   S_f(\mathbf{G}) = \sum_a e^{-i\mathbf{G}\cdot\mathbf{R}_a},

which is used later to construct the nuclear potential.

For example, a hydrogen atom can be defined as:

.. code-block:: text

    a: ℝ = 16.0
    ecut: ℝ = 16.0
    s: ℕ = 60

    Natoms: ℕ = 1
    Nstate: ℕ = 1

    px: ℝ[1] = [0.0]
    py: ℝ[1] = [0.0]
    pz: ℝ[1] = [0.0]

    Z_nuc: ℝ[1] = [1.0]
    f: ℝ[1] = [1.0]

    H_atom: Atoms = Atoms(a, ecut, s, s, s, Natoms,
                          px, py, pz, Nstate, Z_nuc, f)

    H_atom.volume()
    Sf: ℂ[k] = H_atom.sf()
    Sf[0]

Output::

    4096.0 ∈ ℝ
    (1+0j) ∈ ℂ

The volume is :math:`16^3 = 4096` cubic Bohr. At :math:`\mathbf G=0`, the
structure factor is 1 because the hydrogen nucleus is at the origin.

.. note::

   We use the same 16 Bohr cell, 16 Hartree cutoff, and :math:`60^3` grid
   throughout the tutorial. This gives 216,000 points on the full grid, with
   12,533 plane waves in the active basis.

With the system and its reciprocal-space representation defined, we can now
introduce the operators that act on these quantities.


Operators: ``dft_operators``
----------------------------

The DFT++ formulation uses a small set of operators acting on plane-wave
coefficient vectors. In ``dft_operators.phyk``, these are ``op_O``, ``op_L``,
``op_Linv``, ``op_J``, and ``op_I``, together with the adjoints ``op_Idag``
and ``op_Jdag``. The ``_mat`` variants apply the same operations column-wise
to multiple orbitals.

Overlap operator: ``op_O``
~~~~~~~~~~~~~~~~~~~~~~~~~~

The overlap operator multiplies a field by the cell volume,

.. math::

   O W = \Omega W,

where :math:`\Omega = a^3` for the cubic cells used here. The volume factor
provides the volume element when discrete sums are used to represent
integrals.

.. code-block:: text

   def op_O(atoms: Atoms, W: ℂ[k]): ℂ[k]:
       return atoms.volume() * W

Laplacian operator: ``op_L``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In reciprocal space, applying :math:`\nabla^2` to a Fourier component with
wavevector :math:`\mathbf G` multiplies its coefficient by
:math:`-|\mathbf G|^2`:

.. math::

   \nabla^2 e^{i\mathbf G\cdot\mathbf r}
   = -|\mathbf G|^2 e^{i\mathbf G\cdot\mathbf r}.

In the operator used here, the additional volume factor gives

.. math::

   L(\mathbf G) = -\Omega |\mathbf G|^2.

``atoms.g2c()`` supplies :math:`|\mathbf G|^2` on the active basis, while
``atoms.g2()`` supplies it on the full grid. The operator uses the one
matching the size of its input.

.. code-block:: text

   def op_L(atoms: Atoms, W: ℂ[k]): ℂ[k]:
       G2c: ℝ[p] = atoms.g2c()
       if vec_size(W) == vec_size(G2c):
           G2: ℝ[k] = G2c
       else:
           G2: ℝ[k] = atoms.g2()
       return (-atoms.volume()) * G2 * W

Inverse Laplacian operator: ``op_Linv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inverse Laplacian divides by the same factor,

.. math::

   L^{-1}(\mathbf G)
   = -\frac{1}{\Omega |\mathbf G|^2},
   \qquad \mathbf G \neq \mathbf 0.

The :math:`\mathbf G = \mathbf 0` entry is masked to avoid division by zero.
``nonzero`` identifies the valid entries, while ``safe_G2`` replaces the
zero entry before the division. The final mask sets the
:math:`\mathbf G=0` component back to zero. The calculation remains
elementwise, which is useful when differentiating through the operator.
``dft_density`` uses ``op_Linv`` to solve Poisson's equation.

.. code-block:: text

   def op_Linv(atoms: Atoms, W: ℂ[k]): ℂ[k]:
       G2: ℝ[k] = atoms.g2()
       nonzero: ℝ[k] = gt(G2, 0.0) * 1.0
       safe_G2: ℝ[k] = G2 + (1.0 - nonzero)
       return (W / safe_G2 / (-atoms.volume())) * nonzero

Fourier transform operators: ``op_J`` and ``op_I``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Fourier transforms move a field between its real-space values and
reciprocal-space coefficients.

.. math::

   \mathrm{op\_J}:\quad
   x(\mathbf r)\;(\text{real space})
   \;\longrightarrow\;
   \tilde{x}_{\mathbf G}\;(\text{reciprocal space})

The reciprocal-space coefficients are given by

.. math::

   \tilde{x}_{\mathbf G}
   = \frac{1}{N_{\mathrm{grid}}}
     \sum_{\mathbf r}
     x(\mathbf r)e^{-i\mathbf G\cdot\mathbf r}.

``op_J`` applies the forward FFT and divides the result by
``Ngrid``.

The inverse transform is

.. math::

   \mathrm{op\_I}:\quad
   \tilde{x}_{\mathbf G}\;(\text{reciprocal space})
   \;\longrightarrow\;
   x(\mathbf r)\;(\text{real space})

with

.. math::

   x(\mathbf r)
   = \sum_{\mathbf G}
     \tilde{x}_{\mathbf G}e^{i\mathbf G\cdot\mathbf r}.

``op_I`` applies the inverse FFT and multiplies the result by ``Ngrid``.
When its input contains only the active plane-wave coefficients,
``op_I`` first embeds them into the full reciprocal-space grid using
``mask_embed``.

.. code-block:: text

   def op_J(atoms: Atoms, W: ℂ[k]): ℂ[k]:
       s1: ℕ = atoms.s1
       s2: ℕ = atoms.s2
       s3: ℕ = atoms.s3
       Ngrid: ℕ = s1 * s2 * s3
       real_grid: ℂ[s1, s2, s3] = reshape(W, s1, s2, s3)
       reciprocal_grid: ℂ[s1, s2, s3] = fftn(real_grid)
       return reshape(reciprocal_grid, Ngrid) / Ngrid

   def op_I(atoms: Atoms, W: ℂ[k]): ℂ[k]:
       s1: ℕ = atoms.s1
       s2: ℕ = atoms.s2
       s3: ℕ = atoms.s3
       Ngrid: ℕ = s1 * s2 * s3
       active: ℝ[k] = atoms.active()
       if vec_size(W) == vec_size(active):
           reciprocal_grid: ℂ[k] = W
       else:
           reciprocal_grid: ℂ[k] = mask_embed(W, active, Ngrid)
       reciprocal_grid3d: ℂ[s1, s2, s3] = reshape(
           reciprocal_grid, s1, s2, s3
       )
       real_grid: ℂ[s1, s2, s3] = ifftn(reciprocal_grid3d)
       return reshape(real_grid, Ngrid) * Ngrid

The adjoints ``op_Idag`` and ``op_Jdag`` are used when these transforms
appear inside inner products.

These operators provide the basic building blocks for constructing the
electron density, electrostatic potentials, and energy terms used in the
DFT calculation.

Density and electrostatics: ``dft_density``
-------------------------------------------

``dft_density`` turns orbital coefficients into the two quantities the energy
terms need: the electron density and the Hartree potential. For a single
occupied orbital the density is

.. math::

   n(\mathbf r) = f_0 |\psi(\mathbf r)|^2,

where :math:`f_0` is the occupation of the state. Orbitals are stored as
plane-wave coefficients, so they are first normalized and then transformed to
real space.

Normalization: ``orth``
~~~~~~~~~~~~~~~~~~~~~~~

Orbitals must be orthonormal with respect to the overlap operator,
:math:`Y^\dagger O Y = I`. The general recipe is Löwdin orthonormalization [Lowdin1950]_,

.. math::

   Y = W(W^\dagger O W)^{-1/2},

which for the single state used here reduces to a division by a scalar:

.. math::

   Y = \frac{W}{\sqrt{\Omega\sum_{\mathbf G}|W_{\mathbf G}|^2}}.

This normalization is implemented in ``orth``:

.. code-block:: text

   def orth(atoms: Atoms, W: ℂ[p]): ℂ[p]:
       norm: ℝ = sqrt(atoms.volume() * sum(abs(W) ** 2.0))
       return W / norm

Here ``W`` is the orbital coefficient vector used as the optimization
variable, while ``Y`` is the corresponding normalized orbital.

Density: ``get_n_total``
~~~~~~~~~~~~~~~~~~~~~~~~

The normalized orbital ``Y`` is converted to real space using ``op_I``. The
electron density is then

.. math::

   n(\mathbf r) = f_0 |\psi(\mathbf r)|^2.

``get_n_total`` implements this expression directly. The active coefficients
are first placed on the full reciprocal-space grid, then ``op_I`` transforms
the orbital to real space:

.. code-block:: text

   def get_n_total(atoms: Atoms, Y: ℂ[p]): ℝ[k]:
       active: ℝ[k] = atoms.active()
       n_full: ℕ = vec_size(active)
       Y_full: ℂ[k] = mask_embed(Y, active, n_full)
       Yrs: ℂ[k] = op_I(atoms, Y_full)
       f0: ℝ = atoms.f[0]
       return f0 * abs(Yrs) ** 2.0

Here ``abs(Yrs) ** 2.0`` gives :math:`|\psi(\mathbf r)|^2`, and ``f0``
provides the occupation factor.

Hartree potential: ``get_phi``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The electron density produces a classical electrostatic potential, called
the Hartree potential. It describes the Coulomb repulsion generated by the
electron density itself. The potential satisfies Poisson's equation,

.. math::

   \nabla^2\phi(\mathbf r) = -4\pi n(\mathbf r).

Since the density is already represented on the real-space grid, ``get_phi``
first transforms it to reciprocal space. There, the Laplacian acts by
multiplication with :math:`-|\mathbf G|^2`, so Poisson's equation can be
solved by applying the inverse Laplacian:

.. math::

   \phi = -4\pi L^{-1} O J(n).

This maps directly onto the implementation:

.. code-block:: text

   def get_phi(atoms: Atoms, n: ℝ[k]): ℂ[k]:
       n_c: ℂ[k] = n * (1.0 + 0j)
       n_recip: ℂ[k] = op_J(atoms, n_c)
       return (-4.0) * π * op_Linv(atoms, op_O(atoms, n_recip))

``op_J`` transforms the density to reciprocal space, ``op_O`` supplies the
cell-volume factor, and ``op_Linv`` applies the inverse Laplacian. The
:math:`\mathbf G=0` component is handled by the masking in ``op_Linv``.

For example, we can put these steps together using a Gaussian trial orbital:

.. code-block:: text

   W0: ℂ[p] = exp(-0.5 * H_atom.g2c()) * (1.0 + 0j)
   Y0: ℂ[p] = orth(H_atom, W0)

   norm_check: ℝ = H_atom.volume() * sum(abs(Y0) ** 2.0)
   norm_check

   n0: ℝ[k] = get_n_total(H_atom, Y0)
   phi0: ℂ[k] = get_phi(H_atom, n0)
   phi0[0]
   phi0[1]

Output::

   1.0 ∈ ℝ
   0.0 ∈ ℂ
   0.019141973927617073 ∈ ℂ

The normalization check gives 1.0. The :math:`\mathbf G=0` component of
the Hartree potential is zero because ``op_Linv`` masks it. The first
non-zero :math:`|\mathbf G|` component has a small positive value.

All-electron Coulomb potential: ``dft_potentials``
---------------------------------------------------

The electrons are attracted to the nuclei by the Coulomb potential. For a
point nucleus of charge :math:`Z` at the origin,

.. math::

   V(\mathbf r) = -\frac{Z}{r},

whose Fourier transform is

.. math::

   V(\mathbf G) = -\frac{4\pi Z}{|\mathbf G|^2}.

Building the potential: ``coulomb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The potential is constructed in reciprocal space and multiplied by the
structure factor ``atoms.sf()`` to place the nuclei at their actual
positions. ``op_J`` then transforms the result to real space, where it is
used by ``get_Een``:

.. code-block:: text

   def coulomb(atoms: Atoms): ℂ[k]:
       G2: ℝ[k] = atoms.g2()
       nonzero: ℝ[k] = gt(G2, 0.0) * 1.0
       safe_G2: ℝ[k] = G2 + (1.0 - nonzero)
       Vcoul: ℝ[k] = ((-4.0) * π * atoms.Z_nuc[0] / safe_G2) * nonzero
       return op_J(atoms, Vcoul * atoms.sf())

The :math:`\mathbf G=0` term is masked for the same reason as in
``op_Linv``: division by :math:`|\mathbf G|^2` is undefined there. The
masking keeps the calculation elementwise, which is useful when the
potential is differentiated later.

SimpleDFT uses the bare Coulomb potential rather than a pseudopotential.
This keeps the implementation direct and is sufficient for the hydrogen
and helium systems used here. The tradeoff is slower convergence with
respect to ``ecut``, since the :math:`1/r` singularity requires a higher
plane-wave cutoff.

For example, the potential can be evaluated as follows:

.. code-block:: text

   Vext: ℂ[k] = coulomb(H_atom)
   Vext[0]
   Vext[1]

Output::

   -0.17032203078269958 ∈ ℂ
   -0.07142896205186844 ∈ ℂ

Both values are negative, consistent with the attractive electron-nucleus
interaction.

Exchange-correlation functional: ``dft_xc``
--------------------------------------------

Exchange and correlation contain the part of the electron-electron interaction
that is not included in the classical Hartree term. Their exact form is
unknown, so DFT uses an approximation for the exchange-correlation energy.

SimpleDFT uses the local density approximation (LDA) [KS1965]_. In LDA, the
exchange-correlation energy at each point depends only on the local electron
density. Both functionals below use the Wigner-Seitz radius,

.. math::

   r_s = \left(\frac{3}{4\pi n}\right)^{1/3}.

Exchange: ``lda_x``
~~~~~~~~~~~~~~~~~~~

The exchange contribution has a closed form for the uniform electron gas
[Slater1951]_. The exchange energy density ``ex`` and exchange potential
``vx`` are

.. math::

   e_x = \frac{c_x}{r_s},
   \qquad
   c_x = -\frac{3}{4}\left(\frac{3}{2\pi}\right)^{2/3},
   \qquad
   v_x = \frac{4}{3}\,e_x.

Here ``ex`` is the exchange energy density and ``vx`` is the corresponding
exchange potential. This is implemented directly in ``lda_x``:

.. code-block:: text

    def lda_x(n: ℂ[k]): ℂ[2, k]:
        cx: ℝ = (-3.0 / 4.0) * (3.0 / (2.0 * π)) ** (2.0 / 3.0)
        rs: ℂ[k] = (3.0 / (4.0 * π * n)) ** (1.0 / 3.0)
        ex: ℂ[k] = cx / rs
        vx: ℂ[k] = (4.0 / 3.0) * ex
        return [ex, vx]

Correlation: ``lda_c_chachiyo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correlation contribution does not have a simple closed form. SimpleDFT
uses Chachiyo's parametrization [Chachiyo2016]_. The correlation energy
density ``ec`` and correlation potential ``vc`` are

.. math::

   e_c =
   a\ln\!\left(1 + \frac{b}{r_s} + \frac{b}{r_s^2}\right),
   \qquad
   v_c =
   e_c + \frac{a\,b\,(2+r_s)}
   {3\,(b+b\,r_s+r_s^2)},

with :math:`a=-0.01554535` and :math:`b=20.4562557`.

Here ``ec`` is the correlation energy density and ``vc`` is the corresponding
correlation potential. This is implemented in ``lda_c_chachiyo``:

.. code-block:: text

    def lda_c_chachiyo(n: ℂ[k]): ℂ[2, k]:
        a: ℝ = -0.01554535
        b: ℝ = 20.4562557
        rs: ℂ[k] = (3.0 / (4.0 * π * n)) ** (1.0 / 3.0)
        ec: ℂ[k] = a * log(1.0 + b / rs + b / rs ** 2.0)
        vc: ℂ[k] = ec + a * b * (2.0 + rs) / (3.0 * (b + b * rs + rs ** 2.0))
        return [ec, vc]
        
Energy density and potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exchange-correlation quantities are obtained by adding the exchange and
correlation contributions:

.. math::

   e_{xc} = e_x + e_c,
   \qquad
   v_{xc} = v_x + v_c.

The exchange-correlation potential is the functional derivative

.. math::

   v_{xc}(\mathbf r)
   = \frac{\delta E_{xc}}{\delta n(\mathbf r)}.

The functions return both the energy density and the corresponding potential,
so ``ex`` and ``ec`` give the exchange and correlation energy densities,
while ``vx`` and ``vc`` give their corresponding potentials.

For example, both functionals can be evaluated at the average density of one
electron in the 4096 cubic Bohr cell, :math:`n_0 = 1/4096`:

.. code-block:: text

    n0: ℝ = 1.0 / 4096.0
    n: ℂ[4] = [n0, n0, n0, n0]

    ex_vx: ℂ[2, 4] = lda_x(n)
    ec_vc: ℂ[2, 4] = lda_c_chachiyo(n)

    ex: ℂ[4] = ex_vx[0]
    vx: ℂ[4] = ex_vx[1]
    ec: ℂ[4] = ec_vc[0]
    vc: ℂ[4] = ec_vc[1]

    exc: ℂ[4] = ex + ec
    vxc: ℂ[4] = vx + vc

    ex[0]
    vx[0]
    ec[0]
    vc[0]
    exc[0]
    vxc[0]

Output::

    -0.04615992307662964 ∈ ℝ
    -0.06154656410217285 ∈ ℝ
    -0.01841130666434765 ∈ ℝ
    -0.022336944937705994 ∈ ℝ
    -0.06457123160362244 ∈ ℝ
    -0.08388350903987885 ∈ ℝ

Both exchange and correlation give negative contributions at this density.
Here exchange is larger in magnitude than correlation.


Energy terms: ``dft_energies``
------------------------------

The quantities obtained so far can now be combined to form the different
contributions to the DFT energy. For the electronic part, these are the
kinetic, electron-electron Coulomb, exchange-correlation, and
electron-nucleus terms:

.. math::

   E_{\mathrm{kin}}
   = -\frac{1}{2} f_0\,\mathrm{Re}\langle Y|LY\rangle,
   \qquad
   E_{\mathrm{coul}}
   = \frac{1}{2}\,n\cdot J^\dagger O\phi,

.. math::

   E_{xc}
   = n\cdot J^\dagger OJ(\epsilon_{xc}),
   \qquad
   E_{\mathrm{en}}
   = n\cdot V_{\mathrm{ion}}.

Kinetic energy
~~~~~~~~~~~~~~

Kinetic energy is obtained by applying the Laplacian to the normalized
orbital and taking its inner product with the orbital:

.. math::

   E_{\mathrm{kin}}
   = -\frac{1}{2} f_0\,\mathrm{Re}\langle Y|LY\rangle.

Here ``L`` is the Laplacian operator introduced earlier, and ``Y`` is the
normalized orbital. This is implemented in ``get_Ekin``:

.. code-block:: text

    def get_Ekin(atoms: Atoms, Y: ℂ[p]): ℝ:
        LY: ℂ[p] = op_L(atoms, Y)
        real_inner_product: ℝ = sum(real(Y) * real(LY) + real(-i * Y) * real(-i * LY))
        f0: ℝ = atoms.f[0]
        return (-0.5) * f0 * real_inner_product

The real and imaginary parts are handled explicitly to evaluate
:math:`\mathrm{Re}\langle Y|LY\rangle`.

Electron-electron Coulomb energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the Hartree potential obtained earlier, the electron-electron
Coulomb energy is

.. math::

   E_{\mathrm{coul}}
   = \frac{1}{2}\int n(\mathbf r)\phi(\mathbf r)\,d\mathbf r.

In the operator form used by Physika,

.. math::

   E_{\mathrm{coul}}
   = \frac{1}{2}\,n\cdot J^\dagger O\phi.

This is implemented in ``get_Ecoul``:

.. code-block:: text

    def get_Ecoul(atoms: Atoms, n: ℝ[k], φ: ℂ[k]): ℝ:
        φ_rs: ℂ[k] = op_Jdag(atoms, op_O(atoms, φ))
        return 0.5 * sum(n * real(φ_rs))

Here ``op_Jdag`` brings the potential to the real-space representation and
``op_O`` supplies the cell-volume factor.

Exchange-correlation energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the exchange-correlation energy density introduced earlier, the
corresponding energy is

.. math::

   E_{xc}
   = \int n(\mathbf r)\epsilon_{xc}(\mathbf r)\,d\mathbf r.

In the operator form,

.. math::

   E_{xc}
   = n\cdot J^\dagger OJ(\epsilon_{xc}).

``get_Exc`` implements this expression:

.. code-block:: text

    def get_Exc(atoms: Atoms, n: ℝ[k], exc: ℂ[k]): ℝ:
        exc_recip: ℂ[k] = op_J(atoms, exc)
        exc_real: ℂ[k] = op_Jdag(atoms, op_O(atoms, exc_recip))
        return sum(n * real(exc_real))

Electron-nucleus energy
~~~~~~~~~~~~~~~~~~~~~~~

The electron density also interacts with the ionic potential. Its energy is

.. math::

   E_{\mathrm{en}}
   = \int n(\mathbf r)V_{\mathrm{ion}}(\mathbf r)\,d\mathbf r.

In Physika, this is a direct real-space contraction:

.. code-block:: text

    def get_Een(n: ℝ[k], ionic_potential: ℂ[k]): ℝ:
        return sum(n * real(ionic_potential))

Ewald energy
~~~~~~~~~~~~

A final contribution comes from the interaction between the nuclei. In a
periodic cell, each nucleus interacts with the other nuclei and with their
periodic images. Direct summation of these Coulomb interactions converges
slowly, so ``get_Eewald`` uses Ewald summation [Ewald1921]_.

The energy is separated into real-space, reciprocal-space, self, and
neutralizing-background contributions:

.. math::

   E_{\mathrm{ewald}}
   = E_{\mathrm{self}} + E_{\mathrm{neutral}}
   + E_{\mathrm{real}} + E_{\mathrm{recip}}.

The splitting parameter :math:`\nu` is determined from the reciprocal cutoff
``gcut`` and the tolerance :math:`\gamma`. It changes how the calculation is
split between the real- and reciprocal-space sums without changing the
result.

``get_Eewald`` is longer than the other energy functions, so its full
implementation is given in ``examples/dft_energies.phyk``. Since it depends
only on the nuclear positions and charges, it can be evaluated once for a
fixed structure.

Putting the terms together
~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now evaluate all five contributions for hydrogen using the same
Gaussian trial orbital:

.. code-block:: text

    gcut: ℝ = 2.0
    γ: ℝ = 1.0e-8
    Eewald0: ℝ = get_Eewald(H_atom, gcut, γ)

    Ekin0: ℝ = get_Ekin(H_atom, Y0)
    Ecoul0: ℝ = get_Ecoul(H_atom, n0, φ0)
    Exc0: ℝ = get_Exc(H_atom, n0, exc0)
    Een0: ℝ = get_Een(n0, ionic_potential0)

    E0: ℝ = Ekin0 + Ecoul0 + Exc0 + Een0 + Eewald0

    Ekin0
    Ecoul0
    Exc0
    Een0
    Eewald0
    E0

Output::

    0.7499999403953552 ∈ ℝ
    0.3110436797142029 ∈ ℝ
    -0.31641221046447754 ∈ ℝ
    -0.9518148899078369 ∈ ℝ
    -0.08866553008556366 ∈ ℝ
    -0.2958490252494812 ∈ ℝ

The resulting energy is about :math:`-0.296` Hartree. This is not yet the
ground-state energy because the orbital is only a trial orbital. By the
variational principle, its energy is an upper bound to the minimum. Finding
that minimum is the next step.

Energy minimization: ``dft_scf``
--------------------------------

We can now evaluate the total energy for a given orbital. The next step is to
find the orbital that minimizes this energy.

The orbital and density are coupled. The orbital determines the density, which
in turn determines the Hartree and exchange-correlation contributions. These
contributions then enter the energy used to determine the orbital. A
self-consistent solution is reached when this cycle settles to a stationary
state.

A conventional Kohn-Sham calculation solves this problem by repeatedly
building the potential from the current density and solving the Kohn-Sham
eigenvalue problem. SimpleDFT instead minimizes the total energy directly
with respect to the orbital coefficients. This avoids the eigenvalue
problem and allows the energy to be differentiated directly with respect to
the orbital.

The energy of an orbital: ``energy_of_W``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the pieces developed so far are assembled into one function. Its
variable argument is the orbital ``W``:

.. code-block:: text

    def energy_of_W(W: ℂ[p], atoms: Atoms, ionic_potential: ℂ[k], Eewald: ℝ): ℝ:
        Y: ℂ[p] = orth(atoms, W)
        n: ℝ[k] = get_n_total(atoms, Y)
        φ: ℂ[k] = get_phi(atoms, n)
        n_c: ℂ[k] = (n + 1e-10) * (1.0 + 0j)
        ex_vx: ℂ[2, k] = lda_x(n_c)
        ec_vc: ℂ[2, k] = lda_c_chachiyo(n_c)
        exc: ℂ[k] = ex_vx[0] + ec_vc[0]
        Ekin: ℝ = get_Ekin(atoms, Y)
        Ecoul: ℝ = get_Ecoul(atoms, n, φ)
        Exc: ℝ = get_Exc(atoms, n, exc)
        Een: ℝ = get_Een(n, ionic_potential)
        return Ekin + Ecoul + Exc + Een + Eewald

The order follows the calculation itself: ``W`` is normalized to ``Y``,
``Y`` gives the density, the density gives the Hartree potential and
exchange-correlation terms, and these are then used to evaluate the energy.

The small ``1e-10`` shift keeps the LDA expressions well defined in regions
where the density becomes very small. Since the functionals depend on
:math:`n` through :math:`r_s`, the gradient can otherwise become singular in
the low-density tails.

The starting orbital: ``init_W``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before minimizing the energy, we need an initial set of orbital
coefficients. ``init_W`` uses random coefficients for this:

.. code-block:: text

    def init_W(atoms: Atoms, seed: ℕ): ℂ[p]:
        physika.seed(seed)
        n_active: ℝ = len(atoms.g2c())
        W: ℝ[n_active] ~ 𝒰(0.0, 1.0, n_active)
        W: ℂ[p] = W * (1.0 + 0j)
        return orth(atoms, W)

The random coefficients only provide a starting point in the plane-wave
basis. They do not need to be a good approximation to the final orbital.
The ``seed`` keeps the starting point reproducible.

Steepest descent: ``sd``
~~~~~~~~~~~~~~~~~~~~~~~~

At each iteration, ``sd`` moves ``W`` in the direction of decreasing energy.
With step size :math:`\beta`,

.. math::

    W_{k+1} = W_k - \beta \, \nabla_W E(W_k).

The gradient is obtained directly with ``grad`` applied to ``energy_of_W``.
This means the same expression used to compute the energy is differentiated
to obtain its gradient.

Running ``runSCF``
~~~~~~~~~~~~~~~~~~

The pieces developed above are brought together in ``runSCF``. It builds the
ionic potential and the initial orbital, then uses that orbital to obtain the
initial density, Hartree potential, and exchange-correlation quantities before
starting the minimization:

.. code-block:: text

    def runSCF(atoms: Atoms, Eewald: ℝ, Nit: ℕ, β: ℝ, etol: ℝ, seed: ℕ): ℝ:
        ionic_potential: ℂ[k] = coulomb(atoms)
        W0: ℂ[p] = init_W(atoms, seed)
        Y0: ℂ[p] = orth(atoms, W0)
        n0: ℝ[k] = get_n_total(atoms, Y0)
        φ0: ℂ[k] = get_phi(atoms, n0)
        n0_c: ℂ[k] = (n0 + 1e-10) * (1.0 + 0j)
        ex_vx0: ℂ[2, k] = lda_x(n0_c)
        ec_vc0: ℂ[2, k] = lda_c_chachiyo(n0_c)
        exc0: ℂ[k] = ex_vx0[0] + ec_vc0[0]
        vxc0: ℂ[k] = ex_vx0[1] + ec_vc0[1]
        scf: SCF = SCF(
            atoms,
            ionic_potential,
            W0,
            Y0,
            n0,
            φ0,
            exc0,
            vxc0,
            Eewald
        )
        return sd(scf, Nit, β, etol)

``SCF`` is a class that holds these quantities during the calculation:
``atoms``, ``ionic_potential``, ``W``, ``Y``, ``n``, ``φ``, ``exc``, ``vxc``,
and ``Eewald``. ``Eewald`` is fixed because it depends only on the nuclear
configuration.

From this initial state, ``sd`` repeatedly updates the orbital. Each step
therefore goes through the same sequence:

#. normalize ``W`` to ``Y`` with ``orth``
#. build ``n`` with ``get_n_total``
#. obtain ``φ`` with ``get_phi``
#. evaluate ``exc``
#. compute the total energy
#. obtain ``grad(E, W)`` and update ``W``
#. stop updating once :math:`|\Delta E| \le` ``etol``

Running the calculation: ``dft_run``
-------------------------------------

``examples/dft_run.phyk`` puts the complete calculation together for three
systems. The cell, cutoff, grid, and minimization settings are shared by all
three calculations:

.. code-block:: text

   π: ℝ = 3.141592653589793

   a: ℝ = 16.0             # cubic cell side (Bohr)
   ecut: ℝ = 16.0          # plane-wave cutoff (Hartree)
   s: ℕ = 60               # grid points per axis

   gcut: ℝ = 2.0           # Ewald reciprocal-space cutoff
   gamma: ℝ = 1.0e-8       # Ewald screening tolerance

   Nit: ℕ = 1001            # maximum steepest-descent iterations
   β: ℝ = 1.0e-5            # step size
   etol: ℝ = 1.0e-6         # convergence tolerance (Hartree)
   seed: ℕ = 1234           # RNG seed for the initial orbital

   Natoms1: ℕ = 1
   Nstate1: ℕ = 1
   px1: ℝ[1] = [0.0]
   py1: ℝ[1] = [0.0]
   pz1: ℝ[1] = [0.0]

Hydrogen has one nucleus with charge 1 and one electron:

.. code-block:: text

   Z_H: ℝ[1] = [1.0]
   f_H: ℝ[1] = [1.0]

   H_atom: Atoms = Atoms(a, ecut, s, s, s, Natoms1, px1, py1, pz1, Nstate1, Z_H, f_H)
   Eewald_H: ℝ = get_Eewald(H_atom, gcut, gamma)
   E_H: ℝ = runSCF(H_atom, Eewald_H, Nit, β, etol, seed)

For helium, the cell and grid remain unchanged. The nuclear charge and
occupation are changed to 2:

.. code-block:: text

   Z_He: ℝ[1] = [2.0]
   f_He: ℝ[1] = [2.0]

   He_atom: Atoms = Atoms(a, ecut, s, s, s, Natoms1, px1, py1, pz1, Nstate1, Z_He, f_He)
   Eewald_He: ℝ = get_Eewald(He_atom, gcut, gamma)
   E_He: ℝ = runSCF(He_atom, Eewald_He, Nit, β, etol, seed)

For H2, the number of atoms and their positions change. The two hydrogen
nuclei are placed 1.4 Bohr apart along the x axis. Both electrons occupy the
same spatial orbital, so ``Nstate`` remains 1 with occupation 2:

.. code-block:: text

   Natoms2: ℕ = 2
   Nstate2: ℕ = 1

   px2: ℝ[2] = [0.0, 1.4]
   py2: ℝ[2] = [0.0, 0.0]
   pz2: ℝ[2] = [0.0, 0.0]

   Z_H2: ℝ[2] = [1.0, 1.0]
   f_H2: ℝ[1] = [2.0]

   H2_atom: Atoms = Atoms(a, ecut, s, s, s, Natoms2, px2, py2, pz2, Nstate2, Z_H2, f_H2)
   Eewald_H2: ℝ = get_Eewald(H2_atom, gcut, gamma)
   E_H2: ℝ = runSCF(H2_atom, Eewald_H2, Nit, β, etol, seed)

The resulting energies can now be compared directly with the SimpleDFT.jl
reference:

.. list-table::
   :header-rows: 1
   :widths: 20 18 31 31

   * - System
     - Electrons
     - Physika (Hartree)
     - SimpleDFT.jl (Hartree)
   * - H
     - 1
     - -0.438418
     - -0.438418
   * - He
     - 2
     - -2.632035
     - -2.632035
   * - H2
     - 2
     - -1.113969
     - -1.113969

The Physika results reproduce the SimpleDFT.jl values for all three systems.

These are LDA total energies for the chosen cell, cutoff, and grid, rather
than the exact energies of the systems. For comparison, the exact hydrogen
ground-state energy is -0.5 Hartree and helium is near -2.90 Hartree. There
are several reasons for the difference.

- **The functional.** In LDA, the Hartree term includes the repulsion of the
  hydrogen electron with its own charge cloud. Exact exchange cancels this
  self-interaction, while LDA exchange only does so approximately. This
  contributes to the hydrogen energy being near -0.44 rather than -0.5
  Hartree.
- **The basis and grid.** A finite cutoff and grid introduce discretization
  error. The all-electron potential also converges slowly because of the
  nuclear cusp.
- **The cell.** Periodic boundary conditions allow the system to interact
  weakly with its periodic images.

These differences come from the approximations and numerical settings used
in the calculation. The agreement with SimpleDFT.jl shows that the Physika
implementation reproduces the same calculation.

Finding the H2 equilibrium bond length
--------------------------------------

We can use the same total-energy calculation to find the equilibrium H-H
bond length. We calculate the total energy for several bond lengths and
identify the distance at which the energy is lowest.

The cell, cutoff, grid, and SCF settings are the same as in ``dft_run``.
Only the H-H distance changes between calculations. The complete calculation
is shown below.

.. code-block:: text

   from dft_atoms import Atoms
   from dft_operators import op_O, op_L, op_Linv, op_J, op_I, op_Jdag
   from dft_density import orth, get_n_total, get_phi, mask_embed
   from dft_potentials import coulomb
   from dft_xc import lda_x, lda_c_chachiyo
   from dft_energies import get_Ekin, get_Ecoul, get_Exc, get_Een, get_Eewald, axis_index, vec_size
   from dft_scf import SCF, init_W, energy_of_W, sd, runSCF

   π: ℝ = 3.141592653589793

   a: ℝ = 16.0             # cubic cell side (Bohr)
   ecut: ℝ = 16.0          # plane-wave cutoff (Hartree)
   s: ℕ = 60               # grid points per axis

   gcut: ℝ = 2.0           # Ewald reciprocal-space cutoff
   gamma: ℝ = 1.0e-8       # Ewald screening tolerance

   Nit: ℕ = 1001           # maximum steepest-descent iterations
   β: ℝ = 1.0e-5           # step size
   etol: ℝ = 1.0e-6        # convergence tolerance (Hartree)
   seed: ℕ = 1234           # RNG seed for the initial orbital

   Natoms2: ℕ = 2
   Nstate2: ℕ = 1

   Z_H2: ℝ[2] = [1.0, 1.0]
   f_H2: ℝ[1] = [2.0]

   def energy_at_d(d: ℝ): ℝ:
       px: ℝ[2] = [0.0, d]
       py: ℝ[2] = [0.0, 0.0]
       pz: ℝ[2] = [0.0, 0.0]

       H2: Atoms = Atoms(
           a, ecut, s, s, s,
           Natoms2, px, py, pz,
           Nstate2, Z_H2, f_H2
       )

       Eewald: ℝ = get_Eewald(H2, gcut, gamma)
       return runSCF(H2, Eewald, Nit, β, etol, seed)

   ds: ℝ[9] = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
   Es: ℝ[9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

   for i:
       Es[i] = energy_at_d(ds[i])

   Es

The resulting energies are:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - H-H distance (Bohr)
     - Total energy (Hartree)
   * - 1.1
     - -1.078134
   * - 1.2
     - -1.097013
   * - 1.3
     - -1.108265
   * - 1.4
     - -1.113968
   * - 1.5
     - **-1.115552**
   * - 1.6
     - -1.114033
   * - 1.7
     - -1.110198
   * - 1.8
     - -1.104660
   * - 1.9
     - -1.097922

The 1.4 Bohr point reproduces the H2 energy from the ``dft_run`` calculation,
since the numerical settings are unchanged. The energy decreases as the bond
length increases from 1.1 to 1.5 Bohr and then increases again. The lowest
energy in the scan occurs at 1.5 Bohr, so within the 0.1 Bohr spacing used
here, the equilibrium bond length is 1.5 Bohr.


Summary
-------

We went from the many-electron problem to a working DFT calculation:

- DFT replaces the many-electron wavefunction with the electron density,
  made practical by the Kohn-Sham construction.

- In a periodic cell, plane waves turn the Laplacian and Poisson's equation
  into elementwise operations (``dft_operators``).

- The total energy splits into five terms, with one function for each term
  (``dft_energies``).

- Exchange-correlation is the only approximate term: Slater exchange and
  Chachiyo correlation under LDA (``dft_xc``).

- ``dft_scf`` reaches self-consistency by minimizing the energy directly,
  with ``grad`` supplying the gradient.

- The same automatic differentiation can be extended to nuclear positions
  or functional parameters, providing a starting point for forces and
  functional fitting.

- The results reproduce SimpleDFT.jl for hydrogen, helium, and the hydrogen
  molecule.
  
- For H2, varying the bond length and comparing the total energies gives the
  equilibrium bond length.

References
----------

.. [SimpleDFTjl] Schulze, W. T. SimpleDFT.jl: A simple plane wave density
   functional theory Julia code.
   `<https://gitlab.com/wangenau/simpledft.jl>`_.

.. [SimpleDFT] Schulze, W. T. SimpleDFT: A simple plane wave density functional
   theory code. `<https://gitlab.com/wangenau/simpledft>`_.
   Documentation: `<https://wangenau.gitlab.io/simpledft_pages/>`_.

.. [BornOppenheimer1927] Born, M. and Oppenheimer, R. Zur Quantentheorie der
   Molekeln. *Annalen der Physik*, 389(20):457-484, 1927. doi:
   `10.1002/andp.19273892002 <https://doi.org/10.1002/andp.19273892002>`_.

.. [Schrodinger1926] Schrödinger, E. Quantisierung als Eigenwertproblem
   (Erste Mitteilung). *Annalen der Physik*, 384(4):361-376, 1926. doi:
   `10.1002/andp.19263840404 <https://doi.org/10.1002/andp.19263840404>`_.

.. [HK1964] Hohenberg, P. and Kohn, W. Inhomogeneous electron gas.
   *Physical Review*, 136(3B):B864-B871, 1964. doi:
   `10.1103/PhysRev.136.B864 <https://doi.org/10.1103/PhysRev.136.B864>`_.

.. [KS1965] Kohn, W. and Sham, L. J. Self-consistent equations including
   exchange and correlation effects. *Physical Review*, 140(4A):A1133-A1138,
   1965. doi: `10.1103/PhysRev.140.A1133
   <https://doi.org/10.1103/PhysRev.140.A1133>`_.

.. [IsmailBeigiArias2000] Ismail-Beigi, S. and Arias, T. A. New algebraic
   formulation of density functional calculation. *Computer Physics
   Communications*, 128(1-2):1-45, 2000. doi:
   `10.1016/S0010-4655(00)00072-2
   <https://doi.org/10.1016/S0010-4655(00)00072-2>`_.

.. [AriasDFT] Arias, T. A. Practical DFT mini-course. Cornell University.
   `<http://jdftx.org/PracticalDFT.html>`_.

.. [Chachiyo2016] Chachiyo, T. Communication: Simple and accurate uniform
   electron gas correlation energy for the full range of densities. *The
   Journal of Chemical Physics*, 145(2):021101, 2016. doi:
   `10.1063/1.4958669 <https://doi.org/10.1063/1.4958669>`_.

.. [Ewald1921] Ewald, P. P. Die Berechnung optischer und elektrostatischer
   Gitterpotentiale. *Annalen der Physik*, 369(3):253-287, 1921. doi:
   `10.1002/andp.19213690304 <https://doi.org/10.1002/andp.19213690304>`_.

.. [Slater1951] Slater, J. C. A simplification of the Hartree-Fock method.
   *Physical Review*, 81(3):385-390, 1951. doi:
   `10.1103/PhysRev.81.385 <https://doi.org/10.1103/PhysRev.81.385>`_.

.. [Lowdin1950] Löwdin, P.-O. On the non-orthogonality problem connected
   with the use of atomic wave functions in the theory of molecules and
   crystals. *The Journal of Chemical Physics*, 18(3):365-375, 1950. doi:
   `10.1063/1.1747632 <https://doi.org/10.1063/1.1747632>`_.
