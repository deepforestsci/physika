Learning a High-Dimensional Potential Energy Surface with a Behler-Parrinello Neural Network
============================================================================================

Overview
--------

This tutorial builds the neural network potential energy surface (NN-PES) of
Behler and Parrinello [#behler]_ equation by equation. A conventional neural
network that maps Cartesian coordinates straight onto an energy is unusable as
a potential because its output changes when two identical atoms are exchanged, and a
network trained on one system size cannot be applied to another. The
Behler-Parrinello construction removes both limitations by writing the total
energy as a sum of atomic contributions, and by feeding each atom not its
coordinates but a set of *symmetry functions* that describe its local
environment.

Silicon serves as the physical example. The configuration and the reference
energies are three frames of an eight-atom cell taken from a Materials Project
relaxation trajectory [#mp]_.

The tutorial is divided into four parts:

1. **The descriptor:** transform the coordinates of each atom into a
   rotation-, translation- and permutation-invariant description of its
   neighbourhood.
2. **The subnet and the total energy:** map that description onto an atomic
   energy with a small network whose weights are shared by every atom, and sum
   the result.
3. **Energy, forces and parameter gradients:** differentiate the total energy
   with respect to the coordinates and to the weights.
4. **Training:** fit the weights to reference energies.

.. admonition:: Problem statement

   Represent the potential energy of a silicon configuration as a sum of atomic
   contributions, each predicted from an invariant description of the atom's
   local environment by one shared neural network,

   .. math::

      E=\sum_i E_i,\qquad E_i=S\bigl(\{G_i^\mu\}\bigr),

   and obtain the forces as :math:`\mathbf F_i=-\partial E/\partial\mathbf R_i`.

Physical model
--------------

Atomic decomposition
~~~~~~~~~~~~~~~~~~~~

The total energy is written as a sum over atoms [#behler]_,

.. math::

   E=\sum_i E_i.

Invariance requirements
~~~~~~~~~~~~~~~~~~~~~~~

The quantities fed to the subnets must satisfy three conditions:
- They must be invariant under rotation and translation of the system
- Their number must be independent of the atom's coordination, since that changes during a molecular dynamics run
- Two identical local environments must give identical values while two different ones must not.

Permutation invariance is handled separately, by constraining the subnets
rather than the descriptor. In the words of the original Letter [#behler]:

   To ensure the invariance of the total energy with respect to the
   interchanging of two atoms the structure of all subnets and the values of
   the weight parameters are constrained to be identical in each :math:`S_i`.

Cutoff function
~~~~~~~~~~~~~~~

The local environment is delimited by a cutoff function of the interatomic
distance :math:`R_{ij}`,

.. math::

   f_c(R_{ij})=
   \begin{cases}
     0.5\left[\cos\left(\dfrac{\pi R_{ij}}{R_c}\right)+1\right],
       & R_{ij}\le R_c,\\[2ex]
     0, & R_{ij}>R_c.
   \end{cases}

Both the value and the slope vanish at :math:`R_{ij}=R_c`, so atoms enter and
leave the environment smoothly and the energy stays differentiable.

Radial symmetry functions
~~~~~~~~~~~~~~~~~~~~~~~~~

The radial description is a sum of Gaussians over all neighbours, with width
:math:`\eta` and offset :math:`R_s`,

.. math::

   G_i^1=\sum_{j\ne i}e^{-\eta(R_{ij}-R_s)^2}f_c(R_{ij}).

Summing over *all* neighbours is what makes the count of symmetry functions
independent of the coordination number.

Angular symmetry functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The angular description runs over all triplets centred on atom :math:`i`,

.. math::

   G_i^2=2^{1-\zeta}\sum_{j,k\ne i}
   (1+\lambda\cos\theta_{ijk})^{\zeta}
   e^{-\eta(R_{ij}^2+R_{ik}^2+R_{jk}^2)}
   f_c(R_{ij})f_c(R_{ik})f_c(R_{jk}),

with :math:`\lambda=\pm1`, and

.. math::

   \cos\theta_{ijk}=
   \frac{\mathbf R_{ij}\cdot\mathbf R_{ik}}{R_{ij}R_{ik}},
   \qquad \mathbf R_{ij}=\mathbf R_i-\mathbf R_j.

The three cutoff factors and the Gaussian together ensure a smooth decay to
zero at large separation. Because both :math:`G_i^1` and :math:`G_i^2` depend
on every atomic position inside the cutoff, they are genuinely many-body terms.

The atomic subnet
~~~~~~~~~~~~~~~~~

Each atom's symmetry function values enter one small feed-forward network. For
two symmetry functions and three hidden nodes, which is the case implemented
here, the Letter [#behler] writes it as

.. math::

   E_i=f_a^2\left(w_{01}^2+\sum_{j=1}^{3}w_{j1}^2\,
   f_a^1\left(w_{0j}^1+\sum_{\mu=1}^{2}w_{\mu j}^1 G_i^\mu\right)\right),

where :math:`w_{ij}^k` connects node :math:`j` in layer :math:`k` to node
:math:`i` in layer :math:`k+1` and :math:`w_{0j}^k` is a bias. The hidden
activation is the hyperbolic tangent and the output layer is linear.

Part 1: The atomic environment descriptor
-----------------------------------------

1.1 Cutoff function
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   π: ℝ = 3.141592653589793

   def fc(rij: ℝ, rc: ℝ): ℝ:
       θ: ℝ = π * rij / rc
       if rij <= rc:
           return 0.5 * (cos(θ) + 1.0)
       else:
           return 0.0

1.2 Interatomic distance
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def dist(pa: ℝ[3], pb: ℝ[3]): ℝ:
       d: ℝ[3] = pa - pb
       return sqrt(d @ d)

.. note::

   This treats the cell as an isolated cluster. A periodic configuration needs
   the minimum image convention, which this implementation does not yet apply.

1.3 Radial symmetry function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def G1(a: ℕ, pos: ℝ[na, 3], natoms: ℕ, η: ℝ, rs: ℝ, rc: ℝ): ℝ:
       total: ℝ = 0.0
       for j: ℕ(natoms):
           if j != a:
               rij = dist(pos[a], pos[j])
               total += exp(-η * (rij - rs)**2) * fc(rij, rc)
       return total

1.4 Angular symmetry function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def G2(a: ℕ, pos: ℝ[na, 3], natoms: ℕ, η: ℝ, ζ: ℕ, lam: ℤ, rc: ℝ): ℝ:
       total: ℝ = 0.0
       for j: ℕ(natoms):
           if j != a:
               for k: ℕ(natoms):
                   if k != a:
                       if k != j:
                           vij = pos[a] - pos[j]
                           vik = pos[a] - pos[k]
                           rij = dist(pos[a], pos[j])
                           rik = dist(pos[a], pos[k])
                           rjk = dist(pos[j], pos[k])
                           cosθijk = (vij @ vik) / (rij * rik)
                           gauss = exp(-η * (rij**2 + rik**2 + rjk**2))
                           total += (1.0 + lam * cosθijk)**ζ * gauss * fc(rij, rc) * fc(rik, rc) * fc(rjk, rc)
       return 2.0**(1.0 - ζ) * total

Part 2: The atomic subnet and the total energy
----------------------------------------------

2.1 Activation function
~~~~~~~~~~~~~~~~~~~~~~~

The hidden layer activation is the hyperbolic tangent, written through the identity

.. math::

   \tanh x=1-\frac{2}{e^{2x}+1}.

.. code-block:: python

   # Eq. (1): activation
   def fa(x: ℝ): ℝ:
       return 1.0 - 2.0 / (exp(2.0 * x) + 1.0)

2.2 Subnet
~~~~~~~~~~

.. code-block:: python

   def S(g: ℝ[ng], nin: ℕ, nhid: ℕ, w1: ℝ[nh, ng], b1: ℝ[nh], w2: ℝ[nh], b2: ℝ): ℝ:
       total: ℝ = b2
       for j: ℕ(nhid):
           z = b1[j]
           for m: ℕ(nin):
               z += w1[j, m] * g[m]
           total += w2[j] * fa(z)
       return total

2.3 Composing the network
~~~~~~~~~~~~~~~~~~~~~~~~~

Three classes split the model along the lines of the equations:
- ``Descriptor`` holds the symmetry-function parameters and maps coordinates to :math:`\{G_i^\mu\}`
- ``Subnet`` holds the weights and maps :math:`\{G_i^\mu\}` to :math:`E_i`
- ``BPNet`` composes the ``Descriptor`` and ``Subnet``.

Because ``BPNet`` holds exactly one ``Subnet``, every atom is evaluated by the
same weights. That is the architectural constraint the Letter [#behler] requires for
permutation invariance, expressed as object identity rather than as a penalty.

.. code-block:: python

   class Descriptor(rs: ℝ, ζ: ℕ, lam: ℤ):
       def λ(a: ℕ, pos: ℝ[na, 3], natoms: ℕ) → ℝ[2]:
           return [G1(a, pos, natoms, η0, this.rs, rcut), G2(a, pos, natoms, ηa, this.ζ, this.lam, rcut)]
       def update_params(lr: ℝ, learnable_grads: ℝ[1]):
           this.rs = this.rs - lr * learnable_grads[0]

   class Subnet(w1: ℝ[3, 2], b1: ℝ[3], w2: ℝ[3], b2: ℝ, nin: ℕ, nhid: ℕ):
       def λ(g: ℝ[2]) → ℝ:
           return S(g, this.nin, this.nhid, this.w1, this.b1, this.w2, this.b2)
       def update_params(lr: ℝ, learnable_grads: ℝ[4]):
           this.w1 = this.w1 - lr * learnable_grads[0]
           this.b1 = this.b1 - lr * learnable_grads[1]
           this.w2 = this.w2 - lr * learnable_grads[2]
           this.b2 = this.b2 - lr * learnable_grads[3]

   # Eq. (2): E = Σ(i) Ei
   class BPNet:
       desc: Descriptor
       sub: Subnet
       def λ(pos: ℝ[na, 3], natoms: ℕ) → ℝ:
           total: ℝ = 0.0
           for a: ℕ(natoms):
               total += sub(desc(a, pos, natoms))
           return total

2.4 Which parameters are trainable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The symmetry functions are a *choice* of representation rather than part of the
fit.

Four of the six symmetry-function parameters also carry constraints that a
gradient step would break:
- :math:`\eta` must stay positive
- :math:`e^{-\eta R^2}` becomes a divergent :math:`e^{+|\eta|R^2}`
- :math:`\lambda` must stay :math:`\pm1`
- :math:`1+\lambda\cos\theta` can go negative
- :math:`(1+\lambda\cos\theta)^\zeta` returns ``NaN``
- :math:`\zeta` must stay a positive integer
- :math:`R_c` decides which neighbours exist

.. code-block:: python

   # Fixed symmetry-function constants
   η0: ℝ = 0.05
   ηa: ℝ = 0.005
   rcut: ℝ = 4.0

That leaves ``rs`` as the only differentiable symmetry-function parameter,
alongside the subnet weights.

Part 3: Energy, forces and parameter gradients
----------------------------------------------

3.1 Configuration and total energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # A single configuration
   Pc: ℕ = 8
   r: ℝ[Pc, 3] = [[ 3.69468256,  2.44237664,  1.57337608],
               [ 1.22577221,  0.73713025,  6.56769165],
               [ 0.94713127,  0.27985289,  2.65383696],
               [-0.70279321,  0.26258477,  8.05036569],
               [ 1.77357416,  2.44520749,  3.17501106],
               [ 0.68309124, -0.82981877,  4.72394411],
               [-0.40740593,  1.70240007,  5.08154978],
               [-1.40101553, -1.92847942,  4.49800493]]

   rs0: ℝ = 0.0
   ζ1: ℕ = 1
   lam_p: ℤ = 1
   nin0: ℕ = 2
   nhid0: ℕ = 3
   w1v: ℝ[3, 2] = [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]
   b1v: ℝ[3] = [0.05, -0.1, 0.15]
   w2v: ℝ[3] = [0.7, -0.8, 0.9]
   b2v: ℝ = 0.25

   dsc: Descriptor = Descriptor(rs0, ζ1, lam_p)
   sbn: Subnet = Subnet(w1v, b1v, w2v, b2v, nin0, nhid0)
   net: BPNet = BPNet(dsc, sbn)

   # E for the whole 8-atom cell
   net(r, Pc)

.. admonition:: Expected results

   | ✓ No type errors found
   | 1.397019624710083 ∈ ℝ

The weights are untrained, so this energy carries no physical meaning yet.

3.2 Forces
~~~~~~~~~~

The forces are :math:`\mathbf F_i=-\partial E/\partial\mathbf R_i`.

.. code-block:: python

   # Forces:
   grad(net(r, Pc), r)

3.3 Parameter gradients
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BPNet:
       # ...
       def desc_grads(pos: ℝ[na, 3], natoms: ℕ) → ℝ[m]:
           en: ℝ = this(pos, natoms)
           return grad(en, this.desc.learnable_params)
       def sub_grads(pos: ℝ[na, 3], natoms: ℕ) → ℝ[m]:
           en: ℝ = this(pos, natoms)
           return grad(en, this.sub.learnable_params)

.. code-block:: python

   # dE/drs -- the only differentiable symmetry-function parameter left
   net.desc_grads(r, Pc)

   # dE/d(w1, b1, w2, b2)
   net.sub_grads(r, Pc)

.. admonition:: Expected results

   | [-0.6535038352012634] ∈ ℝ[1]
   | [[[2.957629919052124, 0.2866087257862091], [-3.3511080741882324, -0.3236590325832367], [3.7492239475250244, 0.3632603585720062]], [5.550079345703125, -6.3162641525268555, 7.080302715301514], [0.7418038249015808, 0.6403953433036804, -0.6776964664459229], 8.0] ∈ ℝ[4]

``desc_grads`` returns a single number because ``rs`` is the only learnable
descriptor parameter.

Part 4: Training
----------------

4.1 Reference data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Training on 3 frames of the 8 Si atoms MD simulation trajectory
   X3: ℝ[3, 8, 3] = [[[  3.69468256,   2.44237664,   1.57337608],
           ...
           [ -1.40445908,  -1.93079596,   4.51103173]]]

   # Target Energy values
   y3: ℝ[3] = [-39.9157861, -39.91789224, -39.91789594]

   nframes3: ℕ = 3
   nep: ℕ = 10
   lr0: ℝ = 0.0001

4.2 Loss
~~~~~~~~

The loss is the squared error in the total energy,

.. math::

   \mathcal L=\bigl(E^{\mathrm{pred}}-E^{\mathrm{ref}}\bigr)^2 .

.. code-block:: python

   def mse_loss(pred: ℝ, target: ℝ): ℝ:
       return (pred - target)**2

4.3 Training loop
~~~~~~~~~~~~~~~~~

``train`` performs stochastic gradient descent on the subnet weights, one
update per frame.

.. code-block:: python

   class BPNet:
       # ...
       def train(X: ℝ[nf, na, 3], y: ℝ[nf], nframes: ℕ, natoms: ℕ, epochs: ℕ, lr: ℝ) → ℝ:
           last: ℝ = 0.0
           for ep: ℕ(epochs):
               epoch_loss = 0.0
               for s: ℕ(nframes):
                   pred = this(X[s], natoms)
                   current_loss = mse_loss(pred, y[s])
                   epoch_loss = epoch_loss + current_loss
                   learnable_grads = grad(current_loss, this.sub.learnable_params)
                   this.sub.update_params(lr, learnable_grads)
               last = epoch_loss / nframes
           return last

.. code-block:: python

   # loss before training
   mse_loss(net(r, Pc), y3[0])

   # train, returning the final mean loss
   net.train(X3, y3, nframes3, Pc, nep, lr0)

   # predictions after training, against targets
   net(X3[0], Pc)
   net(X3[1], Pc)
   net(X3[2], Pc)

.. admonition:: Expected results

   | 1706.7479248046875 ∈ ℝ
   | 122.2815170288086 ∈ ℝ
   | -29.940744400024414 ∈ ℝ
   | -29.91435432434082 ∈ ℝ
   | -29.91513442993164 ∈ ℝ


4.4 Fitting the descriptor as well
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``train_all`` additionally updates ``rs``, with its own learning rate ``lrd``.
Setting ``lrd`` to zero makes it identical to ``train``.

.. code-block:: python

   class BPNet:
       # ...
       def train_all(X: ℝ[nf, na, 3], y: ℝ[nf], nframes: ℕ, natoms: ℕ, epochs: ℕ, lr: ℝ, lrd: ℝ) → ℝ:
           last: ℝ = 0.0
           for ep: ℕ(epochs):
               epoch_loss = 0.0
               for s: ℕ(nframes):
                   pred = this(X[s], natoms)
                   current_loss = mse_loss(pred, y[s])
                   epoch_loss = epoch_loss + current_loss
                   sub_g = grad(current_loss, this.sub.learnable_params)
                   desc_g = grad(current_loss, this.desc.learnable_params)
                   this.sub.update_params(lr, sub_g)
                   this.desc.update_params(lrd, desc_g)
               last = epoch_loss / nframes
           return last

.. code-block:: python

   # optional: fit rs as well, with train_all
   lrd0: ℝ = 0.0001
   dsc2: Descriptor = Descriptor(0.0, ζ1, lam_p)
   net2: BPNet = BPNet(dsc2, Subnet([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]], [0.05, -0.1, 0.15], [0.7, -0.8, 0.9], 0.25, nin0, nhid0))
   net2.train_all(X3, y3, nframes3, Pc, nep, lr0, lrd0)

   dsc2.rs

.. admonition:: Expected results

   | 120.75401306152344 ∈ ℝ
   | 0.12136517465114594 ∈ ℝ

Scope of this implementation
----------------------------

The implementation is deliberately minimal, and differs from the published
potential in the following ways:

* **One symmetry function of each type.** The [#behler] uses 48 symmetry
  functions in total, over a range of :math:`\eta`, :math:`R_s` and
  :math:`\zeta` values, feeding subnets with two hidden layers of about 40
  nodes each. Here a single :math:`G^1` and a single :math:`G^2` feed three
  hidden nodes.
* **No periodic images.** ``dist`` uses the plain separation vector, so the
  cell is treated as an isolated cluster and neighbours across the periodic
  boundary are missed.
* **Energy-only loss.** Forces are computed but not fitted, although they
  carry far more information: these three frames provide three energies
  against :math:`3\times8\times3=72` force components.

Source code
-----------

The complete Physika implementation is provided in
``tutorials/nn_pes_behler_parrinello.phyk``.

References
----------

.. [#behler] J. Behler and M. Parrinello, "Generalized Neural-Network
   Representation of High-Dimensional Potential-Energy Surfaces,"
   *Phys. Rev. Lett.* **98**, 146401 (2007).
   `doi:10.1103/PhysRevLett.98.146401
   <https://doi.org/10.1103/PhysRevLett.98.146401>`_.

.. [#mp] A. Jain *et al.*, "Commentary: The Materials Project: A materials
   genome approach to accelerating materials innovation," *APL Materials*
   **1**, 011002 (2013). Trajectory data available through the
   `Materials Project <https://next-gen.materialsproject.org/>`_.
