Tensor Field Networks - Moment of Inertia
============================================

Tensor Field Networks for predicting the moment of inertia of a
point-mass cloud.

Introduction
------------

Tensor Field Networks (TFNs) [Thomas2018]_ are a specialized neural
network architecture designed to process 3D data, such as point
clouds or atoms, while respecting 3D geometric transformations. In
this tutorial, we implement a TFN from scratch in Physika and train
it on a synthetic dataset of point masses to predict their moment of
inertia.

.. figure:: /_static/tutorial_files/tfn_message_passing.png
   :alt: Comparison of a GCN and a CNN processing a molecule
   :align: center
   :width: 500px

   A pictorial demonstration of the message passing scheme in the
   Tensor Field Network.

Equivariance
------------

In 3D space, many properties are independent of the orientation of the
object. For example, when predicting moment of inertia, the resulting
rank-2 tensor should rotate accordingly as the point-mass cloud is
rotated. In other words, the model is equivariant to global rotations.

**Mathematical definition:** a function :math:`f` is equivariant to a
transformation :math:`g` if applying the transformation to the input
first and then running :math:`f` gives the same result as running
:math:`f` first and then applying a matching transformation
:math:`\mathcal{T}g` to the output:

.. math::

   f(g \cdot x) = (\mathcal{T}g)\, f(x)


Spherical Tensors
------------------

A **spherical tensor of rank** :math:`k` is a set of :math:`2k+1`
quantities, indexed by :math:`q = -k, ..., k`, that transform under
rotation via the Wigner D-matrix:

.. math::

   T^k_q \;\rightarrow\; \sum_{q'} \mathcal{D}^k_{qq'}(\mathcal{R})\, T^k_{q'}

The key property is **irreducibility**: a rank-:math:`k` spherical
tensor transforms entirely within itself under any rotation. By
contrast, an ordinary Cartesian tensor is *not* irreducible, it
decomposes into a sum of spherical tensors of different ranks.

.. note::
   The :math:`s, p, d, f, ...` orbital shapes are exactly the rank
   :math:`k = 0, 1, 2, 3, ...` spherical tensors, they're the
   angular part of the hydrogen atom's wavefunction,
   :math:`Y_k^q(\theta,\phi)`, the same functions used as
   :math:`Y_0, Y_1, Y_2` throughout this tutorial.

The Edge Tensor
---------------

For every pair of points :math:`(u, v)` in the cloud, the **edge
tensor** is the degree-:math:`\ell` geometric feature built from their
relative position :math:`r = r_u - r_v`:

The edge tensor handles **Translational invariance** by depending only on 
the relative position :math:`r`, which is unchanged if the entire point cloud
is shifted by a constant vector.

.. math::

   \mathsf{r}^\ell_{uv} = \varphi_\ell(r)\, Y_\ell(\hat{\mathbf{r}})

It has two components, intentionally kept separate:

- :math:`\varphi_\ell(r)` -- the *radial* part, a learned scalar
  function of only the distance :math:`r = |r_{uv}|`. Since distance
  is a rotation-invariant scalar, :math:`\varphi_\ell` can be an
  unconstrained neural network (the `Radial Network`_).
- :math:`Y_\ell(\hat{r})` -- the *angular* part, the fixed spherical
  harmonic of the direction :math:`\hat{r} = r/|r|`. This is what
  carries the rotation-equivariant behavior.

Wigner-D Matrices
------------------

A **Wigner D-matrix** :math:`D^\ell(R)` is a :math:`(2\ell+1)\times(2\ell+1)`
unitary matrix assigned to each rotation :math:`R`, one per degree
:math:`\ell`. Composite rotations multiply their D-matrices:
:math:`D^\ell(R_1)D^\ell(R_2) = D^\ell(R_1 R_2)`.

.. math::

   D^0(R) = 1, \qquad D^1(R) = R

:math:`\ell=0` (scalars) is untouched by rotation; :math:`\ell=1` *is*
the ordinary 3x3 rotation matrix itself.

Setup (Radial Basis Functions)
-------------------------------

We now set up the hyperparameters for the point-mass cloud and the
Gaussian radial-basis-function (RBF) expansion used to featurize
pairwise distances.

.. code-block:: text

   num_points: ℝ = 15

   rbf_low: ℝ = 0.0
   rbf_high: ℝ = 2.0
   rbf_count: ℝ = 30
   rbf_spacing: ℝ = (rbf_high - rbf_low) / rbf_count
   centers: ℝ[30] = for i : ℕ(rbf_count) -> rbf_low + i * rbf_spacing
   gamma: ℝ = 1.0 / rbf_spacing
   hidden: ℝ = 16

   center_idx: ℝ = 0.0
   rotation_angle: ℝ = 0.9
   w_init_low: ℝ = -0.2
   w_init_high: ℝ = 0.2

A raw distance :math:`r` is just one number, too little for a small
MLP to shape a geometric profile from. So instead of feeding :math:`r`
directly into the network, we expand it into a series of Gaussian
bumps spread evenly across :math:`[\text{rbf\_low}, \text{rbf\_high}]`,
turning a single distance into a higher-dimensional "soft histogram":

.. math::

   \text{rbf}(r)_c = \exp\!\left(-\gamma\,(r - \text{centers}_c)^2\right), \qquad c = 1, \dots, \text{rbf\_count}

.. figure:: /_static/tutorial_files/gaussian_bumps.png
   :alt: Comparison of a GCN and a CNN processing a molecule
   :align: center
   :width: 500px

   The Gaussian bump functions defined by ``centers`` and ``gamma``,
   spanning ``[rbf_low, rbf_high]``.

Helper Functions
-----------------

.. code-block:: text

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

   def get_3d_array_num_rows(x: ℝ[m, n, p]): ℝ:
       total: ℝ = 0
       temp: ℝ = 0
       for i:
           temp = x[i]
           total += 1
       return total

   def zero_1d(n: ℝ): ℝ[m]:
       results: ℝ[n] = for i:ℕ(n) -> i*0
       return results

   def zero_2d(rows: ℝ, cols: ℝ): ℝ[m, n]:
       results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
       return results

   def zero_3d(a: ℝ, b: ℝ, c: ℝ): ℝ[m, n, p]:
       results: ℝ[a, b, c] = for i:ℕ(a) -> for j:ℕ(b) -> for k:ℕ(c) -> k*0
       return results

   def difference_matrix(r: ℝ[num_points, 3]): ℝ[num_points, num_points, 3]:
       results: ℝ[num_points, num_points, 3] = zero_3d(num_points, num_points, 3)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               for k:ℕ(3):
                   results[i, j, k] = r[i, k] - r[j, k]
       return results

   def distance_matrix(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points]:
       results: ℝ[num_points, num_points] = zero_2d(num_points, num_points)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               acc = 0
               for k:ℕ(3):
                   acc += rij[i, j, k] * rij[i, j, k]
               results[i, j] = sqrt(acc + 1e-12)
       return results

- ``difference_matrix`` -- computes the relative vector
  :math:`r_{ij} = r_i - r_j` for every pair of points,
  making it translation-invariant.
- ``distance_matrix`` -- reduces each relative vector to its scalar
  length :math:`|r_{ij}|` via a regularized norm.

Activation Function
--------------------

After the linear layer produces a hidden pre-activation, we apply the
**shifted softplus** (``ssp``) activation function element-wise to
every value, instead of ReLU.

``ssp`` is used here instead of ReLU because the radial network's
output must be able to go negative, the analytically-derived radial
profile for the moment-of-inertia problem turns out to be
:math:`-r^2`, which would be impossible to learn with ReLU, since ReLU
clips every negative output to zero.

``ssp`` is "shifted" rather than plain softplus because we want the
network to learn a scalar function that can be exactly zero, most
physical applications require the output to be zero at :math:`r=0`,
and a zero-centered nonlinearity gives training a cleaner default to
start from.

Mathematically, ``ssp`` is defined as:

.. math::

   \text{ssp}(x) = \log\left(\tfrac{1}{2}e^x + \tfrac{1}{2}\right)

.. code-block:: text

   def ssp(x: ℝ): ℝ:
       return log(0.5 * exp(x) + 0.5)

Theory: Spherical Harmonics
-----------------------------

Wigner-D matrices tell us how a degree-:math:`\ell` feature is allowed
to transform, but not how to generate one. Spherical harmonics answer
this: for a direction :math:`\hat{r} = r/|r|` on the unit sphere,
:math:`Y_\ell : S^2 \to \mathbb{R}^{2\ell+1}` is a fixed, closed-form
function of only the *direction*, satisfying the equivariance
transformation law:

.. math::

   Y_\ell(\mathcal{R} \cdot \hat{\mathbf{r}}) = \mathcal{D}^\ell_\mathcal{R}\, Y_\ell(\hat{\mathbf{r}})

It's what lets us build the **edge
tensor**, the edge feature between two nodes:

.. math::

   \mathsf{r}^\ell = \varphi_\ell(r)\, Y_\ell(\hat{\mathbf{r}})

:math:`\varphi_\ell` depends only on :math:`|r|` (invariant), whereas
:math:`Y_\ell` depends only on :math:`\hat{r}` (equivariant), following
the theory in [Cheng2022]_.

**The three degrees used here:**

- :math:`Y_0` (1 component) -- the constant :math:`1` (a scalar).
- :math:`Y_1` (3 components) -- :math:`Y_1(\hat{r}) = \hat{r} =
  r/|r|`, the ordinary unit vector. Here :math:`D^1_R = R`, because
  rotating a unit vector is the same as applying the rotation matrix
  to it.
- :math:`Y_2` (5 components) -- forming an irreducible basis (similar to the
  real d-orbital shapes):

.. math::

   Y_2 = \left[\frac{xy}{r^2},\ \frac{yz}{r^2},\ \frac{2z^2-x^2-y^2}{2\sqrt{3}\,r^2},\ \frac{zx}{r^2},\ \frac{x^2-y^2}{2r^2}\right]

.. figure:: /_static/tutorial_files/l2_harmonics.png
   :alt: Comparison of a GCN and a CNN processing a molecule
   :align: center
   :width: 500px

   The five real l=2 spherical harmonic "orbital" shapes, matching the
   component order used by ``Y2`` below (xy, yz, z^2, zx, x^2-y^2).

.. code-block:: text

   def Y0(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points, 1]:
       results: ℝ[num_points, num_points, 1] = zero_3d(num_points, num_points, 1)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               results[i, j, 0] = 1.0
       return results

   def Y1(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points, 3]:
       results: ℝ[num_points, num_points, 3] = zero_3d(num_points, num_points, 3)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               x = rij[i, j, 0]
               y = rij[i, j, 1]
               z = rij[i, j, 2]
               r_norm = sqrt(x*x + y*y + z*z + 1e-12)
               results[i, j, 0] = x / r_norm
               results[i, j, 1] = y / r_norm
               results[i, j, 2] = z / r_norm
       return results

   def Y2(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points, 5]:
       results: ℝ[num_points, num_points, 5] = zero_3d(num_points, num_points, 5)
       sqrt3: ℝ = sqrt(3.0)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               x = rij[i, j, 0]
               y = rij[i, j, 1]
               z = rij[i, j, 2]
               r2 = x*x + y*y + z*z + 1e-12
               results[i, j, 0] = x*y / r2
               results[i, j, 1] = y*z / r2
               results[i, j, 2] = (2.0*z*z - x*x - y*y) / (2.0 * sqrt3 * r2)
               results[i, j, 3] = z*x / r2
               results[i, j, 4] = (x*x - y*y) / (2.0 * r2)
       return results

Radial Network
----------------

Of the edge tensor :math:`\varphi_\ell(r)\cdot Y_\ell(\hat{r})`,
:math:`Y_\ell` is fixed and non-learnable; :math:`\varphi_\ell` is the
learnable half.

We choose a learned function instead of a fixed formula for the radial
component since an MLP is a universal function approximator: it takes
the RBF encoding of :math:`r` as input, and training lets it converge
to the true radial profile dictated by the underlying physics.

.. code-block:: text

   def gaussian_rbf(dij: ℝ[num_points, num_points], centers: ℝ[rbf_count], gamma: ℝ): ℝ[num_points, num_points, rbf_count]:
       results: ℝ[num_points, num_points, rbf_count] = zero_3d(num_points, num_points, rbf_count)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               for c:ℕ(rbf_count):
                   diff = dij[i, j] - centers[c]
                   results[i, j, c] = exp(-gamma * diff * diff)
       return results

   def to_column(x: ℝ[n]): ℝ[n, 1]:
       n_x: ℝ = get_1d_array_length(x)
       results: ℝ[n_x, 1] = zero_2d(n_x, 1)
       for i:ℕ(n_x):
           results[i, 0] = x[i]
       return results

   def ssp_col(x: ℝ[m, 1]): ℝ[m, 1]:
       return log(0.5 * exp(x) + 0.5)

   def radial_net(rbf_features: ℝ[rbf_count], w1: ℝ[hidden, rbf_count], b1: ℝ[hidden, 1], w2: ℝ[1, hidden], b2: ℝ[1, 1]): ℝ:
       x_col: ℝ[rbf_count, 1] = to_column(rbf_features)
       h_pre: ℝ[hidden, 1] = w1 @ x_col + b1
       h: ℝ[hidden, 1] = ssp_col(h_pre)
       out: ℝ[1, 1] = w2 @ h + b2
       return out[0, 0]

   def radial_field(rbf: ℝ[num_points, num_points, rbf_count], w1: ℝ[hidden, rbf_count], b1: ℝ[hidden, 1], w2: ℝ[1, hidden], b2: ℝ[1, 1]): ℝ[num_points, num_points]:
       results: ℝ[num_points, num_points] = zero_2d(num_points, num_points)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               efeat = rbf[i, j]
               results[i, j] = radial_net(efeat, w1, b1, w2, b2)
       return results

Tensor Product Reduction & Clebsch-Gordan Decomposition
----------------------------------------------------------

Before defining tensor product reduction, let's review the ways of
making new equivariant functions. Given two equivariant functions
:math:`f, g : \mathbb{R}^3 \to \mathbb{R}^3`, two immediate
combinations are their **linear combination** :math:`af + bg` and
their **composition** :math:`g \circ f` -- both equivariant.

**Tensor product** offers a third way. :math:`f \otimes g` is also
equivariant, and transforms according to:

.. math::

   \mathcal{D} f \otimes \mathcal{D}' g = (\mathcal{D} \otimes \mathcal{D}')(f \otimes g)

where :math:`D, D'` are Wigner D-matrices and :math:`D \otimes D'` is
their Kronecker product. This is equivariant, but it blows up in
dimension: tensor-producting a degree-:math:`\ell` feature with a
degree-:math:`k` feature would take them jointly to
:math:`(2\ell+1)(2k+1)` dimensions. To fix this, we reduce the product
into a sum of ordinary, smaller-degree features:

.. math::

   L_1 \otimes L_2 = |L_1-L_2| \oplus \cdots \oplus (L_1+L_2)

via the **Clebsch-Gordan coefficients**, a fixed, non-learned table
determined by group theory:

.. math::

   C_{Jm} = \sum_{m_1,m_2} a_{\ell m_1}\, b_{k m_2}\, \langle \ell m_1\, k m_2 \mid Jm \rangle

.. code-block:: text

   def tensor_product_reduce(cg: ℝ[d1, d2, d3], a: ℝ[d1], b: ℝ[d2], dim1: ℝ, dim2: ℝ, dim3: ℝ): ℝ[d3]:
       results: ℝ[dim3] = zero_1d(dim3)
       for k:ℕ(dim3):
           acc = 0
           for i:ℕ(dim1):
               for j:ℕ(dim2):
                   acc += cg[i, j, k] * a[i] * b[j]
           results[k] = acc
       return results

   cg_000: ℝ[1, 1, 1] = [[[1.0]]]
   cg_202: ℝ[5, 1, 5] = [
       [[1.0, 0.0, 0.0, 0.0, 0.0]],
       [[0.0, 1.0, 0.0, 0.0, 0.0]],
       [[0.0, 0.0, 1.0, 0.0, 0.0]],
       [[0.0, 0.0, 0.0, 1.0, 0.0]],
       [[0.0, 0.0, 0.0, 0.0, 1.0]]
   ]

**Naming convention:** ``cg_l1l2l3`` names the tensor reducing a
degree-``l1`` feature tensor-producted with a degree-``l2`` feature
into a degree-``l3`` output. ``cg_000`` is ``CG(0,0,0)`` (a degree-0
edge feature combined with a degree-0 node feature, reduced to degree
0), and ``cg_202`` is ``CG(2,0,2)`` (a degree-2 edge feature combined
with a degree-0 node feature, reduced to degree 2).

Theory: Equivariant Message Passing
--------------------------------------

A TFN layer updates each node :math:`u` by aggregating, over every
neighbor :math:`v`, the tensor product of the edge feature
:math:`Y_J(\hat{r}_{uv})` with the neighbor's feature :math:`x_v^k`:

.. math::

   \mathbf{x}_u^\ell=\sum_{v\in\mathcal{N}(u)}\sum_{k\ge 0}\sum_{J=|k-\ell|}^{k+\ell}\hat{\varphi}_J^{\ell k}(r)\sum_{m=-J}^{J}Y^m_J(\hat{\mathbf{r}})\,Q_{Jm}^{\ell k}\,\mathbf{x}^k_v

.. code-block:: text

   def filter_00(masses: ℝ[num_points], phi0: ℝ[num_points, num_points]): ℝ[num_points]:
       results: ℝ[num_points] = zero_1d(num_points)
       for a:ℕ(num_points):
           acc = [0.0]
           for b:ℕ(num_points):
               edge0 = [phi0[a, b]]
               mass_b = [masses[b]]
               contrib = tensor_product_reduce(cg_000, edge0, mass_b, 1, 1, 1)
               acc = [acc[0] + contrib[0]]
           results[a] = acc[0]
       return results

   def filter_22(masses: ℝ[num_points], phi2: ℝ[num_points, num_points], y2: ℝ[num_points, num_points, 5]): ℝ[num_points, 5]:
       results: ℝ[num_points, 5] = zero_2d(num_points, 5)
       for a:ℕ(num_points):
           acc = zero_1d(5)
           for b:ℕ(num_points):
               edge2 = [phi2[a,b]*y2[a,b,0], phi2[a,b]*y2[a,b,1], phi2[a,b]*y2[a,b,2], phi2[a,b]*y2[a,b,3], phi2[a,b]*y2[a,b,4]]
               mass_b = [masses[b]]
               contrib = tensor_product_reduce(cg_202, edge2, mass_b, 5, 1, 5)
               for m:ℕ(5):
                   acc[m] = acc[m] + contrib[m]
           for m:ℕ(5):
               results[a, m] = acc[m]
       return results

Code: Reassembling the Tensor
--------------------------------

A 3x3 matrix is basically the tensor product of two ordinary 3D
vectors, :math:`1 \otimes 1`, tensor-producted together to give 9
independent numbers. Plugging :math:`L_1=L_2=1` into the reduction
rule from Clebsch-Gordan:

.. math::

   1 \otimes 1 = 0 \oplus 1 \oplus 2, \qquad 1 + 3 + 5 = 9

So any 3x3 matrix :math:`M` decomposes into three pieces, each
transforming under its own clean representation:

.. math::

   M = \underbrace{\tfrac{1}{3}\,\mathrm{tr}(M)\, I}_{\ell=0}
   \;+\;
   \underbrace{\tfrac{1}{2}(M - M^\top)}_{\ell=1}
   \;+\;
   \underbrace{\left[\tfrac{1}{2}(M + M^\top) - \tfrac{1}{3}\,\mathrm{tr}(M)\, I\right]}_{\ell=2}

- degree 0 -- the trace, :math:`\mathrm{tr}(M)`, a single
  rotation-invariant scalar.
- degree 1 -- the antisymmetric part, :math:`(M - M^\top)/2`.
- degree 2 -- the traceless symmetric part, :math:`(M + M^\top)/2 -
  (\mathrm{tr}(M)/3)I`.

.. code-block:: text

   def matrix_from_0_2(out0: ℝ[num_points], out2: ℝ[num_points, 5]): ℝ[num_points, 3, 3]:
       results: ℝ[num_points, 3, 3] = zero_3d(num_points, 3, 3)
       sqrt3: ℝ = sqrt(3.0)
       for a:ℕ(num_points):
           d_xy = out2[a, 0]
           d_yz = out2[a, 1]
           d_z2 = out2[a, 2]
           d_zx = out2[a, 3]
           d_x2y2 = out2[a, 4]
           d_z2_scaled = d_z2 / sqrt3
           Mxx = 0.0 - d_z2_scaled + d_x2y2 + out0[a]
           Myy = 0.0 - d_z2_scaled - d_x2y2 + out0[a]
           Mzz = 2.0 * d_z2_scaled + out0[a]
           results[a, 0, 0] = Mxx
           results[a, 0, 1] = d_xy
           results[a, 0, 2] = d_zx
           results[a, 1, 0] = d_xy
           results[a, 1, 1] = Myy
           results[a, 1, 2] = d_yz
           results[a, 2, 0] = d_zx
           results[a, 2, 1] = d_yz
           results[a, 2, 2] = Mzz
       return results

Defining the Point-Mass Cloud Coordinates
--------------------------------------------

.. code-block:: text

   # sampled from a uniform distribution
   max_coord: ℝ = 0.5
   min_mass: ℝ = 0.5
   max_mass: ℝ = 2.0

   def random_points(n: ℝ, max_coord: ℝ): ℝ[m, 3]:
       pts: ℝ[n, 3] = for i : ℕ(n) → ε : ℝ[3] ~ 𝒰(-max_coord, max_coord, 3)
       return pts

   def random_masses(n: ℝ, min_mass: ℝ, max_mass: ℝ): ℝ[m]:
       masses: ℝ[n] ~ 𝒰(min_mass, max_mass, n)
       return masses


Forward Pass
------------

The forward pass defines how data flows through the complete model.

``points, masses -> difference_matrix -> distance_matrix / Y2 ->
gaussian_rbf -> radial_field (x2) -> filter_00 / filter_22 ->
matrix_from_0_2``

.. code-block:: text

   def λ(points: ℝ[num_points, 3], masses: ℝ[num_points], centers: ℝ[rbf_count], gamma: ℝ) → ℝ[num_points, 3, 3]:
       rij: ℝ[num_points, num_points, 3] = difference_matrix(points)
       dij: ℝ[num_points, num_points] = distance_matrix(rij)
       y2: ℝ[num_points, num_points, 5] = Y2(rij)
       rbf: ℝ[num_points, num_points, rbf_count] = gaussian_rbf(dij, centers, gamma)
       phi0: ℝ[num_points, num_points] = radial_field(rbf, this.w1_0, this.b1_0, this.w2_0, this.b2_0)
       phi2: ℝ[num_points, num_points] = radial_field(rbf, this.w1_2, this.b1_2, this.w2_2, this.b2_2)
       out0: ℝ[num_points] = filter_00(masses, phi0)
       out2: ℝ[num_points, 5] = filter_22(masses, phi2, y2)
       moi: ℝ[num_points, 3, 3] = matrix_from_0_2(out0, out2)
       return moi

Define Loss
-----------

For training the network we use the mean squared error (MSE) loss
function, comparing the model's predicted moment-of-inertia tensor
against the analytic ground truth at the center point.

.. math::

   \mathcal{L} = \frac{1}{9} \sum_{i=1}^{3}\sum_{j=1}^{3} \left(\text{pred}_{ij} - \text{target}_{ij}\right)^2

where:

- ``pred`` is the model's predicted 3x3 moment-of-inertia tensor
- ``target`` is the true tensor from ``moi_tensor`` (`Ground Truth`_)

.. code-block:: text

   def mse(pred: ℝ[3, 3], target: ℝ[3, 3]): ℝ:
       diff: ℝ[3,3] = pred - target
       result = sum(diff * diff) / 9.0
       return result

Model Definition
-----------------

.. code-block:: text

   class MOIModel(w1_0: ℝ[hidden, rbf_count], b1_0: ℝ[hidden, 1], w2_0: ℝ[1, hidden], b2_0: ℝ[1, 1],
                  w1_2: ℝ[hidden, rbf_count], b1_2: ℝ[hidden, 1], w2_2: ℝ[1, hidden], b2_2: ℝ[1, 1]):
       def λ(points: ℝ[num_points, 3], masses: ℝ[num_points], centers: ℝ[rbf_count], gamma: ℝ) → ℝ[num_points, 3, 3]:
           rij: ℝ[num_points, num_points, 3] = difference_matrix(points)
           dij: ℝ[num_points, num_points] = distance_matrix(rij)
           y2: ℝ[num_points, num_points, 5] = Y2(rij)
           rbf: ℝ[num_points, num_points, rbf_count] = gaussian_rbf(dij, centers, gamma)
           phi0: ℝ[num_points, num_points] = radial_field(rbf, this.w1_0, this.b1_0, this.w2_0, this.b2_0)
           phi2: ℝ[num_points, num_points] = radial_field(rbf, this.w1_2, this.b1_2, this.w2_2, this.b2_2)
           out0: ℝ[num_points] = filter_00(masses, phi0)
           out2: ℝ[num_points, 5] = filter_22(masses, phi2, y2)
           moi: ℝ[num_points, 3, 3] = matrix_from_0_2(out0, out2)
           return moi
       def loss_sample() → ℝ:
           points = random_points(num_points, max_coord)
           masses = random_masses(num_points, min_mass, max_mass)
           masses[center_idx] = 0.0
           target = moi_tensor(points, masses, center_idx)
           pred_full = this(points, masses, centers, gamma)
           pred = pred_full[center_idx]
           result = mse(pred, target)
           return result

Ground Truth
------------

The analytic physics: the classical definition of the
moment-of-inertia tensor of a point-mass system about a chosen
reference point. This is what ``MOIModel``'s output is trying to
approximate.

.. code-block:: text

   def moi_tensor(points: ℝ[num_points, 3], masses: ℝ[num_points], center_idx: ℝ): ℝ[3, 3]:
       cx = points[center_idx, 0]
       cy = points[center_idx, 1]
       cz = points[center_idx, 2]
       x: ℝ[num_points] = points[:, 0] - cx
       y: ℝ[num_points] = points[:, 1] - cy
       z: ℝ[num_points] = points[:, 2] - cz
       m: ℝ[num_points] = masses
       Ixx = sum((y*y + z*z) * m)
       Iyy = sum((x*x + z*z) * m)
       Izz = sum((x*x + y*y) * m)
       Ixy = sum((0.0 - x*y) * m)
       Iyz = sum((0.0 - y*z) * m)
       Ixz = sum((0.0 - x*z) * m)
       moi = zero_2d(3, 3)
       moi[0,0] = Ixx
       moi[1,1] = Iyy
       moi[2,2] = Izz
       moi[0,1] = Ixy
       moi[1,0] = Ixy
       moi[1,2] = Iyz
       moi[2,1] = Iyz
       moi[0,2] = Ixz
       moi[2,0] = Ixz
       return moi

.. math::

   I_{ab} = \sum_i m_i \left( |\mathbf{r}_i|^2 \delta_{ab} - r_{i,a}\, r_{i,b} \right)


Training the Model
--------------------

We train the network using stochastic gradient descent (SGD).

.. math::

   \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}

where:

- :math:`\theta` represents model parameters
- :math:`\eta` is the learning rate
- :math:`\nabla_\theta \mathcal{L}` is the gradient of the loss

.. code-block:: text

       def train(steps: ℕ, lr: ℝ) → ℝ:
           last_loss = 0
           for step:ℕ(steps):
               for rep:ℕ(1):
                   current_loss = this.loss_sample()
                   learnable_grads = grad(current_loss, this.learnable_params)
                   this.update(lr, learnable_grads)
                   last_loss = current_loss
           return last_loss

Evaluating the Model
----------------------

.. code-block:: text

       def evaluate() → ℝ:
           total_loss = 0
           for s:ℕ(eval_samples):
               for rep:ℕ(1):
                   current_loss = this.loss_sample()
                   total_loss = total_loss + current_loss
           result = total_loss / eval_samples
           return result

Equivariance Check
--------------------

.. code-block:: text

   def transpose3x3(M: ℝ[3, 3]): ℝ[3, 3]:
       T = zero_2d(3, 3)
       T[0,0] = M[0,0]
       T[0,1] = M[1,0]
       T[0,2] = M[2,0]
       T[1,0] = M[0,1]
       T[1,1] = M[1,1]
       T[1,2] = M[2,1]
       T[2,0] = M[0,2]
       T[2,1] = M[1,2]
       T[2,2] = M[2,2]
       return T

   def rotation_matrix_z(theta: ℝ): ℝ[3, 3]:
       c = cos(theta)
       s = sin(theta)
       Rmat = zero_2d(3, 3)
       Rmat[0,0] = c
       Rmat[0,1] = 0.0 - s
       Rmat[0,2] = 0.0
       Rmat[1,0] = s
       Rmat[1,1] = c
       Rmat[1,2] = 0.0
       Rmat[2,0] = 0.0
       Rmat[2,1] = 0.0
       Rmat[2,2] = 1.0
       return Rmat

   def rotate_points(points: ℝ[num_points, 3], Rmat: ℝ[3, 3]): ℝ[num_points, 3]:
       RmatT: ℝ[3, 3] = transpose3x3(Rmat)
       results: ℝ[num_points, 3] = points @ RmatT
       return results

   def rotate_matrix(Mmat: ℝ[3, 3], Rmat: ℝ[3, 3]): ℝ[3, 3]:
       RmatT: ℝ[3, 3] = transpose3x3(Rmat)
       RM: ℝ[3, 3] = Rmat @ Mmat
       result: ℝ[3, 3] = RM @ RmatT
       return result

   def max_abs_diff(Amat: ℝ[3, 3], Bmat: ℝ[3, 3]): ℝ:
       diff: ℝ[3,3] = Amat - Bmat
       sq: ℝ[3,3] = diff * diff
       worst = 0.0
       for i:ℕ(3):
           for j:ℕ(3):
               if sq[i,j] > worst:
                   worst = sq[i,j]
       result = sqrt(worst)
       return result

Full Code
----------

.. code-block:: text

  
   # Tensor Field Networks 

   # Setup 
   num_points: ℝ = 15

   rbf_low: ℝ = 0.0
   rbf_high: ℝ = 2.0
   rbf_count: ℝ = 30
   rbf_spacing: ℝ = (rbf_high - rbf_low) / rbf_count
   centers: ℝ[30] = for i : ℕ(rbf_count) -> rbf_low + i * rbf_spacing
   gamma: ℝ = 1.0 / rbf_spacing
   hidden: ℝ = 16

   center_idx: ℝ = 0.0
   rotation_angle: ℝ = 0.9
   w_init_low: ℝ = -0.2
   w_init_high: ℝ = 0.2

   #  Helper Functions 
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

   def get_3d_array_num_rows(x: ℝ[m, n, p]): ℝ:
       total: ℝ = 0
       temp: ℝ = 0
       for i:
           temp = x[i]
           total += 1
       return total

   def zero_1d(n: ℝ): ℝ[m]:
       results: ℝ[n] = for i:ℕ(n) -> i*0
       return results

   def zero_2d(rows: ℝ, cols: ℝ): ℝ[m, n]:
       results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
       return results

   def zero_3d(a: ℝ, b: ℝ, c: ℝ): ℝ[m, n, p]:
       results: ℝ[a, b, c] = for i:ℕ(a) -> for j:ℕ(b) -> for k:ℕ(c) -> k*0
       return results

   def difference_matrix(r: ℝ[num_points, 3]): ℝ[num_points, num_points, 3]:
       results: ℝ[num_points, num_points, 3] = zero_3d(num_points, num_points, 3)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               for k:ℕ(3):
                   results[i, j, k] = r[i, k] - r[j, k]
       return results

   def distance_matrix(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points]:
       results: ℝ[num_points, num_points] = zero_2d(num_points, num_points)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               acc = 0
               for k:ℕ(3):
                   acc += rij[i, j, k] * rij[i, j, k]
               results[i, j] = sqrt(acc + 1e-12)
       return results

   def ssp(x: ℝ): ℝ:
       return log(0.5 * exp(x) + 0.5)

   # Spherical Harmonics
   def Y0(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points, 1]:
       results: ℝ[num_points, num_points, 1] = zero_3d(num_points, num_points, 1)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               results[i, j, 0] = 1.0
       return results

   def Y1(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points, 3]:
       results: ℝ[num_points, num_points, 3] = zero_3d(num_points, num_points, 3)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               x = rij[i, j, 0]
               y = rij[i, j, 1]
               z = rij[i, j, 2]
               r_norm = sqrt(x*x + y*y + z*z + 1e-12)
               results[i, j, 0] = x / r_norm
               results[i, j, 1] = y / r_norm
               results[i, j, 2] = z / r_norm
       return results

   def Y2(rij: ℝ[num_points, num_points, 3]): ℝ[num_points, num_points, 5]:
       results: ℝ[num_points, num_points, 5] = zero_3d(num_points, num_points, 5)
       sqrt3: ℝ = sqrt(3.0)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               x = rij[i, j, 0]
               y = rij[i, j, 1]
               z = rij[i, j, 2]
               r2 = x*x + y*y + z*z + 1e-12
               results[i, j, 0] = x*y / r2
               results[i, j, 1] = y*z / r2
               results[i, j, 2] = (2.0*z*z - x*x - y*y) / (2.0 * sqrt3 * r2)
               results[i, j, 3] = z*x / r2
               results[i, j, 4] = (x*x - y*y) / (2.0 * r2)
       return results

   # Radial Network 
   def gaussian_rbf(dij: ℝ[num_points, num_points], centers: ℝ[rbf_count], gamma: ℝ): ℝ[num_points, num_points, rbf_count]:
       results: ℝ[num_points, num_points, rbf_count] = zero_3d(num_points, num_points, rbf_count)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               for c:ℕ(rbf_count):
                   diff = dij[i, j] - centers[c]
                   results[i, j, c] = exp(-gamma * diff * diff)
       return results

   def to_column(x: ℝ[n]): ℝ[n, 1]:
       n_x: ℝ = get_1d_array_length(x)
       results: ℝ[n_x, 1] = zero_2d(n_x, 1)
       for i:ℕ(n_x):
           results[i, 0] = x[i]
       return results

   def ssp_col(x: ℝ[m, 1]): ℝ[m, 1]:
       return log(0.5 * exp(x) + 0.5)

   def radial_net(rbf_features: ℝ[rbf_count], w1: ℝ[hidden, rbf_count], b1: ℝ[hidden, 1], w2: ℝ[1, hidden], b2: ℝ[1, 1]): ℝ:
       x_col: ℝ[rbf_count, 1] = to_column(rbf_features)
       h_pre: ℝ[hidden, 1] = w1 @ x_col + b1
       h: ℝ[hidden, 1] = ssp_col(h_pre)
       out: ℝ[1, 1] = w2 @ h + b2
       return out[0, 0]

   def radial_field(rbf: ℝ[num_points, num_points, rbf_count], w1: ℝ[hidden, rbf_count], b1: ℝ[hidden, 1], w2: ℝ[1, hidden], b2: ℝ[1, 1]): ℝ[num_points, num_points]:
       results: ℝ[num_points, num_points] = zero_2d(num_points, num_points)
       for i:ℕ(num_points):
           for j:ℕ(num_points):
               efeat = rbf[i, j]
               results[i, j] = radial_net(efeat, w1, b1, w2, b2)
       return results

   # Clebsch-Gordan lookup + tensor product reduction 
   def tensor_product_reduce(cg: ℝ[d1, d2, d3], a: ℝ[d1], b: ℝ[d2], dim1: ℝ, dim2: ℝ, dim3: ℝ): ℝ[d3]:
       results: ℝ[dim3] = zero_1d(dim3)
       for k:ℕ(dim3):
           acc = 0
           for i:ℕ(dim1):
               for j:ℕ(dim2):
                   acc += cg[i, j, k] * a[i] * b[j]
           results[k] = acc
       return results

   cg_000: ℝ[1, 1, 1] = [[[1.0]]]
   cg_202: ℝ[5, 1, 5] = [
       [[1.0, 0.0, 0.0, 0.0, 0.0]],
       [[0.0, 1.0, 0.0, 0.0, 0.0]],
       [[0.0, 0.0, 1.0, 0.0, 0.0]],
       [[0.0, 0.0, 0.0, 1.0, 0.0]],
       [[0.0, 0.0, 0.0, 0.0, 1.0]]
   ]

   def filter_00(masses: ℝ[num_points], phi0: ℝ[num_points, num_points]): ℝ[num_points]:
       results: ℝ[num_points] = zero_1d(num_points)
       for a:ℕ(num_points):
           acc = [0.0]
           for b:ℕ(num_points):
               edge0 = [phi0[a, b]]
               mass_b = [masses[b]]
               contrib = tensor_product_reduce(cg_000, edge0, mass_b, 1, 1, 1)
               acc = [acc[0] + contrib[0]]
           results[a] = acc[0]
       return results

   def filter_22(masses: ℝ[num_points], phi2: ℝ[num_points, num_points], y2: ℝ[num_points, num_points, 5]): ℝ[num_points, 5]:
       results: ℝ[num_points, 5] = zero_2d(num_points, 5)
       for a:ℕ(num_points):
           acc = zero_1d(5)
           for b:ℕ(num_points):
               edge2 = [phi2[a,b]*y2[a,b,0], phi2[a,b]*y2[a,b,1], phi2[a,b]*y2[a,b,2], phi2[a,b]*y2[a,b,3], phi2[a,b]*y2[a,b,4]]
               mass_b = [masses[b]]
               contrib = tensor_product_reduce(cg_202, edge2, mass_b, 5, 1, 5)
               for m:ℕ(5):
                   acc[m] = acc[m] + contrib[m]
           for m:ℕ(5):
               results[a, m] = acc[m]
       return results

   # Reassembling the Tensor 
   def matrix_from_0_2(out0: ℝ[num_points], out2: ℝ[num_points, 5]): ℝ[num_points, 3, 3]:
       results: ℝ[num_points, 3, 3] = zero_3d(num_points, 3, 3)
       sqrt3: ℝ = sqrt(3.0)
       for a:ℕ(num_points):
           d_xy = out2[a, 0]
           d_yz = out2[a, 1]
           d_z2 = out2[a, 2]
           d_zx = out2[a, 3]
           d_x2y2 = out2[a, 4]
           d_z2_scaled = d_z2 / sqrt3
           Mxx = 0.0 - d_z2_scaled + d_x2y2 + out0[a]
           Myy = 0.0 - d_z2_scaled - d_x2y2 + out0[a]
           Mzz = 2.0 * d_z2_scaled + out0[a]
           results[a, 0, 0] = Mxx
           results[a, 0, 1] = d_xy
           results[a, 0, 2] = d_zx
           results[a, 1, 0] = d_xy
           results[a, 1, 1] = Myy
           results[a, 1, 2] = d_yz
           results[a, 2, 0] = d_zx
           results[a, 2, 1] = d_yz
           results[a, 2, 2] = Mzz
       return results

   def mse(pred: ℝ[3, 3], target: ℝ[3, 3]): ℝ:
       diff: ℝ[3,3] = pred - target
       result = sum(diff * diff) / 9.0
       return result

   # Model Definition
   class MOIModel(w1_0: ℝ[hidden, rbf_count], b1_0: ℝ[hidden, 1], w2_0: ℝ[1, hidden], b2_0: ℝ[1, 1],
                  w1_2: ℝ[hidden, rbf_count], b1_2: ℝ[hidden, 1], w2_2: ℝ[1, hidden], b2_2: ℝ[1, 1]):
       def λ(points: ℝ[num_points, 3], masses: ℝ[num_points], centers: ℝ[rbf_count], gamma: ℝ) → ℝ[num_points, 3, 3]:
           rij: ℝ[num_points, num_points, 3] = difference_matrix(points)
           dij: ℝ[num_points, num_points] = distance_matrix(rij)
           y2: ℝ[num_points, num_points, 5] = Y2(rij)
           rbf: ℝ[num_points, num_points, rbf_count] = gaussian_rbf(dij, centers, gamma)
           phi0: ℝ[num_points, num_points] = radial_field(rbf, this.w1_0, this.b1_0, this.w2_0, this.b2_0)
           phi2: ℝ[num_points, num_points] = radial_field(rbf, this.w1_2, this.b1_2, this.w2_2, this.b2_2)
           out0: ℝ[num_points] = filter_00(masses, phi0)
           out2: ℝ[num_points, 5] = filter_22(masses, phi2, y2)
           moi: ℝ[num_points, 3, 3] = matrix_from_0_2(out0, out2)
           return moi
       def loss_sample() → ℝ:
           points = random_points(num_points, max_coord)
           masses = random_masses(num_points, min_mass, max_mass)
           masses[center_idx] = 0.0
           target = moi_tensor(points, masses, center_idx)
           pred_full = this(points, masses, centers, gamma)
           pred = pred_full[center_idx]
           result = mse(pred, target)
           return result
       def train(steps: ℕ, lr: ℝ) → ℝ:
           last_loss = 0
           for step:ℕ(steps):
               for rep:ℕ(1):
                   current_loss = this.loss_sample()
                   learnable_grads = grad(current_loss, this.learnable_params)
                   this.update(lr, learnable_grads)
                   last_loss = current_loss
           return last_loss
       def evaluate() → ℝ:
           total_loss = 0
           for s:ℕ(eval_samples):
               for rep:ℕ(1):
                   current_loss = this.loss_sample()
                   total_loss = total_loss + current_loss
           result = total_loss / eval_samples
           return result

   # Ground Truth 
   def moi_tensor(points: ℝ[num_points, 3], masses: ℝ[num_points], center_idx: ℝ): ℝ[3, 3]:
       cx = points[center_idx, 0]
       cy = points[center_idx, 1]
       cz = points[center_idx, 2]
       x: ℝ[num_points] = points[:, 0] - cx
       y: ℝ[num_points] = points[:, 1] - cy
       z: ℝ[num_points] = points[:, 2] - cz
       m: ℝ[num_points] = masses
       Ixx = sum((y*y + z*z) * m)
       Iyy = sum((x*x + z*z) * m)
       Izz = sum((x*x + y*y) * m)
       Ixy = sum((0.0 - x*y) * m)
       Iyz = sum((0.0 - y*z) * m)
       Ixz = sum((0.0 - x*z) * m)
       moi = zero_2d(3, 3)
       moi[0,0] = Ixx
       moi[1,1] = Iyy
       moi[2,2] = Izz
       moi[0,1] = Ixy
       moi[1,0] = Ixy
       moi[1,2] = Iyz
       moi[2,1] = Iyz
       moi[0,2] = Ixz
       moi[2,0] = Ixz
       return moi

   # Defining the point-mass cloud coordinates 
   # sampled from a uniform distribution
   max_coord: ℝ = 0.5
   min_mass: ℝ = 0.5
   max_mass: ℝ = 2.0

   def random_points(n: ℝ, max_coord: ℝ): ℝ[m, 3]:
       pts: ℝ[n, 3] = for i : ℕ(n) → ε : ℝ[3] ~ 𝒰(-max_coord, max_coord, 3)
       return pts

   def random_masses(n: ℝ, min_mass: ℝ, max_mass: ℝ): ℝ[m]:
       masses: ℝ[n] ~ 𝒰(min_mass, max_mass, n)
       return masses

   # Equivariance Check
   def transpose3x3(M: ℝ[3, 3]): ℝ[3, 3]:
       T = zero_2d(3, 3)
       T[0,0] = M[0,0]
       T[0,1] = M[1,0]
       T[0,2] = M[2,0]
       T[1,0] = M[0,1]
       T[1,1] = M[1,1]
       T[1,2] = M[2,1]
       T[2,0] = M[0,2]
       T[2,1] = M[1,2]
       T[2,2] = M[2,2]
       return T

   def rotation_matrix_z(theta: ℝ): ℝ[3, 3]:
       c = cos(theta)
       s = sin(theta)
       Rmat = zero_2d(3, 3)
       Rmat[0,0] = c
       Rmat[0,1] = 0.0 - s
       Rmat[0,2] = 0.0
       Rmat[1,0] = s
       Rmat[1,1] = c
       Rmat[1,2] = 0.0
       Rmat[2,0] = 0.0
       Rmat[2,1] = 0.0
       Rmat[2,2] = 1.0
       return Rmat

   def rotate_points(points: ℝ[num_points, 3], Rmat: ℝ[3, 3]): ℝ[num_points, 3]:
       RmatT: ℝ[3, 3] = transpose3x3(Rmat)
       results: ℝ[num_points, 3] = points @ RmatT
       return results

   def rotate_matrix(Mmat: ℝ[3, 3], Rmat: ℝ[3, 3]): ℝ[3, 3]:
       RmatT: ℝ[3, 3] = transpose3x3(Rmat)
       RM: ℝ[3, 3] = Rmat @ Mmat
       result: ℝ[3, 3] = RM @ RmatT
       return result

   def max_abs_diff(Amat: ℝ[3, 3], Bmat: ℝ[3, 3]): ℝ:
       diff: ℝ[3,3] = Amat - Bmat
       sq: ℝ[3,3] = diff * diff
       worst = 0.0
       for i:ℕ(3):
           for j:ℕ(3):
               if sq[i,j] > worst:
                   worst = sq[i,j]
       result = sqrt(worst)
       return result


   # Main Program:

   # Uniform distribution to initialize the weight tensors within [w_init_low, w_init_high]
   w1_0: ℝ[16, 30] ~ 𝒰(w_init_low, w_init_high, 16, 30)
   b1_0: ℝ[16, 1] = zero_2d(16, 1)
   w2_0: ℝ[1, 16] ~ 𝒰(w_init_low, w_init_high, 1, 16)
   b2_0: ℝ[1, 1] = zero_2d(1, 1)

   w1_2: ℝ[16, 30] ~ 𝒰(w_init_low, w_init_high, 16, 30)
   b1_2: ℝ[16, 1] = zero_2d(16, 1)
   w2_2: ℝ[1, 16] ~ 𝒰(w_init_low, w_init_high, 1, 16)
   b2_2: ℝ[1, 1] = zero_2d(1, 1)

   moi_object: MOIModel = MOIModel(w1_0, b1_0, w2_0, b2_0, w1_2, b1_2, w2_2, b2_2)

   eval_samples: ℕ = 30

   lr: ℝ = 0.002
   epochs: ℕ = 300

   loss_before: ℝ = moi_object.evaluate()
   print(loss_before)

   final_loss: ℝ = moi_object.train(epochs, lr)
   print(final_loss)

   avg_mse_loss: ℝ = moi_object.evaluate()
   print(avg_mse_loss)

   test_points: ℝ[num_points, 3] = random_points(num_points, max_coord)
   test_masses: ℝ[num_points] = random_masses(num_points, min_mass, max_mass)
   test_masses[center_idx] = 0.0

   Rmat: ℝ[3, 3] = rotation_matrix_z(rotation_angle)
   rotated_points: ℝ[num_points, 3] = rotate_points(test_points, Rmat)

   pred_orig_full = moi_object(test_points, test_masses, centers, gamma)
   pred_orig = pred_orig_full[center_idx]
   pred_rot_full = moi_object(rotated_points, test_masses, centers, gamma)
   pred_rot = pred_rot_full[center_idx]
   expected_rot = rotate_matrix(pred_orig, Rmat)

   diff = max_abs_diff(pred_rot, expected_rot)
   print(diff)

References
----------

.. [Thomas2018] Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L.,
   Kohlhoff, K., & Riley, P. (2018). Tensor Field Networks: Rotation-
   and Translation-Equivariant Neural Networks for 3D Point Clouds.
   arXiv:1802.08219. https://arxiv.org/abs/1802.08219

.. [Cheng2022] Cheng, C. (2022). A Less Mathematical Introduction to
   Tensor Field Networks. https://ccr-cheng.github.io/blog/2022/tfn/

.. [SmidtCode] tensorfield-torch: reference PyTorch implementation of
   Tensor Field Networks.
   https://github.com/semodi/tensorfield-torch

.. [MOINotebook] moment_of_inertia.ipynb, the notebook this
   tutorial's model and training loop are adapted from.
   https://github.com/semodi/tensorfield-torch/blob/master/moment_of_inertia.ipynb
