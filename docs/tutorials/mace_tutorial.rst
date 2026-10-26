MACE: Higher-Order Equivariant Message Passing
==============================================

In this tutorial we implement MACE (Message Passing Atomic Cluster Expansion) in Physika and use it to reproduce the energies of the MACE-MP-0 (medium) model [MACEMP0]_.

Introduction
------------

Message Passing Atomic Cluster Expansion (MACE) is a framework for learning interatomic potentials, extensively used in computational chemistry and materials science.
It combines the symmetry-adapted basis of the Atomic Cluster Expansion (ACE) with the equivariant message passing of graph neural networks.
This results in an interatomic potential whose energy is invariant to rotations, translations, and permutations, while being efficient and accurate [Batatia2022]_.

.. figure:: /_static/tutorial_files/mace_overview.png
   :alt: structure to atomic environments, learnable features and atomic energies
   :align: center
   :width: 700px

   Figure 1: Overview of MACE [MACETutorials]_.

Representing Spherical Tensors: Spherical Harmonics
---------------------------------------------------

Descriptors in MACE are represented as 'spherical tensors'.

To explain spherical tensors, we first introduce the spherical harmonics :math:`Y_{\ell m}`,
which are functions of a unit vector that points in a direction in 3D space.
They are labeled by the *angular order* :math:`\ell \geq 0`
and the *component* :math:`m` from :math:`-\ell` to :math:`\ell`, so every angular order has
:math:`2\ell + 1` components.

Spherical harmonics 'transform' in a special way when their argument is rotated. This means that if we take a vector :math:`\mathbf{r}` and rotate it by some matrix :math:`\mathbf{r} \to R\,\mathbf{r}`, then the spherical harmonic :math:`Y_{\ell m}(\mathbf{r})` will also
change in a predictable way.

.. math::

   Y_{\ell m}(R\,\mathbf{r}) = \sum_{m'=-\ell}^{\ell} D^{\ell}_{m m'}(R)\, Y_{\ell m'}(\mathbf{r})

:math:`D^{\ell}(R)` is called a Wigner D-matrix. You can find a more detailed explanation of spherical harmonics and Wigner D-matrices in the TFN tutorial [PhysikaTFN]_.

In MACE, we store the spherical harmonics in an array of shape ``[2l + 1]``.

For :math:`\ell = 0, 1, 2, 3` we compute the harmonics for all valid :math:`m` and store them together in one array, ordered by :math:`\ell` and then by :math:`m`:

.. math::

   \left[ Y_0^0,\;\; Y_1^{-1},\, Y_1^0,\, Y_1^1,\;\; Y_2^{-2},\, Y_2^{-1},\, Y_2^0,\, Y_2^1,\, Y_2^2,\;\; Y_3^{-3},\, Y_3^{-2},\, Y_3^{-1},\, Y_3^0,\, Y_3^1,\, Y_3^2,\, Y_3^3 \right]


Dataset
-------

We evaluate our model by comparing its total energy predictions to the released MACE-MP-0 model, on three small molecules: water, methane and formaldehyde.

.. code-block:: text

   water_positions: ℝ[3, 3] = [[0.0000, 0.0000, 0.0000], [1.3035, 0.1210, -0.0361], [0.0387, 1.0143, -0.0409]]
   water_numbers: ℝ[3] = [8.0, 1.0, 1.0]

   methane_positions: ℝ[5, 3] = [
       [0.000, 0.000, 0.000], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
       [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]
   ]
   methane_numbers: ℝ[5] = [6.0, 1.0, 1.0, 1.0, 1.0]

   formaldehyde_positions: ℝ[4, 3] = [[0.000, 0.000, 0.000], [0.000, 0.000, 1.205], [0.000, 0.940, -0.587], [0.000, -0.940, -0.587]]
   formaldehyde_numbers: ℝ[4] = [6.0, 8.0, 1.0, 1.0]

   reference_energies: ℝ[3] = [-12.655106, -23.912676, -21.853748]

Loading the Released Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We load the weights of the released model and compute the predicted energies of the three molecules using the 'load_mace_weights' function.

.. note::
   ``load_mace_weights`` is not a built-in Physika function. To use it,
   add the following helper to ``physika/runtime.py``:

   .. code-block:: python

      _MACE_URL = ("https://github.com/ACEsuit/mace-foundations/releases/download/"
                   "mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model")
      _MACE_WEIGHTS = []


      def load_mace_weights(index):
          if not _MACE_WEIGHTS:
              import mace
              model = torch.hub.load_state_dict_from_url(_MACE_URL)
              state = model.state_dict()

              def get(key, *shape):
                  return state[key].reshape(*shape)

              def radial(i, out):
                  key = f"interactions.{i}.conv_tp_weights.layer"
                  sizes = [(10, 64), (64, 64), (64, 64), (64, out)]
                  return [get(key + f"{j}.weight", *size)
                          for j, size in enumerate(sizes)]

              def product(i, k, n):
                  key = f"products.{i}.symmetric_contractions.contractions.{k}."
                  return [get(key + "weights.1", 89, 1, 128),
                          get(key + "weights.0", 89, n[0], 128),
                          get(key + "weights_max", 89, n[1], 128)]

              u = "products.0.symmetric_contractions.contractions."
              message1 = get("interactions.0.linear.weight", 4, 128, 128)
              message2 = get("interactions.1.linear.weight", 10, 128, 128)
              w3j = "interactions.1.conv_tp._compiled_main_left_right._w3j_"
              product_linear = get("products.0.linear.weight", 2, 128, 128)
              up2 = get("interactions.1.linear_up.weight", 2, 128, 128)
              weights = ([model.atomic_numbers,
                          get("node_embedding.linear.weight", 89, 128),
                          get("interactions.0.linear_up.weight", 128, 128)]
                         + radial(0, 512) + [message1[j].T for j in range(4)]
                         + [get(u + "0.U_matrix_1", 16, 1),
                            get(u + "0.U_matrix_2", 16, 16, 4),
                            get(u + "0.U_matrix_3", 16, 16, 16, 23),
                            get(u + "1.U_matrix_1", 3, 16, 1),
                            get(u + "1.U_matrix_2", 3, 16, 16, 6),
                            get(u + "1.U_matrix_3", 3, 16, 16, 16, 51)]
                         + product(0, 0, (4, 23)) + product(0, 1, (6, 51))
                         + [get("interactions.0.skip_tp.weight", 128, 89, 128),
                            product_linear[0], product_linear[1].T,
                            get("readouts.0.linear.weight", 128, 1),
                            up2[0], up2[1].T]
                         + radial(1, 1280)
                         + [state[w3j + name] for name in
                            ("1_2_1", "1_1_2", "1_3_2", "1_2_3")]
                         + [message2[j].T for j in range(10)]
                         + product(1, 0, (4, 23))
                         + [get("interactions.1.skip_tp.weight", 128, 89, 128),
                            get("products.1.linear.weight", 128, 128),
                            get("readouts.1.linear_1.weight", 128, 16),
                            get("readouts.1.linear_2.weight", 16, 1),
                            get("atomic_energies_fn.atomic_energies", 89, 1)])
              dtype = torch.get_default_dtype()
              _MACE_WEIGHTS.extend(w.to(dtype).contiguous() for w in weights)
          return _MACE_WEIGHTS[int(index)]

Scale and Shift
~~~~~~~~~~~~~~~

The energies are of order -10 eV, but the output of the network is of order 1. The released model therefore applies a scale and a shift to the network output, and adds the energy of every isolated atom, :math:`E_0(z_i)`:

.. math::

   E = \sum_i E_0(z_i) + s \sum_i \left( E^{(1)}_i + E^{(2)}_i \right) + \mu N

where :math:`E^{(1)}_i` and :math:`E^{(2)}_i` are the node energies of the two layers, :math:`N` is the number of atoms, and :math:`s` and :math:`\mu` are the scale and shift values of the released model.

.. code-block:: text

   scale: ℝ, shift: ℝ = 0.8041538754478097, 0.16409696359187365


MACE Feature Construction
-------------------------

We will now go through some key parts of the feature construction in MACE.

Step 0: The Molecule as a Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step in MACE is to represent the molecules as a graph. To do this we represent the molecule as a list of atoms and 'edges', where
an edge is simply the connection between two atoms. In MACE the cutoff radius determines which atoms are connected by an edge.
We tell MACE which atoms to work with through a ``z_table`` which contains the list of atomic numbers of the atoms in the environment.

We take the water molecule as our example molecule. We use the ``z_table`` to turn its atomic numbers into a one-hot encoding.

.. code-block:: text

   z_table: ℝ[89] = load_mace_weights(0)
   r_cut: ℝ = 6.0

   def one_hot(numbers: ℝ[n], table: ℝ[s]): ℝ[n, s]:
       num_atoms: ℕ, num_species: ℕ = len(numbers), len(table)
       encoding: ℝ[num_atoms, num_species] = zeros(num_atoms, num_species)
       for atom:ℕ(num_atoms):
           for species:ℕ(num_species):
               if numbers[atom] == table[species]:
                   encoding[atom, species] = 1.0
       return encoding

The edges come from a neighbor search using the cutoff radius ``r_cut``. They are all ordered pairs of distinct atoms that are closer than ``r_cut``.

.. code-block:: text

   def neighbor_search(positions: ℝ[n, 3], cutoff: ℝ): ℝ[2, m]:
       num_atoms: ℕ = len(positions)
       vector: ℝ[3], distance: ℝ = zeros(3), 0.0
       num_edges: ℕ = 0
       for sender:ℕ(num_atoms):
           for receiver:ℕ(num_atoms):
               vector = positions[receiver] - positions[sender]
               distance = sqrt(sum(vector * vector))
               if distance > 0.0:
                   if distance < cutoff:
                       num_edges += 1
       edge_index: ℝ[2, num_edges] = zeros(2, num_edges)
       edge: ℕ = 0
       for sender:ℕ(num_atoms):
           for receiver:ℕ(num_atoms):
               vector = positions[receiver] - positions[sender]
               distance = sqrt(sum(vector * vector))
               if distance > 0.0:
                   if distance < cutoff:
                       edge_index[:, edge] = [sender, receiver]
                       edge += 1
       return edge_index

Now we can build the three data tensors of our example molecule:

.. code-block:: text

   positions: ℝ[3, 3] = water_positions
   node_attrs: ℝ[3, 89] = one_hot(water_numbers, z_table)
   edge_index: ℝ[2, 6] = neighbor_search(positions, r_cut)

   print(positions)
   print(edge_index)

``positions`` represents the Cartesian coordinates in Å, one row per atom.

``node_attrs`` describes the species of every atom as a one-hot encoding of the elements, so it has one column for each of the 89 elements.

``edge_index`` represents the edges, which are stored as a list of 'senders' and 'receivers', representing the start and end points of each edge.

Step 1: Embeddings
~~~~~~~~~~~~~~~~~~

We now take this information about the molecule and turn it into a set of features that the model can work with. These are called embeddings.


The equation for the initial node features is given by:

.. math::

   h^{(0)}_{i,k} = \frac{1}{\sqrt{Z}} \sum_{z} W_{zk}\, \delta_{z z_i}

Each atom is given a vector of length :math:`K`, based on its atomic number.
In MACE, :math:`K` is the number of channels. This is the fundamental 'size' of the descriptor.

The atom's chemical species is embedded through a length-:math:`K` vector, and we also embed the lengths and directions of the edges.
The lengths of the edges are mapped through a set of 10 radial Bessel functions, and the directions of the edges are mapped through a set of 16 spherical harmonics.


The three embeddings are shown in Figure 2.

.. figure:: /_static/tutorial_files/mace_embedding.png
   :alt: The node embedding, the radial embedding and the angular embedding
   :align: center
   :width: 700px

   Figure 2: The creation of the node and edge embeddings [MACETutorials]_.


Node features
^^^^^^^^^^^^^

The length-:math:`K` vector of features of atom :math:`i` is computed from its one-hot encoding :math:`\delta_{z z_i}` and a learned matrix :math:`W`, where each atom's embedding is a lookup from the learned matrix.

.. code-block:: text

   num_elements: ℝ, num_channels: ℝ = 89, 128
   w_embed: ℝ[89, 128] = load_mace_weights(1)

   def node_embedding(node_attrs: ℝ[n, s], w: ℝ[s, k]): ℝ[n, k]:
       return (node_attrs @ w) / sqrt(num_elements)

Radial features
^^^^^^^^^^^^^^^

The radial embedding creates features based on the distance between atoms.
We first compute the vector that points from the sender to the receiver of every edge, and its length:

.. math::

   \mathbf{r}_{ji} = \mathbf{x}_i - \mathbf{x}_j, \qquad r_{ji} = \lVert \mathbf{r}_{ji} \rVert

Here atom :math:`j` is the sender and atom :math:`i` the receiver.

A single number is not much for a network to work with, so we expand the distance into a set of ten radial functions that disappear at the cutoff radius.

.. math::

   j_n(r) = \sqrt{\frac{2}{r_\mathrm{cut}}}\,\frac{\sin(n\pi r / r_\mathrm{cut})}{r}, \qquad n = 1, \dots, 10

These are Bessel functions (Figure 3). Each of them vanishes at the cutoff radius. Together they give the model a flexible description of how things depend on distance.

.. figure:: /_static/tutorial_files/mace_bessel_basis.png
   :alt: Eight out of the ten radial edge features as a function of the distance
   :align: center
   :width: 500px

   Figure 3: Eight out of the ten radial Bessel functions as a function of the distance [MACETutorials]_.

Each function is then multiplied by a smooth cutoff envelope, a polynomial that falls to zero at :math:`r_\mathrm{cut}`:

.. math::

   f(x) = 1 - \frac{(p+1)(p+2)}{2}\,x^{p} + p(p+2)\,x^{p+1} - \frac{p(p+1)}{2}\,x^{p+2}, \qquad x = \frac{r}{r_\mathrm{cut}}

Let's define the constants and the functions for the radial embedding. Every edge is shorter than the cutoff, so the envelope can be applied to all edges at once, and the lengths and the radial features are computed for all edges together.

.. code-block:: text

   π: ℝ, num_bessel: ℝ, p_env: ℝ = 3.141592653589793, 10, 5.0

.. code-block:: text

   def edge_vectors(positions: ℝ[n, 3], edge_index: ℝ[2, e]): ℝ[e, 3]:
       num_edges: ℕ = len(edge_index[0])
       vectors: ℝ[num_edges, 3] = zeros(num_edges, 3)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           vectors[edge] = positions[receiver] - positions[sender]
       return vectors

   def edge_lengths(vectors: ℝ[e, 3]): ℝ[e]:
       x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
       return sqrt(x * x + y * y + z * z)

   def radial_embedding(lengths: ℝ[e]): ℝ[e, b]:
       num_edges: ℕ = len(lengths)
       edge_feats: ℝ[num_edges, num_bessel] = zeros(num_edges, num_bessel)
       x = lengths / r_cut
       f_cut = 1.0 - (p_env + 1.0) * (p_env + 2.0) / 2.0 * x ** p_env + p_env * (p_env + 2.0) * x ** (p_env + 1.0) - p_env * (p_env + 1.0) / 2.0 * x ** (p_env + 2.0)
       for n:ℕ(num_bessel):
           edge_feats[:, n] = sqrt(2.0 / r_cut) * sin((n + 1.0) * π * lengths / r_cut) / lengths * f_cut
       return edge_feats

.. code-block:: text

   vectors: ℝ[6, 3] = edge_vectors(positions, edge_index)
   lengths: ℝ[6] = edge_lengths(vectors)
   edge_feats: ℝ[6, 10] = radial_embedding(lengths)

Angular features
^^^^^^^^^^^^^^^^

The angular embedding describes the direction of each edge. We use the spherical harmonics :math:`Y_{\ell m}` of the unit vector :math:`\hat{\mathbf r} = (x, y, z)` for :math:`\ell = 0, 1, 2, 3`.

.. math::

   Y_0 = 1, \qquad
   Y_1 = \sqrt{3}\,(x,\, y,\, z)

.. math::

   Y_2 = \left(\sqrt{15}\,xz,\;\; \sqrt{15}\,xy,\;\; \sqrt{5}\left(y^2 - \tfrac{1}{2}(x^2 + z^2)\right),\;\; \sqrt{15}\,yz,\;\; \tfrac{\sqrt{15}}{2}(z^2 - x^2)\right)

.. math::

   \begin{aligned}
   Y_3 = \Big(&\sqrt{\tfrac{35}{8}}\,(3xz^2 - x^3),\;\; \sqrt{105}\,xyz,\;\; \sqrt{\tfrac{21}{8}}\,x(4y^2 - x^2 - z^2),\;\; \tfrac{\sqrt{7}}{2}\,y(2y^2 - 3x^2 - 3z^2),\\
   &\sqrt{\tfrac{21}{8}}\,z(4y^2 - x^2 - z^2),\;\; \tfrac{\sqrt{105}}{2}\,y(z^2 - x^2),\;\; \sqrt{\tfrac{35}{8}}\,(z^3 - 3x^2 z)\Big)
   \end{aligned}

Here we use *component* normalization: the :math:`2\ell + 1` numbers of :math:`Y_\ell` have a squared norm of :math:`2\ell + 1`:

.. math::

   \lVert Y_\ell \rVert^2 = 2\ell + 1

.. code-block:: text

   def Y0(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 1]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 1] = zeros(num_edges, 1)
       results[:, 0] = 1.0
       return results

   def Y1(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 3]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 3] = zeros(num_edges, 3)
       x, y, z = vectors[:, 0] / lengths, vectors[:, 1] / lengths, vectors[:, 2] / lengths
       results[:, 0] = sqrt(3.0) * x
       results[:, 1] = sqrt(3.0) * y
       results[:, 2] = sqrt(3.0) * z
       return results

   def Y2(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 5]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 5] = zeros(num_edges, 5)
       x, y, z = vectors[:, 0] / lengths, vectors[:, 1] / lengths, vectors[:, 2] / lengths
       results[:, 0] = sqrt(15.0) * x * z
       results[:, 1] = sqrt(15.0) * x * y
       results[:, 2] = sqrt(5.0) * (y * y - 0.5 * (x * x + z * z))
       results[:, 3] = sqrt(15.0) * y * z
       results[:, 4] = sqrt(15.0) / 2.0 * (z * z - x * x)
       return results

   def Y3(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 7]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 7] = zeros(num_edges, 7)
       x, y, z = vectors[:, 0] / lengths, vectors[:, 1] / lengths, vectors[:, 2] / lengths
       results[:, 0] = sqrt(35.0 / 8.0) * (3.0 * x * z * z - x * x * x)
       results[:, 1] = sqrt(105.0) * x * y * z
       results[:, 2] = sqrt(21.0 / 8.0) * x * (4.0 * y * y - x * x - z * z)
       results[:, 3] = sqrt(7.0) / 2.0 * y * (2.0 * y * y - 3.0 * x * x - 3.0 * z * z)
       results[:, 4] = sqrt(21.0 / 8.0) * z * (4.0 * y * y - x * x - z * z)
       results[:, 5] = sqrt(105.0) / 2.0 * y * (z * z - x * x)
       results[:, 6] = sqrt(35.0 / 8.0) * (z * z * z - 3.0 * x * x * z)
       return results

.. code-block:: text

   def spherical_harmonics(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 16]:
       num_edges: ℕ = len(lengths)
       edge_attrs: ℝ[e, 16] = zeros(num_edges, 16)
       edge_attrs[:, 0:1] = Y0(vectors, lengths)
       edge_attrs[:, 1:4] = Y1(vectors, lengths)
       edge_attrs[:, 4:9] = Y2(vectors, lengths)
       edge_attrs[:, 9:16] = Y3(vectors, lengths)
       return edge_attrs

   edge_attrs: ℝ[6, 16] = spherical_harmonics(vectors, lengths)

   print(edge_feats)
   print(edge_attrs)

Step 2: The Interaction Block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is where atoms pool and exchange information. The output is a set of features that contain information about the atomic neighbors.
For every atom we build 'messages' that summarize its neighborhood. The interaction block has four parts:

1. Linear mixing of channels
2. Small network that turns the distances into weights
3. Tensor product that combines the weights, the mixed features, and the direction of each edge
4. Sum over the edges of each atom, followed by another linear mixing of channels

.. figure:: /_static/tutorial_files/mace_interaction.png
   :alt: The interaction block of the first layer
   :align: center
   :width: 700px

   Figure 4: The interaction block of the first layer. Figure from the MACE tutorial notebook [MACETutorials]_.

Mixing the channels
^^^^^^^^^^^^^^^^^^^

We start by mixing the :math:`K` channels of every atom with a learned matrix:

.. math::

   \tilde h_{i,k} = \frac{1}{\sqrt{K}} \sum_{k'} W_{k'k}\, h_{i,k'}


.. code-block:: text

   w_up: ℝ[128, 128] = load_mace_weights(2)

   def linear_up(node_feats: ℝ[n, k], w: ℝ[k, k]): ℝ[n, k]:
       return node_feats @ w / sqrt(num_channels)

Activation function
^^^^^^^^^^^^^^^^^^^

The network and the readout use the silu activation function.
We use silu (sigmoid linear unit), a smoother version of relu that can go negative, thus allowing the network to learn more complex functions.
It is defined as:

.. math::

   \mathrm{silu}(x) = \frac{x}{1 + e^{-x}}

e3nn rescales silu so that its second moment is one. This is defined by the constant ``silu_norm``.

.. code-block:: text

   silu_norm: ℝ = 1.679176792

   def silu(x: ℝ[e, h]): ℝ[e, h]:
       return silu_norm * x / (1.0 + exp(-x))


Radial network
^^^^^^^^^^^^^^

Next, a small network turns the ten radial features of an edge into the weights that act upon the messages sent along it:

.. math::

   R_{kl}(r_{ji}) = \mathrm{MLP}\big( \{ j_n(r_{ji}) \}_{n=1}^{10} \big)

.. code-block:: text

   radial_hidden: ℝ = 64

   w_r1: ℝ[10, 64] = load_mace_weights(3)
   w_r2: ℝ[64, 64] = load_mace_weights(4)
   w_r3: ℝ[64, 64] = load_mace_weights(5)
   w_r4: ℝ[64, 512] = load_mace_weights(6)

.. code-block:: text

   def radial_mlp(edge_feats: ℝ[e, b], w1: ℝ[b, h], w2: ℝ[h, h], w3: ℝ[h, h], w4: ℝ[h, o]): ℝ[e, o]:
       h1: ℝ[e, h] = silu(edge_feats @ w1 / sqrt(num_bessel))
       h2: ℝ[e, h] = silu(h1 @ w2 / sqrt(radial_hidden))
       h3: ℝ[e, h] = silu(h2 @ w3 / sqrt(radial_hidden))
       return h3 @ w4 / sqrt(radial_hidden)

The one-particle basis
^^^^^^^^^^^^^^^^^^^^^^

Now we combine everything along each edge.
For an edge from atom :math:`j` to atom :math:`i` and channel :math:`k`, the message has three components:

- the radial weight :math:`R_{kl}(r_{ji})`
- the mixed feature :math:`\tilde h_{j,k}`
- the direction of the edge :math:`Y_{l}(\hat{\mathbf r}_{ji})`

The tensor product of two spherical tensors of different angular orders is not a spherical tensor itself, so we decompose it into separate spherical
tensors of angular orders :math:`\ell_3 = |\ell_1 - \ell_2|, \dots, \ell_1 + \ell_2`.

The numbers involved in the decomposition are called
Clebsch-Gordan coefficients.
The naming convention for the coefficients is such that ``cg_l1l2l3`` couples a tensor of angular order ``l1`` with one of angular order ``l2`` into one of angular order ``l3``.
For better explanation see the TFN tutorial [PhysikaTFN]_ .

.. math::

   m_{ji,k,lm} = R_{kl}(r_{ji}) \,\big( Y_l(\hat{\mathbf r}_{ji}) \otimes \tilde h_{j,k} \big)_{lm}

.. math::

   \mathrm{out}_{m_3} = \sum_{m_1,\, m_2} C_{m_1 m_2 m_3}\; a_{m_1}\, b_{m_2}

In the first layer of MACE-MP-0 the node features are scalars (:math:`\ell = 0`). 

.. code-block:: text

   def conv_tp(node_feats_up: ℝ[n, k], edge_attrs: ℝ[e, 16], tp_weights: ℝ[e, t], edge_index: ℝ[2, e]): ℝ[e, k, 16]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 16] = zeros(num_edges, num_channels, 16)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           h = node_feats_up[sender]
           s0, s1, s2, s3 = h * tp_weights[edge, 0:128], h * tp_weights[edge, 128:256], h * tp_weights[edge, 256:384], h * tp_weights[edge, 384:512]
           mji[edge, :, 0] = s0 * edge_attrs[edge, 0]
           for m:ℕ(3):
               mji[edge, :, 1 + m] = s1 * edge_attrs[edge, 1 + m]
           for m:ℕ(5):
               mji[edge, :, 4 + m] = s2 * edge_attrs[edge, 4 + m]
           for m:ℕ(7):
               mji[edge, :, 9 + m] = s3 * edge_attrs[edge, 9 + m]
       return mji

In the first layer, the message is just a product of three numbers:

.. math::

   \phi_{ji,k,lm} = R_{kl}(r_{ji})\, Y_l^m(\hat{\mathbf r}_{ji})\, \tilde h_{j,k}


Message Passing
^^^^^^^^^^^^^^^

The messages at each atom are added up and passed through one more linear layer for each :math:`\ell`:

We use the average number of neighbors from the training set of MACE-MP-0, which is 61.964672446250916, to normalize the sum of the messages.

.. math::

   A_{i,k,lm} = \frac{1}{\bar n}\,\frac{1}{\sqrt{K}} \sum_{k'} W^{(l)}_{kk'} \sum_{j \in \mathcal{N}(i)} m_{ji,k',lm}


Step 3: The Product
~~~~~~~~~~~~~~~~~~~

The key distinction of MACE from TFNs is the efficient construction of higher-order features from the output of the interaction block. This is achieved by forming tensor products of the features and then
decomposing them into something that transforms in a known way (:math:`B_{i,\eta_\nu k LM}`). This creates features that contain angular information between the neighbors [TFN2018]_.

The message of atom :math:`i` can be written as a sum over its neighbors, taken one, two, up to :math:`\nu` at a time:

.. math::

   m_i^{(t)} = \sum_j u_1\!\left(\sigma_i^{(t)}; \sigma_j^{(t)}\right) + \sum_{j_1, j_2} u_2\!\left(\sigma_i^{(t)}; \sigma_{j_1}^{(t)}, \sigma_{j_2}^{(t)}\right) + \dots + \sum_{j_1, \dots, j_\nu} u_\nu\!\left(\sigma_i^{(t)}; \sigma_{j_1}^{(t)}, \dots, \sigma_{j_\nu}^{(t)}\right)

Here :math:`t` is the layer and :math:`\sigma_i^{(t)}` is the state of atom :math:`i`. The function :math:`u_\nu` couples :math:`\nu` neighbors at the same time. This is where the many-body information comes from.

.. figure:: /_static/tutorial_files/mace_product.png
   :alt: The product and the update of the node features
   :align: center
   :width: 700px

   Figure 5: The product and the update. The summary :math:`A` is multiplied with itself to give the B-features, which are weighted into :math:`m` and combined with the features of the atom to give the new node features [MACETutorials]_.

Symmetric Products
^^^^^^^^^^^^^^^^^^

The symmetric products are formed channel by channel and are combined through Clebsch-Gordan tables.

The B-features are:

.. math::

   B^{(t)}_{i,\eta_\nu k LM} = \sum_{\mathbf{lm}} \mathcal{C}^{LM}_{\eta_\nu, \mathbf{lm}} \prod_{\xi=1}^{\nu} A^{(t)}_{i,k l_\xi m_\xi}, \qquad \mathbf{lm} = (l_1 m_1, \dots, l_\nu m_\nu)

The index :math:`\eta_\nu` labels the different ways of coupling the :math:`\nu` factors as 'paths'.

A 'path' is one particular way of combining the pieces of :math:`A` (the summary of the atom's neighborhood)
into a result of a certain angular order :math:`L`.
For example, a scalar (:math:`L = 0`) can be made from the scalar :math:`A_0` on its own, from its square :math:`A_0 \otimes A_0`, or from the dot product of two vectors, :math:`A_1 \cdot A_1`, :math:`A_2 \cdot A_2` or :math:`A_3 \cdot A_3`. A vector (:math:`L = 1`) can be made, for example, from :math:`A_1` on its own or from the product of a scalar and a vector, :math:`A_0 \otimes A_1`.

======================  ==================  ==================  ==================
Output                  :math:`\nu = 1`      :math:`\nu = 2`      :math:`\nu = 3`
======================  ==================  ==================  ==================
:math:`L = 0` (scalar)  1 path              4 paths             23 paths
:math:`L = 1` (vector)  1 path              6 paths             51 paths
======================  ==================  ==================  ==================

The Clebsch-Gordan tables, generated with the e3nn library [e3nn]_, are stored as weights.
.. code-block:: text

   U1: ℝ[16, 1] = load_mace_weights(11)
   U2: ℝ[16, 16, 4] = load_mace_weights(12)
   U3: ℝ[16, 16, 16, 23] = load_mace_weights(13)
   U1v: ℝ[3, 16, 1] = load_mace_weights(14)
   U2v: ℝ[3, 16, 16, 6] = load_mace_weights(15)
   U3v: ℝ[3, 16, 16, 16, 51] = load_mace_weights(16)

We define one function for each :math:`\nu`. 

.. code-block:: text

   def B_features_1(A: ℝ[n, k, 16], U: ℝ[16, a]): ℝ[n, k, a]:
       num_atoms: ℕ = len(A)
       num_paths: ℕ = len(U[0])
       B1: ℝ[num_atoms, num_channels, num_paths] = zeros(num_atoms, num_channels, num_paths)
       for atom:ℕ(num_atoms):
           B1[atom] = A[atom] @ U
       return B1

   def B_features_2(A: ℝ[n, k, 16], U: ℝ[16, 16, b]): ℝ[n, k, b]:
       num_atoms: ℕ = len(A)
       num_paths: ℕ = len(U[0, 0])
       B2: ℝ[num_atoms, num_channels, num_paths] = zeros(num_atoms, num_channels, num_paths)
       for atom:ℕ(num_atoms):
           for i:ℕ(16):
               M = A[atom] @ U[i]
               for p:ℕ(num_paths):
                   B2[atom, :, p] = B2[atom, :, p] + A[atom, :, i] * M[:, p]
       return B2

   def B_features_3(A: ℝ[n, k, 16], U: ℝ[16, 16, 16, c]): ℝ[n, k, c]:
       num_atoms: ℕ = len(A)
       num_paths: ℕ = len(U[0, 0, 0])
       B3: ℝ[num_atoms, num_channels, num_paths] = zeros(num_atoms, num_channels, num_paths)
       for atom:ℕ(num_atoms):
           for i:ℕ(16):
               for j:ℕ(16):
                   M = A[atom] @ U[i, j]
                   for p:ℕ(num_paths):
                       B3[atom, :, p] = B3[atom, :, p] + A[atom, :, i] * A[atom, :, j] * M[:, p]
       return B3

Weighting the paths
^^^^^^^^^^^^^^^^^^^

The B-features are combined with learned weights, one weight per path and channel.

.. math::

   m_{i,k,LM} = \sum_{\text{paths}} W^{L}_{z_i,\,\text{path},\,k}\; B_{i,k,\text{path},LM}

The sum over paths is written as a sum over :math:`\nu` and a sum over :math:`\eta_\nu`:

.. math::

   m^{(t)}_{i,kLM} = \sum_{\nu} \sum_{\eta_\nu} W^{(t)}_{z_i kL,\, \eta_\nu}\, B^{(t)}_{i,\eta_\nu kLM}

The weights are selected per element: the one-hot ``node_attrs`` of atom :math:`i` picks the row of the weight tensor that belongs to its element :math:`z_i`.

.. code-block:: text

   w_prod1: ℝ[89, 1, 128] = load_mace_weights(17)
   w_prod2: ℝ[89, 4, 128] = load_mace_weights(18)
   w_prod3: ℝ[89, 23, 128] = load_mace_weights(19)
   w_prod1v: ℝ[89, 1, 128] = load_mace_weights(20)
   w_prod2v: ℝ[89, 6, 128] = load_mace_weights(21)
   w_prod3v: ℝ[89, 51, 128] = load_mace_weights(22)

.. code-block:: text

   def weighted_sum(B1: ℝ[n, k, a], B2: ℝ[n, k, b], B3: ℝ[n, k, c], node_attrs: ℝ[n, s], wp1: ℝ[s, a, k], wp2: ℝ[s, b, k], wp3: ℝ[s, c, k]): ℝ[n, k]:
       num_atoms: ℕ = len(B1)
       m: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for p:ℕ(len(B1[0, 0])):
           m = m + (node_attrs @ wp1[:, p, :]) * B1[:, :, p]
       for p:ℕ(len(B2[0, 0])):
           m = m + (node_attrs @ wp2[:, p, :]) * B2[:, :, p]
       for p:ℕ(len(B3[0, 0])):
           m = m + (node_attrs @ wp3[:, p, :]) * B3[:, :, p]
       return m

.. code-block:: text

   def scalar_product(A: ℝ[n, k, 16], node_attrs: ℝ[n, s]): ℝ[n, k]:
       return weighted_sum(B_features_1(A, U1), B_features_2(A, U2), B_features_3(A, U3), node_attrs, w_prod1, w_prod2, w_prod3)

   def vector_product(A: ℝ[n, k, 16], node_attrs: ℝ[n, s]): ℝ[n, k, 3]:
       num_atoms: ℕ = len(A)
       out: ℝ[num_atoms, num_channels, 3] = zeros(num_atoms, num_channels, 3)
       for m:ℕ(3):
           out[:, :, m] = weighted_sum(B_features_1(A, U1v[m]), B_features_2(A, U2v[m]), B_features_3(A, U3v[m]), node_attrs, w_prod1v, w_prod2v, w_prod3v)
       return out

Updating the node features
^^^^^^^^^^^^^^^^^^^^^^^^^^

The new node features are a linear function of :math:`m` with an additional skip connection that helps in carrying the initial features of the node forward:

.. math::

   h^{(t+1)}_{i,kLM} = \sum_{\tilde k} W^{(t)}_{kL,\tilde k}\, m^{(t)}_{i,\tilde k LM} + \sum_{\tilde k} W^{(t)}_{k z_i L,\tilde k}\, h^{(t)}_{i,\tilde k LM}

In our first layer (:math:`t = 0`) the initial node features are scalars, so the skip connection only exists for :math:`L = 0`:

.. math::

   h^{(1)}_{i,k,LM} = \frac{1}{\sqrt{K}} \sum_{k'} W^{L}_{kk'}\, m_{i,k',LM} \;+\; \delta_{L0}\, s_{i,k}

The skip connection term is:

.. math::

   s_{i,k} = \frac{1}{\sqrt{K Z}} \sum_{k',\,z} W_{k'zk}\; h^{(0)}_{i,k'}\; \delta_{z z_i}

It uses weights that depend on the atomic species and is built from the scalar starting features, so it only adds to the :math:`L = 0` part.
The :math:`L = 1` part gets only the linear mixing of :math:`m`, with its own matrix ``w_pv``.

.. code-block:: text

   w_sc: ℝ[128, 89, 128] = load_mace_weights(23)
   w_p: ℝ[128, 128] = load_mace_weights(24)
   w_pv: ℝ[128, 128] = load_mace_weights(25)

.. code-block:: text

   def skip_tp(node_feats: ℝ[n, k], node_attrs: ℝ[n, s], w: ℝ[k, s, k]): ℝ[n, k]:
       num_atoms: ℕ = len(node_feats)
       sc: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for atom:ℕ(num_atoms):
           for z:ℕ(num_elements):
               if node_attrs[atom, z] == 1.0:
                   sc[atom] = node_feats[atom] @ w[:, z, :] / sqrt(num_channels * num_elements)
       return sc

   def node_update(m: ℝ[n, k], sc: ℝ[n, k], wp: ℝ[k, k]): ℝ[n, k]:
       return m @ wp / sqrt(num_channels) + sc

   def node_update_vectors(mv: ℝ[n, k, 3], wp: ℝ[k, k]): ℝ[n, k, 3]:
       num_atoms: ℕ = len(mv)
       out: ℝ[num_atoms, num_channels, 3] = zeros(num_atoms, num_channels, 3)
       for atom:ℕ(num_atoms):
           out[atom] = wp @ mv[atom] / sqrt(num_channels)
       return out

The outputn scalars are used by the readout layer, and both are the input of the second layer.

Step 4: The Readout
~~~~~~~~~~~~~~~~~~~

Now we can take the newly computed node features and compute the actual energy by passing them through a readout layer.
The readout is a single linear layer on the scalar part of the features:

.. math::

   E_i = \frac{1}{\sqrt{K}} \sum_{k} W_k\, h^{(1)}_{i,k,\ell = 0}, \qquad E = \sum_i E_i

The total energy is simply the sum of the individual atomic contributions.

.. figure:: /_static/tutorial_files/mace_readout.png
   :alt: The readout maps the node features to node energies
   :align: center
   :width: 600px

   Figure 6: The readout layer [MACETutorials]_.


.. code-block:: text

   w_readout: ℝ[128, 1] = load_mace_weights(26)

   def readout(node_feats: ℝ[n, k], w: ℝ[k, 1]): ℝ[n, 1]:
       return node_feats @ w / sqrt(num_channels)

Step 5: The Second Layer
~~~~~~~~~~~~~~~~~~~~~~~~

The second layer repeats Steps 2 to 4 while taking the output of the first layer as its input.
The second layer makes MACE more expressive because:

- Each node now receives messages from neighbors which already summarize their own neighborhood.
- The neighbors now also send directional information (the :math:`\ell = 1` features) to each other.

.. figure:: /_static/tutorial_files/mace_interaction_second_layer.png
   :alt: The interaction block of a general layer s
   :align: center
   :width: 700px

   Figure 7: The interaction block of a general layer :math:`s`, where the input has :math:`l_0`, :math:`l_1` and :math:`l_2` features [MACETutorials]_. In the released medium model the input of the second layer has :math:`l_0` and :math:`l_1` features.

Mixing the inputs
^^^^^^^^^^^^^^^^^

The input now has two parts, scalars and vectors, and each part is mixed with its own matrix. The mixing of the vectors acts on the channels and leaves the three components alone.

.. code-block:: text

   w2_up: ℝ[128, 128] = load_mace_weights(27)
   w2_up_v: ℝ[128, 128] = load_mace_weights(28)

.. code-block:: text

   def linear_up_vectors(node_vecs: ℝ[n, k, 3], w: ℝ[k, k]): ℝ[n, k, 3]:
       num_atoms: ℕ = len(node_vecs)
       out: ℝ[num_atoms, num_channels, 3] = zeros(num_atoms, num_channels, 3)
       for atom:ℕ(num_atoms):
           out[atom] = w @ node_vecs[atom] / sqrt(num_channels)
       return out

The radial network of the second layer is the same function as before, but it outputs 1280 numbers per edge: one weight for each of the 128 channels and each of the ten paths of the tensor product below.

.. code-block:: text

   w2_r1: ℝ[10, 64] = load_mace_weights(29)
   w2_r2: ℝ[64, 64] = load_mace_weights(30)
   w2_r3: ℝ[64, 64] = load_mace_weights(31)
   w2_r4: ℝ[64, 1280] = load_mace_weights(32)

The tensor product with Clebsch-Gordan coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the first layer the node feature was a scalar, so the coupling with the edge direction was trivial. Now the neighbor :math:`j` sends a scalar :math:`s_{j,k}` and a vector :math:`\mathbf{v}_{j,k}`, and we couple each of them with the spherical harmonics :math:`Y_{\ell_2}` of the edge, for :math:`\ell_2 = 0, 1, 2, 3`.
Each allowed way of coupling is one path with its own Clebsch-Gordan coefficient and its own radial weight. Using the naming convention ``cg_l1l2l3`` from Step 2, the ten paths are:

.. list-table::
   :header-rows: 1
   :widths: 8 30 32 30

   * - Path
     - Coupling
     - Constant in the code
     - Output columns of ``mji``
   * - 0
     - :math:`0 \otimes 0 \to 0`
     - 1
     - 0
   * - 1
     - :math:`1 \otimes 1 \to 0`
     - :math:`1/\sqrt{3}`
     - 1
   * - 2
     - :math:`0 \otimes 1 \to 1`
     - 1
     - 2 to 4
   * - 3
     - :math:`1 \otimes 0 \to 1`
     - 1
     - 5 to 7
   * - 4
     - :math:`1 \otimes 2 \to 1`
     - :math:`\sqrt{3}` times ``cg_121``
     - 8 to 10
   * - 5
     - :math:`0 \otimes 2 \to 2`
     - 1
     - 11 to 15
   * - 6
     - :math:`1 \otimes 1 \to 2`
     - :math:`\sqrt{5}` times ``cg_112``
     - 16 to 20
   * - 7
     - :math:`1 \otimes 3 \to 2`
     - :math:`\sqrt{5}` times ``cg_132``
     - 21 to 25
   * - 8
     - :math:`0 \otimes 3 \to 3`
     - 1
     - 26 to 32
   * - 9
     - :math:`1 \otimes 2 \to 3`
     - :math:`\sqrt{7}` times ``cg_123``
     - 33 to 39

In the coupling column, the first factor is the scalar :math:`s_{j,k}` (:math:`0`) or the vector :math:`\mathbf{v}_{j,k}` (:math:`1`), and the second factor is always a spherical harmonic of the edge. 

   m_{ji,k,\ell_3 m_3} = \sum_{\text{paths}} R_{k,\text{path}}(r_{ji}) \, c_{\text{path}} \sum_{m_1, m_2} C^{\ell_3 m_3}_{\ell_1 m_1, \ell_2 m_2}\; h_{j,k,\ell_1 m_1}\, Y_{\ell_2 m_2}(\hat{\mathbf r}_{ji})

.. code-block:: text

   cg_121: ℝ[3, 5, 3] = load_mace_weights(33)
   cg_112: ℝ[3, 3, 5] = load_mace_weights(34)
   cg_132: ℝ[3, 7, 5] = load_mace_weights(35)
   cg_123: ℝ[3, 5, 7] = load_mace_weights(36)

The messages of the second layer have 40 columns, so the the tensor product itself is written out path by path:

.. code-block:: text

   def neighbor_sum_second(mji: ℝ[e, k, 40], edge_index: ℝ[2, e], num_atoms: ℕ): ℝ[num_atoms, k, 40]:
       num_edges: ℕ = len(mji)
       message: ℝ[num_atoms, num_channels, 40] = zeros(num_atoms, num_channels, 40)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           message[receiver] = message[receiver] + mji[edge]
       return message

.. code-block:: text

   def conv_tp_second(up: ℝ[n, k], up_vecs: ℝ[n, k, 3], edge_attrs: ℝ[e, 16], tp_weights: ℝ[e, t], edge_index: ℝ[2, e]): ℝ[e, k, 40]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 40] = zeros(num_edges, num_channels, 40)
       y1: ℝ[3, 1] = zeros(3, 1)
       y2: ℝ[5, 1] = zeros(5, 1)
       y3: ℝ[7, 1] = zeros(7, 1)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           s, v = up[sender], up_vecs[sender]
           y1[:, 0] = edge_attrs[edge, 1:4]
           y2[:, 0] = edge_attrs[edge, 4:9]
           y3[:, 0] = edge_attrs[edge, 9:16]
           d = v @ y1
           mji[edge, :, 0] = tp_weights[edge, 0:128] * s * edge_attrs[edge, 0]
           mji[edge, :, 1] = tp_weights[edge, 128:256] * d[:, 0] / sqrt(3.0)
           for m:ℕ(3):
               a = v @ (cg_121[:, :, m] @ y2)
               mji[edge, :, 2 + m] = tp_weights[edge, 256:384] * s * y1[m, 0]
               mji[edge, :, 5 + m] = tp_weights[edge, 384:512] * v[:, m] * edge_attrs[edge, 0]
               mji[edge, :, 8 + m] = tp_weights[edge, 512:640] * sqrt(3.0) * a[:, 0]
           for m:ℕ(5):
               b = v @ (cg_112[:, :, m] @ y1)
               c = v @ (cg_132[:, :, m] @ y3)
               mji[edge, :, 11 + m] = tp_weights[edge, 640:768] * s * y2[m, 0]
               mji[edge, :, 16 + m] = tp_weights[edge, 768:896] * sqrt(5.0) * b[:, 0]
               mji[edge, :, 21 + m] = tp_weights[edge, 896:1024] * sqrt(5.0) * c[:, 0]
           for m:ℕ(7):
               d3 = v @ (cg_123[:, :, m] @ y2)
               mji[edge, :, 26 + m] = tp_weights[edge, 1024:1152] * s * y3[m, 0]
               mji[edge, :, 33 + m] = tp_weights[edge, 1152:1280] * sqrt(7.0) * d3[:, 0]
       return mji

Then the columns are mixed by learned matrices, one for each path, and added into the 16 columns of the output of the layer.

.. code-block:: text

   w2_0: ℝ[128, 128] = load_mace_weights(37)
   w2_1: ℝ[128, 128] = load_mace_weights(38)
   w2_2: ℝ[128, 128] = load_mace_weights(39)
   w2_3: ℝ[128, 128] = load_mace_weights(40)
   w2_4: ℝ[128, 128] = load_mace_weights(41)
   w2_5: ℝ[128, 128] = load_mace_weights(42)
   w2_6: ℝ[128, 128] = load_mace_weights(43)
   w2_7: ℝ[128, 128] = load_mace_weights(44)
   w2_8: ℝ[128, 128] = load_mace_weights(45)
   w2_9: ℝ[128, 128] = load_mace_weights(46)

.. code-block:: text

   def linear_second(message: ℝ[n, k, 40], w0: ℝ[k, k], w1: ℝ[k, k], w2: ℝ[k, k], w3: ℝ[k, k], w4: ℝ[k, k], w5: ℝ[k, k], w6: ℝ[k, k], w7: ℝ[k, k], w8: ℝ[k, k], w9: ℝ[k, k]): ℝ[n, k, 16]:
       num_atoms: ℕ = len(message)
       out: ℝ[num_atoms, num_channels, 16] = zeros(num_atoms, num_channels, 16)
       out[:, :, 0:1] = (w0 @ message[:, :, 0:1] + w1 @ message[:, :, 1:2]) / sqrt(256.0)
       out[:, :, 1:4] = (w2 @ message[:, :, 2:5] + w3 @ message[:, :, 5:8] + w4 @ message[:, :, 8:11]) / sqrt(384.0)
       out[:, :, 4:9] = (w5 @ message[:, :, 11:16] + w6 @ message[:, :, 16:21] + w7 @ message[:, :, 21:26]) / sqrt(384.0)
       out[:, :, 9:16] = (w8 @ message[:, :, 26:33] + w9 @ message[:, :, 33:40]) / sqrt(256.0)
       return out / avg_num_neighbors

Product, update and readout
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The product of the second layer only produces scalars, The skip connection acts only on the scalar part of the first layer's output.

The readout of the last layer uses a small network instead of a single linear layer. It uses a hidden layer of 16 neurons and the silu activation function to output a single number per atom:

.. math::

   E^{(2)}_i = \frac{1}{\sqrt{16}} \sum_{u=1}^{16} V_u\; \mathrm{silu}\!\left( \frac{1}{\sqrt{K}} \sum_{k} U_{ku}\, h^{(2)}_{i,k} \right)

The model's final energy combines the contributions from both layers with the scale and shift and the isolated atom energies.

.. code-block:: text

   w2_prod1: ℝ[89, 1, 128] = load_mace_weights(47)
   w2_prod2: ℝ[89, 4, 128] = load_mace_weights(48)
   w2_prod3: ℝ[89, 23, 128] = load_mace_weights(49)
   w2_sc: ℝ[128, 89, 128] = load_mace_weights(50)
   w2_p: ℝ[128, 128] = load_mace_weights(51)

   # Readout and total energy
   w2_ro1: ℝ[128, 16] = load_mace_weights(52)
   w2_ro2: ℝ[16, 1] = load_mace_weights(53)
   atomic_energies: ℝ[89, 1] = load_mace_weights(54)

   def readout2(node_feats: ℝ[n, k], w1: ℝ[k, 16], w2: ℝ[16, 1]): ℝ[n, 1]:
       hidden: ℝ[n, 16] = silu(node_feats @ w1 / sqrt(num_channels))
       return hidden @ w2 / sqrt(16.0)

Forward Pass
------------

The forward pass defines how data flows through the complete model.

``positions, numbers -> node_attrs, edge_index, edge_feats, edge_attrs -> node_embedding -> first layer (scalars and vectors) -> readout -> second layer (scalars and vectors in, scalars out) -> readout -> energy``

.. code-block:: text

   def mace_energy(positions: ℝ[n, 3], numbers: ℝ[n]): ℝ:
       num_atoms: ℕ = len(positions)
       node_attrs: ℝ[num_atoms, 89] = one_hot(numbers, z_table)
       edge_index = neighbor_search(positions, r_cut)
       vectors = edge_vectors(positions, edge_index)
       lengths = edge_lengths(vectors)
       edge_feats, edge_attrs = radial_embedding(lengths), spherical_harmonics(vectors, lengths)
       node_feats0 = node_embedding(node_attrs, w_embed)
       node_feats_up = linear_up(node_feats0, w_up)
       tp_weights = radial_mlp(edge_feats, w_r1, w_r2, w_r3, w_r4)
       mji = conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
       message = linear(neighbor_sum(mji, edge_index, num_atoms), w_0, w_1, w_2, w_3)
       sc = skip_tp(node_feats0, node_attrs, w_sc)
       node_feats1 = node_update(scalar_product(message, node_attrs), sc, w_p)
       node_vecs1 = node_update_vectors(vector_product(message, node_attrs), w_pv)
       node_energies = readout(node_feats1, w_readout)
       node_feats_up2 = linear_up(node_feats1, w2_up)
       node_vecs_up2 = linear_up_vectors(node_vecs1, w2_up_v)
       tp_weights2 = radial_mlp(edge_feats, w2_r1, w2_r2, w2_r3, w2_r4)
       mji2 = conv_tp_second(node_feats_up2, node_vecs_up2, edge_attrs, tp_weights2, edge_index)
       message2 = linear_second(neighbor_sum_second(mji2, edge_index, num_atoms), w2_0, w2_1, w2_2, w2_3, w2_4, w2_5, w2_6, w2_7, w2_8, w2_9)
       sc2 = skip_tp(node_feats1, node_attrs, w2_sc)
       node_feats2 = node_update(weighted_sum(B_features_1(message2, U1), B_features_2(message2, U2), B_features_3(message2, U3), node_attrs, w2_prod1, w2_prod2, w2_prod3), sc2, w2_p)
       node_energies2 = readout2(node_feats2, w2_ro1, w2_ro2)
       atom_energies = node_attrs @ atomic_energies
       energy: ℝ = sum(atom_energies) + scale * (sum(node_energies) + sum(node_energies2)) + shift * num_atoms
       return energy

Comparing with MACE-MP-0
------------------------

We now run the model on the three molecules and compare with the energies that ``mace-torch`` gives for the released medium model.

.. code-block:: text

   water_energy: ℝ = mace_energy(water_positions, water_numbers)
   methane_energy: ℝ = mace_energy(methane_positions, methane_numbers)
   formaldehyde_energy: ℝ = mace_energy(formaldehyde_positions, formaldehyde_numbers)

   predictions: ℝ[3] = [water_energy, methane_energy, formaldehyde_energy]
   print(predictions)
   print(reference_energies)
   print(predictions - reference_energies)

Output:

.. code-block:: text

   [-12.655106544494629, -23.912677764892578, -21.853748321533203] ∈ ℝ[3]
   [-12.655105590820312, -23.912675857543945, -21.853748321533203] ∈ ℝ[3]
   [-9.5367431640625e-07, -1.9073486328125e-06, 0.0] ∈ ℝ[3]

The energies match to about :math:`2 \times 10^{-6}` eV.

Full Code
---------

.. code-block:: text

   # MACE: Higher-Order Equivariant Message Passing

   # Dataset
   water_positions: ℝ[3, 3] = [[0.0000, 0.0000, 0.0000], [1.3035, 0.1210, -0.0361], [0.0387, 1.0143, -0.0409]]
   water_numbers: ℝ[3] = [8.0, 1.0, 1.0]

   methane_positions: ℝ[5, 3] = [
       [0.000, 0.000, 0.000], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
       [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]
   ]
   methane_numbers: ℝ[5] = [6.0, 1.0, 1.0, 1.0, 1.0]

   formaldehyde_positions: ℝ[4, 3] = [[0.000, 0.000, 0.000], [0.000, 0.000, 1.205], [0.000, 0.940, -0.587], [0.000, -0.940, -0.587]]
   formaldehyde_numbers: ℝ[4] = [6.0, 8.0, 1.0, 1.0]

   # Total energies (eV) of the three molecules computed with the released model and mace-torch
   reference_energies: ℝ[3] = [-12.655106, -23.912676, -21.853748]

   # Scale and Shift
   scale: ℝ, shift: ℝ = 0.8041538754478097, 0.16409696359187365

   # Step 0: The Molecule as a Graph
   z_table: ℝ[89] = load_mace_weights(0)
   r_cut: ℝ = 6.0

   def one_hot(numbers: ℝ[n], table: ℝ[s]): ℝ[n, s]:
       num_atoms: ℕ, num_species: ℕ = len(numbers), len(table)
       encoding: ℝ[num_atoms, num_species] = zeros(num_atoms, num_species)
       for atom:ℕ(num_atoms):
           for species:ℕ(num_species):
               if numbers[atom] == table[species]:
                   encoding[atom, species] = 1.0
       return encoding

   def neighbor_search(positions: ℝ[n, 3], cutoff: ℝ): ℝ[2, m]:
       num_atoms: ℕ = len(positions)
       vector: ℝ[3], distance: ℝ = zeros(3), 0.0
       num_edges: ℕ = 0
       for sender:ℕ(num_atoms):
           for receiver:ℕ(num_atoms):
               vector = positions[receiver] - positions[sender]
               distance = sqrt(sum(vector * vector))
               if distance > 0.0:
                   if distance < cutoff:
                       num_edges += 1
       edge_index: ℝ[2, num_edges] = zeros(2, num_edges)
       edge: ℕ = 0
       for sender:ℕ(num_atoms):
           for receiver:ℕ(num_atoms):
               vector = positions[receiver] - positions[sender]
               distance = sqrt(sum(vector * vector))
               if distance > 0.0:
                   if distance < cutoff:
                       edge_index[:, edge] = [sender, receiver]
                       edge += 1
       return edge_index

   positions: ℝ[3, 3] = water_positions
   node_attrs: ℝ[3, 89] = one_hot(water_numbers, z_table)
   edge_index: ℝ[2, 6] = neighbor_search(positions, r_cut)

   print(positions)
   print(edge_index)

   # Step 1: Embeddings
   num_elements: ℝ, num_channels: ℝ = 89, 128
   w_embed: ℝ[89, 128] = load_mace_weights(1)

   def node_embedding(node_attrs: ℝ[n, s], w: ℝ[s, k]): ℝ[n, k]:
       return (node_attrs @ w) / sqrt(num_elements)

   # Radial features
   π: ℝ, num_bessel: ℝ, p_env: ℝ = 3.141592653589793, 10, 5.0

   def edge_vectors(positions: ℝ[n, 3], edge_index: ℝ[2, e]): ℝ[e, 3]:
       num_edges: ℕ = len(edge_index[0])
       vectors: ℝ[num_edges, 3] = zeros(num_edges, 3)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           vectors[edge] = positions[receiver] - positions[sender]
       return vectors

   def edge_lengths(vectors: ℝ[e, 3]): ℝ[e]:
       x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
       return sqrt(x * x + y * y + z * z)

   def radial_embedding(lengths: ℝ[e]): ℝ[e, b]:
       num_edges: ℕ = len(lengths)
       edge_feats: ℝ[num_edges, num_bessel] = zeros(num_edges, num_bessel)
       x = lengths / r_cut
       f_cut = 1.0 - (p_env + 1.0) * (p_env + 2.0) / 2.0 * x ** p_env + p_env * (p_env + 2.0) * x ** (p_env + 1.0) - p_env * (p_env + 1.0) / 2.0 * x ** (p_env + 2.0)
       for n:ℕ(num_bessel):
           edge_feats[:, n] = sqrt(2.0 / r_cut) * sin((n + 1.0) * π * lengths / r_cut) / lengths * f_cut
       return edge_feats

   vectors: ℝ[6, 3] = edge_vectors(positions, edge_index)
   lengths: ℝ[6] = edge_lengths(vectors)
   edge_feats: ℝ[6, 10] = radial_embedding(lengths)

   # Angular features
   def Y0(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 1]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 1] = zeros(num_edges, 1)
       results[:, 0] = 1.0
       return results

   def Y1(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 3]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 3] = zeros(num_edges, 3)
       x, y, z = vectors[:, 0] / lengths, vectors[:, 1] / lengths, vectors[:, 2] / lengths
       results[:, 0] = sqrt(3.0) * x
       results[:, 1] = sqrt(3.0) * y
       results[:, 2] = sqrt(3.0) * z
       return results

   def Y2(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 5]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 5] = zeros(num_edges, 5)
       x, y, z = vectors[:, 0] / lengths, vectors[:, 1] / lengths, vectors[:, 2] / lengths
       results[:, 0] = sqrt(15.0) * x * z
       results[:, 1] = sqrt(15.0) * x * y
       results[:, 2] = sqrt(5.0) * (y * y - 0.5 * (x * x + z * z))
       results[:, 3] = sqrt(15.0) * y * z
       results[:, 4] = sqrt(15.0) / 2.0 * (z * z - x * x)
       return results

   def Y3(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 7]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 7] = zeros(num_edges, 7)
       x, y, z = vectors[:, 0] / lengths, vectors[:, 1] / lengths, vectors[:, 2] / lengths
       results[:, 0] = sqrt(35.0 / 8.0) * (3.0 * x * z * z - x * x * x)
       results[:, 1] = sqrt(105.0) * x * y * z
       results[:, 2] = sqrt(21.0 / 8.0) * x * (4.0 * y * y - x * x - z * z)
       results[:, 3] = sqrt(7.0) / 2.0 * y * (2.0 * y * y - 3.0 * x * x - 3.0 * z * z)
       results[:, 4] = sqrt(21.0 / 8.0) * z * (4.0 * y * y - x * x - z * z)
       results[:, 5] = sqrt(105.0) / 2.0 * y * (z * z - x * x)
       results[:, 6] = sqrt(35.0 / 8.0) * (z * z * z - 3.0 * x * x * z)
       return results

   def spherical_harmonics(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 16]:
       num_edges: ℕ = len(lengths)
       edge_attrs: ℝ[e, 16] = zeros(num_edges, 16)
       edge_attrs[:, 0:1] = Y0(vectors, lengths)
       edge_attrs[:, 1:4] = Y1(vectors, lengths)
       edge_attrs[:, 4:9] = Y2(vectors, lengths)
       edge_attrs[:, 9:16] = Y3(vectors, lengths)
       return edge_attrs

   edge_attrs: ℝ[6, 16] = spherical_harmonics(vectors, lengths)

   print(edge_feats)
   print(edge_attrs)

   # Step 2: The Interaction Block
   w_up: ℝ[128, 128] = load_mace_weights(2)

   def linear_up(node_feats: ℝ[n, k], w: ℝ[k, k]): ℝ[n, k]:
       return node_feats @ w / sqrt(num_channels)

   # Activation function
   silu_norm: ℝ = 1.679176792

   def silu(x: ℝ[e, h]): ℝ[e, h]:
       return silu_norm * x / (1.0 + exp(-x))

   # Radial network
   radial_hidden: ℝ = 64

   w_r1: ℝ[10, 64] = load_mace_weights(3)
   w_r2: ℝ[64, 64] = load_mace_weights(4)
   w_r3: ℝ[64, 64] = load_mace_weights(5)
   w_r4: ℝ[64, 512] = load_mace_weights(6)

   def radial_mlp(edge_feats: ℝ[e, b], w1: ℝ[b, h], w2: ℝ[h, h], w3: ℝ[h, h], w4: ℝ[h, o]): ℝ[e, o]:
       h1: ℝ[e, h] = silu(edge_feats @ w1 / sqrt(num_bessel))
       h2: ℝ[e, h] = silu(h1 @ w2 / sqrt(radial_hidden))
       h3: ℝ[e, h] = silu(h2 @ w3 / sqrt(radial_hidden))
       return h3 @ w4 / sqrt(radial_hidden)

   # The one-particle basis
   def conv_tp(node_feats_up: ℝ[n, k], edge_attrs: ℝ[e, 16], tp_weights: ℝ[e, t], edge_index: ℝ[2, e]): ℝ[e, k, 16]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 16] = zeros(num_edges, num_channels, 16)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           h = node_feats_up[sender]
           s0, s1, s2, s3 = h * tp_weights[edge, 0:128], h * tp_weights[edge, 128:256], h * tp_weights[edge, 256:384], h * tp_weights[edge, 384:512]
           mji[edge, :, 0] = s0 * edge_attrs[edge, 0]
           for m:ℕ(3):
               mji[edge, :, 1 + m] = s1 * edge_attrs[edge, 1 + m]
           for m:ℕ(5):
               mji[edge, :, 4 + m] = s2 * edge_attrs[edge, 4 + m]
           for m:ℕ(7):
               mji[edge, :, 9 + m] = s3 * edge_attrs[edge, 9 + m]
       return mji

   # Message Passing
   avg_num_neighbors: ℝ = 61.964672446250916

   w_0: ℝ[128, 128] = load_mace_weights(7)
   w_1: ℝ[128, 128] = load_mace_weights(8)
   w_2: ℝ[128, 128] = load_mace_weights(9)
   w_3: ℝ[128, 128] = load_mace_weights(10)

   def neighbor_sum(mji: ℝ[e, k, 16], edge_index: ℝ[2, e], num_atoms: ℕ): ℝ[num_atoms, k, 16]:
       num_edges: ℕ = len(mji)
       message: ℝ[num_atoms, num_channels, 16] = zeros(num_atoms, num_channels, 16)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           message[receiver] = message[receiver] + mji[edge]
       return message

   def linear(message: ℝ[n, k, 16], w0: ℝ[k, k], w1: ℝ[k, k], w2: ℝ[k, k], w3: ℝ[k, k]): ℝ[n, k, 16]:
       num_atoms: ℕ = len(message)
       out: ℝ[num_atoms, num_channels, 16] = zeros(num_atoms, num_channels, 16)
       out[:, :, 0:1] = w0 @ message[:, :, 0:1] / sqrt(num_channels)
       out[:, :, 1:4] = w1 @ message[:, :, 1:4] / sqrt(num_channels)
       out[:, :, 4:9] = w2 @ message[:, :, 4:9] / sqrt(num_channels)
       out[:, :, 9:16] = w3 @ message[:, :, 9:16] / sqrt(num_channels)
       return out / avg_num_neighbors

   # Step 3: The Product
   U1: ℝ[16, 1] = load_mace_weights(11)
   U2: ℝ[16, 16, 4] = load_mace_weights(12)
   U3: ℝ[16, 16, 16, 23] = load_mace_weights(13)
   U1v: ℝ[3, 16, 1] = load_mace_weights(14)
   U2v: ℝ[3, 16, 16, 6] = load_mace_weights(15)
   U3v: ℝ[3, 16, 16, 16, 51] = load_mace_weights(16)

   def B_features_1(A: ℝ[n, k, 16], U: ℝ[16, a]): ℝ[n, k, a]:
       num_atoms: ℕ = len(A)
       num_paths: ℕ = len(U[0])
       B1: ℝ[num_atoms, num_channels, num_paths] = zeros(num_atoms, num_channels, num_paths)
       for atom:ℕ(num_atoms):
           B1[atom] = A[atom] @ U
       return B1

   def B_features_2(A: ℝ[n, k, 16], U: ℝ[16, 16, b]): ℝ[n, k, b]:
       num_atoms: ℕ = len(A)
       num_paths: ℕ = len(U[0, 0])
       B2: ℝ[num_atoms, num_channels, num_paths] = zeros(num_atoms, num_channels, num_paths)
       for atom:ℕ(num_atoms):
           for i:ℕ(16):
               M = A[atom] @ U[i]
               for p:ℕ(num_paths):
                   B2[atom, :, p] = B2[atom, :, p] + A[atom, :, i] * M[:, p]
       return B2

   def B_features_3(A: ℝ[n, k, 16], U: ℝ[16, 16, 16, c]): ℝ[n, k, c]:
       num_atoms: ℕ = len(A)
       num_paths: ℕ = len(U[0, 0, 0])
       B3: ℝ[num_atoms, num_channels, num_paths] = zeros(num_atoms, num_channels, num_paths)
       for atom:ℕ(num_atoms):
           for i:ℕ(16):
               for j:ℕ(16):
                   M = A[atom] @ U[i, j]
                   for p:ℕ(num_paths):
                       B3[atom, :, p] = B3[atom, :, p] + A[atom, :, i] * A[atom, :, j] * M[:, p]
       return B3

   w_prod1: ℝ[89, 1, 128] = load_mace_weights(17)
   w_prod2: ℝ[89, 4, 128] = load_mace_weights(18)
   w_prod3: ℝ[89, 23, 128] = load_mace_weights(19)
   w_prod1v: ℝ[89, 1, 128] = load_mace_weights(20)
   w_prod2v: ℝ[89, 6, 128] = load_mace_weights(21)
   w_prod3v: ℝ[89, 51, 128] = load_mace_weights(22)

   def weighted_sum(B1: ℝ[n, k, a], B2: ℝ[n, k, b], B3: ℝ[n, k, c], node_attrs: ℝ[n, s], wp1: ℝ[s, a, k], wp2: ℝ[s, b, k], wp3: ℝ[s, c, k]): ℝ[n, k]:
       num_atoms: ℕ = len(B1)
       m: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for p:ℕ(len(B1[0, 0])):
           m = m + (node_attrs @ wp1[:, p, :]) * B1[:, :, p]
       for p:ℕ(len(B2[0, 0])):
           m = m + (node_attrs @ wp2[:, p, :]) * B2[:, :, p]
       for p:ℕ(len(B3[0, 0])):
           m = m + (node_attrs @ wp3[:, p, :]) * B3[:, :, p]
       return m

   def scalar_product(A: ℝ[n, k, 16], node_attrs: ℝ[n, s]): ℝ[n, k]:
       return weighted_sum(B_features_1(A, U1), B_features_2(A, U2), B_features_3(A, U3), node_attrs, w_prod1, w_prod2, w_prod3)

   def vector_product(A: ℝ[n, k, 16], node_attrs: ℝ[n, s]): ℝ[n, k, 3]:
       num_atoms: ℕ = len(A)
       out: ℝ[num_atoms, num_channels, 3] = zeros(num_atoms, num_channels, 3)
       for m:ℕ(3):
           out[:, :, m] = weighted_sum(B_features_1(A, U1v[m]), B_features_2(A, U2v[m]), B_features_3(A, U3v[m]), node_attrs, w_prod1v, w_prod2v, w_prod3v)
       return out

   # Updating the node features
   w_sc: ℝ[128, 89, 128] = load_mace_weights(23)
   w_p: ℝ[128, 128] = load_mace_weights(24)
   w_pv: ℝ[128, 128] = load_mace_weights(25)

   def skip_tp(node_feats: ℝ[n, k], node_attrs: ℝ[n, s], w: ℝ[k, s, k]): ℝ[n, k]:
       num_atoms: ℕ = len(node_feats)
       sc: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for atom:ℕ(num_atoms):
           for z:ℕ(num_elements):
               if node_attrs[atom, z] == 1.0:
                   sc[atom] = node_feats[atom] @ w[:, z, :] / sqrt(num_channels * num_elements)
       return sc

   def node_update(m: ℝ[n, k], sc: ℝ[n, k], wp: ℝ[k, k]): ℝ[n, k]:
       return m @ wp / sqrt(num_channels) + sc

   def node_update_vectors(mv: ℝ[n, k, 3], wp: ℝ[k, k]): ℝ[n, k, 3]:
       num_atoms: ℕ = len(mv)
       out: ℝ[num_atoms, num_channels, 3] = zeros(num_atoms, num_channels, 3)
       for atom:ℕ(num_atoms):
           out[atom] = wp @ mv[atom] / sqrt(num_channels)
       return out

   # Step 4: The Readout
   w_readout: ℝ[128, 1] = load_mace_weights(26)

   def readout(node_feats: ℝ[n, k], w: ℝ[k, 1]): ℝ[n, 1]:
       return node_feats @ w / sqrt(num_channels)

   # Step 5: The Second Layer
   w2_up: ℝ[128, 128] = load_mace_weights(27)
   w2_up_v: ℝ[128, 128] = load_mace_weights(28)

   def linear_up_vectors(node_vecs: ℝ[n, k, 3], w: ℝ[k, k]): ℝ[n, k, 3]:
       num_atoms: ℕ = len(node_vecs)
       out: ℝ[num_atoms, num_channels, 3] = zeros(num_atoms, num_channels, 3)
       for atom:ℕ(num_atoms):
           out[atom] = w @ node_vecs[atom] / sqrt(num_channels)
       return out

   w2_r1: ℝ[10, 64] = load_mace_weights(29)
   w2_r2: ℝ[64, 64] = load_mace_weights(30)
   w2_r3: ℝ[64, 64] = load_mace_weights(31)
   w2_r4: ℝ[64, 1280] = load_mace_weights(32)

   cg_121: ℝ[3, 5, 3] = load_mace_weights(33)
   cg_112: ℝ[3, 3, 5] = load_mace_weights(34)
   cg_132: ℝ[3, 7, 5] = load_mace_weights(35)
   cg_123: ℝ[3, 5, 7] = load_mace_weights(36)

   def neighbor_sum_second(mji: ℝ[e, k, 40], edge_index: ℝ[2, e], num_atoms: ℕ): ℝ[num_atoms, k, 40]:
       num_edges: ℕ = len(mji)
       message: ℝ[num_atoms, num_channels, 40] = zeros(num_atoms, num_channels, 40)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           message[receiver] = message[receiver] + mji[edge]
       return message

   def conv_tp_second(up: ℝ[n, k], up_vecs: ℝ[n, k, 3], edge_attrs: ℝ[e, 16], tp_weights: ℝ[e, t], edge_index: ℝ[2, e]): ℝ[e, k, 40]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 40] = zeros(num_edges, num_channels, 40)
       y1: ℝ[3, 1] = zeros(3, 1)
       y2: ℝ[5, 1] = zeros(5, 1)
       y3: ℝ[7, 1] = zeros(7, 1)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           s, v = up[sender], up_vecs[sender]
           y1[:, 0] = edge_attrs[edge, 1:4]
           y2[:, 0] = edge_attrs[edge, 4:9]
           y3[:, 0] = edge_attrs[edge, 9:16]
           d = v @ y1
           mji[edge, :, 0] = tp_weights[edge, 0:128] * s * edge_attrs[edge, 0]
           mji[edge, :, 1] = tp_weights[edge, 128:256] * d[:, 0] / sqrt(3.0)
           for m:ℕ(3):
               a = v @ (cg_121[:, :, m] @ y2)
               mji[edge, :, 2 + m] = tp_weights[edge, 256:384] * s * y1[m, 0]
               mji[edge, :, 5 + m] = tp_weights[edge, 384:512] * v[:, m] * edge_attrs[edge, 0]
               mji[edge, :, 8 + m] = tp_weights[edge, 512:640] * sqrt(3.0) * a[:, 0]
           for m:ℕ(5):
               b = v @ (cg_112[:, :, m] @ y1)
               c = v @ (cg_132[:, :, m] @ y3)
               mji[edge, :, 11 + m] = tp_weights[edge, 640:768] * s * y2[m, 0]
               mji[edge, :, 16 + m] = tp_weights[edge, 768:896] * sqrt(5.0) * b[:, 0]
               mji[edge, :, 21 + m] = tp_weights[edge, 896:1024] * sqrt(5.0) * c[:, 0]
           for m:ℕ(7):
               d3 = v @ (cg_123[:, :, m] @ y2)
               mji[edge, :, 26 + m] = tp_weights[edge, 1024:1152] * s * y3[m, 0]
               mji[edge, :, 33 + m] = tp_weights[edge, 1152:1280] * sqrt(7.0) * d3[:, 0]
       return mji

   w2_0: ℝ[128, 128] = load_mace_weights(37)
   w2_1: ℝ[128, 128] = load_mace_weights(38)
   w2_2: ℝ[128, 128] = load_mace_weights(39)
   w2_3: ℝ[128, 128] = load_mace_weights(40)
   w2_4: ℝ[128, 128] = load_mace_weights(41)
   w2_5: ℝ[128, 128] = load_mace_weights(42)
   w2_6: ℝ[128, 128] = load_mace_weights(43)
   w2_7: ℝ[128, 128] = load_mace_weights(44)
   w2_8: ℝ[128, 128] = load_mace_weights(45)
   w2_9: ℝ[128, 128] = load_mace_weights(46)

   def linear_second(message: ℝ[n, k, 40], w0: ℝ[k, k], w1: ℝ[k, k], w2: ℝ[k, k], w3: ℝ[k, k], w4: ℝ[k, k], w5: ℝ[k, k], w6: ℝ[k, k], w7: ℝ[k, k], w8: ℝ[k, k], w9: ℝ[k, k]): ℝ[n, k, 16]:
       num_atoms: ℕ = len(message)
       out: ℝ[num_atoms, num_channels, 16] = zeros(num_atoms, num_channels, 16)
       out[:, :, 0:1] = (w0 @ message[:, :, 0:1] + w1 @ message[:, :, 1:2]) / sqrt(256.0)
       out[:, :, 1:4] = (w2 @ message[:, :, 2:5] + w3 @ message[:, :, 5:8] + w4 @ message[:, :, 8:11]) / sqrt(384.0)
       out[:, :, 4:9] = (w5 @ message[:, :, 11:16] + w6 @ message[:, :, 16:21] + w7 @ message[:, :, 21:26]) / sqrt(384.0)
       out[:, :, 9:16] = (w8 @ message[:, :, 26:33] + w9 @ message[:, :, 33:40]) / sqrt(256.0)
       return out / avg_num_neighbors

   w2_prod1: ℝ[89, 1, 128] = load_mace_weights(47)
   w2_prod2: ℝ[89, 4, 128] = load_mace_weights(48)
   w2_prod3: ℝ[89, 23, 128] = load_mace_weights(49)
   w2_sc: ℝ[128, 89, 128] = load_mace_weights(50)
   w2_p: ℝ[128, 128] = load_mace_weights(51)

   # Readout and total energy
   w2_ro1: ℝ[128, 16] = load_mace_weights(52)
   w2_ro2: ℝ[16, 1] = load_mace_weights(53)
   atomic_energies: ℝ[89, 1] = load_mace_weights(54)

   def readout2(node_feats: ℝ[n, k], w1: ℝ[k, 16], w2: ℝ[16, 1]): ℝ[n, 1]:
       hidden: ℝ[n, 16] = silu(node_feats @ w1 / sqrt(num_channels))
       return hidden @ w2 / sqrt(16.0)

   # Model Definition
   def mace_energy(positions: ℝ[n, 3], numbers: ℝ[n]): ℝ:
       num_atoms: ℕ = len(positions)
       node_attrs: ℝ[num_atoms, 89] = one_hot(numbers, z_table)
       edge_index = neighbor_search(positions, r_cut)
       vectors = edge_vectors(positions, edge_index)
       lengths = edge_lengths(vectors)
       edge_feats, edge_attrs = radial_embedding(lengths), spherical_harmonics(vectors, lengths)
       node_feats0 = node_embedding(node_attrs, w_embed)
       node_feats_up = linear_up(node_feats0, w_up)
       tp_weights = radial_mlp(edge_feats, w_r1, w_r2, w_r3, w_r4)
       mji = conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
       message = linear(neighbor_sum(mji, edge_index, num_atoms), w_0, w_1, w_2, w_3)
       sc = skip_tp(node_feats0, node_attrs, w_sc)
       node_feats1 = node_update(scalar_product(message, node_attrs), sc, w_p)
       node_vecs1 = node_update_vectors(vector_product(message, node_attrs), w_pv)
       node_energies = readout(node_feats1, w_readout)
       node_feats_up2 = linear_up(node_feats1, w2_up)
       node_vecs_up2 = linear_up_vectors(node_vecs1, w2_up_v)
       tp_weights2 = radial_mlp(edge_feats, w2_r1, w2_r2, w2_r3, w2_r4)
       mji2 = conv_tp_second(node_feats_up2, node_vecs_up2, edge_attrs, tp_weights2, edge_index)
       message2 = linear_second(neighbor_sum_second(mji2, edge_index, num_atoms), w2_0, w2_1, w2_2, w2_3, w2_4, w2_5, w2_6, w2_7, w2_8, w2_9)
       sc2 = skip_tp(node_feats1, node_attrs, w2_sc)
       node_feats2 = node_update(weighted_sum(B_features_1(message2, U1), B_features_2(message2, U2), B_features_3(message2, U3), node_attrs, w2_prod1, w2_prod2, w2_prod3), sc2, w2_p)
       node_energies2 = readout2(node_feats2, w2_ro1, w2_ro2)
       atom_energies = node_attrs @ atomic_energies
       energy: ℝ = sum(atom_energies) + scale * (sum(node_energies) + sum(node_energies2)) + shift * num_atoms
       return energy

   # Main Program:
   water_energy: ℝ = mace_energy(water_positions, water_numbers)
   methane_energy: ℝ = mace_energy(methane_positions, methane_numbers)
   formaldehyde_energy: ℝ = mace_energy(formaldehyde_positions, formaldehyde_numbers)

   predictions: ℝ[3] = [water_energy, methane_energy, formaldehyde_energy]
   print(predictions)
   print(reference_energies)
   print(predictions - reference_energies)

References
----------

.. [Batatia2022] Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*. Advances in Neural Information Processing Systems (NeurIPS). arXiv:2206.07697. https://arxiv.org/abs/2206.07697

.. [TFN2018] Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). *Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds*. arXiv:1802.08219. https://arxiv.org/abs/1802.08219

.. [PhysikaTFN] Physika documentation. *Tensor Field Networks*. https://physika.readthedocs.io/en/latest/tutorials/tfn_tutorial.html

.. [e3nn] Geiger, M., & Smidt, T. (2022). *e3nn: Euclidean Neural Networks*. arXiv:2207.09453. https://arxiv.org/abs/2207.09453

.. [MACETutorials] ACEsuit. *MACE tutorials*, in particular ``T03_MACE_Theory.ipynb`` and ``MACE_developer.ipynb``. https://github.com/ACEsuit/mace-tutorials

.. [MACEMP0] Batatia, I., Benner, P., Chiang, Y., Elena, A. M., Kovács, D. P., Riebesell, J., et al. (2023). *A foundation model for atomistic materials chemistry*. arXiv:2401.00096. https://arxiv.org/abs/2401.00096
