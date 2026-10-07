MACE: Higher-Order Equivariant Message Passing
==============================================

In this tutorial we implement MACE (Message Passing Atomic Cluster Expansion) in Physika and train it on a dataset of water conformations to predict total energy.

Introduction
------------

Message Passing Atomic Cluster Expansion (MACE) is a framework for learning interatomic potentials, extensively used in computational chemistry and materials science.
It combines the principles of symmetray-adapted basis of the Atomic Cluster Expansion (ACE) with equivariant message passing of graph neural networks. This fusion results
in a complex architecture that can capture intricate interatomic interactions while respecting the underlying symmetries of the system.

.. figure:: /_static/tutorial_files/mace_overview.png
   :alt: structure to atomic environments, learnable features and atomic energies
   :align: center
   :width: 700pxa

   Figure 1: Overview of MACE. [MACETutorials]_.

Representing Spherical Tensors: Ylm's
-------------------------------------

Descriptors in MACE are represented as 'spherical tensors'.

To explain spherical tensors, we first introduce the spherical harmonics :math:`Y_{\ell m}`,
which are functions of a unit vector that points in a direction in 3D space. 
They are labelled by the *angular order* :math:`\ell \geq 0`
and the *component* :math:`m` from s:math:`-\ell` to :math:`\ell`, so every order has
:math:`2\ell + 1` components.

Spherical harmonics 'transform' in a special way when their argument is rotated. This means if we take a vector :math:`\mathbf{r}` and rotate it by some matrix :math:`\mathbf{r} \to R\,\mathbf{r}`, then the spherical harmonic :math:`Y_{\ell m}(\mathbf{r})` will also
change in a predictable way.

.. math::

   Y_{\ell m}(R\,\mathbf{r}) = \sum_{m'=-\ell}^{\ell} D^{\ell}_{m m'}(R)\, Y_{\ell m'}(\mathbf{r})

:math:`D^{\ell}(R)` is called a Wigner D-matrix. You can find a more detailed explanation of spherical harmonics and Wigner D-matrices in the TFN tutorial.

In MACE, we store the spherical harmonics in an array of shape ``[2l + 1]``.

For :math:`\ell = 0, 1, 2` we compute the harmonics for all valid :math:`m` and store them together in one array, ordered by :math:`\ell` and then by :math:`m` :

.. math::

   \left[ Y_0^0,\;\; Y_1^{-1},\, Y_1^0,\, Y_1^1,\;\; Y_2^{-2},\, Y_2^{-1},\, Y_2^0,\, Y_2^1,\, Y_2^2 \right]


Dataset
-------

We train the model on water molecules in random geometries by drawing the two O-H bond lengths uniformly between 0.8 and 1.6 Å and the H-O-H angle between 80 and 140 degrees, and we rotate the molecule in a random direction.
Every structure is labelled with the energy computed by the GFN2-xTB method. 

We use 200 structures, 160 for training and 40 for testing. The structures split at random by DeepChem's ``RandomSplitter``.

.. code-block:: text

   train_test_split: ℕ = 80
   total_dataset_size: ℕ = 200
   seed: ℕ = 0

   dataset = create_water_dataset(train_test_split, total_dataset_size, seed)
   train_dataset = dataset[0]
   test_dataset = dataset[1]

   num_train: ℕ = 160
   num_test: ℕ = 40
   train_positions: ℝ[num_train, 3, 3] = train_dataset[0]
   train_energies: ℝ[num_train] = train_dataset[1]
   test_positions: ℝ[num_test, 3, 3] = test_dataset[0]
   test_energies: ℝ[num_test] = test_dataset[1]

.. note::
   ``create_water_dataset`` is not a built-in Physika function. To use it,
   add the following helper to ``physika/runtime.py``.

   .. code-block:: python

        def create_water_dataset(train_test_split=80, total_dataset_size=200, seed=0):
            import deepchem as dc
            import numpy as np
            import torch
            from ase import Atoms
            from tblite.ase import TBLite

            rng = np.random.default_rng(seed)

            def xtb(unpaired=0):
                return TBLite(method="GFN2-xTB", uhf=unpaired, verbosity=0)
            energy_h = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            energy_h.calc = xtb(1)
            energy_h = energy_h.get_potential_energy()
            energy_o = Atoms("O", positions=[[0.0, 0.0, 0.0]])
            energy_o.calc = xtb(2)
            energy_o = energy_o.get_potential_energy()

            positions, energies = [], []
            for _ in range(total_dataset_size):
                r1, r2 = rng.uniform(0.8, 1.6, size=2)        
                theta = np.deg2rad(rng.uniform(80.0, 140.0)) 
                water = Atoms("OHH", positions=[[0.0, 0.0, 0.0],
                                                [r1, 0.0, 0.0],
                                                [r2 * np.cos(theta), r2 * np.sin(theta), 0.0]])
                water.rotate(rng.uniform(0.0, 360.0), rng.normal(size=3))   
                water.calc = xtb()
                energies.append(water.get_potential_energy() - (energy_o + 2 * energy_h))
                positions.append(water.positions.copy())

            dataset = dc.data.NumpyDataset(X=np.array(positions), y=np.array(energies).reshape(-1, 1))
            splitter = dc.splits.RandomSplitter()
            train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
                dataset, frac_train=train_test_split / 100.0, frac_valid=0.0,
                frac_test=1 - train_test_split / 100.0, seed=seed
            )

            def build(dataset):
                return [torch.tensor(dataset.X, dtype=torch.float32, device=DEVICE),
                        torch.tensor(dataset.y[:, 0], dtype=torch.float32, device=DEVICE)]

            return [build(train_dataset), build(test_dataset)]

Scale and Shift
~~~~~~~~~~~~~~~

The energies are of the order of -12 eV, but the output of the model is of order 1. We therefore train the model on scaled energies:

.. math::

   \hat E = \frac{E - \mu}{\sigma}

where :math:`\mu` and :math:`\sigma` are the mean and the standard deviation of the training energies. To get an energy back in eV we use :math:`E = \mu + \sigma \hat E`. 

.. code-block:: text

   energy_mean: ℝ = sum(train_energies) / num_train
   energy_std: ℝ = sqrt(sum((train_energies - energy_mean) * (train_energies - energy_mean)) / num_train)


MACE Feature Construction
-------------------------

We will now go through some key parts of the feature construction in MACE.

Step 0: The Molecule as a Graph
-------------------------------

The first step in MACE is to represent the molecules as a graph. To do this we represent the molecule as a list of atoms and 'edges' where
an edge is simply the connection between two atoms. In MACE the cutoff radius determines which atoms are connected by an edge.
We tell MACE what atoms to work with through a ``z_table`` which contains the list of atomic numbers of the atoms in the environment.

We take the first structure of the training set as our example molecule. We use the ``z_table`` to turn them into a one-hot encoding.

.. code-block:: text

   r_cut: ℝ = 2.0
   atomic_numbers: ℝ[3] = [8.0, 1.0, 1.0]
   z_table: ℝ[2] = [1.0, 8.0]

   def one_hot(numbers: ℝ[n], table: ℝ[s]): ℝ[n, s]:
       num_atoms: ℕ = len(numbers)
       num_species: ℕ = len(table)
       encoding: ℝ[num_atoms, num_species] = zeros(num_atoms, num_species)
       for atom:ℕ(num_atoms):
           for species:ℕ(num_species):
               if numbers[atom] == table[species]:
                   encoding[atom, species] = 1.0
       return encoding

The edges come from a neighbour search using the cutoff radius ``r_cut``. They are all ordered pairs of distinct atoms that are closer than ``r_cut``.

.. code-block:: text

   def neighbour_search(positions: ℝ[n, 3], cutoff: ℝ): ℝ[2, m]:
       num_atoms: ℕ = len(positions)
       vector: ℝ[3] = zeros(3)
       distance: ℝ = 0.0
       num_edges: ℝ = 0
       for sender:ℕ(num_atoms):
           for receiver:ℕ(num_atoms):
               vector = positions[receiver] - positions[sender]
               distance = sqrt(sum(vector * vector))
               if distance > 0.0:
                   if distance < cutoff:
                       num_edges += 1
       edge_index: ℝ[2, num_edges] = zeros(2, num_edges)
       edge: ℝ = 0
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

   positions: ℝ[3, 3] = train_positions[0]
   node_attrs: ℝ[3, 2] = one_hot(atomic_numbers, z_table)
   edge_index: ℝ[2, 6] = neighbour_search(positions, r_cut)

   print(positions)
   print(node_attrs)
   print(edge_index)

``positions`` represents the Cartesian coordinates in Å, one row per atom.

``node_attrs`` describes the species of every atom as a one-hot encoding of the elements. 

``edge_index`` Represents the edges which are stored as a list of 'senders' and 'receivers', representing the start and end point of each edge. 

The dataset contains many structures, with 103 structures having 6 edges and 97 structures having 4 edges, so we pad all the structures with zeros to keep 6 edges. 

.. code-block:: text

   def batch_edge_index(batch: ℝ[s, 3, 3]): ℝ[s, 2, 6]:
       num_samples: ℕ = len(batch)
       result: ℝ[num_samples, 2, 6] = zeros(num_samples, 2, 6)
       num_edges: ℕ = 0
       for i:ℕ(num_samples):
           edges = neighbour_search(batch[i], r_cut)
           num_edges = len(edges[0])
           result[i, :, 0:num_edges] = edges
       return result

   train_edge_index: ℝ[num_train, 2, 6] = batch_edge_index(train_positions)
   test_edge_index: ℝ[num_test, 2, 6] = batch_edge_index(test_positions)


Step 1: Embeddings
------------------

We now take this information about the molecule and turn it into a set of features that the model can work with. These are called embeddings.


The equation for the initial node features is given by:

.. math::

   h^{(0)}_{i,k} = \frac{1}{\sqrt{Z}} \sum_{z} W_{zk}\, \delta_{z z_i}

For example, in our dataset the ``z_table`` consists of hydrogen and oxygen. If atom :math:`i` is hydrogen, the initial node features are just :math:`W_{0k}`. This means that each atom is given a vector of length :math:`K`, based on its atomic number.

In MACE, :math:`K` is the number of channels. This is the fundamental 'size' of the descriptor.
The atom's chemical species is embedded through a length-:math:`K` vector, and we also embed the lengths and direction of the edges.
The lengths of the edges are mapped through a set of 8 radial Bessel functions, and the directions of the edges are mapped through a set of 9 spherical harmonics.


The three embeddings are shown in Figure 2.

.. figure:: /_static/tutorial_files/mace_embedding.png
   :alt: The node embedding, the radial embedding and the angular embedding
   :align: center
   :width: 700px

   Figure 2: Represents the creation of node and edge embeddings [MACETutorials]_.


Node features
~~~~~~~~~~~~~

The length-:math:`K` vector of features of atom :math:`i` is computed from its one-hot encoding :math:`\delta_{z z_i}` and a learned matrix :math:`W`, where each atom's embedding is a lookup from the learned matrix. 

.. code-block:: text

   num_elements: ℝ = 2
   num_channels: ℝ = 8
   w_embed: ℝ[2, 8] = for z:ℕ(num_elements) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def node_embedding(node_attrs: ℝ[n, s], w: ℝ[s, k]): ℝ[n, k]:
       return (node_attrs @ w) / sqrt(num_elements)

Radial features
~~~~~~~~~~~~~~~

The radial embedding creates features based on the distance between atoms.
We first compute the vector that points from the sender to the receiver of every edge, and its length:

.. math::

   \mathbf{r}_{ji} = \mathbf{x}_i - \mathbf{x}_j, \qquad r_{ji} = \lVert \mathbf{r}_{ji} \rVert

Where atom :math:`j` is the sender and atom :math:`i` the receiver.

A single number is not much for a network to work with, so we expand the distance into a set of eight radial functions that disappear at the cutoff radius.

.. math::

   j_n(r) = \sqrt{\frac{2}{r_\mathrm{cut}}}\,\frac{\sin(n\pi r / r_\mathrm{cut})}{r}, \qquad n = 1, \dots, 8

These are Bessel functions (Figure 3). Each of them vanishes at the cutoff radius. Together they give the model a flexible description of how things depend on distance.

.. figure:: /_static/tutorial_files/mace_bessel_basis.png
   :alt: The eight radial edge features as a function of the distance
   :align: center
   :width: 500px

   Figure 3: The eight radial bessel functions as a function of the distance [MACETutorials]_.

Each function is then multiplied by a smooth cutoff envelope, a polynomial that falls to zero at :math:`r_\mathrm{cut}`:

.. math::

   f(x) = 1 - \frac{(p+1)(p+2)}{2}\,x^{p} + p(p+2)\,x^{p+1} - \frac{p(p+1)}{2}\,x^{p+2}, \qquad x = \frac{r}{r_\mathrm{cut}}

Let's define the constants and the functions for the radial embedding:

.. code-block:: text

   π: ℝ = 3.141592653589793
   num_bessel: ℝ = 8
   p: ℝ = 6.0

.. code-block:: text

   def edge_vectors(positions: ℝ[n, 3], edge_index: ℝ[2, e]): ℝ[e, 3]:
       num_edges: ℕ = len(edge_index[0])
       vectors: ℝ[num_edges, 3] = zeros(num_edges, 3)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           vectors[edge] = positions[receiver] - positions[sender]
       return vectors

   def edge_lengths(vectors: ℝ[e, 3]): ℝ[e]:
       num_edges: ℕ = len(vectors)
       lengths: ℝ[num_edges] = zeros(num_edges)
       for edge:ℕ(num_edges):
           lengths[edge] = sqrt(sum(vectors[edge] * vectors[edge]))
       return lengths

   def radial_embedding(lengths: ℝ[e]): ℝ[e, b]:
       num_edges: ℕ = len(lengths)
       edge_feats: ℝ[num_edges, num_bessel] = zeros(num_edges, num_bessel)
       r: ℝ = 0.0
       x: ℝ = 0.0
       f_cut: ℝ = 0.0
       for edge:ℕ(num_edges):
           r = lengths[edge]
           x = r / r_cut
           if x < 1.0:
               f_cut = 1.0 - (p + 1.0) * (p + 2.0) / 2.0 * x ** p + p * (p + 2.0) * x ** (p + 1.0) - p * (p + 1.0) / 2.0 * x ** (p + 2.0)
               for n:ℕ(num_bessel):
                   edge_feats[edge, n] = sqrt(2.0 / r_cut) * sin((n + 1.0) * π * r / r_cut) / r * f_cut
       return edge_feats

.. code-block:: text

   vectors: ℝ[6, 3] = edge_vectors(positions, edge_index)
   lengths: ℝ[6] = edge_lengths(vectors)
   edge_feats: ℝ[6, 8] = radial_embedding(lengths)

Angular features
~~~~~~~~~~~~~~~~

The angular embedding describes the direction of each edge. We use the spherical harmonics :math:`Y_{\ell m}` of the unit vector :math:`\hat{\mathbf r} = (x, y, z)` for :math:`\ell = 0, 1, 2`.

.. math::

   Y_0 = 1, \qquad
   Y_1 = \sqrt{3}\,(x,\, y,\, z)

.. math::

   Y_2 = \left(\sqrt{15}\,xz,\;\; \sqrt{15}\,xy,\;\; \sqrt{5}\left(y^2 - \tfrac{1}{2}(x^2 + z^2)\right),\;\; \sqrt{15}\,yz,\;\; \tfrac{\sqrt{15}}{2}(z^2 - x^2)\right)


Here we use *component* normalisation: the :math:`2\ell + 1` numbers of :math:`Y_\ell` have a squared norm of :math:`2\ell + 1`:

.. math::

   \lVert Y_\ell \rVert^2 = 2\ell + 1

.. code-block:: text

   max_ell: ℝ = 2

   def Y0(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 1]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 1] = zeros(num_edges, 1)
       for edge:ℕ(num_edges):
           results[edge, 0] = 1.0
       return results

   def Y1(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 3]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 3] = zeros(num_edges, 3)
       for edge:ℕ(num_edges):
           x: ℝ, y: ℝ, z: ℝ = vectors[edge] / lengths[edge]
           results[edge] = [sqrt(3.0) * x, sqrt(3.0) * y, sqrt(3.0) * z]
       return results

   def Y2(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 5]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 5] = zeros(num_edges, 5)
       for edge:ℕ(num_edges):
           x: ℝ, y: ℝ, z: ℝ = vectors[edge] / lengths[edge]
           results[edge] = [sqrt(15.0) * x * z, sqrt(15.0) * x * y, sqrt(5.0) * (y * y - 0.5 * (x * x + z * z)), sqrt(15.0) * y * z, sqrt(15.0) / 2.0 * (z * z - x * x)]
       return results

.. code-block:: text

   def spherical_harmonics(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 9]:
       num_edges: ℕ = len(lengths)
       edge_attrs: ℝ[num_edges, 9] = zeros(num_edges, 9)
       edge_attrs[:, 0:1] = Y0(vectors, lengths)
       edge_attrs[:, 1:4] = Y1(vectors, lengths)
       edge_attrs[:, 4:9] = Y2(vectors, lengths)
       return edge_attrs

   edge_attrs: ℝ[6, 9] = spherical_harmonics(vectors, lengths)

   print(edge_feats)
   print(edge_attrs)

Precompute features for forward pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the edge features and the edge attributes do no depend on the weights, we compute them once for every structure, followed by padding zeros to get zero features.

.. code-block:: text

   def batch_edge_feats(batch: ℝ[s, 3, 3]): ℝ[s, 6, 8]:
       num_samples: ℕ = len(batch)
       result: ℝ[num_samples, 6, 8] = zeros(num_samples, 6, 8)
       num_edges: ℕ = 0
       for i:ℕ(num_samples):
           edges = neighbour_search(batch[i], r_cut)
           num_edges = len(edges[0])
           vecs = edge_vectors(batch[i], edges)
           result[i, 0:num_edges, :] = radial_embedding(edge_lengths(vecs))
       return result

   def batch_edge_attrs(batch: ℝ[s, 3, 3]): ℝ[s, 6, 9]:
       num_samples: ℕ = len(batch)
       result: ℝ[num_samples, 6, 9] = zeros(num_samples, 6, 9)
       num_edges: ℕ = 0
       for i:ℕ(num_samples):
           edges = neighbour_search(batch[i], r_cut)
           num_edges = len(edges[0])
           vecs = edge_vectors(batch[i], edges)
           result[i, 0:num_edges, :] = spherical_harmonics(vecs, edge_lengths(vecs))
       return result

   train_edge_feats: ℝ[num_train, 6, 8] = batch_edge_feats(train_positions)
   test_edge_feats: ℝ[num_test, 6, 8] = batch_edge_feats(test_positions)
   train_edge_attrs: ℝ[num_train, 6, 9] = batch_edge_attrs(train_positions)
   test_edge_attrs: ℝ[num_test, 6, 9] = batch_edge_attrs(test_positions)


Step 2: The Interaction Block
-----------------------------

This is where atoms pool and exchange information. The output is a set of features that contain information about the atomic neighbours.
For every atom we build 'messages' that summarise its neighbourhood. The interaction block has four parts:

1. Linear mixing of channels
2. Small network that turns the distances into weights
3. Tensor product that combines the weights and the mixed features and the direction of each edge
4. Sum over the edges of each atom, followed by another linear mixing of channels

.. figure:: /_static/tutorial_files/mace_interaction.png
   :alt: The interaction block of the first layer
   :align: center
   :width: 700px

   Figure 4: The interaction block of the first layer. Figure from the MACE tutorial notebook [MACETutorials]_.

Mixing the channels
~~~~~~~~~~~~~~~~~~~

We start by mixing the :math:`K` channels of every atom with a learned matrix:

.. math::

   \tilde h_{i,k} = \frac{1}{\sqrt{K}} \sum_{k'} W_{k'k}\, h_{i,k'}


.. code-block:: text

   w_up: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def linear_up(node_feats: ℝ[n, k], w: ℝ[k, k]): ℝ[n, k]:
       num_atoms: ℕ = len(node_feats)
       node_feats_up: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats_up[atom, channel] = sum(node_feats[atom] * w[:, channel]) / sqrt(num_channels)
       return node_feats_up

Activation function
~~~~~~~~~~~~~~~~~~~

The network and the readout use the silu activation function.
We use silu (sigmoid linear unit) which is a smoother version of relu which can go negative, thus allowing the network to learn more complex functions. 
It is defined as:

.. math::

   \mathrm{silu}(x) = \frac{x}{1 + e^{-x}}

e3nn rescales silu so that its second moment is one. This is defined by the constant ``silu_norm``.

.. code-block:: text

   silu_norm: ℝ = 1.679176792

   def silu(x: ℝ[e, h]): ℝ[e, h]:
       return silu_norm * x / (1.0 + exp(-x))


Radial network
~~~~~~~~~~~~~~

Next, a small network turns the eight radial features of an edge into the weights that act upon the messages sent along it:

.. math::

   R_{kl}(r_{ji}) = \mathrm{MLP}\big( \{ j_n(r_{ji}) \}_{n=1}^{8} \big)

.. code-block:: text

   radial_hidden: ℝ = 64

   w_r1: ℝ[8, 64] = for a:ℕ(num_bessel) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w_r2: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w_r3: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w_r4: ℝ[64, 24] = for a:ℕ(radial_hidden) -> row: ℝ[24] ~ Normal(0.0, 1.0, 24)

.. code-block:: text

   def radial_mlp(edge_feats: ℝ[e, 8], w1: ℝ[8, 64], w2: ℝ[64, 64], w3: ℝ[64, 64], w4: ℝ[64, 24]): ℝ[e, 24]:
       h1: ℝ[e, 64] = silu(edge_feats @ w1 / sqrt(num_bessel))
       h2: ℝ[e, 64] = silu(h1 @ w2 / sqrt(radial_hidden))
       h3: ℝ[e, 64] = silu(h2 @ w3 / sqrt(radial_hidden))
       return h3 @ w4 / sqrt(radial_hidden)

The one-particle basis
~~~~~~~~~~~~~~~~~~~~~~

Now we combine everything along each edge.
For an edge from atom :math:`j` to atom :math:`i` and channel :math:`k`, the message has three components:

- the radial weight :math:`R_{kl}(r_{ji})`
- the mixed feature :math:`\tilde h_{j,k}`
- the direction of the edge :math:`Y_{l}(\hat{\mathbf r}_{ji})`

The tensor product between tensors of different orders does not transform in a simple way
when we rotate the inputs. So we decompose it into separate spherical
tensors of orders :math:`\ell_0, \ell_1, \ell_2`. 
The numbers involved in the decomposition are called
Clebsch-Gordan coefficients. 
See the TFN tutorial for a detailed explanation of the tensor product decomposition.

.. math::

   m_{ji,k,lm} = R_{kl}(r_{ji}) \,\big( Y_l(\hat{\mathbf r}_{ji}) \otimes \tilde h_{j,k} \big)_{lm}

.. math::

   \mathrm{out}_{m_3} = \sum_{m_1,\, m_2} C_{m_1 m_2 m_3}\; a_{m_1}\, b_{m_2}

.. code-block:: text

   def tensor_product_reduce(cg: ℝ[d1, d2, d3], a: ℝ[d1], b: ℝ[d2], dim1: ℕ, dim2: ℕ, dim3: ℕ): ℝ[d3]:
       results: ℝ[dim3] = zeros(dim3)
       acc: ℝ = 0.0
       for k:ℕ(dim3):
           acc = 0.0
           for i:ℕ(dim1):
               for j:ℕ(dim2):
                   acc += cg[i, j, k] * a[i] * b[j]
           results[k] = acc
       return results

.. code-block:: text

   cg_000: ℝ[1, 1, 1] = [[[1.0]]]
   cg_101: ℝ[3, 1, 3] = [
       [[1.0, 0.0, 0.0]],
       [[0.0, 1.0, 0.0]],
       [[0.0, 0.0, 1.0]]
   ]
   cg_202: ℝ[5, 1, 5] = [
       [[1.0, 0.0, 0.0, 0.0, 0.0]],
       [[0.0, 1.0, 0.0, 0.0, 0.0]],
       [[0.0, 0.0, 1.0, 0.0, 0.0]],
       [[0.0, 0.0, 0.0, 1.0, 0.0]],
       [[0.0, 0.0, 0.0, 0.0, 1.0]]
   ]

.. code-block:: text

   def conv_tp(node_feats_up: ℝ[n, k], edge_attrs: ℝ[e, 9], tp_weights: ℝ[e, 24], edge_index: ℝ[2, e]): ℝ[e, k, 9]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 9] = zeros(num_edges, num_channels, 9)
       h: ℝ[1] = zeros(1)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           for channel:ℕ(num_channels):
               h = [node_feats_up[sender, channel]]
               mji[edge, channel, 0:1] = tp_weights[edge, channel] * tensor_product_reduce(cg_000, edge_attrs[edge, 0:1], h, 1, 1, 1)
               mji[edge, channel, 1:4] = tp_weights[edge, num_channels + channel] * tensor_product_reduce(cg_101, edge_attrs[edge, 1:4], h, 3, 1, 3)
               mji[edge, channel, 4:9] = tp_weights[edge, 2 * num_channels + channel] * tensor_product_reduce(cg_202, edge_attrs[edge, 4:9], h, 5, 1, 5)
       return mji

The first layer the message is basically just a product of three numbers :

.. math::

   \phi_{ji,k,lm} = R_{kl}(r_{ji})\, Y_l^m(\hat{\mathbf r}_{ji})\, \tilde h_{j,k}


Message Passing
~~~~~~~~~~~~~~~

The messages at each atom are added up, and pass through one more linear layer for each :math:`\ell`:

.. math::

   A_{i,k,lm} = \frac{1}{\bar n}\,\frac{1}{\sqrt{K}} \sum_{k'} W^{(l)}_{kk'} \sum_{j \in \mathcal{N}(i)} m_{ji,k',lm}


.. code-block:: text

   avg_num_neighbors: ℝ = 2.0

   w_0: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_1: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_2: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def neighbour_sum(mji: ℝ[e, k, 9], edge_index: ℝ[2, e], num_atoms: ℕ): ℝ[n, k, 9]:
       num_edges: ℕ = len(mji)
       message: ℝ[num_atoms, num_channels, 9] = zeros(num_atoms, num_channels, 9)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           message[receiver] = message[receiver] + mji[edge]
       return message

   def linear(message: ℝ[n, k, 9], w0: ℝ[k, k], w1: ℝ[k, k], w2: ℝ[k, k]): ℝ[n, k, 9]:
       num_atoms: ℕ = len(message)
       out: ℝ[num_atoms, num_channels, 9] = zeros(num_atoms, num_channels, 9)
       for atom:ℕ(num_atoms):
           out[atom, :, 0:1] = w0 @ message[atom, :, 0:1] / sqrt(num_channels)
           out[atom, :, 1:4] = w1 @ message[atom, :, 1:4] / sqrt(num_channels)
           out[atom, :, 4:9] = w2 @ message[atom, :, 4:9] / sqrt(num_channels)
       return out / avg_num_neighbors

The result, ``message`` thus, summarises the whole neighbourhood.

Step 3: The Product
-------------------

The key distinction of MACE from TFNs is the efficient construction of higher-order features from the output of the interaction block. This is achieved by forming tensor products of the features and then
decomposing them into something which transforms in a known way (B i,n,k,l,m). This creates features which contain angular information between the neighbours. 

The message of atom :math:`i` can be written as a sum over its neighbours at a time:

.. math::

   m_i^{(t)} = \sum_j u_1\!\left(\sigma_i^{(t)}; \sigma_j^{(t)}\right) + \sum_{j_1, j_2} u_2\!\left(\sigma_i^{(t)}; \sigma_{j_1}^{(t)}, \sigma_{j_2}^{(t)}\right) + \dots + \sum_{j_1, \dots, j_\nu} u_\nu\!\left(\sigma_i^{(t)}; \sigma_{j_1}^{(t)}, \dots, \sigma_{j_\nu}^{(t)}\right)

Here :math:`t` is the layer and :math:`\sigma_i^{(t)}` is the state of atom :math:`i`. The function :math:`u_\nu` couples :math:`\nu` neighbours at the same time. This is where the many-body information comes from.

.. figure:: /_static/tutorial_files/mace_product.png
   :alt: The product and the update of the node features
   :align: center
   :width: 700px

   Figure 5: The product and the update. The summary :math:`A` is multiplied with itself to give the B-features, which are weighted into :math:`m` and combined with the features of the atom to give the new node features. [MACETutorials]_.

Symmetric products
~~~~~~~~~~~~~~~~~~

The symmetric products are formed channel by channel and are combined through Clebsch-Gordan tables.

The B-features are:

.. math::

   B^{(t)}_{i,\eta_\nu k LM} = \sum_{\mathbf{lm}} \mathcal{C}^{LM}_{\eta_\nu, \mathbf{lm}} \prod_{\xi=1}^{\nu} A^{(t)}_{i,k l_\xi m_\xi}, \qquad \mathbf{lm} = (l_1 m_1, \dots, l_\nu m_\nu)

The index :math:`\eta_\nu` labels the different ways of coupling the :math:`\nu` factors as 'paths'. 

A 'path' is one particular way of combining the pieces of :math:`A` ( the sumamry of the atom's neighbourhood)
into a result of a certain order :math:`L`. 
For example, a scalar (:math:`L = 0`) can be made from the scalar component :math:`A_0` on its own,
or from the dot product of two vectors, :math:`A_1 \cdot A_1` or :math:`A_2 \cdot A_2`. 
This gives us four paths for the order :math:`L = 0`. 

The Clebsch-Gordan tables that are generated from the e3nn library:

.. code-block:: text

   cg_110: ℝ[3, 3, 1] = [
       [[0.577350], [0.0], [0.0]],
       [[0.0], [0.577350], [0.0]],
       [[0.0], [0.0], [0.577350]]
   ]
   cg_220: ℝ[5, 5, 1] = [
       [[0.447214], [0.0], [0.0], [0.0], [0.0]],
       [[0.0], [0.447214], [0.0], [0.0], [0.0]],
       [[0.0], [0.0], [0.447214], [0.0], [0.0]],
       [[0.0], [0.0], [0.0], [0.447214], [0.0]],
       [[0.0], [0.0], [0.0], [0.0], [0.447214]]
   ]
   cg_011: ℝ[1, 3, 3] = [
       [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
   ]
   cg_121: ℝ[3, 5, 3] = [
       [[0.0, 0.0, 0.547723], [0.0, 0.547723, 0.0], [-0.316228, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.547723, 0.0, 0.0]],
       [[0.0, 0.0, 0.0], [0.547723, 0.0, 0.0], [0.0, 0.632456, 0.0], [0.0, 0.0, 0.547723], [0.0, 0.0, 0.0]],
       [[0.547723, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.316228], [0.0, 0.547723, 0.0], [0.0, 0.0, 0.547723]]
   ]

We define one function for each output order. 

.. code-block:: text

   def B_features_L0(A: ℝ[n, k, 9]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(A)
       B0: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       a0: ℝ[1] = zeros(1)
       a1: ℝ[3] = zeros(3)
       a2: ℝ[5] = zeros(5)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               a0, a1, a2 = A[atom, channel, 0:1], A[atom, channel, 1:4], A[atom, channel, 4:9]
               B0[atom, channel, 0:1] = a0
               B0[atom, channel, 1:2] = tensor_product_reduce(cg_000, a0, a0, 1, 1, 1)
               B0[atom, channel, 2:3] = tensor_product_reduce(cg_110, a1, a1, 3, 3, 1)
               B0[atom, channel, 3:4] = tensor_product_reduce(cg_220, a2, a2, 5, 5, 1)
       return B0

.. code-block:: text

   def B_features_L1(A: ℝ[n, k, 9]): ℝ[n, k, 3, 3]:
       num_atoms: ℕ = len(A)
       B1: ℝ[num_atoms, num_channels, 3, 3] = zeros(num_atoms, num_channels, 3, 3)
       a0: ℝ[1] = zeros(1)
       a1: ℝ[3] = zeros(3)
       a2: ℝ[5] = zeros(5)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               a0, a1, a2 = A[atom, channel, 0:1], A[atom, channel, 1:4], A[atom, channel, 4:9]
               B1[atom, channel, 0] = a1
               B1[atom, channel, 1] = tensor_product_reduce(cg_011, a0, a1, 1, 3, 3)
               B1[atom, channel, 2] = tensor_product_reduce(cg_121, a1, a2, 3, 5, 3)
       return B1

Weighting the paths
~~~~~~~~~~~~~~~~~~~

The B-features are combined with learned weights, one weight per path and channel.

.. math::

   m_{i,k,LM} = \sum_{\text{paths}} W^{L}_{z_i,\,\text{path},\,k}\; B_{i,k,\text{path},LM}

The sum over paths is written as a sum over :math:`\nu` and a sum over :math:`\eta_\nu`:

.. math::

   m^{(t)}_{i,kLM} = \sum_{\nu} \sum_{\eta_\nu} W^{(t)}_{z_i kL,\, \eta_\nu}\, B^{(t)}_{i,\eta_\nu kLM}

The elements are selected through their one-hot rows from the embeddings matrix containing ``node_attrs``.

.. code-block:: text

   w_prod0: ℝ[2, 4, 8] = for z:ℕ(num_elements) -> for q:ℕ(4) -> row: ℝ[8] ~ Normal(0.0, 0.25, 8)
   w_prod1: ℝ[2, 3, 8] = for z:ℕ(num_elements) -> for q:ℕ(3) -> row: ℝ[8] ~ Normal(0.0, 0.333, 8)

   def weighted_sum(B0: ℝ[n, k, 4], B1: ℝ[n, k, 3, 3], node_attrs: ℝ[n, s], wp0: ℝ[2, 4, 8], wp1: ℝ[2, 3, 8]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(B0)
       m: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       w0: ℝ[4, 8] = zeros(4, 8)
       w1: ℝ[3, 8] = zeros(3, 8)
       for atom:ℕ(num_atoms):
           w0 = zeros(4, 8)
           w1 = zeros(3, 8)
           for z:ℕ(num_elements):
               w0 = w0 + node_attrs[atom, z] * wp0[z]
               w1 = w1 + node_attrs[atom, z] * wp1[z]
           for channel:ℕ(num_channels):
               m[atom, channel, 0:1] = [sum(w0[:, channel] * B0[atom, channel])]
               m[atom, channel, 1:4] = w1[:, channel] @ B1[atom, channel]
       return m

Updating the node features
~~~~~~~~~~~~~~~~~~~~~~~~~~

The new node features are a linear function of :math:`m` with an additional skip connection that helps in carryng the initial features of the node forward:

.. math::

   h^{(t+1)}_{i,kLM} = \sum_{\tilde k} W^{(t)}_{kL,\tilde k}\, m^{(t)}_{i,\tilde k LM} + \sum_{\tilde k} W^{(t)}_{k z_i L,\tilde k}\, h^{(t)}_{i,\tilde k LM}

In our first layer (:math:`t = 0`) the initial nodefeatures are scalars,

.. math::

   h^{(1)}_{i,k,L} = \frac{1}{\sqrt{K}} \sum_{k'} W^{L}_{kk'}\, m_{i,k',L} \;+\; \delta_{L0}\, s_{i,k}

The skip connection term is:

.. math::

   s_{i,k} = \frac{1}{\sqrt{K Z}} \sum_{k',\,z} W_{k'zk}\; h^{(0)}_{i,k'}\; \delta_{z z_i}

It uses atomic species-dependent weights and is built from the scalar starting features, so it only adds to the :math:`L = 0` part.

.. code-block:: text

   w_sc: ℝ[8, 2, 8] = for c:ℕ(num_channels) -> for z:ℕ(num_elements) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_p0: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_p1: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def skip_tp(node_feats: ℝ[n, k], node_attrs: ℝ[n, s], w: ℝ[8, 2, 8]): ℝ[n, k]:
       num_atoms: ℕ = len(node_feats)
       sc: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       acc: ℝ = 0.0
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               acc = 0.0
               for z:ℕ(num_elements):
                   acc += node_attrs[atom, z] * sum(node_feats[atom] * w[:, z, channel])
               sc[atom, channel] = acc / sqrt(num_channels * num_elements)
       return sc

   def node_update(m: ℝ[n, k, 4], sc: ℝ[n, k], wp0: ℝ[k, k], wp1: ℝ[k, k]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(m)
       node_feats1: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats1[atom, channel, 0] = sum(wp0[channel] * m[atom, :, 0]) / sqrt(num_channels) + sc[atom, channel]
           node_feats1[atom, :, 1:4] = wp1 @ m[atom, :, 1:4] / sqrt(num_channels)
       return node_feats1

The output ``node_feats1`` has four numbers per channel, one scalar and one vector. These are the features that will be used by the readout layer and the second layer.

Step 4: The Readout
-------------------

Now, we can take the newly computed node features and compute the actual energy by passing them through a readout layer.
The readout is a single linear layer on the scalar part of the features:

.. math::

   E_i = \frac{1}{\sqrt{K}} \sum_{k} W_k\, h^{(1)}_{i,k,\ell = 0}, \qquad E = \sum_i E_i

Since energy is an invariant scalar quantity, the vector part of the node features will not by used, they will be important in the second layer.
The total energy is simply the sum of the individual atomic contriobutions.

.. figure:: /_static/tutorial_files/mace_readout.png
   :alt: The readout maps the node features to node energies
   :align: center
   :width: 600px

   Figure 6: The readout layer [MACETutorials]_.

.. code-block:: text

   w_readout: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def readout(node_feats: ℝ[n, k, 4], w: ℝ[8]): ℝ[n]:
       num_atoms: ℕ = len(node_feats)
       node_energies: ℝ[num_atoms] = zeros(num_atoms)
       for atom:ℕ(num_atoms):
           node_energies[atom] = sum(w * node_feats[atom, :, 0]) / sqrt(num_channels)
       return node_energies

Step 5: Repeating the Layer
---------------------------

The second layer repeats Steps 2 to 4 while taking the output of the first layer as its input.
The second layer makes MACE more expressive in two ways:

- Each node now recieves messages from neighbours which already summarise their own neighbourhood.
- The new features also contain vectors, so the messages now contain directional information about the sender's neighbourhood.

In the second layer the sender carries scalars and vectors, making the interaction step significantly more complex.

.. figure:: /_static/tutorial_files/mace_interaction_second_layer.png
   :alt: The interaction block of a general layer s
   :align: center
   :width: 700px

   Figure 7: The interaction block of the second layer which carries :math:`l_0`, :math:`l_1` and :math:`l_2` features as inputs. [MACETutorials]_.


Mixing the channels (second layer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the input contains scalars and vectors, we mix the channels using one matrix for each:
For a general layer :math:`s` there is one matrix for each :math:`\ell`:

.. math::

   \tilde h^{(s)}_{i,k l_2 m_2} = \sum_{\tilde k} W^{(s)}_{k \tilde k l_2}\, h^{(s)}_{i,\tilde k l_2 m_2}


.. code-block:: text

   w2_up0: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w2_up1: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def linear_up2(node_feats: ℝ[n, k, 4], w0: ℝ[8, 8], w1: ℝ[8, 8]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(node_feats)
       node_feats_up: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats_up[atom, channel, 0] = sum(node_feats[atom, :, 0] * w0[:, channel]) / sqrt(num_channels)
               node_feats_up[atom, channel, 1:4] = w1[:, channel] @ node_feats[atom, :, 1:4] / sqrt(num_channels)
       return node_feats_up

Radial weights
~~~~~~~~~~~~~~

The radial network has the same shape as before, but its own weights and twice the outputs (56 outputs instead of 24), one for each of the 7 paths and each of the 8 channels.

Similar to the first layer, the output of an MLP of the radial features are used as weights. This allows us to use one weight for every channel and every path :math:`\eta_1`, which couples the orders :math:`l_0`, :math:`l_1` and :math:`l_2`:

.. math::

   R^{(s)}_{k \eta_1 l_1 l_2 l_3}(r_{ji}) = \mathrm{MLP}\big( \{ j_n(r_{ji}) \}_{n=1}^{8} \big)

.. code-block:: text

   w2_r1: ℝ[8, 64] = for a:ℕ(num_bessel) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w2_r2: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w2_r3: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w2_r4: ℝ[64, 56] = for a:ℕ(radial_hidden) -> row: ℝ[56] ~ Normal(0.0, 1.0, 56)

   def radial_mlp2(edge_feats: ℝ[e, 8], w1: ℝ[8, 64], w2: ℝ[64, 64], w3: ℝ[64, 64], w4: ℝ[64, 56]): ℝ[e, 56]:
       h1: ℝ[e, 64] = silu(edge_feats @ w1 / sqrt(num_bessel))
       h2: ℝ[e, 64] = silu(h1 @ w2 / sqrt(radial_hidden))
       h3: ℝ[e, 64] = silu(h2 @ w3 / sqrt(radial_hidden))
       return h3 @ w4 / sqrt(radial_hidden)

One-particle basis (7 paths)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An edge now combines the new features of the sender with the direction :math:`Y_l` of the edge.
This allows for exactly seven paths that end up utilizing (:math:`\ell = 0, 1, 2`).

In formulas, the message of layer :math:`s` is a sum over all allowed paths of order :math:`l_0`, :math:`l_1` and :math:`l_2`, 

.. math::

   \phi^{(s)}_{ji,k l_3 m_3} = \sum_{l_2 m_2} \sum_{l_1 m_1} c^{l_1 m_1 l_2 m_2}_{l_3 m_3}\, R^{(s)}_{k l_1 l_2 l_3}(r_{ji})\, Y_{l_1}^{m_1}(\hat{\mathbf r}_{ji})\, \tilde h^{(s)}_{j,k l_2 m_2}

They are listed in the table below:

.. list-table::
   :header-rows: 1
   :widths: 8 22 70

   * - Path
     - Sender :math:`\times` edge :math:`\to` out
     - What it computes
   * - 0
     - 0 × 0 → 0
     - the scalar of the sender, weighted by :math:`l_0` feature of the edge 
   * - 1
     - 1 × 1 → 0
     - the dot product of the sender's vector with the unit vector of the edge
   * - 2
     - 0 × 1 → 1
     - the scalar of the sender times the direction of the edge
   * - 3
     - 1 × 0 → 1
     - the vector of the sender, weighted by :math:`l_0` feature of the edge 
   * - 4
     - 1 × 2 → 1
     - the quadrupole(:math:`l_2`) of the edge acting on the vector of the sender
   * - 5
     - 0 × 2 → 2
     - the scalar of the sender times the quadrupole(:math:`l_2`) of the edge
   * - 6
     - 1 × 1 → 2
     - the traceless outer product of the sender's vector and the edge direction 


Let's define the tables for the two new paths from e3nn, ``cg_022`` and ``cg_112``.

.. code-block:: text

   cg_022: ℝ[1, 5, 5] = [
       [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]
   ]
   cg_112: ℝ[3, 3, 5] = [
       [[0.0, 0.0, -0.408248, 0.0, -0.707107], [0.0, 0.707107, 0.0, 0.0, 0.0], [0.707107, 0.0, 0.0, 0.0, 0.0]],
       [[0.0, 0.707107, 0.0, 0.0, 0.0], [0.0, 0.0, 0.816497, 0.0, 0.0], [0.0, 0.0, 0.0, 0.707107, 0.0]],
       [[0.707107, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.707107, 0.0], [0.0, 0.0, -0.408248, 0.0, 0.707107]]
   ]

.. code-block:: text

   def conv_tp2(node_feats_up: ℝ[n, k, 4], edge_attrs: ℝ[e, 9], tp_weights: ℝ[e, 56], edge_index: ℝ[2, e]): ℝ[e, k, 21]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 21] = zeros(num_edges, num_channels, 21)
       h0: ℝ[1] = zeros(1)
       h1: ℝ[3] = zeros(3)
       y0: ℝ[1] = zeros(1)
       y1: ℝ[3] = zeros(3)
       y2: ℝ[5] = zeros(5)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           y0, y1, y2 = edge_attrs[edge, 0:1], edge_attrs[edge, 1:4], edge_attrs[edge, 4:9]
           for channel:ℕ(num_channels):
               h0, h1 = [node_feats_up[sender, channel, 0]], node_feats_up[sender, channel, 1:4]
               mji[edge, channel, 0:1] = tp_weights[edge, channel] * tensor_product_reduce(cg_000, h0, y0, 1, 1, 1)
               mji[edge, channel, 1:2] = tp_weights[edge, num_channels + channel] * tensor_product_reduce(cg_110, h1, y1, 3, 3, 1)
               mji[edge, channel, 2:5] = tp_weights[edge, 2 * num_channels + channel] * tensor_product_reduce(cg_011, h0, y1, 1, 3, 3)
               mji[edge, channel, 5:8] = tp_weights[edge, 3 * num_channels + channel] * tensor_product_reduce(cg_101, h1, y0, 3, 1, 3)
               mji[edge, channel, 8:11] = tp_weights[edge, 4 * num_channels + channel] * tensor_product_reduce(cg_121, h1, y2, 3, 5, 3)
               mji[edge, channel, 11:16] = tp_weights[edge, 5 * num_channels + channel] * tensor_product_reduce(cg_022, h0, y2, 1, 5, 5)
               mji[edge, channel, 16:21] = tp_weights[edge, 6 * num_channels + channel] * tensor_product_reduce(cg_112, h1, y1, 3, 3, 5)
       return mji

Message Passing (second layer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the same logic as in the first layer, but now the messages are 21-dimensional instead of 9-dimensional. 
The sum over the neighbours and the linear mixing are given by:

.. math::

   A^{(s)}_{i,k l_3 m_3} = \sum_{\tilde k, \eta_1} W^{(s)}_{k \tilde k \eta_1 l_3} \sum_{j \in \mathcal{N}(i)} \phi^{(s)}_{ji,\tilde k \eta_1 l_3 m_3}

.. code-block:: text

   w2_msg: ℝ[7, 8, 8] = for p:ℕ(7) -> for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def neighbour_sum2(mji: ℝ[e, k, 21], edge_index: ℝ[2, e], num_atoms: ℕ): ℝ[n, k, 21]:
       num_edges: ℕ = len(mji)
       summed: ℝ[num_atoms, num_channels, 21] = zeros(num_atoms, num_channels, 21)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           summed[receiver] = summed[receiver] + mji[edge]
       return summed

   def linear2(summed: ℝ[n, k, 21], w: ℝ[7, k, k]): ℝ[n, k, 9]:
       num_atoms: ℕ = len(summed)
       message: ℝ[num_atoms, num_channels, 9] = zeros(num_atoms, num_channels, 9)
       for atom:ℕ(num_atoms):
           message[atom, :, 0:1] = (w[0] @ summed[atom, :, 0:1] + w[1] @ summed[atom, :, 1:2]) / sqrt(2.0 * num_channels)
           message[atom, :, 1:4] = (w[2] @ summed[atom, :, 2:5] + w[3] @ summed[atom, :, 5:8] + w[4] @ summed[atom, :, 8:11]) / sqrt(3.0 * num_channels)
           message[atom, :, 4:9] = (w[5] @ summed[atom, :, 11:16] + w[6] @ summed[atom, :, 16:21]) / sqrt(2.0 * num_channels)
       return message / avg_num_neighbors

Product and update
~~~~~~~~~~~~~~~~~~

The second layer is the last one, since it only needs to produce scalars, we remove the vector features for the vector update and the skip conenction keeping only scalars.

.. code-block:: text

   w2_prod0: ℝ[2, 4, 8] = for z:ℕ(num_elements) -> for q:ℕ(4) -> row: ℝ[8] ~ Normal(0.0, 0.25, 8)

   def weighted_sum2(B0: ℝ[n, k, 4], node_attrs: ℝ[n, s], wp: ℝ[2, 4, 8]): ℝ[n, k]:
       num_atoms: ℕ = len(B0)
       m: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       w0: ℝ[4, 8] = zeros(4, 8)
       for atom:ℕ(num_atoms):
           w0 = zeros(4, 8)
           for z:ℕ(num_elements):
               w0 = w0 + node_attrs[atom, z] * wp[z]
           for channel:ℕ(num_channels):
               m[atom, channel] = sum(w0[:, channel] * B0[atom, channel])
       return m

.. code-block:: text

   w2_sc: ℝ[8, 2, 8] = for c:ℕ(num_channels) -> for z:ℕ(num_elements) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w2_p: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def node_update2(m: ℝ[n, k], sc: ℝ[n, k], wp: ℝ[k, k]): ℝ[n, k]:
       num_atoms: ℕ = len(m)
       node_feats: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats[atom, channel] = sum(wp[channel] * m[atom]) / sqrt(num_channels) + sc[atom, channel]
       return node_feats

Readout and total energy
~~~~~~~~~~~~~~~~~~~~~~~~

The readout of the last layer uses a small network instead of a single linear layer. It uses a hidden layer of 16 neurons and the SilU activation function to output a single number per atom. :

.. math::

   E^{(2)}_i = \frac{1}{\sqrt{16}} \sum_{u=1}^{16} V_u\; \mathrm{silu}\!\left( \frac{1}{\sqrt{K}} \sum_{k} U_{ku}\, h^{(2)}_{i,k} \right)

The model's final energy is the sum of the contributions from both layers:

.. math::

   E = \sum_i \big( E^{(1)}_i + E^{(2)}_i \big)

.. code-block:: text

   w2_ro1: ℝ[8, 16] = for a:ℕ(num_channels) -> row: ℝ[16] ~ Normal(0.0, 1.0, 16)
   w2_ro2: ℝ[16] ~ Normal(0.0, 1.0, 16)

   def readout2(node_feats: ℝ[n, k], w1: ℝ[k, 16], w2: ℝ[16]): ℝ[n]:
       num_atoms: ℕ = len(node_feats)
       hidden: ℝ[num_atoms, 16] = silu(node_feats @ w1 / sqrt(num_channels))
       node_energies: ℝ[num_atoms] = zeros(num_atoms)
       for atom:ℕ(num_atoms):
           node_energies[atom] = sum(hidden[atom] * w2) / sqrt(16.0)
       return node_energies


Forward Pass
------------

The forward pass defines how data flows through the complete model.

``node_attrs, edge_index, edge_feats, edge_attrs -> node_embedding -> linear_up -> radial_mlp -> conv_tp -> neighbour_sum, linear -> B_features -> weighted_sum -> update -> readout -> second layer -> energy``

.. code-block:: text

       def λ(node_attrs: ℝ[3, 2], edge_index: ℝ[2, 6], edge_feats: ℝ[6, 8], edge_attrs: ℝ[6, 9]) → ℝ:
           node_feats0: ℝ[3, 8] = node_embedding(node_attrs, this.w_embed)
           node_feats_up: ℝ[3, 8] = linear_up(node_feats0, this.w_up)
           tp_weights: ℝ[6, 24] = radial_mlp(edge_feats, this.w_r1, this.w_r2, this.w_r3, this.w_r4)
           mji: ℝ[6, 8, 9] = conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
           message: ℝ[3, 8, 9] = linear(neighbour_sum(mji, edge_index, 3), this.w_0, this.w_1, this.w_2)
           B0: ℝ[3, 8, 4] = B_features_L0(message)
           B1: ℝ[3, 8, 3, 3] = B_features_L1(message)
           m: ℝ[3, 8, 4] = weighted_sum(B0, B1, node_attrs, this.w_prod0, this.w_prod1)
           sc: ℝ[3, 8] = skip_tp(node_feats0, node_attrs, this.w_sc)
           node_feats1: ℝ[3, 8, 4] = node_update(m, sc, this.w_p0, this.w_p1)
           node_energies: ℝ[3] = readout(node_feats1, this.w_readout)
           node_feats_up2: ℝ[3, 8, 4] = linear_up2(node_feats1, this.w2_up0, this.w2_up1)
           tp_weights2: ℝ[6, 56] = radial_mlp2(edge_feats, this.w2_r1, this.w2_r2, this.w2_r3, this.w2_r4)
           mji2: ℝ[6, 8, 21] = conv_tp2(node_feats_up2, edge_attrs, tp_weights2, edge_index)
           message2: ℝ[3, 8, 9] = linear2(neighbour_sum2(mji2, edge_index, 3), this.w2_msg)
           B0_2: ℝ[3, 8, 4] = B_features_L0(message2)
           m2: ℝ[3, 8] = weighted_sum2(B0_2, node_attrs, this.w2_prod0)
           sc2: ℝ[3, 8] = skip_tp(node_feats1[:, :, 0], node_attrs, this.w2_sc)
           node_feats2: ℝ[3, 8] = node_update2(m2, sc2, this.w2_p)
           node_energies2: ℝ[3] = readout2(node_feats2, this.w2_ro1, this.w2_ro2)
           energy: ℝ = sum(node_energies) + sum(node_energies2)
           return energy


Define Loss
-----------

For training we use the squared error of the scaled energy of one structure:

.. math::

   \mathcal{L} = \left( \hat E_\mathrm{pred} - \hat E_\mathrm{target} \right)^2

where:

- :math:`\hat E_\mathrm{pred}` is the total energy that the model predicts
- :math:`\hat E_\mathrm{target}` is the computed,scaled GFN2-xTB energy of the structure

.. code-block:: text

   def mse(pred: ℝ, target: ℝ): ℝ:
       diff: ℝ = pred - target
       result: ℝ = diff * diff
       return result

Model Definition
----------------

Let us now define the model class.

.. code-block:: text

   class MACEModel(w_embed: ℝ[2, 8], w_up: ℝ[8, 8], w_r1: ℝ[8, 64], w_r2: ℝ[64, 64], w_r3: ℝ[64, 64], w_r4: ℝ[64, 24],
                   w_0: ℝ[8, 8], w_1: ℝ[8, 8], w_2: ℝ[8, 8], w_prod0: ℝ[2, 4, 8], w_prod1: ℝ[2, 3, 8],
                   w_sc: ℝ[8, 2, 8], w_p0: ℝ[8, 8], w_p1: ℝ[8, 8], w_readout: ℝ[8],
                   w2_up0: ℝ[8, 8], w2_up1: ℝ[8, 8], w2_r1: ℝ[8, 64], w2_r2: ℝ[64, 64], w2_r3: ℝ[64, 64], w2_r4: ℝ[64, 56],
                   w2_msg: ℝ[7, 8, 8], w2_prod0: ℝ[2, 4, 8], w2_sc: ℝ[8, 2, 8], w2_p: ℝ[8, 8], w2_ro1: ℝ[8, 16], w2_ro2: ℝ[16]):

       def λ(node_attrs: ℝ[3, 2], edge_index: ℝ[2, 6], edge_feats: ℝ[6, 8], edge_attrs: ℝ[6, 9]) → ℝ:
           node_feats0: ℝ[3, 8] = node_embedding(node_attrs, this.w_embed)
           node_feats_up: ℝ[3, 8] = linear_up(node_feats0, this.w_up)
           tp_weights: ℝ[6, 24] = radial_mlp(edge_feats, this.w_r1, this.w_r2, this.w_r3, this.w_r4)
           mji: ℝ[6, 8, 9] = conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
           message: ℝ[3, 8, 9] = linear(neighbour_sum(mji, edge_index, 3), this.w_0, this.w_1, this.w_2)
           B0: ℝ[3, 8, 4] = B_features_L0(message)
           B1: ℝ[3, 8, 3, 3] = B_features_L1(message)
           m: ℝ[3, 8, 4] = weighted_sum(B0, B1, node_attrs, this.w_prod0, this.w_prod1)
           sc: ℝ[3, 8] = skip_tp(node_feats0, node_attrs, this.w_sc)
           node_feats1: ℝ[3, 8, 4] = node_update(m, sc, this.w_p0, this.w_p1)
           node_energies: ℝ[3] = readout(node_feats1, this.w_readout)
           node_feats_up2: ℝ[3, 8, 4] = linear_up2(node_feats1, this.w2_up0, this.w2_up1)
           tp_weights2: ℝ[6, 56] = radial_mlp2(edge_feats, this.w2_r1, this.w2_r2, this.w2_r3, this.w2_r4)
           mji2: ℝ[6, 8, 21] = conv_tp2(node_feats_up2, edge_attrs, tp_weights2, edge_index)
           message2: ℝ[3, 8, 9] = linear2(neighbour_sum2(mji2, edge_index, 3), this.w2_msg)
           B0_2: ℝ[3, 8, 4] = B_features_L0(message2)
           m2: ℝ[3, 8] = weighted_sum2(B0_2, node_attrs, this.w2_prod0)
           sc2: ℝ[3, 8] = skip_tp(node_feats1[:, :, 0], node_attrs, this.w2_sc)
           node_feats2: ℝ[3, 8] = node_update2(m2, sc2, this.w2_p)
           node_energies2: ℝ[3] = readout2(node_feats2, this.w2_ro1, this.w2_ro2)
           energy: ℝ = sum(node_energies) + sum(node_energies2)
           return energy

       def loss_sample(sample: ℕ) → ℝ:
           pred: ℝ = this(node_attrs, train_edge_index[sample], train_edge_feats[sample], train_edge_attrs[sample])
           target: ℝ = (train_energies[sample] - energy_mean) / energy_std
           result: ℝ = mse(pred, target)
           return result

       def error_sample(sample: ℕ) → ℝ:
           scaled: ℝ = this(node_attrs, test_edge_index[sample], test_edge_feats[sample], test_edge_attrs[sample])
           result: ℝ = energy_mean + energy_std * scaled - test_energies[sample]
           return result

       def train(epochs: ℕ, lr: ℝ) → ℝ:
           last_loss: ℝ = 0
           current_loss: ℝ = 0
           for epoch:ℕ(epochs):
               for sample:ℕ(num_train):
                   for rep:ℕ(1):
                       current_loss = this.loss_sample(sample)
                       learnable_grads = grad(current_loss, this.learnable_params)
                       this.update(lr, learnable_grads)
                       last_loss = current_loss
           return last_loss
           
       def evaluate() → ℝ:
           total_error: ℝ = 0
           current_error: ℝ = 0
           for sample:ℕ(num_test):
               for rep:ℕ(1):
                   current_error = this.error_sample(sample)
                   total_error = total_error + current_error * current_error
           result: ℝ = sqrt(total_error / num_test)
           return result


Training the Model
------------------

We train the network using stochastic gradient descent (SGD), with one structure for every step.

.. math::

   \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}

where:

- :math:`\theta` represents the 27 weights of the model
- :math:`\eta` is the learning rate
- :math:`\nabla_\theta \mathcal{L}` is the gradient of the loss of one structure

.. code-block:: text

       def train(epochs: ℕ, lr: ℝ) → ℝ:
           last_loss: ℝ = 0
           current_loss: ℝ = 0
           for epoch:ℕ(epochs):
               for sample:ℕ(num_train):
                   for rep:ℕ(1):
                       current_loss = this.loss_sample(sample)
                       learnable_grads = grad(current_loss, this.learnable_params)
                       this.update(lr, learnable_grads)
                       last_loss = current_loss
           return last_loss

We train for 100 epochs with a learning rate of 0.05.

.. code-block:: text

   mace_object: MACEModel = MACEModel(w_embed, w_up, w_r1, w_r2, w_r3, w_r4,
                                      w_0, w_1, w_2, w_prod0, w_prod1,
                                      w_sc, w_p0, w_p1, w_readout,
                                      w2_up0, w2_up1, w2_r1, w2_r2, w2_r3, w2_r4,
                                      w2_msg, w2_prod0, w2_sc, w2_p, w2_ro1, w2_ro2)

   example_energy: ℝ = mace_object(node_attrs, edge_index, edge_feats, edge_attrs)
   print(example_energy)

   lr: ℝ = 0.05
   epochs: ℕ = 100

   rmse_before: ℝ = mace_object.evaluate()
   print(rmse_before)

   final_loss: ℝ = mace_object.train(epochs, lr)
   print(final_loss)

   rmse_after: ℝ = mace_object.evaluate()
   print(rmse_after)

Evaluating the Model
--------------------

``evaluate`` returns the RMSE in eV over the test structures. 

.. code-block:: text

       def error_sample(sample: ℕ) → ℝ:
           scaled: ℝ = this(node_attrs, test_edge_index[sample], test_edge_feats[sample], test_edge_attrs[sample])
           result: ℝ = energy_mean + energy_std * scaled - test_energies[sample]
           return result

       def evaluate() → ℝ:
           total_error: ℝ = 0
           current_error: ℝ = 0
           for sample:ℕ(num_test):
               for rep:ℕ(1):
                   current_error = this.error_sample(sample)
                   total_error = total_error + current_error * current_error
           result: ℝ = sqrt(total_error / num_test)
           return result

The predicted energies of the test set are collected in ``test_predictions``:

.. code-block:: text

   test_predictions: ℝ[num_test] = zeros(num_test)
   for sample:ℕ(num_test):
       test_predictions[sample] = test_energies[sample] + mace_object.error_sample(sample)
   print(test_predictions)


Visualizing the Loss Curve
--------------------------

.. code-block:: text

   loss_history: ℝ[epochs] = zeros(epochs)
   one_epoch: ℕ = 1
   current_loss: ℝ = 0
   for epoch:ℕ(epochs):
       current_loss = mace_object.train(one_epoch, lr)
       loss_history[epoch] = current_loss

   plot_training_loss(loss_history)

.. note::
   ``plot_training_loss`` is not a built-in Physika function. To use it,
   add the following helper to ``physika/runtime.py``:

   .. code-block:: python

        def plot_training_loss(loss_history):
            import matplotlib.pyplot as plt

            plt.plot(loss_history.detach().cpu().numpy())

            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.yscale("log")
            plt.title("Training Loss")
            plt.show()

.. figure:: /_static/tutorial_files/mace_loss_curve.png
   :alt: Training loss of MACE on water
   :align: center
   :width: 450px

   Figure 8: Training loss after every epoch, for the first 6 epochs.


Full Code
---------

.. code-block:: text

   # MACE: Higher-Order Equivariant Message Passing

   # Dataset
   num_train: ℕ = 8
   num_test: ℕ = 4

   train_positions: ℝ[num_train, 3, 3] = [
       [[0.0000, 0.0000, 0.0000], [1.3035, 0.1210, -0.0361], [0.0387, 1.0143, -0.0409]],
       [[0.0000, 0.0000, 0.0000], [0.5566, 0.1041, -1.2624], [-0.4404, 0.7846, 0.8458]],
       [[0.0000, 0.0000, 0.0000], [-0.9748, 0.9579, 0.2162], [0.8859, 0.0659, -0.3090]],
       [[0.0000, 0.0000, 0.0000], [-0.6634, 0.3849, -0.4699], [-0.2285, -0.3502, 1.2694]],
       [[0.0000, 0.0000, 0.0000], [-0.8672, -0.2552, -1.0005], [-0.2882, 0.8669, 0.9533]],
       [[0.0000, 0.0000, 0.0000], [0.9919, -0.2859, -0.1822], [-0.5818, 1.0325, -0.0912]],
       [[0.0000, 0.0000, 0.0000], [1.0513, -0.3912, 0.6070], [0.2607, 0.9788, -0.3458]],
       [[0.0000, 0.0000, 0.0000], [1.1268, -0.4287, -0.8342], [0.0260, 1.3507, -0.4679]]
   ]
   train_energies: ℝ[num_train] = [-12.4103, -10.7186, -11.8415, -12.2896, -10.8181, -12.6558, -12.6144, -9.8413]

   test_positions: ℝ[num_test, 3, 3] = [
       [[0.0000, 0.0000, 0.0000], [1.1014, 0.3095, 0.1934], [-0.4569, 1.3549, -0.1436]],
       [[0.0000, 0.0000, 0.0000], [0.6110, -0.8965, -0.6491], [0.5299, 0.6870, 0.5715]],
       [[0.0000, 0.0000, 0.0000], [1.2824, 0.0804, -0.2181], [-0.5724, 1.4111, -0.2409]],
       [[0.0000, 0.0000, 0.0000], [-0.6127, 0.4966, 0.8616], [0.4113, -1.2876, 0.3875]]
   ]
   test_energies: ℝ[num_test] = [-11.3831, -12.5956, -10.0866, -11.4604]

   # Scale and Shift
   energy_mean: ℝ = sum(train_energies) / num_train
   energy_std: ℝ = sqrt(sum((train_energies - energy_mean) * (train_energies - energy_mean)) / num_train)

   # Step 0: The Molecule as a Graph
   r_cut: ℝ = 2.0
   atomic_numbers: ℝ[3] = [8.0, 1.0, 1.0]
   z_table: ℝ[2] = [1.0, 8.0]

   def one_hot(numbers: ℝ[n], table: ℝ[s]): ℝ[n, s]:
       num_atoms: ℕ = len(numbers)
       num_species: ℕ = len(table)
       encoding: ℝ[num_atoms, num_species] = zeros(num_atoms, num_species)
       for atom:ℕ(num_atoms):
           for species:ℕ(num_species):
               if numbers[atom] == table[species]:
                   encoding[atom, species] = 1.0
       return encoding

   def neighbour_search(positions: ℝ[n, 3], cutoff: ℝ): ℝ[2, m]:
       num_atoms: ℕ = len(positions)
       vector: ℝ[3] = zeros(3)
       distance: ℝ = 0.0
       num_edges: ℝ = 0
       for sender:ℕ(num_atoms):
           for receiver:ℕ(num_atoms):
               vector = positions[receiver] - positions[sender]
               distance = sqrt(sum(vector * vector))
               if distance > 0.0:
                   if distance < cutoff:
                       num_edges += 1
       edge_index: ℝ[2, num_edges] = zeros(2, num_edges)
       edge: ℝ = 0
       for sender:ℕ(num_atoms):
           for receiver:ℕ(num_atoms):
               vector = positions[receiver] - positions[sender]
               distance = sqrt(sum(vector * vector))
               if distance > 0.0:
                   if distance < cutoff:
                       edge_index[:, edge] = [sender, receiver]
                       edge += 1
       return edge_index

   positions: ℝ[3, 3] = train_positions[0]
   node_attrs: ℝ[3, 2] = one_hot(atomic_numbers, z_table)
   edge_index: ℝ[2, 6] = neighbour_search(positions, r_cut)

   print(positions)
   print(node_attrs)
   print(edge_index)

   def batch_edge_index(batch: ℝ[s, 3, 3]): ℝ[s, 2, 6]:
       num_samples: ℕ = len(batch)
       result: ℝ[num_samples, 2, 6] = zeros(num_samples, 2, 6)
       num_edges: ℕ = 0
       for i:ℕ(num_samples):
           edges = neighbour_search(batch[i], r_cut)
           num_edges = len(edges[0])
           result[i, :, 0:num_edges] = edges
       return result

   train_edge_index: ℝ[num_train, 2, 6] = batch_edge_index(train_positions)
   test_edge_index: ℝ[num_test, 2, 6] = batch_edge_index(test_positions)

   # Step 1: Embeddings
   num_elements: ℝ = 2
   num_channels: ℝ = 8
   # sampled from a normal distribution
   w_embed: ℝ[2, 8] = for z:ℕ(num_elements) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def node_embedding(node_attrs: ℝ[n, s], w: ℝ[s, k]): ℝ[n, k]:
       return (node_attrs @ w) / sqrt(num_elements)

   # Radial features
   π: ℝ = 3.141592653589793
   num_bessel: ℝ = 8
   p: ℝ = 6.0

   def edge_vectors(positions: ℝ[n, 3], edge_index: ℝ[2, e]): ℝ[e, 3]:
       num_edges: ℕ = len(edge_index[0])
       vectors: ℝ[num_edges, 3] = zeros(num_edges, 3)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           vectors[edge] = positions[receiver] - positions[sender]
       return vectors

   def edge_lengths(vectors: ℝ[e, 3]): ℝ[e]:
       num_edges: ℕ = len(vectors)
       lengths: ℝ[num_edges] = zeros(num_edges)
       for edge:ℕ(num_edges):
           lengths[edge] = sqrt(sum(vectors[edge] * vectors[edge]))
       return lengths

   def radial_embedding(lengths: ℝ[e]): ℝ[e, b]:
       num_edges: ℕ = len(lengths)
       edge_feats: ℝ[num_edges, num_bessel] = zeros(num_edges, num_bessel)
       r: ℝ = 0.0
       x: ℝ = 0.0
       f_cut: ℝ = 0.0
       for edge:ℕ(num_edges):
           r = lengths[edge]
           x = r / r_cut
           if x < 1.0:
               f_cut = 1.0 - (p + 1.0) * (p + 2.0) / 2.0 * x ** p + p * (p + 2.0) * x ** (p + 1.0) - p * (p + 1.0) / 2.0 * x ** (p + 2.0)
               for n:ℕ(num_bessel):
                   edge_feats[edge, n] = sqrt(2.0 / r_cut) * sin((n + 1.0) * π * r / r_cut) / r * f_cut
       return edge_feats

   vectors: ℝ[6, 3] = edge_vectors(positions, edge_index)
   lengths: ℝ[6] = edge_lengths(vectors)
   edge_feats: ℝ[6, 8] = radial_embedding(lengths)

   # Angular features
   max_ell: ℝ = 2

   def Y0(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 1]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 1] = zeros(num_edges, 1)
       for edge:ℕ(num_edges):
           results[edge, 0] = 1.0
       return results

   def Y1(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 3]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 3] = zeros(num_edges, 3)
       for edge:ℕ(num_edges):
           x: ℝ, y: ℝ, z: ℝ = vectors[edge] / lengths[edge]
           results[edge] = [sqrt(3.0) * x, sqrt(3.0) * y, sqrt(3.0) * z]
       return results

   def Y2(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 5]:
       num_edges: ℕ = len(lengths)
       results: ℝ[num_edges, 5] = zeros(num_edges, 5)
       for edge:ℕ(num_edges):
           x: ℝ, y: ℝ, z: ℝ = vectors[edge] / lengths[edge]
           results[edge] = [sqrt(15.0) * x * z, sqrt(15.0) * x * y, sqrt(5.0) * (y * y - 0.5 * (x * x + z * z)), sqrt(15.0) * y * z, sqrt(15.0) / 2.0 * (z * z - x * x)]
       return results

   def spherical_harmonics(vectors: ℝ[e, 3], lengths: ℝ[e]): ℝ[e, 9]:
       num_edges: ℕ = len(lengths)
       edge_attrs: ℝ[num_edges, 9] = zeros(num_edges, 9)
       edge_attrs[:, 0:1] = Y0(vectors, lengths)
       edge_attrs[:, 1:4] = Y1(vectors, lengths)
       edge_attrs[:, 4:9] = Y2(vectors, lengths)
       return edge_attrs

   edge_attrs: ℝ[6, 9] = spherical_harmonics(vectors, lengths)

   print(edge_feats)
   print(edge_attrs)

   # Precompute features for forward pass
   def batch_edge_feats(batch: ℝ[s, 3, 3]): ℝ[s, 6, 8]:
       num_samples: ℕ = len(batch)
       result: ℝ[num_samples, 6, 8] = zeros(num_samples, 6, 8)
       num_edges: ℕ = 0
       for i:ℕ(num_samples):
           edges = neighbour_search(batch[i], r_cut)
           num_edges = len(edges[0])
           vecs = edge_vectors(batch[i], edges)
           result[i, 0:num_edges, :] = radial_embedding(edge_lengths(vecs))
       return result

   def batch_edge_attrs(batch: ℝ[s, 3, 3]): ℝ[s, 6, 9]:
       num_samples: ℕ = len(batch)
       result: ℝ[num_samples, 6, 9] = zeros(num_samples, 6, 9)
       num_edges: ℕ = 0
       for i:ℕ(num_samples):
           edges = neighbour_search(batch[i], r_cut)
           num_edges = len(edges[0])
           vecs = edge_vectors(batch[i], edges)
           result[i, 0:num_edges, :] = spherical_harmonics(vecs, edge_lengths(vecs))
       return result

   train_edge_feats: ℝ[num_train, 6, 8] = batch_edge_feats(train_positions)
   test_edge_feats: ℝ[num_test, 6, 8] = batch_edge_feats(test_positions)
   train_edge_attrs: ℝ[num_train, 6, 9] = batch_edge_attrs(train_positions)
   test_edge_attrs: ℝ[num_test, 6, 9] = batch_edge_attrs(test_positions)

   # Step 2: The Interaction Block
   # sampled from a normal distribution
   w_up: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def linear_up(node_feats: ℝ[n, k], w: ℝ[k, k]): ℝ[n, k]:
       num_atoms: ℕ = len(node_feats)
       node_feats_up: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats_up[atom, channel] = sum(node_feats[atom] * w[:, channel]) / sqrt(num_channels)
       return node_feats_up

   # Activation function
   silu_norm: ℝ = 1.679176792

   def silu(x: ℝ[e, h]): ℝ[e, h]:
       return silu_norm * x / (1.0 + exp(-x))

   # Radial network
   radial_hidden: ℝ = 64

   # sampled from a normal distribution
   w_r1: ℝ[8, 64] = for a:ℕ(num_bessel) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w_r2: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w_r3: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w_r4: ℝ[64, 24] = for a:ℕ(radial_hidden) -> row: ℝ[24] ~ Normal(0.0, 1.0, 24)

   def radial_mlp(edge_feats: ℝ[e, 8], w1: ℝ[8, 64], w2: ℝ[64, 64], w3: ℝ[64, 64], w4: ℝ[64, 24]): ℝ[e, 24]:
       h1: ℝ[e, 64] = silu(edge_feats @ w1 / sqrt(num_bessel))
       h2: ℝ[e, 64] = silu(h1 @ w2 / sqrt(radial_hidden))
       h3: ℝ[e, 64] = silu(h2 @ w3 / sqrt(radial_hidden))
       return h3 @ w4 / sqrt(radial_hidden)

   # The one-particle basis
   def tensor_product_reduce(cg: ℝ[d1, d2, d3], a: ℝ[d1], b: ℝ[d2], dim1: ℕ, dim2: ℕ, dim3: ℕ): ℝ[d3]:
       results: ℝ[dim3] = zeros(dim3)
       acc: ℝ = 0.0
       for k:ℕ(dim3):
           acc = 0.0
           for i:ℕ(dim1):
               for j:ℕ(dim2):
                   acc += cg[i, j, k] * a[i] * b[j]
           results[k] = acc
       return results

   cg_000: ℝ[1, 1, 1] = [[[1.0]]]
   cg_101: ℝ[3, 1, 3] = [
       [[1.0, 0.0, 0.0]],
       [[0.0, 1.0, 0.0]],
       [[0.0, 0.0, 1.0]]
   ]
   cg_202: ℝ[5, 1, 5] = [
       [[1.0, 0.0, 0.0, 0.0, 0.0]],
       [[0.0, 1.0, 0.0, 0.0, 0.0]],
       [[0.0, 0.0, 1.0, 0.0, 0.0]],
       [[0.0, 0.0, 0.0, 1.0, 0.0]],
       [[0.0, 0.0, 0.0, 0.0, 1.0]]
   ]

   def conv_tp(node_feats_up: ℝ[n, k], edge_attrs: ℝ[e, 9], tp_weights: ℝ[e, 24], edge_index: ℝ[2, e]): ℝ[e, k, 9]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 9] = zeros(num_edges, num_channels, 9)
       h: ℝ[1] = zeros(1)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           for channel:ℕ(num_channels):
               h = [node_feats_up[sender, channel]]
               mji[edge, channel, 0:1] = tp_weights[edge, channel] * tensor_product_reduce(cg_000, edge_attrs[edge, 0:1], h, 1, 1, 1)
               mji[edge, channel, 1:4] = tp_weights[edge, num_channels + channel] * tensor_product_reduce(cg_101, edge_attrs[edge, 1:4], h, 3, 1, 3)
               mji[edge, channel, 4:9] = tp_weights[edge, 2 * num_channels + channel] * tensor_product_reduce(cg_202, edge_attrs[edge, 4:9], h, 5, 1, 5)
       return mji

   # Message Passing
   avg_num_neighbors: ℝ = 2.0

   # sampled from a normal distribution
   w_0: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_1: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_2: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def neighbour_sum(mji: ℝ[e, k, 9], edge_index: ℝ[2, e], num_atoms: ℕ): ℝ[n, k, 9]:
       num_edges: ℕ = len(mji)
       message: ℝ[num_atoms, num_channels, 9] = zeros(num_atoms, num_channels, 9)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           message[receiver] = message[receiver] + mji[edge]
       return message

   def linear(message: ℝ[n, k, 9], w0: ℝ[k, k], w1: ℝ[k, k], w2: ℝ[k, k]): ℝ[n, k, 9]:
       num_atoms: ℕ = len(message)
       out: ℝ[num_atoms, num_channels, 9] = zeros(num_atoms, num_channels, 9)
       for atom:ℕ(num_atoms):
           out[atom, :, 0:1] = w0 @ message[atom, :, 0:1] / sqrt(num_channels)
           out[atom, :, 1:4] = w1 @ message[atom, :, 1:4] / sqrt(num_channels)
           out[atom, :, 4:9] = w2 @ message[atom, :, 4:9] / sqrt(num_channels)
       return out / avg_num_neighbors

   # Step 3: The Product
   cg_110: ℝ[3, 3, 1] = [
       [[0.577350], [0.0], [0.0]],
       [[0.0], [0.577350], [0.0]],
       [[0.0], [0.0], [0.577350]]
   ]
   cg_220: ℝ[5, 5, 1] = [
       [[0.447214], [0.0], [0.0], [0.0], [0.0]],
       [[0.0], [0.447214], [0.0], [0.0], [0.0]],
       [[0.0], [0.0], [0.447214], [0.0], [0.0]],
       [[0.0], [0.0], [0.0], [0.447214], [0.0]],
       [[0.0], [0.0], [0.0], [0.0], [0.447214]]
   ]
   cg_011: ℝ[1, 3, 3] = [
       [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
   ]
   cg_121: ℝ[3, 5, 3] = [
       [[0.0, 0.0, 0.547723], [0.0, 0.547723, 0.0], [-0.316228, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.547723, 0.0, 0.0]],
       [[0.0, 0.0, 0.0], [0.547723, 0.0, 0.0], [0.0, 0.632456, 0.0], [0.0, 0.0, 0.547723], [0.0, 0.0, 0.0]],
       [[0.547723, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.316228], [0.0, 0.547723, 0.0], [0.0, 0.0, 0.547723]]
   ]

   def B_features_L0(A: ℝ[n, k, 9]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(A)
       B0: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       a0: ℝ[1] = zeros(1)
       a1: ℝ[3] = zeros(3)
       a2: ℝ[5] = zeros(5)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               a0, a1, a2 = A[atom, channel, 0:1], A[atom, channel, 1:4], A[atom, channel, 4:9]
               B0[atom, channel, 0:1] = a0
               B0[atom, channel, 1:2] = tensor_product_reduce(cg_000, a0, a0, 1, 1, 1)
               B0[atom, channel, 2:3] = tensor_product_reduce(cg_110, a1, a1, 3, 3, 1)
               B0[atom, channel, 3:4] = tensor_product_reduce(cg_220, a2, a2, 5, 5, 1)
       return B0

   def B_features_L1(A: ℝ[n, k, 9]): ℝ[n, k, 3, 3]:
       num_atoms: ℕ = len(A)
       B1: ℝ[num_atoms, num_channels, 3, 3] = zeros(num_atoms, num_channels, 3, 3)
       a0: ℝ[1] = zeros(1)
       a1: ℝ[3] = zeros(3)
       a2: ℝ[5] = zeros(5)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               a0, a1, a2 = A[atom, channel, 0:1], A[atom, channel, 1:4], A[atom, channel, 4:9]
               B1[atom, channel, 0] = a1
               B1[atom, channel, 1] = tensor_product_reduce(cg_011, a0, a1, 1, 3, 3)
               B1[atom, channel, 2] = tensor_product_reduce(cg_121, a1, a2, 3, 5, 3)
       return B1

   # Weighting the paths
   # sampled from a normal distribution
   w_prod0: ℝ[2, 4, 8] = for z:ℕ(num_elements) -> for q:ℕ(4) -> row: ℝ[8] ~ Normal(0.0, 0.25, 8)
   w_prod1: ℝ[2, 3, 8] = for z:ℕ(num_elements) -> for q:ℕ(3) -> row: ℝ[8] ~ Normal(0.0, 0.333, 8)

   def weighted_sum(B0: ℝ[n, k, 4], B1: ℝ[n, k, 3, 3], node_attrs: ℝ[n, s], wp0: ℝ[2, 4, 8], wp1: ℝ[2, 3, 8]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(B0)
       m: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       w0: ℝ[4, 8] = zeros(4, 8)
       w1: ℝ[3, 8] = zeros(3, 8)
       for atom:ℕ(num_atoms):
           w0 = zeros(4, 8)
           w1 = zeros(3, 8)
           for z:ℕ(num_elements):
               w0 = w0 + node_attrs[atom, z] * wp0[z]
               w1 = w1 + node_attrs[atom, z] * wp1[z]
           for channel:ℕ(num_channels):
               m[atom, channel, 0:1] = [sum(w0[:, channel] * B0[atom, channel])]
               m[atom, channel, 1:4] = w1[:, channel] @ B1[atom, channel]
       return m

   # Updating the node features
   # sampled from a normal distribution
   w_sc: ℝ[8, 2, 8] = for c:ℕ(num_channels) -> for z:ℕ(num_elements) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_p0: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w_p1: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def skip_tp(node_feats: ℝ[n, k], node_attrs: ℝ[n, s], w: ℝ[8, 2, 8]): ℝ[n, k]:
       num_atoms: ℕ = len(node_feats)
       sc: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       acc: ℝ = 0.0
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               acc = 0.0
               for z:ℕ(num_elements):
                   acc += node_attrs[atom, z] * sum(node_feats[atom] * w[:, z, channel])
               sc[atom, channel] = acc / sqrt(num_channels * num_elements)
       return sc

   def node_update(m: ℝ[n, k, 4], sc: ℝ[n, k], wp0: ℝ[k, k], wp1: ℝ[k, k]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(m)
       node_feats1: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats1[atom, channel, 0] = sum(wp0[channel] * m[atom, :, 0]) / sqrt(num_channels) + sc[atom, channel]
           node_feats1[atom, :, 1:4] = wp1 @ m[atom, :, 1:4] / sqrt(num_channels)
       return node_feats1

   # Step 4: The Readout
   # sampled from a normal distribution
   w_readout: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def readout(node_feats: ℝ[n, k, 4], w: ℝ[8]): ℝ[n]:
       num_atoms: ℕ = len(node_feats)
       node_energies: ℝ[num_atoms] = zeros(num_atoms)
       for atom:ℕ(num_atoms):
           node_energies[atom] = sum(w * node_feats[atom, :, 0]) / sqrt(num_channels)
       return node_energies

   # Step 5: Repeating the Layer
   # sampled from a normal distribution
   w2_up0: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w2_up1: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def linear_up2(node_feats: ℝ[n, k, 4], w0: ℝ[8, 8], w1: ℝ[8, 8]): ℝ[n, k, 4]:
       num_atoms: ℕ = len(node_feats)
       node_feats_up: ℝ[num_atoms, num_channels, 4] = zeros(num_atoms, num_channels, 4)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats_up[atom, channel, 0] = sum(node_feats[atom, :, 0] * w0[:, channel]) / sqrt(num_channels)
               node_feats_up[atom, channel, 1:4] = w1[:, channel] @ node_feats[atom, :, 1:4] / sqrt(num_channels)
       return node_feats_up

   # Radial weights
   # sampled from a normal distribution
   w2_r1: ℝ[8, 64] = for a:ℕ(num_bessel) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w2_r2: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w2_r3: ℝ[64, 64] = for a:ℕ(radial_hidden) -> row: ℝ[64] ~ Normal(0.0, 1.0, 64)
   w2_r4: ℝ[64, 56] = for a:ℕ(radial_hidden) -> row: ℝ[56] ~ Normal(0.0, 1.0, 56)

   def radial_mlp2(edge_feats: ℝ[e, 8], w1: ℝ[8, 64], w2: ℝ[64, 64], w3: ℝ[64, 64], w4: ℝ[64, 56]): ℝ[e, 56]:
       h1: ℝ[e, 64] = silu(edge_feats @ w1 / sqrt(num_bessel))
       h2: ℝ[e, 64] = silu(h1 @ w2 / sqrt(radial_hidden))
       h3: ℝ[e, 64] = silu(h2 @ w3 / sqrt(radial_hidden))
       return h3 @ w4 / sqrt(radial_hidden)

   # One-particle basis (7 paths)
   cg_022: ℝ[1, 5, 5] = [
       [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]
   ]
   cg_112: ℝ[3, 3, 5] = [
       [[0.0, 0.0, -0.408248, 0.0, -0.707107], [0.0, 0.707107, 0.0, 0.0, 0.0], [0.707107, 0.0, 0.0, 0.0, 0.0]],
       [[0.0, 0.707107, 0.0, 0.0, 0.0], [0.0, 0.0, 0.816497, 0.0, 0.0], [0.0, 0.0, 0.0, 0.707107, 0.0]],
       [[0.707107, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.707107, 0.0], [0.0, 0.0, -0.408248, 0.0, 0.707107]]
   ]

   def conv_tp2(node_feats_up: ℝ[n, k, 4], edge_attrs: ℝ[e, 9], tp_weights: ℝ[e, 56], edge_index: ℝ[2, e]): ℝ[e, k, 21]:
       num_edges: ℕ = len(edge_attrs)
       mji: ℝ[num_edges, num_channels, 21] = zeros(num_edges, num_channels, 21)
       h0: ℝ[1] = zeros(1)
       h1: ℝ[3] = zeros(3)
       y0: ℝ[1] = zeros(1)
       y1: ℝ[3] = zeros(3)
       y2: ℝ[5] = zeros(5)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           y0, y1, y2 = edge_attrs[edge, 0:1], edge_attrs[edge, 1:4], edge_attrs[edge, 4:9]
           for channel:ℕ(num_channels):
               h0, h1 = [node_feats_up[sender, channel, 0]], node_feats_up[sender, channel, 1:4]
               mji[edge, channel, 0:1] = tp_weights[edge, channel] * tensor_product_reduce(cg_000, h0, y0, 1, 1, 1)
               mji[edge, channel, 1:2] = tp_weights[edge, num_channels + channel] * tensor_product_reduce(cg_110, h1, y1, 3, 3, 1)
               mji[edge, channel, 2:5] = tp_weights[edge, 2 * num_channels + channel] * tensor_product_reduce(cg_011, h0, y1, 1, 3, 3)
               mji[edge, channel, 5:8] = tp_weights[edge, 3 * num_channels + channel] * tensor_product_reduce(cg_101, h1, y0, 3, 1, 3)
               mji[edge, channel, 8:11] = tp_weights[edge, 4 * num_channels + channel] * tensor_product_reduce(cg_121, h1, y2, 3, 5, 3)
               mji[edge, channel, 11:16] = tp_weights[edge, 5 * num_channels + channel] * tensor_product_reduce(cg_022, h0, y2, 1, 5, 5)
               mji[edge, channel, 16:21] = tp_weights[edge, 6 * num_channels + channel] * tensor_product_reduce(cg_112, h1, y1, 3, 3, 5)
       return mji

   # Message Passing (second layer)
   # sampled from a normal distribution
   w2_msg: ℝ[7, 8, 8] = for p:ℕ(7) -> for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def neighbour_sum2(mji: ℝ[e, k, 21], edge_index: ℝ[2, e], num_atoms: ℕ): ℝ[n, k, 21]:
       num_edges: ℕ = len(mji)
       summed: ℝ[num_atoms, num_channels, 21] = zeros(num_atoms, num_channels, 21)
       for edge:ℕ(num_edges):
           sender: ℝ, receiver: ℝ = edge_index[:, edge]
           summed[receiver] = summed[receiver] + mji[edge]
       return summed

   def linear2(summed: ℝ[n, k, 21], w: ℝ[7, k, k]): ℝ[n, k, 9]:
       num_atoms: ℕ = len(summed)
       message: ℝ[num_atoms, num_channels, 9] = zeros(num_atoms, num_channels, 9)
       for atom:ℕ(num_atoms):
           message[atom, :, 0:1] = (w[0] @ summed[atom, :, 0:1] + w[1] @ summed[atom, :, 1:2]) / sqrt(2.0 * num_channels)
           message[atom, :, 1:4] = (w[2] @ summed[atom, :, 2:5] + w[3] @ summed[atom, :, 5:8] + w[4] @ summed[atom, :, 8:11]) / sqrt(3.0 * num_channels)
           message[atom, :, 4:9] = (w[5] @ summed[atom, :, 11:16] + w[6] @ summed[atom, :, 16:21]) / sqrt(2.0 * num_channels)
       return message / avg_num_neighbors

   # Product and update
   # sampled from a normal distribution
   w2_prod0: ℝ[2, 4, 8] = for z:ℕ(num_elements) -> for q:ℕ(4) -> row: ℝ[8] ~ Normal(0.0, 0.25, 8)

   def weighted_sum2(B0: ℝ[n, k, 4], node_attrs: ℝ[n, s], wp: ℝ[2, 4, 8]): ℝ[n, k]:
       num_atoms: ℕ = len(B0)
       m: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       w0: ℝ[4, 8] = zeros(4, 8)
       for atom:ℕ(num_atoms):
           w0 = zeros(4, 8)
           for z:ℕ(num_elements):
               w0 = w0 + node_attrs[atom, z] * wp[z]
           for channel:ℕ(num_channels):
               m[atom, channel] = sum(w0[:, channel] * B0[atom, channel])
       return m

   # sampled from a normal distribution
   w2_sc: ℝ[8, 2, 8] = for c:ℕ(num_channels) -> for z:ℕ(num_elements) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)
   w2_p: ℝ[8, 8] = for c:ℕ(num_channels) -> row: ℝ[8] ~ Normal(0.0, 1.0, 8)

   def node_update2(m: ℝ[n, k], sc: ℝ[n, k], wp: ℝ[k, k]): ℝ[n, k]:
       num_atoms: ℕ = len(m)
       node_feats: ℝ[num_atoms, num_channels] = zeros(num_atoms, num_channels)
       for atom:ℕ(num_atoms):
           for channel:ℕ(num_channels):
               node_feats[atom, channel] = sum(wp[channel] * m[atom]) / sqrt(num_channels) + sc[atom, channel]
       return node_feats

   # Readout and total energy
   # sampled from a normal distribution
   w2_ro1: ℝ[8, 16] = for a:ℕ(num_channels) -> row: ℝ[16] ~ Normal(0.0, 1.0, 16)
   w2_ro2: ℝ[16] ~ Normal(0.0, 1.0, 16)

   def readout2(node_feats: ℝ[n, k], w1: ℝ[k, 16], w2: ℝ[16]): ℝ[n]:
       num_atoms: ℕ = len(node_feats)
       hidden: ℝ[num_atoms, 16] = silu(node_feats @ w1 / sqrt(num_channels))
       node_energies: ℝ[num_atoms] = zeros(num_atoms)
       for atom:ℕ(num_atoms):
           node_energies[atom] = sum(hidden[atom] * w2) / sqrt(16.0)
       return node_energies

   # Define Loss
   def mse(pred: ℝ, target: ℝ): ℝ:
       diff: ℝ = pred - target
       result: ℝ = diff * diff
       return result

   # Model Definition
   class MACEModel(w_embed: ℝ[2, 8], w_up: ℝ[8, 8], w_r1: ℝ[8, 64], w_r2: ℝ[64, 64], w_r3: ℝ[64, 64], w_r4: ℝ[64, 24],
                   w_0: ℝ[8, 8], w_1: ℝ[8, 8], w_2: ℝ[8, 8], w_prod0: ℝ[2, 4, 8], w_prod1: ℝ[2, 3, 8],
                   w_sc: ℝ[8, 2, 8], w_p0: ℝ[8, 8], w_p1: ℝ[8, 8], w_readout: ℝ[8],
                   w2_up0: ℝ[8, 8], w2_up1: ℝ[8, 8], w2_r1: ℝ[8, 64], w2_r2: ℝ[64, 64], w2_r3: ℝ[64, 64], w2_r4: ℝ[64, 56],
                   w2_msg: ℝ[7, 8, 8], w2_prod0: ℝ[2, 4, 8], w2_sc: ℝ[8, 2, 8], w2_p: ℝ[8, 8], w2_ro1: ℝ[8, 16], w2_ro2: ℝ[16]):
       def λ(node_attrs: ℝ[3, 2], edge_index: ℝ[2, 6], edge_feats: ℝ[6, 8], edge_attrs: ℝ[6, 9]) → ℝ:
           node_feats0: ℝ[3, 8] = node_embedding(node_attrs, this.w_embed)
           node_feats_up: ℝ[3, 8] = linear_up(node_feats0, this.w_up)
           tp_weights: ℝ[6, 24] = radial_mlp(edge_feats, this.w_r1, this.w_r2, this.w_r3, this.w_r4)
           mji: ℝ[6, 8, 9] = conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
           message: ℝ[3, 8, 9] = linear(neighbour_sum(mji, edge_index, 3), this.w_0, this.w_1, this.w_2)
           B0: ℝ[3, 8, 4] = B_features_L0(message)
           B1: ℝ[3, 8, 3, 3] = B_features_L1(message)
           m: ℝ[3, 8, 4] = weighted_sum(B0, B1, node_attrs, this.w_prod0, this.w_prod1)
           sc: ℝ[3, 8] = skip_tp(node_feats0, node_attrs, this.w_sc)
           node_feats1: ℝ[3, 8, 4] = node_update(m, sc, this.w_p0, this.w_p1)
           node_energies: ℝ[3] = readout(node_feats1, this.w_readout)
           node_feats_up2: ℝ[3, 8, 4] = linear_up2(node_feats1, this.w2_up0, this.w2_up1)
           tp_weights2: ℝ[6, 56] = radial_mlp2(edge_feats, this.w2_r1, this.w2_r2, this.w2_r3, this.w2_r4)
           mji2: ℝ[6, 8, 21] = conv_tp2(node_feats_up2, edge_attrs, tp_weights2, edge_index)
           message2: ℝ[3, 8, 9] = linear2(neighbour_sum2(mji2, edge_index, 3), this.w2_msg)
           B0_2: ℝ[3, 8, 4] = B_features_L0(message2)
           m2: ℝ[3, 8] = weighted_sum2(B0_2, node_attrs, this.w2_prod0)
           sc2: ℝ[3, 8] = skip_tp(node_feats1[:, :, 0], node_attrs, this.w2_sc)
           node_feats2: ℝ[3, 8] = node_update2(m2, sc2, this.w2_p)
           node_energies2: ℝ[3] = readout2(node_feats2, this.w2_ro1, this.w2_ro2)
           energy: ℝ = sum(node_energies) + sum(node_energies2)
           return energy
       def loss_sample(sample: ℕ) → ℝ:
           pred: ℝ = this(node_attrs, train_edge_index[sample], train_edge_feats[sample], train_edge_attrs[sample])
           target: ℝ = (train_energies[sample] - energy_mean) / energy_std
           result: ℝ = mse(pred, target)
           return result
       def error_sample(sample: ℕ) → ℝ:
           scaled: ℝ = this(node_attrs, test_edge_index[sample], test_edge_feats[sample], test_edge_attrs[sample])
           result: ℝ = energy_mean + energy_std * scaled - test_energies[sample]
           return result
       def train(epochs: ℕ, lr: ℝ) → ℝ:
           last_loss: ℝ = 0
           current_loss: ℝ = 0
           for epoch:ℕ(epochs):
               for sample:ℕ(num_train):
                   for rep:ℕ(1):
                       current_loss = this.loss_sample(sample)
                       learnable_grads = grad(current_loss, this.learnable_params)
                       this.update(lr, learnable_grads)
                       last_loss = current_loss
           return last_loss
       def evaluate() → ℝ:
           total_error: ℝ = 0
           current_error: ℝ = 0
           for sample:ℕ(num_test):
               for rep:ℕ(1):
                   current_error = this.error_sample(sample)
                   total_error = total_error + current_error * current_error
           result: ℝ = sqrt(total_error / num_test)
           return result

   # Main Program:
   mace_object: MACEModel = MACEModel(w_embed, w_up, w_r1, w_r2, w_r3, w_r4,
                                      w_0, w_1, w_2, w_prod0, w_prod1,
                                      w_sc, w_p0, w_p1, w_readout,
                                      w2_up0, w2_up1, w2_r1, w2_r2, w2_r3, w2_r4,
                                      w2_msg, w2_prod0, w2_sc, w2_p, w2_ro1, w2_ro2)

   example_energy: ℝ = mace_object(node_attrs, edge_index, edge_feats, edge_attrs)
   print(example_energy)

   lr: ℝ = 0.05
   epochs: ℕ = 100

   rmse_before: ℝ = mace_object.evaluate()
   print(rmse_before)

   final_loss: ℝ = mace_object.train(epochs, lr)
   print(final_loss)

   rmse_after: ℝ = mace_object.evaluate()
   print(rmse_after)

   test_predictions: ℝ[num_test] = zeros(num_test)
   for sample:ℕ(num_test):
       test_predictions[sample] = test_energies[sample] + mace_object.error_sample(sample)
   print(test_predictions)

References
----------

.. [Batatia2022] Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*. Advances in Neural Information Processing Systems (NeurIPS). arXiv:2206.07697. https://arxiv.org/abs/2206.07697

.. [Batatia2022Design] Batatia, I., Batzner, S., Kovács, D. P., Musaelian, A., Simm, G. N. C., Drautz, R., Ortner, C., Kozinsky, B., & Csányi, G. (2022). *The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials*. arXiv:2205.06643. https://arxiv.org/abs/2205.06643

.. [Thomas2018] Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). *Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds*. arXiv:1802.08219. https://arxiv.org/abs/1802.08219

.. [MACETutorials] ACEsuit. *MACE tutorials*, in particular ``T03_MACE_Theory.ipynb`` and ``MACE_developer.ipynb``. https://github.com/ACEsuit/mace-tutorials

.. [tblite] *tblite: light-weight tight-binding framework*. https://github.com/tblite/tblite

.. [ASE] *Atomic Simulation Environment (ASE)*. https://wiki.fysik.dtu.dk/ase/

.. [MACEPractice] *MACE in Practice I*, notebook of the CAMML tutorials, which labels molecular configurations with GFN2-xTB energies and forces through tblite. https://workshop.camml.ac.uk/notebooks/day-4/t01-mace-practice-i
