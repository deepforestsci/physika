Graph Convolutional Networks
=============================

In this tutorial we implemented a Graph Convolutional Network (GCN) in physika
and trained it on Hydration free energy prediction with the FreeSolv dataset.

Why Use a GCN?
------------------------

Molecular data is naturally graph-structured, atoms as nodes and bonds as
edges. Standard neural networks cannot directly exploit this structure,
whereas GCNs operate on the graph itself via its adjacency matrix which
yields a more data-efficient model for predicting graph level properties
[DistillGNN2021]_.

GCN Architecture
------------------------

.. figure:: /_static/tutorial_files/GCN_vs_CNN_overview.png
   :alt: Comparison of a GCN and a CNN processing a molecule
   :align: center
   :width: 500px

   *Comparison of a GCN and a CNN processing a molecule* [Bernstein2023]_

Each atom is initialized with a feature vector (e.g. atomic number,
degree). A graph convolution layer updates these via message passing,
each node aggregates its neighbor's features with its own.

The final node features are pooled into a graph-level representation
and passed through fully connected layers to predict the target
property. 


Dataset
--------

We trained the GCN model on the FreeSolv dataset [FreeSolv]_ (generated
using the DeepChem library) to predict hydration free energy (kcal/mol)
for small molecules.

.. code-block:: text

    train_test_split: ℕ = 80
    total_dataset_size: ℕ = 642
    max_atoms: ℕ = 44

    dataset = create_dataset(train_test_split, total_dataset_size, max_atoms)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    train_A: ℝ[513, 44, 44] = train_dataset[0]
    train_H: ℝ[513, 44, 30] = train_dataset[1]
    train_sizes: ℝ[513] = train_dataset[2]
    train_y: ℝ[513] = train_dataset[3]

    test_A: ℝ[65, 44, 44] = test_dataset[0]
    test_H: ℝ[65, 44, 30] = test_dataset[1]
    test_sizes: ℝ[65] = test_dataset[2]
    test_y: ℝ[65] = test_dataset[3]


.. note::
   ``create_dataset`` is not a built-in Physika function. To use it,
   add the following helper to ``physika/runtime.py``:

   .. code-block:: python

        def create_dataset(train_test_split=80, total_dataset_size=642, max_atoms=44):
            import deepchem as dc
            import torch
            import torch.nn.functional as F
            from rdkit import Chem
            from rdkit.Chem import rdmolops

            tasks, datasets, transformers = dc.molnet.load_sampl(
                featurizer='Raw', splitter='random',
                frac_train=train_test_split/100.0, frac_valid=0.0,
                frac_test=1-train_test_split/100.0, seed=42
            )
            train_dataset, valid_dataset, test_dataset = datasets
            featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=False)

            def build(dataset, limit):
                A_list, H_list, sizes, y_list = [], [], [], []
                for i in range(min(limit, len(dataset.X))):
                    mol = Chem.AddHs(dataset.X[i])
                    n = mol.GetNumAtoms()
                    graph = featurizer.featurize([mol])[0]
                    A = torch.tensor(rdmolops.GetAdjacencyMatrix(mol), dtype=torch.float32)
                    H = torch.tensor(graph.node_features, dtype=torch.float32)
                    A_list.append(F.pad(A, (0, max_atoms-n, 0, max_atoms-n)))
                    H_list.append(F.pad(H, (0, 0, 0, max_atoms-n)))
                    sizes.append(n)
                    y_list.append(dataset.y[i][0])
                return [torch.stack(A_list), torch.stack(H_list),
                        torch.tensor(sizes, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32)]

            train_data = build(train_dataset, total_dataset_size)
            test_data = build(test_dataset, total_dataset_size)
            return [train_data, test_data]


Every molecule's adjacency matrix and feature matrix are padded upto a 
fixed size (molecule with maximum atoms) using ``torch.nn.functional.pad``, 
which pads zeros to the matrix.
Bond connectivity is figured out from RDKit's "GetAjacencyMatrix" and 
the 30 dimensional per-atom features come from DeepChem's "MolGraphConvFeaturizer"


Helper functions
------------------------

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

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols: ℝ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results

    def get_sum_of_1d_array(x: ℝ[m]): ℝ:
        total: ℝ  = 0
        for i:
            total += x[i]
        return total

    def diag_matrix(d: ℝ[n]): ℝ[n, n]:
        sz: ℝ = get_1d_array_length(d)
        result: ℝ[sz, sz] = zero_2d_array(sz, sz)
        for i:
            result[i, i] = d[i]
        return result

    def ones_vector(n: ℝ): ℝ[n]:
        return for i:ℕ(n) -> 0.0*i + 1.0


Activation functions
------------------------

After the normalization step produces the updated node features, we apply
the sigmoid activation function element-wise to every value.

Sigmoid keeps every node feature in the range :math:`(0, 1)`, letting the
model learn non-linear patterns across layers.

Mathematically, sigmoid is defined as:

.. math::

    \sigma(x) = \frac{1}{1 + e^{-x}}


.. code-block:: text

    def sigma(x: ℝ[a,b]): ℝ[a,b]:
        return 1.0 / (1.0 + exp(0.0 - x))


GCNModel class
------------------------

Lets build the full GCNModel class step by step.
The overall architecture of our network (forward pass) follows the pipeline:

``Normalize adjacency -> Propagate + Sigmoid -> Pool -> Linear``

We will implement each block in separate functions.

1. Normalization block
~~~~~~~~~~~~~~~~~~~~~~~~

The normalization step is the core building block of a graph convolutional
network.

It expects:

- an adjacency matrix, one row/column per node in the graph

Mathematically, this step rescales the raw adjacency matrix so that a node's
own features survive into the next layer, and high-degree nodes don't
dominate the aggregation just by having many connections.

The input tensor is defined as:

.. math::

    A \in \mathbb{R}^{n \times n}

We add self-loops first:

.. math::

    \tilde{A} = A + I

Each node's degree is the number of nodes it's connected to (including the
self loops), and the normalized adjacency matrix is:

.. math::

    \hat{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\,\tilde{D}^{-\frac{1}{2}}

where:

- :math:`A` is the raw adjacency matrix
- :math:`I` is the identity matrix
- :math:`\tilde{D}` is the diagonal degree matrix of :math:`\tilde{A}`

The output is the normalized adjacency matrix:

.. math::

    \hat{A} \in \mathbb{R}^{n \times n}

.. code-block:: text

    def normalize_adj(A: ℝ[n, n]): ℝ[n, n]:
        sz: ℝ = get_2d_array_num_rows(A)
        I: ℝ[sz, sz] = diag_matrix(ones_vector(sz))
        A_plus: ℝ[sz, sz] = A + I
        deg: ℝ[sz] = for i:ℕ(sz) -> get_sum_of_1d_array(A_plus[i])
        d_inv_sqrt: ℝ[sz] = for i:ℕ(sz) -> 1.0/sqrt(deg[i])
        D: ℝ[sz, sz] = diag_matrix(d_inv_sqrt)
        return D @ A_plus @ D


2. Pooling Block
~~~~~~~~~~~~~~~~~~~~~~~

The pooling layer aggregates the node-wise features into a single vector for
each molecule.

The output length is computed using:

.. math::

    d = \text{number of feature columns in } H

where:

- :math:`n` is the real number of atoms in the molecule
- :math:`H` is the node feature matrix

.. code-block:: text

    def masked_graph_sum_pool(H: ℝ[n, d], sz: ℝ): ℝ[d]:
        cols: ℝ = get_1d_array_length(H[0])
        result: ℝ[cols] = zero_1d_array(cols)
        for i:ℕ(sz):
            row = H[i]
            for j:ℕ(cols):
                result[j] += row[j]
        return result


Forward Pass
~~~~~~~~~~~~

The forward pass defines how data flows through the complete graph
convolutional network.

``Normalize adjacency -> Propagate + Sigmoid -> Pool -> Linear``

The Physika implementation is:

.. code-block:: text

    class GCNModel(W1: ℝ[30, 4], W2: ℝ[4]):
        def λ(A: ℝ[n,n], H: ℝ[n,30], sz: ℝ) -> ℝ:
            A_hat: ℝ[n,n] = normalize_adj(A)
            H1: ℝ[n,4] = sigma(A_hat @ (H @ W1))
            pooled: ℝ[4] = masked_graph_sum_pool(H1, sz)
            pred: ℝ = pooled @ W2
            return pred

For a molecule with :math:`n` atoms:

.. math::

    A \in \mathbb{R}^{n \times n}, \qquad H \in \mathbb{R}^{n \times 30}

After one propagation layer:

.. math::

    H_1 = \sigma(\hat{A}\,H\,W_1) \in \mathbb{R}^{n \times 4}

After pooling:

.. math::

    \text{pooled} \in \mathbb{R}^{4}

The final linear layer then computes a single predicted value:

.. math::

    pred = \text{pooled} \cdot W_2

This follows the message-passing formulation of Kipf and Welling [Kipf2017]_.

where:

.. math::

    W_1 \in \mathbb{R}^{30 \times 4}, \qquad W_2 \in \mathbb{R}^{4}


Initializing GCNModel object
------------------------------

.. code-block:: text

    # Normal distribution to generate random values with sigma as 1.0 and μ as 0.0

    μ : ℝ = 0.0
    σ : ℝ = 1.0

    W1: ℝ[30, 4] = for i:ℕ(30) -> row: ℝ[4] ~ Normal(μ, σ, 4)
    W2: ℝ[4] ~ Normal(μ, σ, 4)

    gcn_object: GCNModel = GCNModel(W1, W2)


Define loss
---------------------------

For training the network we use mean squared error, since hydration free
energy is a continuous value rather than a discrete class. 

.. math::

    \mathcal{L}(y, \hat{y}) = (\hat{y} - y)^2

where:

- :math:`y` is the true hydration free energy
- :math:`\hat{y}` is the predicted hydration free energy

.. code-block:: text

    def mse(pred: ℝ, target: ℝ): ℝ:
        return (pred - target) ** 2.0


Training the Model
------------------

We train the network using stochastic gradient descent (SGD).

.. math::

    \theta = \theta - \eta \nabla_{\theta}\mathcal{L}

where:

- :math:`\theta` represents model parameters
- :math:`\eta` is the learning rate
- :math:`\nabla_{\theta}\mathcal{L}` is the gradient of the loss

.. code-block:: text

    def train(epochs: ℕ, lr: ℝ) -> ℝ:
        len_train_X: ℝ = get_1d_array_length(train_sizes)
        loss: ℝ = 0
        for i:ℕ(epochs):
            loss = 0
            for j:ℕ(len_train_X):
                A_j = train_A[j]
                H_j = train_H[j]
                sz_j = train_sizes[j]
                label = train_y[j]
                z = this(A_j, H_j, sz_j)
                current_loss = mse(z, label)
                loss += current_loss
                learnable_grads = grad(current_loss, this.learnable_params)
                this.update_params(lr, learnable_grads)
            loss = loss / len_train_X
        return loss

    def update_params(lr: ℝ, learnable_grads: ℝ[m]):
        this.W1 = this.W1 - lr * learnable_grads[0]
        this.W2 = this.W2 - lr * learnable_grads[1]

Training is then a single call:

.. code-block:: text

    epochs: ℕ = 100
    lr: ℝ = 0.0005
    final_loss: ℝ = gcn_object.train(epochs, lr)
    print(final_loss)


Testing the Model
-----------------

Since hydration free energy is a continuous value, a prediction is counted
as correct when it falls within a tolerance of the true value, rather than
using an exact class match.

The final accuracy is computed as:

.. math::

    \mathrm{accuracy} =
    \frac{\mathrm{correct\ predictions}}
    {\mathrm{total\ predictions}}

.. code-block:: text

    def within_tolerance(pred: ℝ, target: ℝ, tol: ℝ): ℝ:
        diff: ℝ = pred - target
        if diff < 0.0:
            diff = 0.0 - diff
        if diff < tol:
            return 1.0
        else:
            return 0.0

    def evaluate() -> ℝ:
        len_test_X: ℝ = get_1d_array_length(test_sizes)
        correct: ℝ = 0
        for i:ℕ(len_test_X):
            A_i = test_A[i]
            H_i = test_H[i]
            sz_i = test_sizes[i]
            y_pred = this(A_i, H_i, sz_i)
            correct += within_tolerance(y_pred, test_y[i], 1.0)
        return correct / len_test_X

Evaluation is then a single call too:

.. code-block:: text

    accuracy: ℝ = gcn_object.evaluate()
    print(accuracy)


Full Code
---------

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

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def zero_2d_array(rows: ℝ, cols: ℝ): ℝ[m, n]:
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:ℕ(cols) -> j*0
        return results

    def get_sum_of_1d_array(x: ℝ[m]): ℝ:
        total: ℝ = 0
        for i:
            total += x[i]
        return total

    def diag_matrix(d: ℝ[n]): ℝ[n, n]:
        sz: ℝ = get_1d_array_length(d)
        result: ℝ[sz, sz] = zero_2d_array(sz, sz)
        for i:
            result[i, i] = d[i]
        return result

    def ones_vector(n: ℝ): ℝ[n]:
        return for i:ℕ(n) -> 0.0*i + 1.0

    def sigma(x: ℝ[a,b]): ℝ[a,b]:
        return 1.0 / (1.0 + exp(0.0 - x))

    def normalize_adj(A: ℝ[n, n]): ℝ[n, n]:
        sz: ℝ = get_2d_array_num_rows(A)
        I: ℝ[sz, sz] = diag_matrix(ones_vector(sz))
        A_plus: ℝ[sz, sz] = A + I
        deg: ℝ[sz] = for i:ℕ(sz) -> get_sum_of_1d_array(A_plus[i])
        d_inv_sqrt: ℝ[sz] = for i:ℕ(sz) -> 1.0/sqrt(deg[i])
        D: ℝ[sz, sz] = diag_matrix(d_inv_sqrt)
        return D @ A_plus @ D

    def masked_graph_sum_pool(H: ℝ[n, d], sz: ℝ): ℝ[d]:
        cols: ℝ = get_1d_array_length(H[0])
        result: ℝ[cols] = zero_1d_array(cols)
        for i:ℕ(sz):
            row = H[i]
            for j:ℕ(cols):
                result[j] += row[j]
        return result

    def mse(pred: ℝ, target: ℝ): ℝ:
        return (pred - target) ** 2.0

    def within_tolerance(pred: ℝ, target: ℝ, tol: ℝ): ℝ:
        diff: ℝ = pred - target
        if diff < 0.0:
            diff = 0.0 - diff
        if diff < tol:
            return 1.0
        else:
            return 0.0

    class GCNModel(W1: ℝ[30, 4], W2: ℝ[4]):
        def λ(A: ℝ[n,n], H: ℝ[n,30], sz: ℝ) -> ℝ:
            A_hat: ℝ[n,n] = normalize_adj(A)
            H1: ℝ[n,4] = sigma(A_hat @ (H @ W1))
            pooled: ℝ[4] = masked_graph_sum_pool(H1, sz)
            pred: ℝ = pooled @ W2
            return pred
        def train(epochs: ℕ, lr: ℝ) -> ℝ:
            len_train_X: ℝ = get_1d_array_length(train_sizes)
            loss: ℝ = 0
            for i:ℕ(epochs):
                loss = 0
                for j:ℕ(len_train_X):
                    A_j = train_A[j]
                    H_j = train_H[j]
                    sz_j = train_sizes[j]
                    label = train_y[j]
                    z = this(A_j, H_j, sz_j)
                    current_loss = mse(z, label)
                    loss += current_loss
                    learnable_grads = grad(current_loss, this.learnable_params)
                    this.update_params(lr, learnable_grads)
                loss = loss / len_train_X
            return loss
        def evaluate() -> ℝ:
            len_test_X: ℝ = get_1d_array_length(test_sizes)
            correct: ℝ = 0
            for i:ℕ(len_test_X):
                A_i = test_A[i]
                H_i = test_H[i]
                sz_i = test_sizes[i]
                y_pred = this(A_i, H_i, sz_i)
                correct += within_tolerance(y_pred, test_y[i], 1.0)
            return correct / len_test_X
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W1 = this.W1 - lr * learnable_grads[0]
            this.W2 = this.W2 - lr * learnable_grads[1]

    # Normal distribution to generate random values with sigma as 1.0 and μ as 0.0
    μ : ℝ = 0.0
    σ : ℝ = 1.0

    W1: ℝ[30, 4] = for i:ℕ(30) -> row: ℝ[4] ~ Normal(μ, σ, 4)
    W2: ℝ[4] ~ Normal(μ, σ, 4)

    gcn_object: GCNModel = GCNModel(W1, W2)

    train_test_split: ℕ = 80
    total_dataset_size: ℕ = 642
    max_atoms: ℕ = 44

    dataset = create_dataset(train_test_split, total_dataset_size, max_atoms)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    train_A: ℝ[513, 44, 44] = train_dataset[0]
    train_H: ℝ[513, 44, 30] = train_dataset[1]
    train_sizes: ℝ[513] = train_dataset[2]
    train_y: ℝ[513] = train_dataset[3]

    test_A: ℝ[65, 44, 44] = test_dataset[0]
    test_H: ℝ[65, 44, 30] = test_dataset[1]
    test_sizes: ℝ[65] = test_dataset[2]
    test_y: ℝ[65] = test_dataset[3]

    epochs: ℕ = 100
    lr: ℝ = 0.0005
    final_loss: ℝ = gcn_object.train(epochs, lr)
    print(final_loss)

    accuracy: ℝ = gcn_object.evaluate()
    print(accuracy)


References
----------

.. [Kipf2017] Kipf, T. N., & Welling, M. (2017). *Semi-Supervised
   Classification with Graph Convolutional Networks*. International
   Conference on Learning Representations (ICLR).
   https://arxiv.org/abs/1609.02907

.. [Bernstein2023] Bernstein, M. N. (2023). *Graph convolutional neural
   networks*. https://mbernste.github.io/posts/gcn/

.. [DistillGNN2021] Sanchez-Lengeling, B., Reif, E., Pearce, A., &
   Wiltschko, A. B. (2021). *A Gentle Introduction to Graph Neural
   Networks*. Distill. https://distill.pub/2021/gnn-intro/

.. [FreeSolv] Mobley Lab. *FreeSolv: The Free Solvation Database*.
   https://github.com/MobleyLab/FreeSolv

- `RDKit: Open-source cheminformatics <https://www.rdkit.org/>`_
- `DeepChem Documentation <https://deepchem.readthedocs.io/>`_
