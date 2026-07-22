Fourier Neural Operator
=======================


Many engineering and scientific problems are governed by partial differential equations (PDEs).
Whether we want to predict the temperature distribution on a rod, simulate flow around an aircraft the underlying
workflow is similar. We define the physical system, specify the boundary conditions and then use a numerical solver to 
compute the solution.

Now to do this, we use traditional numerical methods such as Finite Difference Method (FDM), Finite Element Method (FEM) and Finite
Volume Method (FVM). They are accurate but they come with a fundamental limitation which is,
every new set of boundary conditions or parameters requires solving the PDE from scratch.

Fourier Neural Operators take a different approach entirely. Instead of solving the PDE pointwise in the spatial domain, FNO lifts the input 
function into a higher-dimensional latent space and operates on it in the frequency domain. [LiKovachki2021]_ , [HoraKapoorMatveev2026]_


FNO Architecture
----------------

.. figure:: /_static/tutorial_files/fno/fno_architecture.png
   :alt: FNO Architecture
   :align: center
   :width: 700px

   FNO Architecture

Lets try to understand the FNO architecture in 2 parts:

Part 1 :-
~~~~~~~~~~~~~~~~~~

.. figure:: /_static/tutorial_files/fno/fno_architecture_first_half.png
   :alt: FNO Architecture
   :align: center
   :width: 700px

This is a outer structure and we can call this as main layer and it gets further divided into 3 parts. The input function :math:`a(x)`
is first passed through a pointwise lifting operator :math:`P`, which projects the input data from its original low-dimensional 
space into a higher-dimensional latent representation. This lifted representation then passes through a series of Fourier layers, 
where the actual learning happens. Finally, a pointwise projection operator :math:`Q`  maps the latent representation back down to 
the target output function :math:`u(x)`. Think of :math:`P` and :math:`Q` as the entry and exit gates which handles the dimensionality
, while the Fourier layers handle the physics.

In our implementation code, :math:`P` is a simple ``Conv1d`` block, and :math:`Q` is a ``MLP`` block:

.. code-block:: text

    class Conv1d(W: ℝ[out, in], b: ℝ[out, 1]):
        def λ(x: ℝ[in, n]) -> ℝ[out, n]:
            z: ℝ[out, n] = this.W @ x + this.b
            return z
        def update_params(lr: ℝ, learnable_grads: ℝ[List]):
            this.W = this.W - lr * learnable_grads[0]
            this.b = this.b - lr * learnable_grads[1]


    class MLP(W1: ℝ[mid, in],  b1: ℝ[mid, 1], W2: ℝ[out, mid], b2: ℝ[out, 1]):
        def λ(x: ℝ[in, n]) -> ℝ[out, n]:
            z1: ℝ[mid, n] = this.W1 @ x + this.b1
            a1: ℝ[mid, n] = gelu(z1)
            z2: ℝ[mid, n] = this.W2 @ a1 + this.b2
            return z2
        def update_params(lr: ℝ, learnable_grads: ℝ[List]):
            this.W1 = this.W1 - lr * learnable_grads[0]
            this.b1 = this.b1 - lr * learnable_grads[1]
            this.W2 = this.W2 - lr * learnable_grads[2]
            this.b2 = this.b2 - lr * learnable_grads[3]




Part 2 :-
~~~~~~~~~~~~~~~~~~

.. figure:: /_static/tutorial_files/fno/fno_architecture_second_half.png
   :alt: FNO Architecture
   :align: center
   :width: 700px

Now here where the magic happens, Each Fourier layer takes an input :math:`v(x)` and splits it into two parallel branches. In the upper branch, the input is 
transformed into the frequency domain via the Fast Fourier Transform. Since the input is real-valued, only the positive frequencies are retained using
:math:`\text{rFFT}`, . The first :math:`modes` frequencies are then passed through a learnable MLP that applies complex-valued weights directly in frequency space 
this is where the operator learns the global structure of the solution. An inverse FFT then brings the result back to the spatial domain. 

In our implementation, this entire upper branch is captured by the ``SpectralConv`` block:

.. code-block:: text

    class SpectralConv(weights1: ℂ[in_ch, out_ch, modes], in_ch: ℕ, out_ch: ℕ, modes: ℕ):
        def λ(x: ℂ[in_ch, n]) -> ℂ[out_ch, n]:
            x_ft: ℂ[in_ch, n_ft] = rfft(x)
            x_ft: ℂ[in_ch, modes] = x_ft[:, :self.modes]
            out_ft: ℂ[out_ch, modes] = compl_mul1d(x_ft, self.weights1)
            results: ℂ[out_ch, n] = irfft(out_ft, len(x[0]))
            return results
        def update_params(lr: ℝ, learnable_grads: ℝ[List]):
            this.weights1 = this.weights1 - lr * learnable_grads[0]


In the lower branch, the same input :math:`v(x)` passes through a simple Conv1d block (W), which acts as a local, pointwise correction. The outputs of both branches are
summed together and passed through a GELU activation, giving the layer the ability to capture both global frequency-domain patterns and local spatial features simultaneously.
The lower branch is represented by `Conv1d` blocks.

Here is the Physika implementation for 1d FNO model:

.. code-block:: text

    class FNO1d:
        p: Conv1d
        conv0: SpectralConv
        conv1: SpectralConv
        conv2: SpectralConv
        conv3: SpectralConv
        mlp0: MLP
        mlp1: MLP
        mlp2: MLP
        mlp3: MLP
        w0: Conv1d
        w1: Conv1d
        w2: Conv1d
        w3: Conv1d
        q: MLP
        def λ(x: R[1, n]) -> R[1, n]:
            # Lifting
            x: ℝ[width, n] = p(x)
            # Layer 1
            x1: ℝ[width, n] = conv0(x)
            x1: ℝ[width, n] = mlp0(x1)
            x2: ℝ[width, n] = w0(x)
            x: ℝ[width, n] = gelu(x1 + x2)
            # Layer 2
            x1: ℝ[width, n] = conv1(x)
            x1: ℝ[width, n] = mlp1(x1)
            x2: ℝ[width, n] = w1(x)
            x: ℝ[width, n] = gelu(x1 + x2)
            # Layer 3
            x1: ℝ[width, n] = conv2(x)
            x1: ℝ[width, n] = mlp2(x1)
            x2: ℝ[width, n] = w2(x)
            x: ℝ[width, n] = gelu(x1 + x2)
            # Layer 4
            x1: ℝ[width, n] = conv3(x)
            x1: ℝ[width, n] = mlp3(x1)
            x2: ℝ[width, n] = w3(x)
            x: ℝ[width, n] = x1 + x2
            # Projection
            x: ℝ[1, n] = q(x)
            return x


To summarize the architecture through Physika code, lets see how forward pass works:

.. code-block:: text

    # Lifting
    x: ℝ[width, n] = p(x)

    # Layer 1
    x1: ℝ[width, n] = conv0(x)
    x1: ℝ[width, n] = mlp0(x1)
    x2: ℝ[width, n] = w0(x)
    x: ℝ[width, n] = gelu(x1 + x2)

    # Layer 2
    .....
    .....

    # Projection
    x: ℝ[1, n] = q(x)
    return x


``x = p(x)`` :- takens spatial inputs (data) and lifts it from its original 
low-dimensional channel space into a higher-dimensional latent representation.

``x1 = conv0(x)`` :- passes the latent input through the SpectralConv block; applies FFT, truncates to the first 
modes frequencies, multiplies by learned complex weights, and maps back to spatial domain via inverse FFT.


``x1 = mlp0(x1)`` :- applies a pointwise MLP on top of the spectral output to add nonlinear mixing across channels
in the spatial domain.

``x2 = w0(x)`` :-  runs the same input through a simple Conv1d in parallel, acting as a local bypass that preserves 
pointwise spatial information.

``x = gelu(x1 + x2)`` :-  sums both branches and applies GELU, combining global frequency-domain features with local spatial
corrections before passing to the next layer.

``x = q(x)`` :- projects the final latent representation back down to the target output dimension, giving us :math:`u(x)`.


Dataset
-------

For this tutorial, we use the ``heat1d-pde-dataset`` [HeatDataset]_ available on Hugging Face.
The dataset is built around the 1D heat equation:

.. math::

    \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}


Each sample in the dataset consists of an input data representing the initial temperature distribution and a corresponding output label representing 
the final temperature distribution after the system has evolved forward in time. The task for the FNO is to learn the mapping between these two states.
For this tutorial we are only using ``tests/`` samples and then splitting total samples into train/test.


.. note::
   ``create_heat_dataset`` is not a built-in Physika function. To use it,
   add the following helper function to ``physika/runtime.py``.

   .. code-block:: python

        def create_heat_dataset(samples):
            import gzip
            import numpy as np
            import torch
            from torch.utils.data import TensorDataset, random_split
            from huggingface_hub import hf_hub_download

            repo_id = "nick-leland/heat1d-pde-dataset"

            initial_states_file_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="test/initial_states.npy.gz",
            )

            final_states_file_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="test/final_states.npy.gz",
            )

            with gzip.open(initial_states_file_path, "rb") as f:
                initial_states_data = np.load(f)

            with gzip.open(final_states_file_path, "rb") as f:
                final_states_data = np.load(f)

            initial_states_data = initial_states_data[:samples]
            final_states_data = final_states_data[:samples]

            X = torch.tensor(initial_states_data, dtype=torch.float32)
            Y = torch.tensor(final_states_data, dtype=torch.float32)

            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)

            dataset = TensorDataset(X, Y)

            train_len = int(0.8 * samples)
            test_len = samples - train_len

            train_subset, test_subset = random_split(
                dataset,
                [train_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )

            # Extract tensors from subsets
            train_X = X[train_subset.indices]
            train_Y = Y[train_subset.indices]

            test_X = X[test_subset.indices]
            test_Y = Y[test_subset.indices]

            return train_X, train_Y, test_X, test_Y



Runtime Helper functions
------------------------

Add this following code snippet to ``physika/runtime.py`` file.

.. code-block:: python

    def random_complex(*shape, scale=0.1):
        real = torch.randn(*shape) * scale
        imag = torch.randn(*shape) * scale
        return torch.complex(real, imag)


    def compl_mul1d(x_ft, weights1):
        return torch.einsum("ix,iox->ox", x_ft, weights1)

``random_complex`` :-  Physika's default random initializers only produce real tensors, so we need this to initialize the spectral
weights as proper complex numbers with separate real and imaginary components.

``compl_mul1d`` :- Physika currently dont have String as data type, so we can't pass ``ix, iox->ox`` as argument to einsum through
physika, thats why we have to add this helper function in runtime.


Training the FNO model
------------------------

The loss and train methods below are defined as part of the FNO1d class. 
For the loss function we use Mean Squared Error (MSE):

.. math::

    \mathcal{L}(\hat{u}, u) = \frac{1}{n} \sum_{i=1}^{n} (\hat{u}(x_i) - u(x_i))^2

which measures the average squared difference between the predicted and true output functions at each spatial point.
For optimization we use Stochastic Gradient Descent (SGD), which updates each parameter by stepping in the direction that reduces the loss:

.. math::

    \theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}


where :math:`\theta` represents any learnable parameter in the network, :math:`\eta` is the learning rate, and :math:`\nabla_\theta \mathcal{L}` is the
gradient of the loss with respect to that parameter. Each component of the model — p, conv0–conv3, mlp0–mlp3, w0–w3, and q — maintains its own parameters and receives its own gradient update independently.

.. code-block:: text

    def loss(pred: ℝ[1, n], label: ℝ[1, n]) -> ℝ:
        diff: ℝ[1, n] = pred - label
        return mean(diff**2)
    def train(X: ℝ[m, 1, n], y: ℝ[m, 1, n], epochs: ℕ, lr: ℝ) -> ℝ:
        len_dataset = len(X)
        for i:ℕ(epochs):
            epoch_loss = 0
            for j:ℕ(len_dataset):
                pred = this(X[j])
                current_loss = this.loss(pred, y[j])
                epoch_loss = epoch_loss + current_loss
                # ------------------------------------------------
                # Calculate gradients
                # ------------------------------------------------
                dp = grad(current_loss, this.p.learnable_params)
                dconv0 = grad(current_loss, this.conv0.learnable_params)
                dconv1 = grad(current_loss, this.conv1.learnable_params)
                dconv2 = grad(current_loss, this.conv2.learnable_params)
                dconv3 = grad(current_loss, this.conv3.learnable_params)
                dmlp0 = grad(current_loss, this.mlp0.learnable_params)
                dmlp1 = grad(current_loss, this.mlp1.learnable_params)
                dmlp2 = grad(current_loss, this.mlp2.learnable_params)
                dmlp3 = grad(current_loss, this.mlp3.learnable_params)
                dw0 = grad(current_loss, this.w0.learnable_params)
                dw1 = grad(current_loss, this.w1.learnable_params)
                dw2 = grad(current_loss, this.w2.learnable_params)
                dw3 = grad(current_loss, this.w3.learnable_params)
                dq = grad(current_loss, this.q.learnable_params)
                # ------------------------------------------------
                # update parameters
                # ------------------------------------------------
                this.p.update_params(lr, dp)
                this.conv0.update_params(lr, dconv0)
                this.conv1.update_params(lr, dconv1)
                this.conv2.update_params(lr, dconv2)
                this.conv3.update_params(lr, dconv3)
                this.mlp0.update_params(lr, dmlp0)
                this.mlp1.update_params(lr, dmlp1)
                this.mlp2.update_params(lr, dmlp2)
                this.mlp3.update_params(lr, dmlp3)
                this.q.update_params(lr, dq)
            last_loss = epoch_loss / len_dataset
        return last_loss


Results after training the model
---------------------------------

After training, to visualize the model's output we are using python code snippet which compares the predicted
final state and ground truth final state, Please add the below code snippet in ``physika/runtime.py`` 


.. code-block:: python

    def compare_states(pred_state, true_state):
        import matplotlib.pyplot as plt
        import numpy as np
        true_flat = true_state.detach().cpu().numpy().squeeze()
        pred_flat = pred_state.detach().cpu().numpy().squeeze()
        plt.figure(figsize=(10, 5))
        plt.plot(true_flat, color="black", linestyle="-", label="True state", lw=2)
        plt.plot(pred_flat, color="crimson", linestyle="--", label="Predicted State", lw=2)

        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best", fontsize=10)

        plt.tight_layout()
        plt.show()


.. figure:: /_static/tutorial_files/fno/fno_predictions_1.png
   :alt: FNO results 1
   :align: center
   :width: 700px

   FNO results 1

.. figure:: /_static/tutorial_files/fno/fno_predictions_2.png
   :alt: FNO results 2
   :align: center
   :width: 700px

   FNO results 2


Full code
---------

.. code-block:: text

    class Conv1d(W: ℝ[out, in], b: ℝ[out, 1]):
        def λ(x: ℝ[in, n]) -> ℝ[out, n]:
            z: ℝ[out, n] = this.W @ x + this.b
            return z
        def update_params(lr: ℝ, learnable_grads: ℝ[List]):
            this.W = this.W - lr * learnable_grads[0]
            this.b = this.b - lr * learnable_grads[1]


    class MLP(W1: ℝ[mid, in],  b1: ℝ[mid, 1], W2: ℝ[out, mid], b2: ℝ[out, 1]):
        def λ(x: ℝ[in, n]) -> ℝ[out, n]:
            z1: ℝ[mid, n] = this.W1 @ x + this.b1
            a1: ℝ[mid, n] = gelu(z1)
            z2: ℝ[mid, n] = this.W2 @ a1 + this.b2
            return z2
        def update_params(lr: ℝ, learnable_grads: ℝ[List]):
            this.W1 = this.W1 - lr * learnable_grads[0]
            this.b1 = this.b1 - lr * learnable_grads[1]
            this.W2 = this.W2 - lr * learnable_grads[2]
            this.b2 = this.b2 - lr * learnable_grads[3]


    class SpectralConv(weights1: ℂ[in_ch, out_ch, modes], in_ch: ℕ, out_ch: ℕ, modes: ℕ):
        def λ(x: ℂ[in_ch, n]) -> ℂ[out_ch, n]:
            x_ft: ℂ[in_ch, n_ft] = rfft(x)
            x_ft: ℂ[in_ch, modes] = x_ft[:, :self.modes]
            out_ft: ℂ[out_ch, modes] = compl_mul1d(x_ft, self.weights1)
            results: ℂ[out_ch, n] = irfft(out_ft, len(x[0]))
            return results
        def update_params(lr: ℝ, learnable_grads: ℝ[List]):
            this.weights1 = this.weights1 - lr * learnable_grads[0]


    width: ℕ = 16
    modes: ℕ = 8


    class FNO1d:
        p: Conv1d
        conv0: SpectralConv
        conv1: SpectralConv
        conv2: SpectralConv
        conv3: SpectralConv
        mlp0: MLP
        mlp1: MLP
        mlp2: MLP
        mlp3: MLP
        w0: Conv1d
        w1: Conv1d
        w2: Conv1d
        w3: Conv1d
        q: MLP
        def λ(x: R[1, n]) -> R[1, n]:
            # Lifting
            x: ℝ[width, n] = p(x)
            # Layer 1
            x1: ℝ[width, n] = conv0(x)
            x1: ℝ[width, n] = mlp0(x1)
            x2: ℝ[width, n] = w0(x)
            x: ℝ[width, n] = gelu(x1 + x2)
            # Layer 2
            x1: ℝ[width, n] = conv1(x)
            x1: ℝ[width, n] = mlp1(x1)
            x2: ℝ[width, n] = w1(x)
            x: ℝ[width, n] = gelu(x1 + x2)
            # Layer 3
            x1: ℝ[width, n] = conv2(x)
            x1: ℝ[width, n] = mlp2(x1)
            x2: ℝ[width, n] = w2(x)
            x: ℝ[width, n] = gelu(x1 + x2)
            # Layer 4
            x1: ℝ[width, n] = conv3(x)
            x1: ℝ[width, n] = mlp3(x1)
            x2: ℝ[width, n] = w3(x)
            x: ℝ[width, n] = x1 + x2
            # Projection
            x: ℝ[1, n] = q(x)
            return x
        def loss(pred: ℝ[1, n], label: ℝ[1, n]) -> ℝ:
            diff: ℝ[1, n] = pred - label
            return mean(diff**2)
        def train(X: ℝ[m, 1, n], y: ℝ[m, 1, n], epochs: ℕ, lr: ℝ) -> ℝ:
            len_dataset = len(X)
            for i:ℕ(epochs):
                epoch_loss = 0
                for j:ℕ(len_dataset):
                    pred = this(X[j])
                    current_loss = this.loss(pred, y[j])
                    epoch_loss = epoch_loss + current_loss
                    # ------------------------------------------------
                    # Calculate gradients
                    # ------------------------------------------------
                    dp = grad(current_loss, this.p.learnable_params)
                    dconv0 = grad(current_loss, this.conv0.learnable_params)
                    dconv1 = grad(current_loss, this.conv1.learnable_params)
                    dconv2 = grad(current_loss, this.conv2.learnable_params)
                    dconv3 = grad(current_loss, this.conv3.learnable_params)
                    dmlp0 = grad(current_loss, this.mlp0.learnable_params)
                    dmlp1 = grad(current_loss, this.mlp1.learnable_params)
                    dmlp2 = grad(current_loss, this.mlp2.learnable_params)
                    dmlp3 = grad(current_loss, this.mlp3.learnable_params)
                    dw0 = grad(current_loss, this.w0.learnable_params)
                    dw1 = grad(current_loss, this.w1.learnable_params)
                    dw2 = grad(current_loss, this.w2.learnable_params)
                    dw3 = grad(current_loss, this.w3.learnable_params)
                    dq = grad(current_loss, this.q.learnable_params)
                    # ------------------------------------------------
                    # update parameters
                    # ------------------------------------------------
                    this.p.update_params(lr, dp)
                    this.conv0.update_params(lr, dconv0)
                    this.conv1.update_params(lr, dconv1)
                    this.conv2.update_params(lr, dconv2)
                    this.conv3.update_params(lr, dconv3)
                    this.mlp0.update_params(lr, dmlp0)
                    this.mlp1.update_params(lr, dmlp1)
                    this.mlp2.update_params(lr, dmlp2)
                    this.mlp3.update_params(lr, dmlp3)
                    this.q.update_params(lr, dq)
                last_loss = epoch_loss / len_dataset
            return last_loss


    # --------------------------------------------------------
    # Initialize objects for each block
    # --------------------------------------------------------


    # --------------------------------------------------------
    # Lifting layer / Encoder (P)
    # --------------------------------------------------------

    Wp: ℝ[width, 1] = for i:ℕ(width) -> ε: ℝ[1] ~ Normal(0.0, 0.1, 1)
    Bp: ℝ[width, 1] = for i:ℕ(width) -> ε: ℝ[1] ~ Normal(0.0, 0.1, 1)

    p = Conv1d(Wp, Bp)


    # --------------------------------------------------------
    # Fourier layers - spectral (global) path
    # --------------------------------------------------------

    weights0: ℂ[width, width, modes] = random_complex(width, width, modes)
    weights1: ℂ[width, width, modes] = random_complex(width, width, modes)
    weights2: ℂ[width, width, modes] = random_complex(width, width, modes)
    weights3: ℂ[width, width, modes] = random_complex(width, width, modes)

    conv0: SpectralConv = SpectralConv(weights0, width, width, modes)
    conv1: SpectralConv = SpectralConv(weights1, width, width, modes)
    conv2: SpectralConv = SpectralConv(weights2, width, width, modes)
    conv3: SpectralConv = SpectralConv(weights3, width, width, modes)


    # --------------------------------------------------------
    # Local bypass path (W) — pointwise residual connection
    # --------------------------------------------------------

    Ww0: ℝ[width, width] = for i:ℕ(width) -> ε: ℝ[width] ~ Normal(0.0, 0.1, width)
    Bw0: ℝ[width, 1] = for i:ℕ(width) -> ε: ℝ[1] ~ Normal(0.0, 0.1, 1)

    w0: Conv1d = Conv1d(Ww0, Bw0)
    w1: Conv1d = Conv1d(Ww0, Bw0)
    w2: Conv1d = Conv1d(Ww0, Bw0)
    w3: Conv1d = Conv1d(Ww0, Bw0)

    # --------------------------------------------------------
    # MLP blocks (applied after each SpectralConv)
    # --------------------------------------------------------

    W1: ℝ[width, width] = for i:ℕ(width) -> ε: ℝ[width] ~ Normal(0.0, 0.1, width)
    b1: ℝ[width,1] = zeros(width, 1)
    W2: ℝ[width, width] = for i:ℕ(width) -> ε: ℝ[width] ~ Normal(0.0, 0.1, width)
    b2: ℝ[width,1] = zeros(width, 1)


    mlp0: MLP = MLP(W1, b1, W2, b2)
    mlp1: MLP = MLP(W1, b1, W2, b2)
    mlp2: MLP = MLP(W1, b1, W2, b2)
    mlp3: MLP = MLP(W1, b1, W2, b2)


    # --------------------------------------------------------
    # Projection / decoder (Q)
    # --------------------------------------------------------

    q_width: N = 2 * width
    Wq1: ℝ[q_width, width] = for i:ℕ(q_width) -> ε: ℝ[width] ~ Normal(0.0, 0.1, width)
    bq1: ℝ[q_width,1] = zeros(q_width, 1)
    Wq2: ℝ[1, q_width] = for i:ℕ(1) -> ε: ℝ[q_width] ~ Normal(0.0, 0.1, q_width)
    bq2: ℝ[1,1] = zeros(1, 1)

    q = MLP(Wq1, bq1, Wq2, bq2)



    # ----------------------------------------------------------------------------------------
    # Heat equation dataset (https://huggingface.co/datasets/nick-leland/heat1d-pde-dataset)
    # ----------------------------------------------------------------------------------------

    dataset = create_heat_dataset(50)
    train_X: ℝ[m,1,n] = dataset[0]
    train_y: ℝ[m,1,n] = dataset[1]
    test_X: ℝ[m,1,n] = dataset[0]
    test_y: ℝ[m,1,n] = dataset[1]


    # --------------------------------------------------------


    fno_obj: FNO1d = FNO1d(p, conv0, conv1, conv2, conv3, mlp0, mlp1, mlp2, mlp3, w0, w1, w2, w3, q)


    # --------------------------------------------------------
    # Training and evaluation FNO model
    # --------------------------------------------------------


    epochs: ℕ = 1
    loss: ℝ = fno_obj.train(train_X, train_y, epochs, 0.0001)
    loss


    test_idx: ℕ = 5
    pred: ℝ[m, n] = fno_obj(test_X[test_idx])
    compare_states(pred, test_y[test_idx])



References
----------

.. [LiKovachki2021] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K.,
  Stuart, A., & Anandkumar, A. (2021). *Fourier Neural Operator for
  Parametric Partial Differential Equations*. International Conference
  on Learning Representations (ICLR). https://arxiv.org/pdf/2010.08895

.. [HoraKapoorMatveev2026] Hora, G. S., Kapoor, P., & Matveev, A. (2026). *How a Fourier
  Neural Operator Learns to Solve PDEs — and Where It Falls Short*.
  PhysicsX Newsroom. https://www.physicsx.ai/newsroom/how-a-fourier-neural-operator-learns-to-solve-pdes----and-where-it-falls-short

.. [HeatDataset] Leland, N. (nick-leland). *heat1d-pde-dataset*. Hugging Face Datasets.
  https://huggingface.co/datasets/nick-leland/heat1d-pde-dataset

- `fourier_neural_operator: 1D Implementation (GitHub, wenhangao21) <https://github.com/wenhangao21/fourier_neural_operator/blob/master/fourier_1d.py>`_