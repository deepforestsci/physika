Group Equivariant convolution networks
======================================


In :doc:`Convolution neural network tutorial <cnn_tutorial>` we implemented a simple CNN model and trained it to classify
MNIST dataset.
Generally CNNs handles translations effectively. In simple terms when image is shifted the features detected by convolution filters
also shifts accordingly. This is also called as translation invariance as shown in below picture.


.. figure:: /_static/tutorial_files/gcnn/invariance_example.png
   :alt: Invariance example
   :align: center
   :width: 500px

   Figure 1: Invariance example [invariance_image]_ 


However CNNs do not handle rotations in the same way. 
For example, a handwritten digit can appear in different orientations so ideally a model should recognize same digit regardless of its orientations.
which is also called as translation equivariance as show in below picture:


.. figure:: /_static/tutorial_files/gcnn/equivariance_example.png
   :alt: Equivariance example
   :align: center
   :width: 600px

   Figure 2: Equivariance example [equivariance_image]_ 


To handle this we use Group Equivariant convolution networks, which extends the CNNs to handle transformations such as rotations. They use group convolution
to detect features accross allowing different orientation.

In this tutorial, we will implement a GCNN model using :math:`P4` group. The :math:`P4` group contains four rotations: : 0°, 90°, 180°, and 270°.
These rotations allow the network detect features at different orientation.


Dataset
-------

We are using MNIST dataset, which are rotated by 0°, 90°, 180°, and 270°. This gives the network examples of the same
digit in four different orientations during both training and testing.
Add this below python script in ``physika/runtime.py`` file:  

.. code-block:: python


    def create_dataset(train_test_split=80, total_dataset_size=100, batch_size=8):
        import torch
        from torchvision import datasets, transforms

        mnist = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        # Number of original images needed
        num_original = total_dataset_size // 4

        X = []
        y = []

        for i in range(num_original):
            image, label = mnist[i]
            X.append(image)
            y.append(label)

        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.long)

        # Shuffle original images
        indices = torch.randperm(num_original)
        X = X[indices]
        y = y[indices]

        # Split original images
        split_index = int(train_test_split / 100.0 * num_original)

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Generate four orientations
        def rotate_images(X, y):
            X_rotated = torch.cat([
                X,
                torch.rot90(X, 1, [2, 3]),
                torch.rot90(X, 2, [2, 3]),
                torch.rot90(X, 3, [2, 3])
            ], dim=0)

            y_rotated = torch.cat([y, y, y, y], dim=0)

            indices = torch.randperm(len(X_rotated))
            return X_rotated[indices], y_rotated[indices]

        X_train, y_train = rotate_images(X_train, y_train)
        X_test, y_test = rotate_images(X_test, y_test)

        # Keep complete batches
        train_count = (len(X_train) // batch_size) * batch_size
        test_count = (len(X_test) // batch_size) * batch_size

        X_train, y_train = X_train[:train_count], y_train[:train_count]
        X_test, y_test = X_test[:test_count], y_test[:test_count]

        # Convert into batched tensors
        num_train_batches = train_count // batch_size
        num_test_batches = test_count // batch_size

        train_X_batches = X_train.view(num_train_batches, batch_size, 1, 28, 28)
        train_y_batches = y_train.view(num_train_batches, batch_size)

        test_X_batches = X_test.view(num_test_batches, batch_size, 1, 28, 28)
        test_y_batches = y_test.view(num_test_batches, batch_size)

        train_dataset = [train_X_batches, train_y_batches]
        test_dataset = [test_X_batches, test_y_batches]

        return [train_dataset, test_dataset]



Group convolution
-----------------

In general CNN model, convolution is performed over spatial dimensions
of an input image. Whereas in GCNN, convolution is extended to operate over
transformations represented by a group. [Cohen_Welling]_ [GCNN_GitHub]_ 


.. math::

   [f * \psi](g) =
   \sum_{h \in G} f(h)\psi(g^{-1}h)


where :math:`f` is the input feature map, :math:`\psi` is the convolution
filter, and :math:`g,h` are elements of the group.


We can divide this further into 2 classes to implement




Lifting Convolution
^^^^^^^^^^^^^^^^^^^


The first convolution layer maps the input image from the spatial domain
:math:`\mathbb{Z}^2` to feature maps defined over :math:`P4`.

For each filter, we generate four rotated filter versions of 0°, 90°, 180°, and 270°.
We then apply these rotated filters to the input image to obtain feature maps,
Mathematically, this operation can be written as:

.. math::

   [f * \psi](g) =
   \sum_{x \in \mathbb{Z}^2}
   f(x)\psi(g^{-1}x)


where :math:`f` is the input image, :math:`\psi` is the filter, and
:math:`g \in P4` represents an orientation and spatial translation.

we can represent this with physika code as:

.. code-block:: text

    class Z2P4Conv2d(w: ℝ[O, I, K, K], b: ℝ[O]):
        def λ(x: ℝ[B, I, H, W]) -> ℝ[B, O, 4, m, n]:
            B, I, H, W: ℝ = size(x)
            m: ℕ = H - K + 1
            n: ℕ = W - K + 1
            y: ℝ[B, O, 4, m, n] = zeros(B, O, 4, m, n)
            bias: ℝ[1, O, 1, 1] = reshape(b, 1, O, 1, 1)
            for r: ℕ(4):
                w_r: ℝ[O, I, K, K] = rot90(w, r, [2, 3])
                z: ℝ[B, O, m, n] = conv2d(x, w_r)
                y[:, :, r, :, :] = z + bias
            return y


P4-to-P4 Group Convolution
--------------------------

Since the first convolution layer produces four different feature maps, The next layer takes them as input
and performs convolutions over both spatial positions and orientation.

This operation is defined as:

.. math::

   [f * \psi]_j(g) =
   \sum_{h \in P4}\sum_k
   f_k(h)\psi_{j,k}(g^{-1}h)

where :math:`f_k` represents the input feature maps, :math:`\psi_{j,k}`
represents the filters, and :math:`j` and :math:`k` denote output and
input channels, respectively.

We can implement this layer in physika as:


.. code-block:: text

    class P4P4GConv2d(w: ℝ[O, I, 4, K, K], b: ℝ[O]):
        def λ(x: ℝ[B, I, 4, H, W]) -> ℝ[B, O, 4, m, n]:
            B, I, r, H, W: ℝ = size(x)
            m: ℕ = H - K + 1
            n: ℕ = W - K + 1
            x: ℝ[B, I*4, H, W] = reshape(x, B, I * 4, H, W)
            y: ℝ[B, O, 4, m, n] = zeros(B, O, 4, m, n)
            bias: ℝ[1, O, 1, 1] = reshape(b, 1, O, 1, 1)
            for r: ℕ(4):
                w_r: ℝ[O, I, 4, K, K] = roll(w, r, 2)
                w_r = rot90(w_r, r, [3, 4])
                w_r: ℝ[O, I*4, K, K] = reshape(w_r, O, I * 4, K, K)
                y[:, :, r, :, :] = conv2d(x, w_r) + bias
            return y


Both this layers uses ``conv2d`` which can be implement as:

.. code-block:: text

    def conv2d(input: ℝ[B, I, H, W], kernel: ℝ[O, I, K, K]): ℝ[B, O, m, n]:
        B, I, H, W: ℝ = size(input)
        O, I, K, K: ℝ = size(kernel)
        H_out: ℕ = H - K + 1
        W_out: ℕ = W - K + 1
        results: ℝ[B, O, m, n] = zeros(B, O, H_out, W_out)
        for b: ℕ(B):
            for o: ℕ(O):
                for h: ℕ(H_out):
                    for w: ℕ(W_out):
                        patch: ℝ[I, K, K] = input[b, :, h:h+K, w:w+K]
                        weights: ℝ[I, K, K] = kernel[o]
                        results[b, o, h, w] = sum(patch * weights)
        return results




Building the P4-GCNN
--------------------

we can combine this 2 convolution layers to build complete P4-GCNN model as:

.. code-block:: text

    class P4GConvNet(gconv1: Z2P4Conv2d, gconv2: P4P4GConv2d, fc: Linear):
        def λ(x: ℝ[B, 1, H, W]) -> ℝ[B, 10]:
            x: ℝ[B, O, 4, H, W] = relu5d(gconv1(pad2d(x, 1)))
            x: ℝ[B, O, 4, H, W] = p4_maxpooling(x)
            x: ℝ[B, O, 4, H, W] = relu5d(gconv2(x))
            x: ℝ[B, O, H, W] = gconv2d_maxpooling(x)
            x: ℝ[B, n] = sort_and_flatten(x)
            x: ℝ[B, 10] = fc(x)
            return softmax(x)


Zero padding
^^^^^^^^^^^^

Before applying the first convolution layer, we add zero padding to the
input image. Padding will add zeros around the image which helps
preserve its spatial dimensions during convolution.

For an input image of size :math:`H \times W` and padding size :math:`p`,
the output dimensions become:

.. math::

   H_{\text{out}} = H + 2p

   W_{\text{out}} = W + 2p

.. code-block:: text

    def pad2d(x: ℝ[B, C, H, W], p: ℕ): ℝ[B, C, m, n]:
       B, C, H, W = size(x)
       m: ℕ = H + 2 * p
       n: ℕ = W + 2 * p
       Y: ℝ[B, C, m, n] = zeros(B, C, m, n)
       Y[:, :, p:p + H, p:p + W] = x
       return Y


ReLU activation function
^^^^^^^^^^^^^^^^^^^^^^^^

We are using ReLU (Recitified Linear unit) activation function implemented as:


.. math::

   \operatorname{ReLU}(x) =
   \begin{cases}
       x & \text{if } x > 0, \\
       0 & \text{otherwise}.
   \end{cases}


.. code-block:: text

    def relu(x: ℝ): ℝ:
        if x > 0.0:
            return x
        else:
            return 0.0

    def relu5d(x: ℝ[B, O, 4, H, W]): ℝ[B, O, 4, H, W]:
        B, O, r, H, W = size(x)
        results: ℝ[B, O, 4, H, W] = zeros(B, O, 4, H, W)
        for b: ℕ(B):
            for o: ℕ(O):
                for r: ℕ(r):
                    for i: ℕ(H):
                        for j: ℕ(W):
                            results[b, o, r, i, j] = relu(x[b, o, r, i, j])
        return results



p4_maxpooling
^^^^^^^^^^^^^

Max pooling reduces the spatial dimensions of feature maps by selecting
the maximum value from each :math:`2 \times 2` region.

Mathematically, it is defined as:

.. math::

   Y[b,o,r,i,j] =
   \max_{\substack{0 \leq u < 2 \\ 0 \leq v < 2}}
   X[b,o,r,2i+u,2j+v]

.. code-block:: text

    def p4_maxpooling(x: ℝ[B, O, 4, H, W]): ℝ[B, O, 4, m, n]:
        B, O, r, H, W: ℝ = size(x)
        x: ℝ[B, O*4, H, W] = view(x, B, O * 4, H, W)
        x = maxpool2d(x)
        B, C, m, n: ℝ = size(x)
        x = view(x, B, O, 4, m, n)
        return x


P4 Max Pooling over Orientations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the second convolution layer, we apply max pooling over the
orientation dimension. For each spatial position and output channel,
we select the maximum value among the four orientations.

.. math::

   Y[B,O,i,j] = \max_{r \in \{0,1,2,3\}} X[B,O,r,i,j]


.. code-block:: text

   def gconv2d_maxpooling(x: ℝ[B, O, 4, H, W]) -> ℝ[B, O, m, n]:
       return max_value(p4_maxpooling(x), 2)

The ``p4_maxpooling`` function first reduces the spatial dimensions.
We then take the maximum over the orientation dimension, producing
feature maps of shape :math:`[B,O,m,n]`.


Sorting and Flattening
^^^^^^^^^^^^^^^^^^^^^^

After pooling over the orientation dimension, we obtain feature maps
of shape :math:`[B,O,H,W]`.

Before passing these feature maps to the fully connected layer, we
sort the spatial values of each channel and then flatten all channels
into a single vector.

The sorted feature maps are then flattened into a vector of size
:math:`O \times H \times W` for each batch element.


.. code-block:: text

    def sort_and_flatten(x: ℝ[B, O, H, W]): ℝ[B, n]:
        B, O, H, W: ℝ = size(x)
        x: ℝ[B, O, H*W] = view(x, B, O, H * W)
        x = sort(x, 2)
        x: ℝ[B, n] = view(x, B, O * H * W)
        return x


Fully connected network
^^^^^^^^^^^^^^^^^^^^^^^

After sorting and flattening, we pass the resulting feature vectors
to a fully connected layer. The layer applies a linear transformation
using learnable weights and bias.

Mathematically, it is defined as:

.. math::

   Y = XW^T + b

where :math:`X` is the input, :math:`W` is the weight matrix, and
:math:`b` is the bias vector.

In Physika, we implement this layer as follows:

.. code-block:: text

    class Linear(w: ℝ[O, I], b: ℝ[O]):
        def λ(x: ℝ[B, in]) -> ℝ[B, O]:
            B, I: ℝ = size(x)
            O, I: ℝ = size(w)
            y: ℝ[B, O] = zeros(B, O)
            for i: ℕ(B):
                xb = reshape(x[i], I, 1)
                yb = w @ xb
                y[i] = reshape(yb, O) + b
            return y

and we use batch wise ``softmax`` activation function for logits to classify between 10 MNIST digits, which can we implemented as:

.. code-block:: text

    def softmax(x: ℝ[B, M]): ℝ[B, M]:
        B, M: ℝ = size(x)
        E: ℝ[B, M] = zeros(B, M)
        Y: ℝ[B, M] = zeros(B, M)
        for i: ℕ(B):
            for j: ℕ(M):
                E[i, j] = exp(x[i, j])
        for i: ℕ(B):
            S: ℝ = sum(E[i])
            for j: ℕ(M):
                Y[i, j] = E[i, j] / S
        return Y


Training and evaluation
^^^^^^^^^^^^^^^^^^^^^^^

We can implement batched training loop in same ``P4GConvNet`` as a ``train`` function, During training, the network makes predictions, computes the loss, and
updates its learnable parameters using gradients.

we can implement train function as:

.. code-block:: text

    def train(train_X: ℝ[nB, B, 1, H, W], train_y: ℝ[nB, B], epochs: ℕ, lr: ℝ) -> ℝ:
        nB, B, C, H, W: ℝ = size(train_X)
        print(nB)
        loss: ℝ = 0
        for epoch: ℕ(epochs):
            print(epoch)
            epoch_loss: ℝ = 0
            for bidx: ℕ(nB):
                x: ℝ[B, C, H, W] = train_X[bidx]
                y: ℝ[B] = train_y[bidx]
                z: ℝ[B, 10] = this(x)
                current_loss: ℝ = cross_entropy(z, y)
                epoch_loss += current_loss
                learnable_grads: list = ∇(
                    current_loss,
                    this.learnable_params
                )
                this.update_params(lr, learnable_grads)
            epoch_loss = epoch_loss / nB
            print(epoch_loss)
        loss = epoch_loss
        return loss

Once training is complete, we evaluate the network on the test dataset.
We compare the predicted labels with the actual labels and calculate
the accuracy of model.

.. code-block:: text

    def evaluate(test_X: ℝ[nB, B, 1, H, W], test_y: ℝ[nB, B]) -> ℝ:
        nB, B, C, H, W: ℝ = size(test_X)
        correct: ℕ = 0
        total: ℕ = 0
        for bidx: ℕ(nB):
            x: ℝ[B, C, H, W] = test_X[bidx]
            y: ℝ[B] = test_y[bidx]
            logits: ℝ[B, 10] = this(x)
            predictions: ℝ[B] = argmax(logits)
            for idx: ℕ(B):
                if predictions[idx] == y[idx]:
                    correct = correct + 1
            total = total + B
        print(correct)
        accuracy: ℝ = correct / total
        return accuracy * 100.0


Full code
---------


.. code-block:: text

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def rand_5d_array(m: N, n: N, o: N, p: N, q: N, μ:ℝ): ℝ[m, n, o, p, q]:
        return for i:ℕ(m) → for j:ℕ(n) → for k:N(o) → for s:N(p) → for b:N(q) → μ * eps ~ 𝒩(0.0, 1.0)

    def rand_4d_array(m: N, n: N, o: N, p: N, μ:ℝ): ℝ[m, n, o, p]:
        return for i:ℕ(m) → for j:ℕ(n) → for k:N(o) → for s:N(p) → μ * eps ~ 𝒩(0.0, 1.0)

    def rand_3d_array(m: N, n: N, o: N, μ:ℝ): ℝ[m, n, o]:
        return for i:ℕ(m) → for j:ℕ(n) → for k:N(o) → μ * eps ~ 𝒩(0.0, 1.0)

    def rand_2d_array(n:N, m:N, μ:ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → μ * eps ~ 𝒩(0.0, 1.0)

    def rand_array(n: N, μ:ℝ): R[n]:
        return for i:ℕ(n) → μ * eps ~ 𝒩(0.0, 1.0)


    def pad2d(x: ℝ[B, C, H, W], p: ℕ): ℝ[B, C, m, n]:
        B, C, H, W = size(x)
        m: ℕ = H + 2 * p
        n: ℕ = W + 2 * p
        Y: ℝ[B, C, m, n] = zeros(B, C, m, n)
        Y[:, :, p:p + H, p:p + W] = x
        return Y

    def cross_entropy(probs: ℝ[B, M], labels: ℝ[B]): ℝ:
        B, M: ℝ = size(probs)
        q: ℝ[B] = for i:ℕ(B) -> probs[i, labels[i]]
        l = -sum(log(q)) / B
        return l

    def softmax(x: ℝ[B, M]): ℝ[B, M]:
        B, M: ℝ = size(x)
        E: ℝ[B, M] = zeros(B, M)
        Y: ℝ[B, M] = zeros(B, M)
        for i: ℕ(B):
            for j: ℕ(M):
                E[i, j] = exp(x[i, j])
        for i: ℕ(B):
            S: ℝ = sum(E[i])
            for j: ℕ(M):
                Y[i, j] = E[i, j] / S
        return Y


    def argmax1d(iterable: ℝ[m]): ℝ:
        idx: ℝ = 0
        max_val: ℝ = iterable[0]
        len_iterable: ℝ = get_1d_array_length(iterable)
        for i:ℕ(1, len_iterable):
            if iterable[i] > max_val:
                max_val: ℝ = iterable[i]
                idx: ℝ = i
        return idx

    def argmax(x: ℝ[B, C]): ℝ[B]:
        B, C: ℝ = size(x)
        I: ℝ[B] = zeros(B)
        for b: ℕ(B):
            I[b] = argmax1d(x[b])
        return I


    def relu(x: ℝ): ℝ:
        if x > 0.0:
            return x
        else:
            return 0.0

    def relu5d(x: ℝ[B, O, 4, H, W]): ℝ[B, O, 4, H, W]:
        B, O, r, H, W = size(x)
        results: ℝ[B, O, 4, H, W] = zeros(B, O, 4, H, W)
        for b: ℕ(B):
            for o: ℕ(O):
                for r: ℕ(r):
                    for i: ℕ(H):
                        for j: ℕ(W):
                            results[b, o, r, i, j] = relu(x[b, o, r, i, j])
        return results


    def conv2d(input: ℝ[B, I, H, W], kernel: ℝ[O, I, K, K]): ℝ[B, O, m, n]:
        B, I, H, W: ℝ = size(input)
        O, I, K, K: ℝ = size(kernel)
        H_out: ℕ = H - K + 1
        W_out: ℕ = W - K + 1
        results: ℝ[B, O, m, n] = zeros(B, O, H_out, W_out)
        for b: ℕ(B):
            for o: ℕ(O):
                for h: ℕ(H_out):
                    for w: ℕ(W_out):
                        patch: ℝ[I, K, K] = input[b, :, h:h+K, w:w+K]
                        weights: ℝ[I, K, K] = kernel[o]
                        results[b, o, h, w] = sum(patch * weights)
        return results


    def maxpool2d(x: ℝ[B, C, H, W]): ℝ[B, C, m, n]:
        B, C, H, W: ℝ = size(x)
        m: ℕ = H / 2
        n: ℕ = W / 2
        Y: ℝ[B, C, m, n] = zeros(B, C, m, n)
        for b: ℕ(B):
            for i: ℕ(m):
                for j: ℕ(n):
                    patch: ℝ[C, 2, 2] = x[b, :, i*2:i*2+2, j*2:j*2+2]
                    patch = max_value(patch, 2)
                    patch = max_value(patch, 1)
                    Y[b, :, i, j] = patch
        return Y


    def p4_maxpooling(x: ℝ[B, O, 4, H, W]): ℝ[B, O, 4, m, n]:
        B, O, r, H, W: ℝ = size(x)
        x: ℝ[B, O*4, H, W] = view(x, B, O * 4, H, W)
        x = maxpool2d(x)
        B, C, m, n: ℝ = size(x)
        x = view(x, B, O, 4, m, n)
        return x

    def gconv2d_maxpooling(x: ℝ[B, O, 4, H, W]): ℝ[B, O, m, n]:
        return max_value(p4_maxpooling(x), 2)

    def sort_and_flatten(x: ℝ[B, O, H, W]): ℝ[B, n]:
        B, O, H, W: ℝ = size(x)
        x: ℝ[B, O, H*W] = view(x, B, O, H * W)
        x = sort(x, 2)
        x: ℝ[B, n] = view(x, B, O * H * W)
        return x

    class Z2P4Conv2d(w: ℝ[O, I, K, K], b: ℝ[O]):
        def λ(x: ℝ[B, I, H, W]) -> ℝ[B, O, 4, m, n]:
            B, I, H, W: ℝ = size(x)
            m: ℕ = H - K + 1
            n: ℕ = W - K + 1
            y: ℝ[B, O, 4, m, n] = zeros(B, O, 4, m, n)
            bias: ℝ[1, O, 1, 1] = reshape(b, 1, O, 1, 1)
            for r: ℕ(4):
                w_r: ℝ[O, I, K, K] = rot90(w, r, [2, 3])
                z: ℝ[B, O, m, n] = conv2d(x, w_r)
                y[:, :, r, :, :] = z + bias
            return y
        def update_params(lr: R, learnable_grads: list):
            this.w = this.w - lr * learnable_grads[0]
            this.b = this.b - lr * learnable_grads[1]



    class P4P4GConv2d(w: ℝ[O, I, 4, K, K], b: ℝ[O]):
        def λ(x: ℝ[B, I, 4, H, W]) -> ℝ[B, O, 4, m, n]:
            B, I, r, H, W: ℝ = size(x)
            m: ℕ = H - K + 1
            n: ℕ = W - K + 1
            x: ℝ[B, I*4, H, W] = reshape(x, B, I * 4, H, W)
            y: ℝ[B, O, 4, m, n] = zeros(B, O, 4, m, n)
            bias: ℝ[1, O, 1, 1] = reshape(b, 1, O, 1, 1)
            for r: ℕ(4):
                w_r: ℝ[O, I, 4, K, K] = roll(w, r, 2)
                w_r = rot90(w_r, r, [3, 4])
                w_r: ℝ[O, I*4, K, K] = reshape(w_r, O, I * 4, K, K)
                y[:, :, r, :, :] = conv2d(x, w_r) + bias
            return y
        def update_params(lr: R, learnable_grads: list):
            this.w = this.w - lr * learnable_grads[0]
            this.b = this.b - lr * learnable_grads[1]

    class Linear(w: ℝ[O, I], b: ℝ[O]):
        def λ(x: ℝ[B, in]) -> ℝ[B, O]:
            B, I: ℝ = size(x)
            O, I: ℝ = size(w)
            y: ℝ[B, O] = zeros(B, O)
            for i: ℕ(B):
                xb = reshape(x[i], I, 1)
                yb = w @ xb
                y[i] = reshape(yb, O) + b
            return y
        def update_params(lr: R, learnable_grads: list):
            this.w = this.w - lr * learnable_grads[0]
            this.b = this.b - lr * learnable_grads[1]

    class P4GConvNet(gconv1: Z2P4Conv2d, gconv2: P4P4GConv2d, fc: Linear):
        def λ(x: ℝ[B, 1, H, W]) -> ℝ[B, 10]:
            x: ℝ[B, O, 4, H, W] = relu5d(gconv1(pad2d(x, 1)))
            x: ℝ[B, O, 4, H, W] = p4_maxpooling(x)
            x: ℝ[B, O, 4, H, W] = relu5d(gconv2(x))
            x: ℝ[B, O, H, W] = gconv2d_maxpooling(x)
            x: ℝ[B, n] = sort_and_flatten(x)
            x: ℝ[B, 10] = fc(x)
            return softmax(x)
        def train(train_X: ℝ[nB, B, 1, H, W], train_y: ℝ[nB, B], epochs: ℕ, lr: ℝ) -> ℝ:
            nB, B, C, H, W: ℝ = size(train_X)
            print(nB)
            loss: ℝ = 0
            for epoch: ℕ(epochs):
                print(epoch)
                epoch_loss: ℝ = 0
                for bidx: ℕ(nB):
                    x: ℝ[B, C, H, W] = train_X[bidx]
                    y: ℝ[B] = train_y[bidx]
                    z: ℝ[B, 10] = this(x)
                    current_loss: ℝ = cross_entropy(z, y)
                    epoch_loss += current_loss
                    learnable_grads: list = ∇(
                        current_loss,
                        this.learnable_params
                    )
                    this.update_params(lr, learnable_grads)
                epoch_loss = epoch_loss / nB
                print(epoch_loss)
            loss  = epoch_loss
            return loss
        def evaluate(test_X: ℝ[nB, B, 1, H, W], test_y: ℝ[nB, B]) -> ℝ:
            nB, B, C, H, W: ℝ = size(test_X)
            correct: ℕ = 0
            total: ℕ = 0
            for bidx: ℕ(nB):
                x: ℝ[B, C, H, W] = test_X[bidx]
                y: ℝ[B] = test_y[bidx]
                logits: ℝ[B, 10] = this(x)
                predictions: ℝ[B] = argmax(logits)
                for idx: ℕ(B):
                    if predictions[idx] == y[idx]:
                        correct = correct + 1
                total = total + B
            print(correct)
            accuracy: ℝ = correct / total
            return accuracy * 100.0
        def update_params(lr: N, learnable_grads: list):
            this.gconv1.update_params(lr, learnable_grads[:2])
            this.gconv2.update_params(lr, learnable_grads[2:4])
            this.fc.update_params(lr, learnable_grads[4:6])

    I: ℕ = 1
    O: ℕ = 6
    K: ℕ = 3

    H: ℕ = 28
    W: ℕ = 28
    B: ℕ = 4

    gconv1: Z2P4Conv2d = Z2P4Conv2d(
        rand_4d_array(O, I, K, K, 0.1),
        rand_array(O, 0.1)
    )
    gconv2: P4P4GConv2d = P4P4GConv2d(
        rand_5d_array(O, O, 4, K, K, 0.1),
        rand_array(O, 0.1)
    )


    H1: ℕ = H / 2
    W1: ℕ = W / 2
    H2: ℕ = H1 - K + 1
    W2: ℕ = W1 - K + 1
    H3: ℕ = H2 / 2
    W3: ℕ = W2 / 2
    flattened_size: ℕ = O * H3 * W3


    fc: Linear = Linear(
        rand_2d_array(10, flattened_size, 0.1),
        rand_array(10, 0.1)
    )


    model: P4GConvNet = P4GConvNet(gconv1, gconv2, fc)



    dataset = create_dataset(80, 10, 2)
    train_data = dataset[0]
    test_data = dataset[1]
    train_X: ℝ[nB, B, 1, 28, 28] = train_data[0]
    train_y: ℝ[nB, B] = train_data[1]
    test_X: ℝ[nB, B, 1, 28, 28] = test_data[0]
    test_y: ℝ[nB, B] = test_data[1]

    epochs: ℕ = 1
    lr: ℝ = 0.1


    train_loss: ℝ = model.train(train_X, train_y, epochs, lr)
    print(train_loss)

    #test_loss: ℝ = model.evaluate(test_X, test_y)
    accuracy: ℝ = model.evaluate(train_X, train_y)
    print(accuracy)


.. note::

    Training this model is currently slow in Physika, so we are using small dataset and overfitting it.



References
----------

.. [invariance_image] DPHI, *Image illustrating image rotation*.
   `<https://dphi-live.s3.amazonaws.com/media_uploads/image_9f667ea28fba4324902ae5d06d8d8907.png>`_.

.. [equivariance_image] Apple, *Image Augmentation: Rotation*.
   `<https://developer.apple.com/tutorials/images/com.apple.createml/MLImageClassifier-ImageAugmentationOptions-rotation-1@2x.png>`_.

.. [Cohen_Welling] T. S. Cohen and M. Welling, *Group Equivariant Convolutional Networks*,
   Proceedings of the 33rd International Conference on Machine Learning, 2016.
   `<https://arxiv.org/pdf/1602.07576>`_.

.. [GCNN_GitHub] Biiiilly, *Group Equivariant Network*, GitHub.
   `<https://github.com/Biiiilly/group_equivariant_network>`_.
