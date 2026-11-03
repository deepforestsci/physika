Convolutional Neural Networks 
=============================

In this tutorial we implemented Convolutional Neural networks in physika
and trained it on simple classification task with MNIST dataset. [CS231n]_, [DeepLizardCNN]_, [PineconeCNN]_ 

Dataset
--------

We trained CNN model on simple MNIST dataset for classifications task, for 10 classes numbers ranging from 0 to 10. [MNIST]_


.. code-block:: text

    dataset = create_dataset(80, 100)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    train_X = train_dataset[0]
    train_y = train_dataset[1]
    test_X = test_dataset[0]
    test_y = test_dataset[1]


.. note::
   ``create_dataset`` is not a built-in Physika function. To use it,
   add the following helper to ``physika/runtime.py``:

   .. code-block:: python

        def create_dataset(train_test_split = 80, total_dataset_size = 40):
            import torch
            from torchvision import datasets, transforms

            transform = transforms.ToTensor()

            mnist = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transform
            )

            X = []
            y = []

            # take first total_dataset_size samples
            for i in range(total_dataset_size):
                image, label = mnist[i]

                # [1,28,28] -> [28,28]
                image = image.squeeze(0)
                X.append(image)
                y.append(label)
            X = torch.stack(X)
            y = torch.tensor(y)

            # split index
            split_index = int(
                (train_test_split / 100.0)
                *
                total_dataset_size
            )

            # train split
            X_train = X[:split_index]
            y_train = y[:split_index]

            # test split
            X_test = X[split_index:]
            y_test = y[split_index:]
            train_data = [X_train, y_train]
            test_data = [X_test, y_test]
            return [train_data, test_data]


Each input or each single digit is size of 1x28x28, 1 is single channel image
of grayscale and 28x28 is Height and width

.. figure:: /_static/tutorial_files/cnn/single_digit.png
   :alt: 
   :align: center
   :width: 500px


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
        results: ℝ[rows, cols] = for i:ℕ(rows) -> for j:N(cols) -> j*0
        return results

    def get_sum_of_1d_array(x: ℝ[m]): ℝ:
        total = 0
        for i:
            total += x[i]
        return total

    def max(x: ℝ, y: ℝ): ℝ:
        if x>y:
            return x
        else:
            return y
    
    def rand_2d_array(n: ℝ, m: ℝ, μ: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → μ * eps ~ 𝒩(0.0, 1.0)



Activation functions
------------------------

After the convolution operation produces an output feature map,
we apply the ReLU activation function element-wise to every value.

ReLU helps the model learn non-linear patterns by removing negative values.

Mathematically, ReLU is defined as:

.. math::

    \text{ReLU}(x) = \max(0, x) = 
    \begin{cases} 
    x & \text{if } x > 0 \\ 
    0 & \text{if } x \le 0 
    \end{cases}


In the final layer of a convolutional neural network, the model produces raw output scores known as logits. We apply the softmax function to convert these 
logits into a probability distribution over the output classes.
Mathematically:

.. math::

    \sigma(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{m} e^{x_j}} \quad \text{for } i = 1, \dots, m


.. code-block:: text
        
    def relu(x: ℝ): ℝ:
        if x>0:
            return x
        else: 
            return 0.0

    def relu2d(x: ℝ[H, W]): ℝ[H, W]:
        rows: ℝ = get_2d_array_num_rows(x)
        cols: ℝ = get_1d_array_length(x[0])
        results: ℝ[rows, cols] = zero_2d_array(rows, cols)
        for i:ℕ(rows):
            for j:ℕ(cols):
                results[i, j] = relu(x[i, j])
        return results

    def softmax(x: ℝ[m]): ℝ[m]:
        len_x: ℝ = get_1d_array_length(x)
        exps_array: ℝ[m] = for i:ℕ(len_x) -> exp(x[i])
        total: ℝ = get_sum_of_1d_array(exps_array)
        results: ℝ[len_x] = for i:ℕ(len_x) -> exps_array[i] / total
        return results



ConvNet class 
------------------------

Lets bulid the full convnets class step by step.
The overall architecture of our network (forward pass) follows the pipeline:

``Convolution -> ReLU -> MaxPool -> Flatten -> Linear -> Softmax``

We will implement this each block in separate functions

1. Convolution block
~~~~~~~~~~~~~~~~~~~~~~

The convolution layer is the core building block of a convolutional neural
network.

It expects:

- an input image or feature map  (shape :- 28x28 for this dataset)
- a convolution kernel
- a bias term

Mathematically, the layer slides the kernel across the input tensor and computes
local weighted sums at every spatial location.


The input tensor is defined as:

.. math::

    input \in \mathbb{R}^{H \times W}

The convolution kernel is defined as:

.. math::

    kernel \in \mathbb{R}^{K \times K}

The layer produces a new feature map containing extracted spatial features such
as:

- edges
- textures
- local patterns


The output spatial dimensions are computed using:

.. math::

    H_{\text{out}} = \frac{(H - K) + 2P}{S} + 1

.. math::

    W_{\text{out}} = \frac{(W - K) + 2P}{S} + 1

where:

- :math:`H, W` are the input dimensions
- :math:`K` is the kernel size
- :math:`P` is the padding
- :math:`S` is the stride


The output feature map is therefore:

.. math::

    results \in
    \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}

.. figure:: /_static/tutorial_files/cnn/conv.gif
   :alt: 
   :align: center
   :width: 500px


Here we assume padding as 1 and stride as 1

.. code-block:: text

    def conv2d(input: ℝ[H, W], kernel: ℝ[K, K], b: ℝ): ℝ[m, n]:
        H_out: ℝ = get_2d_array_num_rows(input) - get_2d_array_num_rows(kernel) + 1
        W_out: ℝ = get_1d_array_length(input[0]) - get_1d_array_length(kernel[0]) + 1
        results: ℝ[H_out, W_out] = zeros(H_out, W_out)
        for i:ℕ(H_out):
            for j:ℕ(W_out):
                s: ℝ = 0
                for ki:ℕ(len(kernel)):
                    for kj:ℕ(len(kernel[0])):
                        s += input[i+ki, j+kj] * kernel[ki, kj]
                results[i, j] = s + b
        return results


2. Max Pooling Block
~~~~~~~~~~~~~~~~~~~~~~~


The max pooling layer reduces the spatial dimensions of the feature map while
preserving the most important activations. [PyTorchMaxPool]_


.. figure:: /_static/tutorial_files/cnn/max_pool.png
   :alt: 
   :align: center
   :width: 500px



The output spatial dimensions are computed using:

.. math::

    H_{\text{out}} =
    \frac{(H - K)}{S} + 1

.. math::

    W_{\text{out}} =
    \frac{(W - K)}{S} + 1

where:

- :math:`K` is the pooling window size
- :math:`S` is the stride


.. code-block:: text

    def maxpool2d(x: ℝ[m, n]): ℝ[m, n]:
        S: ℝ = 2
        r: ℝ = get_2d_array_num_rows(x) / 2
        c: ℝ = get_1d_array_length(x[0]) / 2
        Y: ℝ[r, c] = zeros(r, c)
        for i:ℕ(r):
            for j:ℕ(c):
                a = x[i*S, j*S]
                b = x[i*S, j*S + 1]
                c = x[i*2+1, j*2]
                d = x[i*2+1, j*2+1]
                Y[i, j] = max(a, max(b, max(c, d)))
        return Y


Flatten Block
~~~~~~~~~~~~~

The flatten layer converts a 2D feature map into a 1D vector so that it can be
fed into the linear classification layer.

The input tensor is defined as:

.. math::

    x \in \mathbb{R}^{H \times W}

The flattened vector length is computed as:

.. math::

    n = H \cdot W

The output tensor therefore becomes:

.. math::

    results \in \mathbb{R}^{n}

The flatten operation maps every spatial coordinate:

.. math::

    (i,j)

into a 1D index:

.. math::

    index = i \cdot W + j


Visualization of Convolution block + pooling block + neural network

.. figure:: /_static/tutorial_files/cnn/cnn_fcn.png
   :alt: 
   :align: center
   :width: 500px


.. code-block:: text

    def flatten(x: ℝ[m, n]): ℝ[n]:
        m: ℝ = get_2d_array_num_rows(x)
        n: ℝ = get_1d_array_length(x[0])
        L: ℝ = m*n
        z: ℝ[L] = zeros(L)
        for i:ℕ(m):
            for j:ℕ(n):
                z[i*n + j] = x[i, j]
        return z

    def linear(x: ℝ[I], weight: ℝ[O, I], bias: ℝ[O]): ℝ[O]:
        O: ℝ = get_1d_array_length(bias)
        I: ℝ = get_1d_array_length(x)
        results: ℝ[O] = zeros(O)
        for i:ℕ(O):
            s: ℝ = 0
            for j:ℕ(I):
                s += weight[i, j] * x[j]
            results[i] = s + bias[i]
        return results




Forward Pass
~~~~~~~~~~~~

The forward pass defines how data flows through the complete convolutional
neural network.

``onvolution -> ReLU -> MaxPool -> Convolution -> ReLU -> MaxPool -> Flatten -> Linear -> Softmax``

The Physika implementation is:


.. code-block:: text

    def λ(input: ℝ[H, W]) -> ℝ[m]:
        # layer 1
        conv1 = conv2d(input, kernel1, b1)
        relu1 = relu2d(conv1)
        pool1 = maxpool2d(relu1)
        # layer 2
        conv2 = conv2d(pool1, kernel2, b2)
        relu2 = relu2d(conv2)
        pool2 = maxpool2d(relu2)
        flat = flatten(pool2)
        out = linear(flat, w, b3)
        results = softmax(out)
        return results

For an MNIST image:

.. math::

    input \in \mathbb{R}^{28 \times 28}

After the first convolution with a :math:`3 \times 3` kernel:

.. math::

    28 \times 28 \rightarrow 26 \times 26

After the first max pooling operation with a stride of :math:`2`:

.. math::

    26 \times 26 \rightarrow 13 \times 13

The second convolution produces:

.. math::

    13 \times 13 \rightarrow 11 \times 11

After the second max pooling operation:

.. math::

    11 \times 11 \rightarrow 5 \times 5

The flatten layer converts the resulting feature map into a vector:

.. math::

    5 \times 5 \rightarrow 25

Therefore, the linear layer receives:

.. math::

    x \in \mathbb{R}^{25}

The linear layer computes:

.. math::

    y = Wx + b

where:

.. math::

    W \in \mathbb{R}^{10 \times 25}, \qquad b \in \mathbb{R}^{10}

Finally, softmax converts the output logits into probabilities over the 10 MNIST classes.


.. figure:: /_static/tutorial_files/cnn/softmax_output.png
   :alt: 
   :align: center
   :width: 500px



Initializing Convnet object
---------------------------

.. code-block:: text

    kernel1: ℝ[3, 3] = rand_2d_array(3, 3, 0.01)
    kernel2: ℝ[3, 3] = rand_2d_array(3, 3, 0.01)

    b1: ℝ = 0.0
    b2: ℝ = 0.0

    w: ℝ[10, 25] = rand_2d_array(10, 25, 0.1)

    b3: ℝ[10] = zeros(10)

    cnn_object: ConvNets = ConvNets(kernel1, b1, kernel2, b2, w, b3)


Define loss
---------------------------

For training the network we use the cross entropy loss function.
Cross entropy increases when the model predicts a low probability for the
correct class and decreases when the predicted probability is high.

.. math::

    \mathcal{L}(y, \hat{y}) = -\log(\hat{y}_y)

where:

- :math:`y` is the true class label
- :math:`\hat{y}_y` is the predicted probability of the correct class


.. code-block:: text

    def cross_entropy(probs: ℝ[m], label: ℝ): ℝ:
        p: ℝ = probs[label]
        return -log(p)


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

    def train(train_X: R[80, 28, 28], train_y: R[80], epochs: N, lr: R, train_dataset_len: R): R:
        loss: ℝ = 0
        for epoch:ℕ(epochs):
            print(epoch)
            epoch_loss: ℝ = 0
            for j:ℕ(train_dataset_len):
                input = train_X[j]
                label = train_y[j]
                z = this(input)
                current_loss = cross_entropy(z, label)
                epoch_loss += current_loss
                learnable_grads = grad(current_loss, this.learnable_params)
                this.update_params(lr, learnable_grads)
            epoch_loss: ℝ = epoch_loss / train_dataset_len
            print(epoch_loss)
        loss: ℝ = epoch_loss
        return loss


Testing the Model
-----------------

After training, we evaluate the model on unseen test data.

To obtain the predicted class from the output probability distribution, we use
the argmax operation.


The final classification accuracy is computed as:

.. math::

    \mathrm{accuracy} =
    \frac{\mathrm{correct\ predictions}}
    {\mathrm{total\ predictions}}


.. code-block:: text

    def argmax(iterable: R[m]): R:
        idx: ℝ = 0
        max_val: ℝ = iterable[0]
        len_iterable: ℝ = get_1d_array_length(iterable)
        for i:ℕ(1, len_iterable):
            if iterable[i] > max_val:
                max_val = iterable[i]
                idx = i
        return idx


    correct: ℝ = 0
    len_test_X: ℝ = get_1d_array_length(test_X)
    y_true: ℝ = 0

    for i:ℕ(len_test_X):
        x = test_X[i]
        y_true = test_y[i]
        y_pred = cnn_object(x)
        pred_class = argmax(y_pred)
        if pred_class == y_true:
            correct += 1
    accuracy = correct / len_test_X



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
        results: ℝ[rows, cols] = for i:N(rows) -> for j:N(cols) -> j*0
        return results

    def get_sum_of_1d_array(x: ℝ[m]): ℝ:
        total = 0
        for i:
            total += x[i]
        return total

    def max(x: ℝ, y: ℝ): ℝ:
        if x>y:
            return x
        else:
            return y

    def relu(x: ℝ): ℝ:
        if x>0:
            return x
        else: 
            return 0.0

    def relu2d(x: ℝ[H, W]): ℝ[H, W]:
        rows: ℝ = get_2d_array_num_rows(x)
        cols: ℝ = get_1d_array_length(x[0])
        results: ℝ[rows, cols] = zero_2d_array(rows, cols)
        for i:N(rows):
            for j:N(cols):
                results[i, j] = relu(x[i, j])
        return results

    def softmax(x: ℝ[m]): ℝ[m]:
        len_x: ℝ = get_1d_array_length(x)
        exps_array: ℝ[m] = for i:N(len_x) -> exp(x[i])
        total: ℝ = get_sum_of_1d_array(exps_array)
        results: ℝ[len_x] = for i:N(len_x) -> exps_array[i] / total
        return results


    def conv2d(input: ℝ[H, W], kernel: ℝ[K, K], b: ℝ): ℝ[m, n]:
        H_out: ℝ = get_2d_array_num_rows(input) - get_2d_array_num_rows(kernel) + 1
        W_out: ℝ = get_1d_array_length(input[0]) - get_1d_array_length(kernel[0]) + 1
        results: ℝ[H_out, W_out] = zeros(H_out, W_out)
        for i:ℕ(H_out):
            for j:ℕ(W_out):
                s: ℝ = 0
                for ki:ℕ(len(kernel)):
                    for kj:ℕ(len(kernel[0])):
                        s += input[i+ki, j+kj] * kernel[ki, kj]
                results[i, j] = s + b
        return results



    def maxpool2d(x: ℝ[m, n]): ℝ[m, n]:
        S: ℝ = 2
        r: ℝ = get_2d_array_num_rows(x) / 2
        c: ℝ = get_1d_array_length(x[0]) / 2
        Y: ℝ[r, c] = zeros(r, c)
        for i:ℕ(r):
            for j:ℕ(c):
                a = x[i*S, j*S]
                b = x[i*S, j*S + 1]
                c = x[i*2+1, j*2]
                d = x[i*2+1, j*2+1]
                Y[i, j] = max(a, max(b, max(c, d)))
        return Y

    def flatten(x: ℝ[m, n]): ℝ[n]:
        m: ℝ = get_2d_array_num_rows(x)
        n: ℝ = get_1d_array_length(x[0])
        L: ℝ = m*n
        z: ℝ[L] = zeros(L)
        for i:ℕ(m):
            for j:ℕ(n):
                z[i*n + j] = x[i, j]
        return z

    def linear(x: ℝ[I], weight: ℝ[O, I], bias: ℝ[O]): ℝ[O]:
        O: ℝ = get_1d_array_length(bias)
        I: ℝ = get_1d_array_length(x)
        results: ℝ[O] = zeros(O)
        for i:ℕ(O):
            s: ℝ = 0
            for j:ℕ(I):
                s += weight[i, j] * x[j]
            results[i] = s + bias[i]
        return results


    class ConvNets(kernel1: ℝ[K, K], b1: ℝ, kernel2: ℝ[K, K], b2: ℝ, w: ℝ[m, n], b3: ℝ[m]):
        def λ(input: ℝ[H, W]) -> ℝ[m]:
            # layer 1
            conv1 = conv2d(input, kernel1, b1)
            relu1 = relu2d(conv1)
            pool1 = maxpool2d(relu1)
            # layer 2
            conv2 = conv2d(pool1, kernel2, b2)
            relu2 = relu2d(conv2)
            pool2 = maxpool2d(relu2)
            flat = flatten(pool2)
            out = linear(flat, w, b3)
            results = softmax(out)
            return results
        def train(train_X: R[D, 28, 28], train_y: R[D], epochs: N, lr: R, train_dataset_len: R): R:
            loss: ℝ = 0
            for epoch:ℕ(epochs):
                print(epoch)
                epoch_loss: ℝ = 0
                for j:ℕ(train_dataset_len):
                    input = train_X[j]
                    label = train_y[j]
                    z = this(input)
                    current_loss = cross_entropy(z, label)
                    epoch_loss += current_loss
                    learnable_grads: list = ∇(current_loss, this.learnable_params)
                    this.update_params(lr, learnable_grads)
                epoch_loss: ℝ = epoch_loss / train_dataset_len
                print(epoch_loss)
            loss: ℝ = epoch_loss
            return loss
        def update_params(lr: R, learnable_grads: list):
            this.kernel1 = this.kernel1 - lr * learnable_grads[0]
            this.b1 = this.b1 - lr * learnable_grads[1]
            this.kernel2 = this.kernel2 - lr * learnable_grads[2]
            this.b2 = this.b2 - lr * learnable_grads[3]
            this.w = this.w - lr * learnable_grads[4]
            this.b3 = this.b3 - lr * learnable_grads[5]


    def rand_2d_array(n: ℝ, m: ℝ, μ: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → μ * eps ~ 𝒩(0.0, 1.0)

    kernel1: ℝ[3, 3] = rand_2d_array(3, 3, 0.01)
    kernel2: ℝ[3, 3] = rand_2d_array(3, 3, 0.01)

    b1: ℝ = 0.0
    b2: ℝ = 0.0

    w: ℝ[10, 25] = rand_2d_array(10, 25, 0.1)

    b3: ℝ[10] = zeros(10)

    cnn_object: ConvNets = ConvNets(kernel1, b1, kernel2, b2, w, b3)


    dataset = create_dataset(80, 10)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    train_dataset_len: ℝ = get_1d_array_length(train_dataset[0])
    test_dataset_len: ℝ = get_1d_array_length(test_dataset[0])

    train_X: ℝ[train_dataset_len, 28, 28] = train_dataset[0]
    train_y: ℝ[train_dataset_len] = train_dataset[1]
    test_X = test_dataset[0]
    test_y = test_dataset[1]

    def cross_entropy(probs: ℝ[m], label: ℝ): ℝ:
        p: ℝ = probs[label]
        return -log(p)

    epochs: ℕ = 50
    lr: ℝ = 0.1

    cnn_object.train(train_X, train_y, epochs, lr, train_dataset_len)

    def argmax(iterable: R[m]): R:
        idx: ℝ = 0
        max_val: ℝ = iterable[0]
        len_iterable: ℝ = get_1d_array_length(iterable)
        for i:ℕ(1, len_iterable):
            if iterable[i] > max_val:
                max_val = iterable[i]
                idx = i
        return idx


    correct: ℝ = 0
    len_test_X: ℝ = len(test_X[0])
    y_true: ℝ = 0

    for i:ℕ(test_dataset_len):
        x: ℝ[28, 28] = test_X[i]
        y_true = test_y[i]
        y_pred = cnn_object(x)
        pred_class = argmax(y_pred)
        if pred_class == y_true:
            correct += 1
    accuracy: ℝ = correct / len_test_X
    print(accuracy)


References
----------

.. [CS231n] Stanford University. *Convolutional Networks*. CS231n: Deep Learning for Computer Vision.
   https://cs231n.github.io/convolutional-networks/

.. [MNIST] Raschka, S. *MNIST Dataset*. mlxtend Documentation.
   https://rasbt.github.io/mlxtend/user_guide/data/mnist_data/

.. [PyTorchMaxPool] GeeksforGeeks. *Apply a 2D Max Pooling in PyTorch*.
   https://www.geeksforgeeks.org/computer-vision/apply-a-2d-max-pooling-in-pytorch/

.. [DeepLizardCNN] DeepLizard. *Convolutional Neural Networks (CNN) Tutorial*.
   https://deeplizard.com/resource/pavq7noze2

.. [PineconeCNN] Pinecone. *Convolutional Neural Networks*.
   https://www.pinecone.io/learn/series/image-search/cnn/