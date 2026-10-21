Recurrent Neural networks (RNNs)
================================

The complete code, including parameter estimation, is provided in the appendix ~\ref{app:schrodinger_equation}

In this tutorial, we will implement a simple RNN model from scratch in physika.
Recurrent Neural networks are models that are designed to process sequential data,
such as text, speech, and time series, where the order of elements is important. Unlike fully connected
networks, which process inputs independently, RNNs utilize recurrent connections, where the output of a neuron
at one time step is fed back as input to the network at the next time step


.. figure:: /_static/tutorial_files/rnn/rnn_cell.png
   :alt: rnn architecture
   :align: center
   :width: 500px
   :name: rnn architecture

   Figure 1: Flow of RNN architecture [RNN_Murf]_

Each RNN cell maintains a hidden state, which is a form of memory that gets updated at 
each time step based on the current input and previous hidden state. This allows the network
to learn from past inputs. [RNN_Wikipedia]_, [RNN_JakeTae]_


Dataset
--------

For this tutorial, we are using a dataset from PyTorch's RNN tutorial [RNN_PyTorch]_, we can also use this
shell script directly to download the dataset

.. code-block:: console

    !curl -O https://download.pytorch.org/tutorial/data.zip; unzip data.zip

Once the dataset gets downloaded, add the below python script into ``physika/runtime.py`` file:

.. code-block:: python

    def create_dataset(train_split):
        import os
        import random
        from string import ascii_letters
        import torch
        from unidecode import unidecode

        data_dir = "./data/names"

        lang2label = {
            file_name.split(".")[0]: torch.tensor([i], dtype=torch.long)
            for i, file_name in enumerate(os.listdir(data_dir))
        }

        # dict -> keys as name (data/names) and values as indexed tensors
        num_letters = len(lang2label)


        # all ascii_letters + some punctuation things
        char2idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}

        # total - 59 characters which is our vocabulory
        num_letters = len(char2idx)
        #print(num_letters)

        def name2tensor(name):
            tensor = torch.zeros(len(name), 1, num_letters)
            for i, char in enumerate(name):
                tensor[i][0][char2idx[char]] = 1
            return tensor

        tensor_names = []
        target_langs = []

        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file)) as f:
                lang = file.split(".")[0]
                names = [unidecode(line.rstrip()) for line in f]
                names = names[:100]
                for name in names:
                    try:
                        tensor_names.append(name2tensor(name))
                        target_langs.append(lang2label[lang])
                    except KeyError:
                        pass
        
        dataset = list(zip(tensor_names, target_langs))

        random.shuffle(dataset)

        split_idx = int(train_split * len(dataset))

        train_dataset = dataset[:split_idx]
        test_dataset = dataset[split_idx:]
        return [train_dataset, test_dataset]

``lang2label`` is a dictionary where keys are language classes (as indices) and its labels are represents as tensors in keys.
Therefore, we have a total of 18 different classes of different languages, and each class has names in the form of strings.


.. code-block:: text

    {
        'Arabic': tensor([1]),
        'Chinese': tensor([15]),
        'Czech': tensor([10]),
        'Dutch': tensor([0]),
        'English': tensor([13]),
        'French': tensor([8]),
        'German': tensor([2]),
        'Greek': tensor([11]),
        'Irish': tensor([5]),
        'Italian': tensor([17]),
        'Japanese': tensor([6]),
        'Korean': tensor([9]),
        'Polish': tensor([4]),
        'Portuguese': tensor([14]),
        'Russian': tensor([7]),
        'Scottish': tensor([12]),
        'Spanish': tensor([16]),
        'Vietnamese': tensor([3])
    }


After that we create ``char2idx`` which represents vocabulory as:

.. code-block:: text

    {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
        'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13,
        'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20,
        'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'A': 26, 'B': 27,
        'C': 28, 'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34,
        'J': 35, 'K': 36, 'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41,
        'Q': 42, 'R': 43, 'S': 44, 'T': 45, 'U': 46, 'V': 47, 'W': 48,
        'X': 49, 'Y': 50, 'Z': 51, ' ': 52, '.': 53, ',': 54, ':': 55,
        ';': 56, '-': 57, "'": 58
    }


So, in total, we have 59 distinct characters, each represented as its unique index. We then merge all.
names and its class label in ``tensor_names`` and ``target_langs`` respectively which then gets split
into ``train_dataset`` and ``test_dataset``.

Here is how the input tensor (name) and label (class) are represented for training:



.. code-block:: text

    dataset = create_dataset(0.9)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    sample = train_dataset[0]
    name = sample[0]
    target = sample[1]

    print(name)
    print(target)

.. code-block:: text

    tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0.]]])

    tensor([3])

The first tensor represents the name, while the second tensor represents
its class label. In this example, the class label is ``3``, which
corresponds to Vietnamese.

We can also recover the name from the first tensor. Its shape is
``[5, 1, 59]``:

* ``5`` represents the length of the name.
* ``1`` represents the batch size.
* ``59`` represents the vocabulary size, i.e., the total number of
  possible characters.

To recover the name, we look for the ``1`` in each one-hot encoded
character vector and use its index to look up the corresponding
character in our vocabulary.

The following table shows how the name can be recovered:


+----------+------------+-----------+
| Position | Index      | Character |
+==========+============+===========+
| 0        | 33         | ``H``     |
+----------+------------+-----------+
| 1        | 20         | ``u``     |
+----------+------------+-----------+
| 2        | 24         | ``y``     |
+----------+------------+-----------+
| 3        | 13         | ``n``     |
+----------+------------+-----------+
| 4        | 7          | ``h``     |
+----------+------------+-----------+


Therefore, the name represented by the tensor is ``Huynh`` and its class is Vietnamese.

.. note::

    The dataset is randomly sampled, so the name and its corresponding
    class may be different each time the dataset is created.


Defining the RNN model
----------------------

Before defining the RNN class itself, we first need Linear layers. The RNN class
uses linear layers to transform the input and hidden state into the next hidden state and 
next output.

Linear layer
~~~~~~~~~~~~

.. math::

    y = Wx + b


we can define this as:


.. code-block:: text

    class Linear:
        W: ℝ[out_features, in_features]
        b: ℝ[out_features]
        adam_W: AdamOptimizer2D
        adam_b: AdamOptimizer1D
        def λ(x: ℝ[in_features]) → ℝ[out_features]:
            in_features: ℝ = get_1d_array_length(x)
            col: ℝ[in_features, 1] = zeros(in_features, 1)
            col[:, 0] = x
            res: ℝ[out_features, 1] = W @ col
            out = res[:, 0] + b
            return out
        def update_params(learnable_grads: ℝ[m]):
            this.W = this.adam_W.step(this.W,learnable_grads[0])
            this.b = this.adam_b.step(this.b,learnable_grads[1])

RNN class
~~~~~~~~~~~~

In our RNN class, we are using 2 ``Linear`` layers. We call the first one ``in2hidden`` which transforms.
input into the next hidden state, and we call the second one ``in2output``, which transforms input into
output (prediction).


.. code-block:: text

    class RNN(H: ℕ, in2hidden: Linear, in2output: Linear):
        def λ(X: ℝ[B, T, I]) → ℝ[B, T, O]:
            dims = shape(X)
            B: ℕ = dims[0]
            T: ℕ = dims[1]
            out: ℝ[B, T, O] = zeros(B, T, O)
            for i:ℕ(B):
                hidden_state: ℝ[H] = this.init_hidden()
                for t:ℕ(T):
                    x: ℝ[I] = X[i, t]
                    combined: ℝ[I + H] = concat(x, hidden_state)
                    hidden_state: ℝ[H] = σ(in2hidden(combined))
                    out[i, t] = in2output(hidden_state)
            return out
        def init_hidden(): ℝ[m]:    
            return zeros(this.H)


Here ``σ`` is our sigmoid activation function defined as: [Sigmoid_Wikipedia]_

.. code-block:: text

    def σ(x: ℝ[a]): ℝ[a]:
        n: ℝ = get_1d_array_length(x)
        result: ℝ[a] = zeros(n)
        for i:ℕ(n):
            result[i] = 1.0 / (1.0 + exp(-x[i]))
        return result

At each timestep, the RNN receives two pieces of information:

* the current character (input) ``x``
* the hiddent state from previous timestep ``hidden_state``

we concatenate them into single vector called as ``combined`` which first gets passed into ``in2hidden`` linear layer
to generated next hidden state and then same ``combined`` gets passed into ``in2output`` which gives us
model's prediction. while ``init_hidden`` is used to generate the initial hidden state.

We can visualize this architecture as:

.. figure:: /_static/tutorial_files/rnn/rnn_arch.png
   :alt: rnn_architecture
   :align: center
   :width: 500px
   :name: rnn_architecture

   Figure 2: RNN architecture



Training loop
~~~~~~~~~~~~~

Since this is a character level model, in our training loop we will loop over each character at a time.
At each timestep, the model takes the current character and the previous hidden state, and produces a prediction for the class and
the next hidden state.

The prediction is represented by a vector of logits, with one value for each
of the 18 classes. We use cross-entropy loss to compare these logits with the
target class:

.. math::

    \mathcal{L} = -o_y + \log\left(\sum_{i=1}^{18} e^{o_i}\right)

where o is the vector of output logits and y is the target class index.
In Physika, we define the loss as:

.. code-block:: text

    def cross_entropy(logits: ℝ[m], label: N): ℝ:
        total: ℝ = 0
        for i:ℕ(18):
            total += exp(logits[i])
        return -logits[label] + log(total)

The loss is then used to compute gradients for the model parameters, which
are updated using the Adam optimizer.

Here is how the train function looks like:

.. code-block:: text

    def train(epochs: ℕ): ℝ:
        for epoch:ℕ(epochs):
            print(epoch)
            epoch_loss: ℝ = 0
            for idx:ℕ(train_dataset_len):
                sample = train_dataset[idx]
                name = sample[0]
                dims = shape(name)
                T: ℕ = dims[0]
                name: ℝ[1, T, I] = reshape(sample[0], 1, T, I)
                target: ℕ = sample[1]
                out: ℝ[1, T, O] = this(name)
                output: ℝ[O] = out[0, T - 1]
                loss: ℝ = cross_entropy(output, target)
                epoch_loss += detach(loss)
                grad_in2hidden = grad(loss, this.in2hidden.learnable_params)
                grad_in2output = grad(loss, this.in2output.learnable_params)
                this.in2hidden.update_params(grad_in2hidden)
                this.in2output.update_params(grad_in2output)
            epoch_loss: ℝ = epoch_loss / train_dataset_len
            print(epoch_loss)
        return loss


Evaluate the model
~~~~~~~~~~~~~~~~~~~

After training the model, we evaluate it on the test dataset. The test dataset
contains names that were not used during training.
For each name, we start with an initial hidden state and process the name one
character at a time, just as we did during training. 

After the last character has been processed, the model produces an output
logit for each of the 18 language classes. We use ``argmax`` to select the
class with the highest logit and compare it with the target class.

we can define argmax as:

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

and here is the complete evaluate function:

.. code-block:: text

    def evaluate() → ℝ:
        correct: ℝ = 0
        for idx: ℕ(test_dataset_len):
            sample = test_dataset[idx]
            name = sample[0]
            dims = shape(name)
            T: ℕ = dims[0]
            name: ℝ[1, T, I] = reshape(name, 1, T, 59)
            target: ℕ = sample[1]
            out: ℝ[1, T, O] = this(name)
            output: ℝ[O] = out[0, T - 1]
            predicted: ℝ = argmax(output)
            if predicted == target:
                correct += 1
        return correct / test_dataset_len


Full code
---------

.. code-block:: text


    def get_1d_array_length(x: ℝ[m]): ℝ:
        total, temp: ℝ = 0, 0
        for i:
            temp = x[i]
            total += 1
        return total

    def rand_2d_array(n:ℝ, m:ℝ, μ:ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → μ * eps ~ 𝒩(0.0, 1.0)

    def rand_array(x: ℝ): ℝ[m]:
        t ~ 𝒩(0, 1, x)
        return t

    def σ(x: ℝ[a]): ℝ[a]:
        n: ℝ = get_1d_array_length(x)
        result: ℝ[a] = zeros(n)
        for i:ℕ(n):
            result[i] = 1.0 / (1.0 + exp(-x[i]))
        return result

    def argmax(iterable: R[m]): R:
        idx: ℝ = 0
        max_val: ℝ = iterable[0]
        len_iterable: ℝ = get_1d_array_length(iterable)
        for i:ℕ(1, len_iterable):
            if iterable[i] > max_val:
                max_val = iterable[i]
                idx = i
        return idx


    dataset = create_dataset(0.9)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    train_dataset_len = get_1d_array_length(train_dataset)
    test_dataset_len = get_1d_array_length(test_dataset)
    print(train_dataset_len)
    print(test_dataset_len)


    class AdamOptimizer2D:
        lr, beta1, beta2, eps, t: ℝ
        m, v: ℝ[m, n]
        def step(param: ℝ[m, n], grad: ℝ[m, n]) → ℝ[m, n]:
            this.t = this.t + 1.0
            this.m = this.beta1 * this.m + (1.0 - this.beta1) * grad
            this.v = this.beta2 * this.v + (1.0 - this.beta2) * grad**2
            m_hat: ℝ[m, n] = this.m / (1.0 - this.beta1**this.t)
            v_hat: ℝ[m, n] = this.v / (1.0 - this.beta2**this.t)
            param_new: ℝ[m, n] = (
                param
                - this.lr * m_hat / (sqrt(v_hat) + this.eps)
            )
            return param_new


    class AdamOptimizer1D:
        lr, beta1, beta2, eps, t: ℝ
        m, v: ℝ[m]
        def step(param: ℝ[m], grad: ℝ[m]) → ℝ[m]:
            this.t = this.t + 1.0
            this.m = this.beta1 * this.m + (1.0 - this.beta1) * grad
            this.v = this.beta2 * this.v + (1.0 - this.beta2) * grad**2
            m_hat: ℝ[m] = this.m / (1.0 - this.beta1**this.t)
            v_hat: ℝ[m] = this.v / (1.0 - this.beta2**this.t)
            param_new: ℝ[m] = (
                param
                - this.lr * m_hat / (sqrt(v_hat) + this.eps)
            )
            return param_new

    def cross_entropy(logits: ℝ[m], label: N): ℝ:
        total: ℝ = 0
        for i:ℕ(18):
            total += exp(logits[i])
        return -logits[label] + log(total)

    class Linear:
        W: ℝ[out_features, in_features]
        b: ℝ[out_features]
        adam_W: AdamOptimizer2D
        adam_b: AdamOptimizer1D
        def λ(x: ℝ[in_features]) → ℝ[out_features]:
            in_features: ℝ = get_1d_array_length(x)
            col: ℝ[in_features, 1] = zeros(in_features, 1)
            col[:, 0] = x
            res: ℝ[out_features, 1] = W @ col
            out = res[:, 0] + b
            return out
        def update_params(learnable_grads: ℝ[m]):
            this.W = this.adam_W.step(this.W,learnable_grads[0])
            this.b = this.adam_b.step(this.b,learnable_grads[1])

    # I -> Input feature size (vocab)
    # O -> output class size
    # H -> hidden size (used in hidden state)

    I: ℝ = 59
    O: ℝ = 18
    H: ℕ = 128


    class RNN(H: ℕ, in2hidden: Linear, in2output: Linear):
        def λ(X: ℝ[B, T, I]) → ℝ[B, T, O]:
            dims = shape(X)
            B: ℕ = dims[0]
            T: ℕ = dims[1]
            out: ℝ[B, T, O] = zeros(B, T, O)
            for i:ℕ(B):
                hidden_state: ℝ[H] = this.init_hidden()
                for t:ℕ(T):
                    x: ℝ[I] = X[i, t]
                    combined: ℝ[I + H] = concat(x, hidden_state)
                    hidden_state: ℝ[H] = σ(in2hidden(combined))
                    out[i, t] = in2output(hidden_state)
            return out
        def init_hidden(): ℝ[m]:    
            return zeros(this.H)
        def train(epochs: ℕ): ℝ:
            for epoch:ℕ(epochs):
                print(epoch)
                epoch_loss: ℝ = 0
                for idx:ℕ(train_dataset_len):
                    sample = train_dataset[idx]
                    name = sample[0]
                    dims = shape(name)
                    T: ℕ = dims[0]
                    name: ℝ[1, T, I] = reshape(sample[0], 1, T, I)
                    target: ℕ = sample[1]
                    out: ℝ[1, T, O] = this(name)
                    output: ℝ[O] = out[0, T - 1]
                    loss: ℝ = cross_entropy(output, target)
                    epoch_loss += detach(loss)
                    grad_in2hidden = grad(loss, this.in2hidden.learnable_params)
                    grad_in2output = grad(loss, this.in2output.learnable_params)
                    this.in2hidden.update_params(grad_in2hidden)
                    this.in2output.update_params(grad_in2output)
                epoch_loss: ℝ = epoch_loss / train_dataset_len
                print(epoch_loss)
            return loss
        def evaluate() → ℝ:
            correct: ℝ = 0
            for idx: ℕ(test_dataset_len):
                sample = test_dataset[idx]
                name = sample[0]
                dims = shape(name)
                T: ℕ = dims[0]
                name: ℝ[1, T, I] = reshape(name, 1, T, 59)
                target: ℕ = sample[1]
                out: ℝ[1, T, O] = this(name)
                output: ℝ[O] = out[0, T - 1]
                predicted: ℝ = argmax(output)
                if predicted == target:
                    correct += 1
            return correct / test_dataset_len



    input_size: ℝ = 59
    output_size: ℝ = 18
    hidden_size: ℕ = 128


    # --------------------
    # weights for RNN
    # --------------------

    W1: ℝ[H, I+H] = rand_2d_array(H, I+H, 0.01)
    b1: ℝ[H] = rand_array(H)

    W2: ℝ[O, H] = rand_2d_array(O, H, 0.01)
    b2: ℝ[O] = rand_array(O)

    lr: ℝ = 0.001

    adam_W1: AdamOptimizer2D = AdamOptimizer2D(lr, 0.9, 0.999, 1e-8, 0.0, zeros(H, I + H), zeros(H, I + H))
    adam_b1: AdamOptimizer1D = AdamOptimizer1D(lr, 0.9, 0.999, 1e-8, 0.0, zeros(H), zeros(H))
    adam_W2: AdamOptimizer2D = AdamOptimizer2D(lr, 0.9, 0.999, 1e-8, 0.0,zeros(O, H),zeros(O, H))
    adam_b2: AdamOptimizer1D = AdamOptimizer1D(lr, 0.9, 0.999, 1e-8, 0.0, zeros(O), zeros(O) )


    in2hidden: Linear = Linear(W1, b1, adam_W1, adam_b1)
    in2output: Linear = Linear(W2, b2, adam_W2, adam_b2)

    rnn: RNN = RNN(H, in2hidden, in2output)

    epochs: ℕ = 10

    loss: ℝ = rnn.train(epochs)
    print(loss)

    accuracy: ℝ = rnn.evaluate()
    print(accuracy)




References
----------

.. [RNN_Wikipedia] Wikipedia. *Recurrent neural network*.
   https://en.wikipedia.org/wiki/Recurrent_neural_network

.. [RNN_Murf] Murf AI. *Recurrent Neural Network (RNN)*.
   https://murf.ai/ai-glossary/recurrent-neural-network

.. [Sigmoid_Wikipedia] Wikipedia. *Sigmoid function*.
   https://en.wikipedia.org/wiki/Sigmoid_function

.. [RNN_JakeTae] Jake Tae. *PyTorch RNN*.
   https://jaketae.github.io/study/pytorch-rnn/

.. [RNN_PyTorch] PyTorch. *Classifying Names with a Character-Level RNN*.
   https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
