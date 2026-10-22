Tiny Transformers
=================


In this tutorial we implement a decoder only Transformer architecture in Physika. [Vaswani2017]_ [ZeroEntropyDecoderOnly]_
We call it Tinytransformer because it has approx 280k total number of parameters. The model learns to predict the
next token based on given prompt.


Dataset and tokenizer
---------------------


The dataset consists total 50 sentences as a small text corpus. To use it paste the below text into ``corpus.txt`` file
which we will use in tokenizer. [SuperMK15TinyTransformer]_


.. code-block:: text

    i like dogs.
    you like cats.
    we love animals.
    cats are cute.
    dogs are loyal.
    the cat sleeps on the mat.
    the dog runs in the park.
    birds fly in the sky.
    a small bird sings a song.
    fish swim in the blue water.
    the big dog barks at the cat.
    the kitten plays with yarn.
    rabbits eat green grass.
    horses run fast in the field.
    the sun is bright and warm.
    the moon shines at night.
    i see a brown dog.
    you see a white cat.
    we see the green trees.
    animals need food and water.
    the puppy is very happy.
    the tiger lives in the forest.
    lions are strong and brave.
    elephants are very big.
    monkeys climb the tall trees.
    the rain falls on the flowers.
    flowers grow in the garden.
    bees make sweet honey.
    the bear sleeps in a cave.
    owls hunt at night.
    a frog jumps into the pond.
    the water is cold and clear.
    i feed the hungry birds.
    you pet the soft cat.
    we walk the dog every day.
    the hamster runs on a wheel.
    turtles move very slowly.
    the eagle flies high.
    butterflies have colorful wings.
    the grass is wet with dew.
    the fox hides in the bushes.
    a deer runs through the woods.
    the farm has many animals.
    cows give fresh milk.
    sheep have thick wool.
    the rooster wakes up the farm.
    ducks swim in a row.
    the sky is full of stars.
    nature is beautiful and quiet.
    i love my pet.


We are using word-level tokenizer which tokenizes each sentences into separate words. So for example the sentence
``I like dogs.`` gets tokenized as:

.. code-block:: python

    ['i', 'like', 'dogs', '.']


To create dataset add the below helper function in ``physika/runtime.py`` file

.. code-block:: python

    import re
    _CORPUS = "corpus.txt"

    with open(_CORPUS, "r") as f:
        _sentences = [line.strip() for line in f if line.strip()]


    def _tokenize(text):
        return re.findall(r"[\w']+|[.,!?;]|<end>", text)


    # Add end-of-sentence token
    _sentences = [s + " <end>" for s in _sentences]

    # Build vocabulary
    _all_tokens = []
    for sentence in _sentences:
        _all_tokens.extend(_tokenize(sentence))

    _vocab = sorted(set(_all_tokens))
    _vocab.append("<pad>")
    _vocab.append("<unk>")

    _stoi = {word: i for i, word in enumerate(_vocab)}
    _itos = {i: word for word, i in _stoi.items()}

    _PAD_IDX = _stoi["<pad>"]
    _UNK_IDX = _stoi["<unk>"]

    _VOCAB_SIZE = len(_vocab)

    # Longest tokenized sentence, including <end>
    _MAX_LEN = max(len(_tokenize(s)) for s in _sentences)

    def encode(text):
        """Convert text into token IDs."""
        return [_stoi.get(token, _UNK_IDX) for token in _tokenize(text)]


    def decode(tokens):
        """Convert token IDs back into text."""
        words = []

        for token in tokens:
            token_id = int(token)
            word = _itos[token_id]

            if word == "<end>":
                break

            if word == "<pad>":
                continue

            words.append(word)

        return " ".join(words)


    def create_dataset():
        """
        Create next-token prediction dataset.

        X contains all tokens except the final token.
        Y contains all tokens except the first token.
        """

        seq_len = _MAX_LEN - 1

        X = []
        Y = []

        for sentence in _sentences:
            tokens = encode(sentence)

            x = tokens[:-1]
            y = tokens[1:]

            x += [_PAD_IDX] * (seq_len - len(x))
            y += [_PAD_IDX] * (seq_len - len(y))

            X.append(x)
            Y.append(y)

        return (
            torch.tensor(X, dtype=torch.long),
            torch.tensor(Y, dtype=torch.long),
            _VOCAB_SIZE,
            _MAX_LEN
        )




Decoder only architecture
--------------------------

In this section we will build the decoder only architecture step by step, this section will get further divided into
sub-sections for each component of the architecture. Here is how the architecture looks like:



.. figure:: /_static/tutorial_files/tiny_transformers/tiny_transformer_decoder_architecture.png
   :alt: transformer_architecture
   :align: center
   :width: 700px

   Figure 1: TinyTransformer architecture (image generated using AI)


Embedding block
^^^^^^^^^^^^^^^

The embedding block converts input token IDs into learnable vectors which gets processed into next layers. for example
sentence ``i like dogs`` gets converted into token IDs as ``[12, 45, 87]``. The embedding block accepts these tokens
and creates vectors for each token

we implement embedding block in physika as:


.. code-block:: text

    class Embedding:
        weight: R[vocab_size, d_model]
        def λ(x: N[b, n]) → R[b, n, d_model]:
            batch_size: N = size(x, 0)
            seq_len: N = size(x, 1)
            return for i:ℕ(batch_size) → for j:ℕ(seq_len) → this.weight[x[i, j]]
        def update_params(learnable_grads: R[m, n]):
            this.weight = this.opt_weight.step(this.weight,learnable_grads[0])

here, `vocab_size` is total number of unique words present in our text corpus (for word-level tokenizer) plus additional words as ``<end>``,
``<pad>``, ``<unk>``, which gets us total vocab size as 146 and ``d_model`` as 128. The weight matrix therefore has shape (146, 128) and serves
as a lookup table, with each row storing the embedding vector for one token.


Positional embeddings
^^^^^^^^^^^^^^^^^^^^^

Unlike word embeddings which represents individual meaning of tokens, Positional embeddings provide information about
information about their position in a sequence. For examples the sentences ``i like dogs`` and ``dogs like i`` contains
same tokens but in a differen order. Positional embeddings helps model to distinguish these sequences by assinging a vector
to each position. These vectors are then added to token embeddings as a input to the transformer block.

In physika we pass positional embeddings as a matrix of shape (1, max_len, d_model) as a parameter to Tinytransformer which will
showcase later in this tutorial. Here `max_len` is maximum sequence length of our dataset returned by tokenizer.


Multihead attention
^^^^^^^^^^^^^^^^^^^

Multihead attention block is core of transformers, which allows transformer to learn relationships between tokens
in a sequence. For example in the sentence `i like dogs`, the model can learn how the word `like` relates to `i` and
`dogs`.

Here is the complete picture of Multihead attention block

.. figure:: /_static/tutorial_files/tiny_transformers/multi_head_attention_internals.png
   :alt: multi_head_attention
   :align: center
   :width: 700px

   Figure 2: Multi-head attention block (image generated using AI)

we implement this in physika as:

.. code-block:: text

    class MultiHeadAttention:
        d_model, nhead, d_k: N
        W_q, W_k, W_v, W_o: Linear3DnoBias
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            batch_size: N = size(x, 0)
            seq_len: N = size(x, 1)
            Q: R[b, n, d_model] = W_q(x)
            K: R[b, n, d_model] = W_k(x)
            V: R[b, n, d_model] = W_v(x)
            Q: R[b, nhead, n, d_k] = transpose(view(Q, batch_size, seq_len, nhead, d_k), 1, 2)
            K: R[b, nhead, n, d_k] = transpose(view(K, batch_size, seq_len, nhead, d_k), 1, 2)
            V: R[b, nhead, n, d_k] = transpose(view(V, batch_size, seq_len, nhead, d_k), 1, 2)
            scores = (Q @ transpose(K, 2, 3)) / sqrt(d_k)
            scores = scores + causal_mask(seq_len)
            weights: R[b, nhead, n, n] = softmax(scores, 3)
            context: R[b, nhead, n, d_k] = weights @ V
            context: R[b, n, nhead, d_k] = transpose(context, 1, 2)
            context: R[b, n, d_model] = reshape(context, batch_size, seq_len, d_model)
            return W_o(context)


Layernorm
^^^^^^^^^

Layer Normalization (LayerNorm) helps stabilize training by normalizing the features of each token independently, rather than across the batch.
It adjusts the values to have a mean of zero and variance of one, then applies learnable scaling and shifting parameters.

The operation is defined as

.. math::

    y = \frac{x - E[x]}
    {\sqrt{\operatorname{Var}(x) + \epsilon}}
    \ast \gamma + \beta

here is the physika implementation:

.. code-block:: text

    class LayerNorm:
        d_model: N
        γ, β: R[d_model]
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            μ: R[b, n, 1] = mean_keepdim(x, 2)
            variance: R[b, n, 1] = mean_keepdim((x - μ) * (x - μ), 2)
            normalized: R[b, n, d_model] = (
                (x - μ) / sqrt(variance + 1e-5)
            )
            γ_expanded: R[1, 1, d_model] = reshape(this.γ, 1, 1, d_model)
            β_expanded: R[1, 1, d_model] = reshape(this.β, 1, 1, d_model)
            return normalized * γ_expanded + β_expanded


Feed-forward network
^^^^^^^^^^^^^^^^^^^^

The feed-forward network processes each token independently to learn more complex features.
It consists of two fully connected layers with a ReLU activation function in between.

we can implement it in physika as:

.. code-block:: text
    
    class FeedForward:
        W1, W2: Linear3D
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            x = W1(x)
            x = relu(x)
            x = W2(x)
            return x

Transformer-block
^^^^^^^^^^^^^^^^^

We combine multi-head attention, layer normalization, and the feed-forward network to create a Transformer block.
In this tutorial, we use three Transformer blocks.

.. code-block:: text

    class TransformerBlock:
        attention: MultiHeadAttention
        norm1: LayerNorm
        ffn: FeedForward
        norm2: LayerNorm
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            x = norm1(x + attention(x))
            x = norm2(x + ffn(x))
            return x

Linear layer
^^^^^^^^^^^^

The final linear layer maps the Transformer output to the vocabulary size, producing a score for each possible next token.
These scores, called logits, are used to predict the next token.

We implement it in Physika as follows:

.. code-block:: text

    class Linear3D:
        W: R[out, in]
        b: R[out]
        def λ(x: R[b, n, in]) → R[b, n, out]:
            out_size: N = size(this.b, 0)
            bias: R[1, 1, out] = reshape(this.b, 1, 1, out_size)
            res: R[b, n, out] = x @ transpose(this.W, 0, 1) + bias
            return res

Tinytransformer class
---------------------

We can now combine the embedding layer, positional embeddings, Transformer blocks, and final linear layer to build the complete TinyTransformer model.

.. code-block:: text

    class TinyTransformer:
        embed: Embedding
        pos_embed: R[1, max_len, d_model]
        layers: list
        fc: Linear3D
        opt_pos_embed: AdamOptimizer3D
        def λ(x: N[b, n]) → R[b, n, vocab_size]:
            token_embeddings: R[b, n, d_model] = embed(x)
            seq_len: N = size(x, 1)
            position_embeddings: R[1, n, d_model] = pos_embed[:, :seq_len, :]
            x = token_embeddings + position_embeddings
            for i:N(len(layers)):
                layer
                 = layers[i]
                x = layer(x)
            return fc(x)


here layers is a list initialized with 3 Transformer-block class objects. We train this model
by minimizing the cross-entropy loss between the predicted logits and the target tokens.
The sequence loss is calculated by averaging the cross-entropy over all tokens in the batch.


For a sequence of logits (z) and target token (y), the cross-entropy loss is

.. math::

    \mathcal{L}(z,y) = -z_y + \log\left(\sum_{k=1}^{V} e^{z_k}\right)

where (V) is the vocabulary size, (z_k) is the logit for token (k), and (z_y) is the logit of the target token. For a batch of sequences, the average loss is

.. math::

    \mathcal{L}{\text{seq}} =
    \frac{1}{bn}\sum{i=1}^{b}\sum_{j=1}^{n}
    \mathcal{L}(z_{i,j}, y_{i,j})

we can implement this training loop as function to TinyTransformer class:

.. code-block:: text

    def train(X: N[b, n], Y: N[b, n], epochs: N):
        for epoch:N(epochs):
            logits = this(X)
            loss = cross_entropy_sequence(logits, Y)
            learnable_grads = grad(loss, this.learnable_params)
            this.update_params(learnable_grads)
            print(epoch)
            print(loss)

One the model is trained, we can pass prompts to the model using ``predict_tokens`` function as:

.. code-block:: text

    def predict_tokens(prompt: string, max_tokens: N): string:
        tokens: list = encode(prompt)
        generated: list = tokens
        prompt_len: N = len(tokens)
        remaining: N = max_len - prompt_len
        num_tokens: N = min(max_tokens, remaining)
        for i: ℕ(num_tokens):
            x = reshape(to_tensor(generated), 1, len(generated))
            logits = this(x)
            seq_len: ℕ = size(logits, 1)
            last_logits = logits[0, seq_len - 1]
            next_token_id: ℕ = argmax(last_logits)
            generated = append(generated, next_token_id)
        return decode(generated)

we can use this function as:

.. code-block:: text

    model.predict_tokens("i like", 2)
    model.predict_tokens("the puppy", 5)

See the `Full code` section for complete code.

Save and load model
-------------------

After training, we can save the model's learned parameters and load them later without training again.
Add below functions in ``physika/runtime.py`` file.

.. code-block:: python

    def save_model(model, name: str):
        import os
        path = os.path.abspath(f"{name}.pth")
        torch.save(model.state_dict(), path)
        print(f"Model saved to: {path}")

    def load_model(model, name: str):
        model.load_state_dict(torch.load(f"{name}.pth", map_location="cpu", weights_only=True))
        return model

We can train the TinyTransformer model, generate text, and save its learned parameters as follows:

.. code-block:: text

    model: TinyTransformer = TinyTransformer(
        embed,
        pos_embed,
        layers,
        fc,
        opt_pos_embed
    )

    epochs: N = 500
    model.train(X, Y, epochs)

    model.predict_tokens("i like", 2)


    save_model(model, "tiny_transformer")
    new_model = load_model(model, "tiny_transformer")

    model.predict_tokens("i like", 2)

The saved file, tiny_transformer.pth, contains the model's learned parameters.
Loading these parameters into a model allows us to reuse the trained model for text generation.


Full code
---------


.. code-block:: text

    def rand_3d_array(m: N, n: N, o: N, μ:ℝ): ℝ[m, n, o]:
        return for i:ℕ(m) → for j:ℕ(n) → for k:N(o) → μ * eps ~ 𝒩(0.0, 1.0)

    def rand_2d_array(n:N, m:N, μ:ℝ): ℝ[n, m]:
            return for i:ℕ(n) → for j:ℕ(m) → μ * eps ~ 𝒩(0.0, 1.0)

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total, temp: ℝ = 0, 0
        for i:
            temp = x[i]
            total += 1
        return total

    def argmax(iterable: ℝ[m]): ℕ:
        idx: ℕ = 0
        max_val: ℝ = iterable[0]
        len_iterable: ℕ = get_1d_array_length(iterable)
        for i:ℕ(1, len_iterable):
            if iterable[i] > max_val:
                max_val = iterable[i]
                idx = i
        return idx


    def cross_entropy(logits: ℝ[m], label: N): ℝ:
        vocab_size: N = size(logits, 0)
        total: ℝ = 0
        for i:ℕ(vocab_size):
            total += exp(logits[i])
        return -logits[label] + log(total)


    def cross_entropy_sequence(logits: ℝ[b, n, vocab_size],labels: N[b, n]): ℝ:
        batch_size: N = size(logits, 0)
        seq_len: N = size(logits, 1)
        total_loss: ℝ = 0
        for i:ℕ(batch_size):
            for j:ℕ(seq_len):
                total_loss += cross_entropy(logits[i, j], labels[i, j])
        return total_loss / (batch_size * seq_len)




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

    class AdamOptimizer3D:
        lr, beta1, beta2, eps, t: ℝ
        m, v: ℝ[m, n, o]
        def step(param: ℝ[m, n, o],grad: ℝ[m, n, o]) → ℝ[m, n, o]:
            this.t = this.t + 1.0
            this.m = this.beta1 * this.m + (1.0 - this.beta1) * grad
            this.v = this.beta2 * this.v + (1.0 - this.beta2) * grad**2
            m_hat: ℝ[m, n, o] = (this.m / (1.0 - this.beta1**this.t))
            v_hat: ℝ[m, n, o] = (this.v / (1.0 - this.beta2**this.t))
            param_new: ℝ[m, n, o] = param - this.lr * m_hat / (sqrt(v_hat) + this.eps)
            return param_new


    class Linear3DnoBias:
        W: R[out, in]
        opt_W: AdamOptimizer2D
        def λ(x: R[b, n, in]) → R[b, n, out]:
            res: R[b, n, out] = x @ transpose(this.W, 0, 1)
            return res
        def update_params(learnable_grads: R[m, n]):
            this.W = this.opt_W.step(this.W, learnable_grads[0])

    class Linear3D:
        W: R[out, in]
        b: R[out]
        opt_W: AdamOptimizer2D
        opt_b: AdamOptimizer1D
        def λ(x: R[b, n, in]) → R[b, n, out]:
            out_size: N = size(this.b, 0)
            bias: R[1, 1, out] = reshape(this.b, 1, 1, out_size)
            res: R[b, n, out] = x @ transpose(this.W, 0, 1) + bias
            return res
        def update_params(learnable_grads: ℝ[m]):
            this.W = this.opt_W.step(this.W,learnable_grads[0])
            this.b = this.opt_b.step(this.b,learnable_grads[1])


    class MultiHeadAttention:
        d_model, nhead, d_k: N
        W_q, W_k, W_v, W_o: Linear3DnoBias
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            batch_size: N = size(x, 0)
            seq_len: N = size(x, 1)
            Q: R[b, n, d_model] = W_q(x)
            K: R[b, n, d_model] = W_k(x)
            V: R[b, n, d_model] = W_v(x)
            Q: R[b, nhead, n, d_k] = transpose(view(Q, batch_size, seq_len, nhead, d_k), 1, 2)
            K: R[b, nhead, n, d_k] = transpose(view(K, batch_size, seq_len, nhead, d_k), 1, 2)
            V: R[b, nhead, n, d_k] = transpose(view(V, batch_size, seq_len, nhead, d_k), 1, 2)
            scores = (Q @ transpose(K, 2, 3)) / sqrt(d_k)
            scores = scores + causal_mask(seq_len)
            weights: R[b, nhead, n, n] = softmax(scores, 3)
            context: R[b, nhead, n, d_k] = weights @ V
            context: R[b, n, nhead, d_k] = transpose(context, 1, 2)
            context: R[b, n, d_model] = reshape(context, batch_size, seq_len, d_model)
            return W_o(context)
        def update_params(learnable_grads: R[m]):
            this.W_q.update_params(learnable_grads[0])
            this.W_k.update_params(learnable_grads[1])
            this.W_v.update_params(learnable_grads[2])
            this.W_o.update_params(learnable_grads[3])


    class LayerNorm:
        d_model: N
        γ, β: R[d_model]
        opt_γ: AdamOptimizer1D
        opt_β: AdamOptimizer1D
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            μ: R[b, n, 1] = mean_keepdim(x, 2)
            variance: R[b, n, 1] = mean_keepdim((x - μ) * (x - μ), 2)
            normalized: R[b, n, d_model] = (
                (x - μ) / sqrt(variance + 1e-5)
            )
            γ_expanded: R[1, 1, d_model] = reshape(this.γ, 1, 1, d_model)
            β_expanded: R[1, 1, d_model] = reshape(this.β, 1, 1, d_model)
            return normalized * γ_expanded + β_expanded
        def update_params(learnable_grads: R[m]):
            this.γ = this.opt_γ.step(this.γ,learnable_grads[0])
            this.β = this.opt_β.step(this.β,learnable_grads[1])


    class FeedForward:
        W1, W2: Linear3D
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            x = W1(x)
            x = relu(x)
            x = W2(x)
            return x
        def update_params(learnable_grads: R[m]):
            W1.update_params(learnable_grads[0])
            W2.update_params(learnable_grads[1])


    class TransformerBlock:
        attention: MultiHeadAttention
        norm1: LayerNorm
        ffn: FeedForward
        norm2: LayerNorm
        def λ(x: R[b, n, d_model]) → R[b, n, d_model]:
            x = norm1(x + attention(x))
            x = norm2(x + ffn(x))
            return x
        def update_params(learnable_grads: R[m]):
            attention.update_params(learnable_grads[0])
            norm1.update_params(learnable_grads[1])
            ffn.update_params(learnable_grads[2])
            norm2.update_params(learnable_grads[3])



    class Embedding:
        weight: R[vocab_size, d_model]
        opt_weight: AdamOptimizer2D
        def λ(x: N[b, n]) → R[b, n, d_model]:
            batch_size: N = size(x, 0)
            seq_len: N = size(x, 1)
            return for i:ℕ(batch_size) → for j:ℕ(seq_len) → this.weight[x[i, j]]
        def update_params(learnable_grads: R[m, n]):
            this.weight = this.opt_weight.step(this.weight,learnable_grads[0])


    def get_list_length(x: list): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def zero_1d_array(len: ℝ): list:
        results: list = for i: ℕ(len) -> i*0
        return results

    def append(x: list, var: ℝ): list:
        new_length: ℝ = get_list_length(x) + 1
        results: list = zero_1d_array(new_length)
        len_x: ℕ = get_list_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results


    class TinyTransformer:
        embed: Embedding
        pos_embed: R[1, max_len, d_model]
        layers: list
        fc: Linear3D
        opt_pos_embed: AdamOptimizer3D
        def λ(x: N[b, n]) → R[b, n, vocab_size]:
            token_embeddings: R[b, n, d_model] = embed(x)
            seq_len: N = size(x, 1)
            position_embeddings: R[1, n, d_model] = pos_embed[:, :seq_len, :]
            x = token_embeddings + position_embeddings
            for i:N(len(layers)):
                layer = layers[i]
                x = layer(x)
            return fc(x)
        def update_params(learnable_grads: R[m]):
            this.embed.update_params([learnable_grads[1]])
            this.pos_embed = this.opt_pos_embed.step(
                this.pos_embed, learnable_grads[0]
            )
            block_grads_size: N = 12
            for idx:N(len(this.layers)):
                start = 2 + idx * block_grads_size
                end = start + block_grads_size
                grads = learnable_grads[start:end]
                layer = layers[idx]
                layer.update_params(grads)
            fc_start = 2 + len(layers) * block_grads_size
            this.fc.update_params(learnable_grads[fc_start:fc_start + 2])
        def train(X: N[b, n], Y: N[b, n], epochs: N):
            for epoch:N(epochs):
                logits = this(X)
                loss = cross_entropy_sequence(logits, Y)
                learnable_grads = grad(loss, this.learnable_params)
                this.update_params(learnable_grads)
                print(epoch)
                print(loss)
        def predict_tokens(prompt: string, max_tokens: N): string:
            tokens: list = encode(prompt)
            generated: list = tokens
            prompt_len: N = len(tokens)
            remaining: N = max_len - prompt_len
            num_tokens: N = min(max_tokens, remaining)
            for i: ℕ(num_tokens):
                x = reshape(to_tensor(generated), 1, len(generated))
                logits = this(x)
                seq_len: ℕ = size(logits, 1)
                last_logits = logits[0, seq_len - 1]
                next_token_id: ℕ = argmax(last_logits)
                generated = append(generated, next_token_id)
            return decode(generated)



    # embedding - (146, 128)
    # pos_embed - (1, 9, 128)
    # 


    # --------------------------------------------------
    # Model configuration
    # --------------------------------------------------

    d_model: N = 128
    nhead: N = 8
    d_k: N = d_model / nhead

    dataset = create_dataset()
    X = dataset[0]
    Y = dataset[1]
    vocab_size = dataset[2]
    max_len = dataset[3]



    # --------------------------------------------------
    # Model construction
    # --------------------------------------------------

    embed: Embedding = Embedding(
        rand_2d_array(vocab_size, d_model, 0.02),
        AdamOptimizer2D(
            0.001, 0.9, 0.999, 1e-8, 0.0,
            zeros(vocab_size, d_model),
            zeros(vocab_size, d_model)
        )
    )

    pos_embed: R[1, max_len, d_model] = rand_3d_array(1, max_len, d_model, 1.0)

    opt_pos_embed: AdamOptimizer3D = AdamOptimizer3D(
        0.001, 0.9, 0.999, 1e-8, 0.0,
        zeros(1, max_len, d_model),
        zeros(1, max_len, d_model)
    )

    attention: MultiHeadAttention = MultiHeadAttention(
        d_model,
        nhead,
        d_k,
        Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
        Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
        Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
        Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model)))
    )


    norm1: LayerNorm = LayerNorm(
        d_model,
        ones(d_model),
        zeros(d_model),
        AdamOptimizer1D(
            0.001, 0.9, 0.999, 1e-8, 0.0,
            zeros(d_model),
            zeros(d_model)
        ),
        AdamOptimizer1D(
            0.001, 0.9, 0.999, 1e-8, 0.0,
            zeros(d_model),
            zeros(d_model)
        )
    )


    ffn: FeedForward = FeedForward(
        Linear3D(
            rand_2d_array(64, d_model, 0.02),
            zeros(64),
            AdamOptimizer2D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(64, d_model),
                zeros(64, d_model)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(64),
                zeros(64)
            )
        ),
        Linear3D(
            rand_2d_array(d_model, 64, 0.02),
            zeros(d_model),
            AdamOptimizer2D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model, 64),
                zeros(d_model, 64)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            )
        )
    )


    norm2: LayerNorm = LayerNorm(
        d_model,
        ones(d_model),
        zeros(d_model),
        AdamOptimizer1D(
            0.001, 0.9, 0.999, 1e-8, 0.0,
            zeros(d_model),
            zeros(d_model)
        ),
        AdamOptimizer1D(
            0.001, 0.9, 0.999, 1e-8, 0.0,
            zeros(d_model),
            zeros(d_model)
        )
    )



    block1: TransformerBlock = TransformerBlock(
        MultiHeadAttention(
            d_model,
            nhead,
            d_k,
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model)))
        ),
        LayerNorm(
            d_model,
            ones(d_model),
            zeros(d_model),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            )
        ),
        FeedForward(
            Linear3D(
                rand_2d_array(64, d_model, 0.02),
                zeros(64),
                AdamOptimizer2D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(64, d_model),
                    zeros(64, d_model)
                ),
                AdamOptimizer1D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(64),
                    zeros(64)
                )
            ),
            Linear3D(
                rand_2d_array(d_model, 64, 0.02),
                zeros(d_model),
                AdamOptimizer2D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(d_model, 64),
                    zeros(d_model, 64)
                ),
                AdamOptimizer1D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(d_model),
                    zeros(d_model)
                )
            )
        ),
        LayerNorm(
            d_model,
            ones(d_model),
            zeros(d_model),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            )
        )
    )
    block2: TransformerBlock = TransformerBlock(
        MultiHeadAttention(
            d_model,
            nhead,
            d_k,
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model)))
        ),
        LayerNorm(
            d_model,
            ones(d_model),
            zeros(d_model),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            )
        ),
        FeedForward(
            Linear3D(
                rand_2d_array(64, d_model, 0.02),
                zeros(64),
                AdamOptimizer2D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(64, d_model),
                    zeros(64, d_model)
                ),
                AdamOptimizer1D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(64),
                    zeros(64)
                )
            ),
            Linear3D(
                rand_2d_array(d_model, 64, 0.02),
                zeros(d_model),
                AdamOptimizer2D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(d_model, 64),
                    zeros(d_model, 64)
                ),
                AdamOptimizer1D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(d_model),
                    zeros(d_model)
                )
            )
        ),
        LayerNorm(
            d_model,
            ones(d_model),
            zeros(d_model),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            )
        )
    )
    block3: TransformerBlock = TransformerBlock(
        MultiHeadAttention(
            d_model,
            nhead,
            d_k,
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model))),
            Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001,0.9,0.999,1e-8,0.0,zeros(d_model, d_model),zeros(d_model, d_model)))
        ),
        LayerNorm(
            d_model,
            ones(d_model),
            zeros(d_model),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            )
        ),
        FeedForward(
            Linear3D(
                rand_2d_array(64, d_model, 0.02),
                zeros(64),
                AdamOptimizer2D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(64, d_model),
                    zeros(64, d_model)
                ),
                AdamOptimizer1D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(64),
                    zeros(64)
                )
            ),
            Linear3D(
                rand_2d_array(d_model, 64, 0.02),
                zeros(d_model),
                AdamOptimizer2D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(d_model, 64),
                    zeros(d_model, 64)
                ),
                AdamOptimizer1D(
                    0.001, 0.9, 0.999, 1e-8, 0.0,
                    zeros(d_model),
                    zeros(d_model)
                )
            )
        ),
        LayerNorm(
            d_model,
            ones(d_model),
            zeros(d_model),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            ),
            AdamOptimizer1D(
                0.001, 0.9, 0.999, 1e-8, 0.0,
                zeros(d_model),
                zeros(d_model)
            )
        )
    )

    layers: list = [block1, block2, block3]


    fc: Linear3D = Linear3D(
        rand_2d_array(vocab_size, d_model, 0.02),
        zeros(vocab_size),
        AdamOptimizer2D(
            0.001, 0.9, 0.999, 1e-8, 0.0,
            zeros(vocab_size, d_model),
            zeros(vocab_size, d_model)
        ),
        AdamOptimizer1D(
            0.001, 0.9, 0.999, 1e-8, 0.0,
            zeros(vocab_size),
            zeros(vocab_size)
        )
    )



    model: TinyTransformer = TinyTransformer(
        embed,
        pos_embed,
        layers,
        fc,
        opt_pos_embed
    )


    epochs: N = 500
    model.train(X, Y, epochs)

    model.predict_tokens("i like", 2)


    save_model(model, "tiny_transformer")
    new_model = load_model(model, "tiny_transformer")

    model.predict_tokens("i like", 2)



References
----------

.. [Vaswani2017] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones,
   A. N. Gomez, Ł. Kaiser, and I. Polosukhin, *Attention Is All You Need*,
   Advances in Neural Information Processing Systems 30, pp. 5998–6008, 2017.
   https://proceedings.neurips.cc/paper_files/paper/2017/file/
   3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

.. [ZeroEntropyDecoderOnly] ZeroEntropy, *Decoder-Only Models*.
   https://zeroentropy.dev/concepts/decoder-only-model/

.. [SuperMK15TinyTransformer] SuperMK15, *TinyTransformer: A Multihead
   Attention Tiny Transformer Built in Python*, GitHub repository.
   https://github.com/SuperMK15/TinyTransformer
