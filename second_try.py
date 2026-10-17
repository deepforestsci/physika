import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import compute_grad, size, transpose, view, causal_mask, softmax, relu, encode_text, to_tensor, decode_token, get_vocab_size, get_max_len, create_dataset

# === Functions ===
def rand_3d_array(m, n, o, μ):
    return torch.stack([torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_k in range(int(o)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(m)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_2d_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def get_1d_array_length(x):
    total, temp = 0, 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def argmax(iterable):
    idx = 0
    max_val = iterable[int(0)]
    len_iterable = get_1d_array_length(iterable)
    for i in range(int(1), int(len_iterable)):
        if iterable[int(i)] > max_val:
            max_val = iterable[int(i)]
            idx = i
    return idx

def cross_entropy(logits, label):
    vocab_size = size(logits, 0)
    total = 0
    for i in range(int(0), int(vocab_size)):
        total = total + torch.exp(logits[int(i)] if isinstance(logits[int(i)], torch.Tensor) else torch.tensor(float(logits[int(i)])))
    return ((-logits[int(label)]) + torch.log(total if isinstance(total, torch.Tensor) else torch.tensor(float(total))))

def cross_entropy_sequence(logits, labels):
    batch_size = size(logits, 0)
    seq_len = size(logits, 1)
    total_loss = 0
    for i in range(int(0), int(batch_size)):
        for j in range(int(0), int(seq_len)):
            total_loss = total_loss + cross_entropy(logits[int(i), int(j)], labels[int(i), int(j)])
    return (total_loss / (batch_size * seq_len))

# === Classes ===
class AdamOptimizer1D(nn.Module):
    def __init__(self, lr, beta1, beta2, eps, t, m, v):
        super().__init__()
        self.lr = torch.as_tensor(lr).float()
        self.beta1 = torch.as_tensor(beta1).float()
        self.beta2 = torch.as_tensor(beta2).float()
        self.eps = torch.as_tensor(eps).float()
        self.t = torch.as_tensor(t).float()
        self.m = torch.as_tensor(m).float()
        self.v = torch.as_tensor(v).float()

    def step(self, param, grad):
        this = self
        param = torch.as_tensor(param, device=DEVICE).float()
        grad = torch.as_tensor(grad, device=DEVICE).float()
        with torch.no_grad():
            self.t.copy_((self.t + 1.0))
        with torch.no_grad():
            self.m.copy_(((self.beta1 * self.m) + ((1.0 - self.beta1) * grad)))
        with torch.no_grad():
            self.v.copy_(((self.beta2 * self.v) + ((1.0 - self.beta2) * (grad ** 2))))
        m_hat = (self.m / (1.0 - (self.beta1 ** self.t)))
        v_hat = (self.v / (1.0 - (self.beta2 ** self.t)))
        param_new = (param - ((self.lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + self.eps)))
        return param_new

class AdamOptimizer2D(nn.Module):
    def __init__(self, lr, beta1, beta2, eps, t, m, v):
        super().__init__()
        self.lr = torch.as_tensor(lr).float()
        self.beta1 = torch.as_tensor(beta1).float()
        self.beta2 = torch.as_tensor(beta2).float()
        self.eps = torch.as_tensor(eps).float()
        self.t = torch.as_tensor(t).float()
        self.m = torch.as_tensor(m).float()
        self.v = torch.as_tensor(v).float()

    def step(self, param, grad):
        this = self
        param = torch.as_tensor(param, device=DEVICE).float()
        grad = torch.as_tensor(grad, device=DEVICE).float()
        with torch.no_grad():
            self.t.copy_((self.t + 1.0))
        with torch.no_grad():
            self.m.copy_(((self.beta1 * self.m) + ((1.0 - self.beta1) * grad)))
        with torch.no_grad():
            self.v.copy_(((self.beta2 * self.v) + ((1.0 - self.beta2) * (grad ** 2))))
        m_hat = (self.m / (1.0 - (self.beta1 ** self.t)))
        v_hat = (self.v / (1.0 - (self.beta2 ** self.t)))
        param_new = (param - ((self.lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + self.eps)))
        return param_new

class AdamOptimizer3D(nn.Module):
    def __init__(self, lr, beta1, beta2, eps, t, m, v):
        super().__init__()
        self.lr = torch.as_tensor(lr).float()
        self.beta1 = torch.as_tensor(beta1).float()
        self.beta2 = torch.as_tensor(beta2).float()
        self.eps = torch.as_tensor(eps).float()
        self.t = torch.as_tensor(t).float()
        self.m = torch.as_tensor(m).float()
        self.v = torch.as_tensor(v).float()

    def step(self, param, grad):
        this = self
        param = torch.as_tensor(param, device=DEVICE).float()
        grad = torch.as_tensor(grad, device=DEVICE).float()
        with torch.no_grad():
            self.t.copy_((self.t + 1.0))
        with torch.no_grad():
            self.m.copy_(((self.beta1 * self.m) + ((1.0 - self.beta1) * grad)))
        with torch.no_grad():
            self.v.copy_(((self.beta2 * self.v) + ((1.0 - self.beta2) * (grad ** 2))))
        m_hat = (self.m / (1.0 - (self.beta1 ** self.t)))
        v_hat = (self.v / (1.0 - (self.beta2 ** self.t)))
        param_new = (param - ((self.lr * m_hat) / (torch.sqrt(v_hat if isinstance(v_hat, torch.Tensor) else torch.tensor(float(v_hat))) + self.eps)))
        return param_new

class Linear3DnoBias(nn.Module):
    def __init__(self, W, opt_W):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.add_module('opt_W', opt_W)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        res = (x @ transpose(self.W, 0, 1))
        return res

    def update_params(self, learnable_grads):
        this = self
        with torch.no_grad():
            self.W.copy_(self.opt_W.step(self.W, learnable_grads[int(0)]))

class Linear3D(nn.Module):
    def __init__(self, W, b, opt_W, opt_b):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.add_module('opt_W', opt_W)
        self.add_module('opt_b', opt_b)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        out_size = size(self.b, 0)
        bias = torch.reshape(self.b, (1, 1, out_size))
        res = ((x @ transpose(self.W, 0, 1)) + bias)
        return res

    def update_params(self, learnable_grads):
        this = self
        with torch.no_grad():
            self.W.copy_(self.opt_W.step(self.W, learnable_grads[int(0)]))
        with torch.no_grad():
            self.b.copy_(self.opt_b.step(self.b, learnable_grads[int(1)]))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, d_k, W_q, W_k, W_v, W_o):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.d_k = int(d_k)
        self.add_module('W_q', W_q)
        self.add_module('W_k', W_k)
        self.add_module('W_v', W_v)
        self.add_module('W_o', W_o)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        batch_size = size(x, 0)
        seq_len = size(x, 1)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = transpose(view(Q, batch_size, seq_len, self.nhead, self.d_k), 1, 2)
        K = transpose(view(K, batch_size, seq_len, self.nhead, self.d_k), 1, 2)
        V = transpose(view(V, batch_size, seq_len, self.nhead, self.d_k), 1, 2)
        scores = ((Q @ transpose(K, 2, 3)) / torch.sqrt(self.d_k if isinstance(self.d_k, torch.Tensor) else torch.tensor(float(self.d_k))))
        scores = (scores + causal_mask(seq_len))
        weights = softmax(scores, 3)
        context = (weights @ V)
        context = transpose(context, 1, 2)
        context = torch.reshape(context, (batch_size, seq_len, self.d_model))
        return self.W_o(context)

class LayerNorm(nn.Module):
    def __init__(self, d_model, γ, β, opt_γ, opt_β):
        super().__init__()
        self.d_model = int(d_model)
        self.γ = nn.Parameter(torch.as_tensor(γ))
        self.β = nn.Parameter(torch.as_tensor(β))
        self.add_module('opt_γ', opt_γ)
        self.add_module('opt_β', opt_β)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        μ = torch.mean(x, dim=2, keepdim=True)
        variance = torch.mean(((x - μ) * (x - μ)), dim=2, keepdim=True)
        normalized = ((x - μ) / torch.sqrt((variance + 1e-05) if isinstance((variance + 1e-05), torch.Tensor) else torch.tensor(float((variance + 1e-05)))))
        γ_expanded = torch.reshape(self.γ, (1, 1, self.d_model))
        β_expanded = torch.reshape(self.β, (1, 1, self.d_model))
        return ((normalized * γ_expanded) + β_expanded)

    def update_params(self, learnable_grads):
        this = self
        with torch.no_grad():
            self.γ.copy_(self.opt_γ.step(self.γ, learnable_grads[int(0)]))
        with torch.no_grad():
            self.β.copy_(self.opt_β.step(self.β, learnable_grads[int(1)]))

class FeedForward(nn.Module):
    def __init__(self, W1, W2):
        super().__init__()
        self.add_module('W1', W1)
        self.add_module('W2', W2)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x = self.W1(x)
        x = relu(x)
        x = self.W2(x)
        return x

    def update_params(self, learnable_grads):
        this = self
        self.W1.update_params(learnable_grads[int(0)])
        self.W2.update_params(learnable_grads[int(1)])

class TransformerBlock(nn.Module):
    def __init__(self, attention, norm1, ffn, norm2):
        super().__init__()
        self.add_module('attention', attention)
        self.add_module('norm1', norm1)
        self.add_module('ffn', ffn)
        self.add_module('norm2', norm2)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        x = self.norm1((x + self.attention(x)))
        x = self.norm2((x + self.ffn(x)))
        return x

    def update_params(self, learnable_grads):
        this = self
        self.attention.update_params(learnable_grads[int(0)])
        self.norm1.update_params(learnable_grads[int(1)])
        self.ffn.update_params(learnable_grads[int(2)])
        self.norm2.update_params(learnable_grads[int(3)])

class Embedding(nn.Module):
    def __init__(self, weight, opt_weight):
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(weight))
        self.add_module('opt_weight', opt_weight)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        batch_size = size(x, 0)
        seq_len = size(x, 1)
        return torch.stack([torch.stack([self.weight[int(x[int(i), int(j)])] for _fi_j in range(int(seq_len)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(batch_size)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

    def update_params(self, learnable_grads):
        this = self
        with torch.no_grad():
            self.weight.copy_(self.opt_weight.step(self.weight, learnable_grads[int(0)]))


class TinyTransformer(nn.Module):
    def __init__(self, embed, pos_embed, layers, fc, opt_pos_embed):
        super().__init__()
        self.add_module('embed', embed)
        self.pos_embed = nn.Parameter(torch.as_tensor(pos_embed))
        #self.layers = torch.as_tensor(layers).float() if isinstance(layers, (int, float, torch.Tensor)) else layers
        self.layers = nn.ModuleList(layers)
        self.add_module('fc', fc)
        self.add_module('opt_pos_embed', opt_pos_embed)
        self.learnable_params = list(self.parameters())

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        token_embeddings = self.embed(x)
        seq_len = size(x, 1)
        position_embeddings = self.pos_embed[:, :int(seq_len), :]
        x = (token_embeddings + position_embeddings)
        for i in range(int(0), int(len(self.layers))):
            layer = self.layers[int(i)]
            x = layer(x)
        return self.fc(x)

    def update_params(self, learnable_grads):
        this = self
        self.embed.update_params(learnable_grads[int(1)])
        with torch.no_grad():
            self.pos_embed.copy_(self.opt_pos_embed.step(self.pos_embed, learnable_grads[int(0)]))
        block_grads_size = 12

        for i in range(len(self.layers)):
            start = 2 + i * block_grads_size
            end = start + block_grads_size
            grads = learnable_grads[start:end]

            layer = self.layers[i]

            layer.attention.W_q.update_params([grads[0]])
            layer.attention.W_k.update_params([grads[1]])
            layer.attention.W_v.update_params([grads[2]])
            layer.attention.W_o.update_params([grads[3]])

            layer.norm1.update_params([grads[4], grads[5]])

            layer.ffn.W1.update_params([grads[6], grads[7]])
            layer.ffn.W2.update_params([grads[8], grads[9]])

            layer.norm2.update_params([grads[10], grads[11]])
        # Output projection
        fc_start = 2 + len(self.layers) * block_grads_size
        self.fc.update_params(learnable_grads[fc_start:fc_start + 2])

    def train(self, X, Y, epochs):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        Y = torch.as_tensor(Y, device=DEVICE).float()
        for epoch in range(int(0), int(epochs)):
            logits = self(X)
            loss = cross_entropy_sequence(logits, Y)
            learnable_grads = compute_grad(loss, self.learnable_params)
            self.update_params(learnable_grads)
            print(epoch)
            print(loss)

    def predict(self, prompt):
        this = self
        tokens = encode_text(prompt)
        x = torch.reshape(to_tensor(tokens), (1, len(tokens)))
        logits = self(x)
        seq_len = size(logits, 1)
        last_logits = logits[int(0), int((seq_len - 1))]
        next_token_id = argmax(last_logits)
        return decode_token(next_token_id)

# === Program ===
d_model = 128
nhead = 2
d_k = (d_model / nhead)
max_len = 9
vocab_size = 146
embed = Embedding(rand_2d_array(vocab_size, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(vocab_size), int(d_model), device=DEVICE), torch.zeros(int(vocab_size), int(d_model), device=DEVICE))).to(DEVICE)
pos_embed = rand_3d_array(1, max_len, d_model, 1.0)
opt_pos_embed = AdamOptimizer3D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(1), int(max_len), int(d_model), device=DEVICE), torch.zeros(int(1), int(max_len), int(d_model), device=DEVICE)).to(DEVICE)
attention = MultiHeadAttention(d_model, nhead, d_k, Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE)))).to(DEVICE)
norm1 = LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE))).to(DEVICE)
ffn = FeedForward(Linear3D(rand_2d_array(64, d_model, 0.02), torch.zeros(int(64), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), int(d_model), device=DEVICE), torch.zeros(int(64), int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), device=DEVICE), torch.zeros(int(64), device=DEVICE))), Linear3D(rand_2d_array(d_model, 64, 0.02), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(64), device=DEVICE), torch.zeros(int(d_model), int(64), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)))).to(DEVICE)
norm2 = LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE))).to(DEVICE)

block1 = TransformerBlock(MultiHeadAttention(d_model, nhead, d_k, Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE)))), LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE))), FeedForward(Linear3D(rand_2d_array(64, d_model, 0.02), torch.zeros(int(64), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), int(d_model), device=DEVICE), torch.zeros(int(64), int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), device=DEVICE), torch.zeros(int(64), device=DEVICE))), Linear3D(rand_2d_array(d_model, 64, 0.02), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(64), device=DEVICE), torch.zeros(int(d_model), int(64), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)))), LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)))).to(DEVICE)
block2 = TransformerBlock(MultiHeadAttention(d_model, nhead, d_k, Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE)))), LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE))), FeedForward(Linear3D(rand_2d_array(64, d_model, 0.02), torch.zeros(int(64), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), int(d_model), device=DEVICE), torch.zeros(int(64), int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), device=DEVICE), torch.zeros(int(64), device=DEVICE))), Linear3D(rand_2d_array(d_model, 64, 0.02), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(64), device=DEVICE), torch.zeros(int(d_model), int(64), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)))), LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)))).to(DEVICE)
block3 = TransformerBlock(MultiHeadAttention(d_model, nhead, d_k, Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE))), Linear3DnoBias(rand_2d_array(d_model, d_model, 0.02), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(d_model), device=DEVICE), torch.zeros(int(d_model), int(d_model), device=DEVICE)))), LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE))), FeedForward(Linear3D(rand_2d_array(64, d_model, 0.02), torch.zeros(int(64), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), int(d_model), device=DEVICE), torch.zeros(int(64), int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(64), device=DEVICE), torch.zeros(int(64), device=DEVICE))), Linear3D(rand_2d_array(d_model, 64, 0.02), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), int(64), device=DEVICE), torch.zeros(int(d_model), int(64), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)))), LayerNorm(d_model, torch.ones(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(d_model), device=DEVICE), torch.zeros(int(d_model), device=DEVICE)))).to(DEVICE)


layers = [block1, block2, block3]
fc = Linear3D(rand_2d_array(vocab_size, d_model, 0.02), torch.zeros(int(vocab_size), device=DEVICE), AdamOptimizer2D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(vocab_size), int(d_model), device=DEVICE), torch.zeros(int(vocab_size), int(d_model), device=DEVICE)), AdamOptimizer1D(0.001, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(vocab_size), device=DEVICE), torch.zeros(int(vocab_size), device=DEVICE))).to(DEVICE)
model = TinyTransformer(embed, pos_embed, layers, fc, opt_pos_embed).to(DEVICE)
vocab_size = get_vocab_size()
max_len = get_max_len()
dataset = create_dataset()
X = dataset[int(0)]
Y = dataset[int(1)]
epochs = 50


print(model.train(X, Y, epochs))
print(model.predict('I like'))
print(model.predict('birds fly'))
#print(model.predict('we see'))
#print(model.predict('bees make'))

