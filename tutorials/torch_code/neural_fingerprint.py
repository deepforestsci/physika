import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def get_sum_of_1d_array(x):
    total = 0
    for i in range(len(x)):
        total = total + x[int(i)]
    return total

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def sigma(x):
    return (1.0 / (1.0 + torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))))

def softmax_row(x):
    sz = get_1d_array_length(x)
    e = torch.stack([torch.exp(x[int(i)] if isinstance(x[int(i)], torch.Tensor) else torch.tensor(float(x[int(i)]))) for _fi_i in range(int(sz)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    total = get_sum_of_1d_array(e)
    return torch.stack([(e[int(i)] / total) for _fi_i in range(int(sz)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def mse(pred, target):
    return ((pred - target) ** 2.0)

def empty_graph(n_vertices):
    z = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(n_vertices)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(n_vertices)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    g = UndirectedGraph()
    g.adjacency = z
    return g

# === Classes ===
class UndirectedGraph(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.adjacency = None

    def num_vertices(self):
        this = self
        return (get_2d_array_num_rows(self.adjacency) * 1.0)

    def add_weighted_edge(self, u, v, w):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        v = torch.as_tensor(v, device=DEVICE).float()
        w = torch.as_tensor(w, device=DEVICE).float()
        m = self.adjacency
        k = get_2d_array_num_rows(m)
        new_adj = torch.stack([torch.stack([m[int(a), int(b)] for _fi_b in range(int(k)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        new_adj[int(u), int(v)] = w
        new_adj[int(v), int(u)] = w
        self.adjacency = new_adj

    def add_edge(self, u, v):
        this = self
        u = torch.as_tensor(u, device=DEVICE).float()
        v = torch.as_tensor(v, device=DEVICE).float()
        self.add_weighted_edge(u, v, 1.0)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class NeuralFingerprint(nn.Module):
    def __init__(self, Hw, Wout, Wr):
        super().__init__()
        self.Hw = nn.Parameter(torch.as_tensor(Hw))
        self.Wout = nn.Parameter(torch.as_tensor(Wout))
        self.Wr = nn.Parameter(torch.as_tensor(Wr))
        self.learnable_params = [self.Hw, self.Wout, self.Wr]

    def forward(self, A, H0):
        this = self
        A = torch.as_tensor(A, device=DEVICE).float()
        H0 = torch.as_tensor(H0, device=DEVICE).float()
        f = zero_1d_array(8.0)
        r = H0
        n_atoms = get_2d_array_num_rows(A)
        for L in range(int(0), int(2)):
            nbr_sum = (A @ r)
            v = (r + nbr_sum)
            HwL = self.Hw[int(L)]
            r = sigma((v @ HwL))
            WoutL = self.Wout[int(L)]
            logits = (r @ WoutL)
            for a in range(int(0), int(n_atoms)):
                f = f + softmax_row(logits[int(a)])
        return f

    def train(self, epochs, lr):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        loss = 0
        for step in range(int(0), int(epochs)):
            fp_h2o = self(A_h2o, H0_h2o)
            fp_ch4 = self(A_ch4, H0_ch4)
            pred_h2o = get_sum_of_1d_array((fp_h2o * self.Wr))
            pred_ch4 = get_sum_of_1d_array((fp_ch4 * self.Wr))
            loss = (mse(pred_h2o, target_h2o) + mse(pred_ch4, target_ch4))
            g = compute_grad(loss, self.learnable_params)
            with torch.no_grad():
                self.Hw.copy_((self.Hw - (lr * g[int(0)])))
            with torch.no_grad():
                self.Wout.copy_((self.Wout - (lr * g[int(1)])))
            with torch.no_grad():
                self.Wr.copy_((self.Wr - (lr * g[int(2)])))
        return loss

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
μ = 0.0
σ = 1.0
Hw = torch.stack([torch.stack([torch.distributions.Normal(μ, σ).rsample((int(4),)) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(2)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
Wout = torch.stack([torch.stack([torch.distributions.Normal(μ, σ).rsample((int(8),)) for _fi_j in range(int(4)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(2)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
Wr = torch.distributions.Normal(μ, σ).rsample((int(8),))
fp_model = NeuralFingerprint(Hw, Wout, Wr).to(DEVICE)
n0 = 3
g = empty_graph(n0)
print(g.add_edge(0.0, 1.0))
print(g.add_edge(1.0, 2.0))
H0 = torch.as_tensor(torch.stack([torch.distributions.Normal(μ, σ).rsample((int(4),)) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])).requires_grad_(True).to(DEVICE)
fingerprint = fp_model(g.adjacency, H0)
print(fingerprint)
fp_sum = get_sum_of_1d_array(fingerprint)
print(compute_grad(fp_sum, H0))
print(compute_grad(fp_sum, fp_model.learnable_params))
A_h2o = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)
H0_h2o = torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=DEVICE)
A_ch4 = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)
H0_ch4 = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=DEVICE)
fp_h2o = fp_model(A_h2o, H0_h2o)
fp_ch4 = fp_model(A_ch4, H0_ch4)
print(fp_h2o)
print(fp_ch4)
target_h2o = 1.0
target_ch4 = 0.0
pred_h2o_0 = get_sum_of_1d_array((fp_h2o * fp_model.Wr))
pred_ch4_0 = get_sum_of_1d_array((fp_ch4 * fp_model.Wr))
loss0 = (mse(pred_h2o_0, target_h2o) + mse(pred_ch4_0, target_ch4))
print(pred_h2o_0)
print(pred_ch4_0)
print(loss0)
epochs = 300
lr = 0.01
final_loss = fp_model.train(epochs, lr)
print(final_loss)
fp_h2o_trained = fp_model(A_h2o, H0_h2o)
fp_ch4_trained = fp_model(A_ch4, H0_ch4)
pred_h2o_1 = get_sum_of_1d_array((fp_h2o_trained * fp_model.Wr))
pred_ch4_1 = get_sum_of_1d_array((fp_ch4_trained * fp_model.Wr))
print(pred_h2o_1)
print(pred_ch4_1)