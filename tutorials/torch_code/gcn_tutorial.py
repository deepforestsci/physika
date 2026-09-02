import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
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

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_3d_A(n):
    results = torch.stack([torch.stack([torch.stack([(k * 0) for _fi_k in range(int(44)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(44)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def zero_3d_H(n):
    results = torch.stack([torch.stack([torch.stack([(k * 0) for _fi_k in range(int(30)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(44)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def get_sum_of_1d_array(x):
    total = 0
    for i in range(len(x)):
        total = total + x[int(i)]
    return total

def diag_matrix(d):
    sz = get_1d_array_length(d)
    result = zero_2d_array(sz, sz)
    for i in range(len(d)):
        result[int(i), int(i)] = d[int(i)]
    return result

def ones_vector(n):
    return torch.stack([((0.0 * i) + 1.0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def sigma(x):
    return (1.0 / (1.0 + torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))))

def normalize_adj(A):
    sz = get_2d_array_num_rows(A)
    I = diag_matrix(ones_vector(sz))
    A_plus = (A + I)
    deg = torch.stack([get_sum_of_1d_array(A_plus[int(i)]) for _fi_i in range(int(sz)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    d_inv_sqrt = torch.stack([(1.0 / torch.sqrt(deg[int(i)] if isinstance(deg[int(i)], torch.Tensor) else torch.tensor(float(deg[int(i)])))) for _fi_i in range(int(sz)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    D = diag_matrix(d_inv_sqrt)
    return ((D @ A_plus) @ D)

def masked_graph_sum_pool(H, sz):
    cols = get_1d_array_length(H[int(0)])
    result = zero_1d_array(cols)
    for i in range(int(0), int(sz)):
        row = H[int(i)]
        for j in range(int(0), int(cols)):
            result[int(j)] += row[int(j)]
    return result

def mse(pred, target):
    return ((pred - target) ** 2.0)

def within_tolerance(pred, target, tol):
    diff = (pred - target)
    if diff < 0.0:
        diff = (0.0 - diff)
    if diff < tol:
        return 1.0
    else:
        return 0.0

# === Classes ===
class GCNModel(nn.Module):
    def __init__(self, W1, W2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.W2 = nn.Parameter(torch.as_tensor(W2))
        self.learnable_params = [self.W1, self.W2]

    def forward(self, A, H, sz):
        this = self
        A = torch.as_tensor(A, device=DEVICE).float()
        H = torch.as_tensor(H, device=DEVICE).float()
        sz = torch.as_tensor(sz, device=DEVICE).float()
        A_hat = normalize_adj(A)
        H1 = sigma((A_hat @ (H @ self.W1)))
        pooled = masked_graph_sum_pool(H1, sz)
        pred = (pooled @ self.W2)
        return pred

    def train(self, epochs, lr):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        len_train_X = get_1d_array_length(train_sizes)
        loss = 0
        for i in range(int(0), int(epochs)):
            loss = 0
            for j in range(int(0), int(len_train_X)):
                A_j = train_A[int(j)]
                H_j = train_H[int(j)]
                sz_j = train_sizes[int(j)]
                label = train_y[int(j)]
                z = self(A_j, H_j, sz_j)
                current_loss = mse(z, label)
                loss = loss + current_loss
                learnable_grads = compute_grad(current_loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
            loss = (loss / len_train_X)
        return loss

    def evaluate(self):
        this = self
        len_test_X = get_1d_array_length(test_sizes)
        correct = 0
        for i in range(int(0), int(len_test_X)):
            A_i = test_A[int(i)]
            H_i = test_H[int(i)]
            sz_i = test_sizes[int(i)]
            y_pred = self(A_i, H_i, sz_i)
            correct = correct + within_tolerance(y_pred, test_y[int(i)], 1.0)
        return (correct / len_test_X)

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.W1.copy_((self.W1 - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.W2.copy_((self.W2 - (lr * learnable_grads[int(1)])))

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
W1 = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(4),)) for _fi_i in range(int(30)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2 = torch.distributions.Normal(μ, σ).rsample((int(4),))
gcn_object = GCNModel(W1, W2).to(DEVICE)
train_A = zero_3d_A(513)
train_H = zero_3d_H(513)
train_sizes = zero_1d_array(513)
train_y = zero_1d_array(513)
test_A = zero_3d_A(65)
test_H = zero_3d_H(65)
test_sizes = zero_1d_array(65)
test_y = zero_1d_array(65)
lr = 0.0005
epochs = 1
final_loss = gcn_object.train(epochs, lr)
print(print(final_loss))
accuracy = gcn_object.evaluate()
print(print(accuracy))