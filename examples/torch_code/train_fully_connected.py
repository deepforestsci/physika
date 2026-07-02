import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

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

def zero_2d_array(rows, cols):
    results = torch.stack([torch.stack([(j * 0) for _fi_j in range(int(cols)) for j in [torch.tensor(float(_fi_j))]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i))]])
    return results

def get_3d_array_depth(x):
    depth = get_1d_array_length(X)
    return depth

def sigma(x):
    rows = get_2d_array_num_rows(x)
    cols = get_1d_array_length(x[int(0)])
    results = zero_2d_array(rows, cols)
    for i in range(int(0), int(rows)):
        for j in range(int(0), int(cols)):
            results[int(i), int(j)] = (1.0 / (1.0 + torch.exp((0.0 - x[int(i), int(j)]) if isinstance((0.0 - x[int(i), int(j)]), torch.Tensor) else torch.tensor(float((0.0 - x[int(i), int(j)]))))))
    return results

# === Classes ===
class FullyConnectedNetwork(nn.Module):
    def __init__(self, W, B, w, b):
        super().__init__()
        self.W = nn.Parameter(torch.as_tensor(W))
        self.B = nn.Parameter(torch.as_tensor(B))
        self.w = nn.Parameter(torch.as_tensor(w))
        self.b = nn.Parameter(torch.as_tensor(b))
        self.learnable_params = [self.W, self.B, self.w, self.b]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x).float()
        for k in range(int(0), int(2)):
            x = sigma(((self.W[int(k)] @ x) + self.B[int(k)]))
        results = ((self.w @ x) + self.b)
        return results[int(0), int(0)]

    def loss(self, y, target):
        this = self
        y = torch.as_tensor(y).float()
        target = torch.as_tensor(target).float()
        return ((y - target) ** 2.0)

    def train(self, X, y, epochs, lr):
        this = self
        X = torch.as_tensor(X).float()
        y = torch.as_tensor(y).float()
        lr = torch.as_tensor(lr).float()
        len_dataset = get_3d_array_depth(X)
        last_loss = 0
        for i in range(int(0), int(epochs)):
            epoch_loss = 0
            for j in range(int(0), int(len_dataset)):
                pred = self(X[int(j)])
                current_loss = self.loss(pred, y[int(j)])
                epoch_loss = (epoch_loss + current_loss)
                learnable_grads = compute_grad(current_loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
            last_loss = (epoch_loss / len_dataset)
        return last_loss

    def evaluate(self, X, y):
        this = self
        X = torch.as_tensor(X).float()
        y = torch.as_tensor(y).float()
        x = 1.5
        len_dataset = get_3d_array_depth(X)
        total_loss = 0
        for j in range(int(0), int(len_dataset)):
            pred = self(X[int(j)])
            current_loss = self.loss(pred, y[int(j)])
            total_loss = (total_loss + current_loss)
        return (total_loss / len_dataset)

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr).float()
        with torch.no_grad():
            self.W.copy_((self.W - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.B.copy_((self.B - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.w.copy_((self.w - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.b.copy_((self.b - (lr * learnable_grads[int(3)])))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
X = torch.tensor([[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]], [[1.0], [1.0], [1.0]]])
y = torch.tensor([0.2, 0.4, 0.6, 0.9])
W = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]])
B = torch.tensor([[[0.1], [0.2], [0.3]], [[0.1], [0.2], [0.3]]])
w = torch.tensor([[0.5, 0.5, 0.5]])
b = 0.1
net = FullyConnectedNetwork(W, B, w, b)
loss_before = net.evaluate(X, y)
physika_print(loss_before)
epochs = 1000
lr = 0.1
net_trained = net.train(X, y, epochs, lr)
physika_print(net_trained)
loss_after = net.evaluate(X, y)
physika_print(loss_after)
physika_print(net(torch.tensor([1.0, 0.0, 0.0])))
physika_print(net(torch.tensor([0.0, 1.0, 0.0])))
physika_print(net(torch.tensor([0.0, 0.0, 1.0])))
physika_print(net(torch.tensor([1.0, 1.0, 1.0])))