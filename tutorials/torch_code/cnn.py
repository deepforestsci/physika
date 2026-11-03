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

def get_sum_of_1d_array(x):
    total = 0
    for i in range(len(x)):
        total = total + x[int(i)]
    return total

def max(x, y):
    if x > y:
        return x
    else:
        return y

def relu(x):
    if x > 0:
        return x
    else:
        return 0.0

def relu2d(x):
    rows = get_2d_array_num_rows(x)
    cols = get_1d_array_length(x[int(0)])
    results = zero_2d_array(rows, cols)
    for i in range(int(0), int(rows)):
        for j in range(int(0), int(cols)):
            results[int(i), int(j)] = relu(x[int(i), int(j)])
    return results

def softmax(x):
    len_x = get_1d_array_length(x)
    exps_array = torch.stack([torch.exp(x[int(i)] if isinstance(x[int(i)], torch.Tensor) else torch.tensor(float(x[int(i)]))) for _fi_i in range(int(len_x)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    total = get_sum_of_1d_array(exps_array)
    results = torch.stack([(exps_array[int(i)] / total) for _fi_i in range(int(len_x)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def rand_1d_array(n, μ):
    return torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_2d_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def rand_3d_array(n, m, o, μ):
    return torch.stack([torch.stack([torch.stack([(μ * torch.distributions.Normal(0.0, 1.0).rsample()) for _fi_k in range(int(o)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def cross_entropy(probs, label, m=None):
    if m is None:
        m = int(probs.shape[0])
    p = probs[int(label)]
    return (-torch.log(torch.as_tensor(p).float()))

def argmax(iterable):
    idx = 0
    max_val = iterable[int(0)]
    len_iterable = get_1d_array_length(iterable)
    for i in range(int(1), int(len_iterable)):
        if iterable[int(i)] > max_val:
            max_val = iterable[int(i)]
            idx = i
    return idx

# === Classes ===
class ConvNets(nn.Module):
    def __init__(self, kernel, b1, w, b2):
        super().__init__()
        self.kernel = nn.Parameter(torch.as_tensor(kernel))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.w = nn.Parameter(torch.as_tensor(w))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.learnable_params = list(self.parameters())

    def conv2d(self, input):
        this = self
        input = torch.as_tensor(input, device=DEVICE).float()
        out_H = ((get_2d_array_num_rows(input) - get_2d_array_num_rows(self.kernel)) + 1)
        out_W = ((get_1d_array_length(input[int(0)]) - get_1d_array_length(self.kernel[int(0)])) + 1)
        results = zero_2d_array(out_H, out_W)
        for i in range(int(0), int(out_H)):
            for j in range(int(0), int(out_W)):
                acc = 0
                for ki in range(int(0), int(len(self.kernel))):
                    for kj in range(int(0), int(len(self.kernel[int(0)]))):
                        acc = acc + (input[int((i + ki)), int((j + kj))] * self.kernel[int(ki), int(kj)])
                results[int(i), int(j)] = (acc + self.b1)
        return results

    def maxpool2d(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        stride = 2
        rows = (get_2d_array_num_rows(x) / 2)
        cols = (get_1d_array_length(x[int(0)]) / 2)
        results = zero_2d_array(rows, cols)
        for i in range(int(0), int(rows)):
            for j in range(int(0), int(cols)):
                a = x[int((i * stride)), int((j * stride))]
                b = x[int((i * stride)), int(((j * stride) + 1))]
                c = x[int(((i * 2) + 1)), int((j * 2))]
                d = x[int(((i * 2) + 1)), int(((j * 2) + 1))]
                results[int(i), int(j)] = max(a, max(b, max(c, d)))
        return results

    def flatten(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        rows = get_2d_array_num_rows(x)
        cols = get_1d_array_length(x[int(0)])
        new_len = (rows * cols)
        results = zero_1d_array(new_len)
        for i in range(int(0), int(rows)):
            for j in range(int(0), int(cols)):
                results[int(((i * cols) + j))] = x[int(i), int(j)]
        return results

    def linear(self, x, weight, bias):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        weight = torch.as_tensor(weight, device=DEVICE).float()
        bias = torch.as_tensor(bias, device=DEVICE).float()
        out = get_1d_array_length(bias)
        inp = get_1d_array_length(x)
        results = zero_1d_array(out)
        for i in range(int(0), int(out)):
            acc = 0
            for j in range(int(0), int(inp)):
                acc = acc + (weight[int(i), int(j)] * x[int(j)])
            results[int(i)] = (acc + bias[int(i)])
        return results

    def forward(self, input):
        this = self
        input = torch.as_tensor(input, device=DEVICE).float()
        conv1 = self.conv2d(input)
        relu1 = relu2d(conv1)
        pool1 = self.maxpool2d(relu1)
        flat = self.flatten(pool1)
        out = self.linear(flat, self.w, self.b2)
        results = softmax(out)
        return results

    def train(self, train_X, train_y, epochs, lr, train_dataset_len):
        this = self
        train_X = torch.as_tensor(train_X, device=DEVICE).float()
        train_y = torch.as_tensor(train_y, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        train_dataset_len = torch.as_tensor(train_dataset_len, device=DEVICE).float()
        loss = 0
        for epoch in range(int(0), int(epochs)):
            print(epoch)
            epoch_loss = 0
            for j in range(int(0), int(train_dataset_len)):
                input = train_X[int(j)]
                label = train_y[int(j)]
                z = self(input)
                current_loss = cross_entropy(z, label)
                epoch_loss = epoch_loss + current_loss
                learnable_grads = compute_grad(current_loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
            epoch_loss = (epoch_loss / train_dataset_len)
            print(epoch_loss)
        loss = epoch_loss
        return loss

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.kernel.copy_((self.kernel - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b1.copy_((self.b1 - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.w.copy_((self.w - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.b2.copy_((self.b2 - (lr * learnable_grads[int(3)])))

# === Program ===
kernel = rand_2d_array(3, 3, 0.01)
b1 = 0.0
w = rand_2d_array(10, 169, 0.1)
b2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=DEVICE)
cnn_object = ConvNets(kernel, b1, w, b2).to(DEVICE)
train_dataset_len = 1
test_dataset_len = 1
train_X = rand_3d_array(20, 28, 28, 0.5)
train_y = rand_1d_array(20, 0.0)
test_X = rand_3d_array(10, 28, 28, 0.5)
test_y = rand_1d_array(10, 0.0)
epochs = 1
lr = 0.1
print(cnn_object.train(train_X, train_y, epochs, lr, train_dataset_len))
correct = 0
len_test_X = len(test_X[int(0)])
y_true = 0
for i in range(int(0), int(test_dataset_len)):
    x = test_X[int(i)]
    y_true = test_y[int(i)]
    y_pred = cnn_object(x)
    pred_class = argmax(y_pred)
    if pred_class == y_true:
        correct = correct + 1
accuracy = (correct / len_test_X)
print(print(accuracy))