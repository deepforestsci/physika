import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def rand_2d_array(n, m, μ):
    return torch.stack([torch.stack([(μ * torch.distributions.Normal(0.5, 1.0).rsample()) for _fi_j in range(int(m)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def softplus(x):
    return torch.log((1 + torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))) if isinstance((1 + torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))), torch.Tensor) else torch.tensor(float((1 + torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))))))

def mse_loss(pred, true):
    return ((pred - true) ** 2)

def ode_derivative(state, lnn):
    derivative = torch.zeros(int(2), int(1), device=DEVICE)
    derivative[int(0), int(0)] = state[int(1), int(0)]
    derivative[int(1), int(0)] = lnn(state)[int(0)]
    return derivative

def rk4_step(state, Δt, lnn):
    k1 = ode_derivative(state, lnn)
    k2_state = (state + ((0.5 * Δt) * k1))
    k2 = ode_derivative(k2_state, lnn)
    k3_state = (state + ((0.5 * Δt) * k2))
    k3 = ode_derivative(k3_state, lnn)
    k4_state = (state + (Δt * k3))
    k4 = ode_derivative(k4_state, lnn)
    return (state + ((Δt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))

def ode_solver(lnn, initial_state, Δt, num_steps):
    trajectory = torch.zeros(int((num_steps + 1)), int(2), int(1), device=DEVICE)
    state = initial_state
    trajectory[int(0), :, :] = state
    for i in range(int(0), int(num_steps)):
        state = rk4_step(state, Δt, lnn)
        trajectory[int((i + 1)), :, :] = state
    return trajectory

def gaussian_solve(A, b):
    a_row = get_2d_array_num_rows(A)
    a_col = get_2d_array_num_cols(A)
    new_col = (a_col + 1)
    aug = torch.zeros(int(a_row), int(new_col), device=DEVICE)
    for i in range(int(0), int(a_row)):
        aug[int(i), :int(a_col)] = A[int(i), :]
        aug[int(i), int(a_col)] = b[int(i)]
    for i in range(int(0), int(a_row)):
        max_row = i
        for k in range(int((i + 1)), int(a_row)):
            if torch.abs(aug[int(k), int(i)] if isinstance(aug[int(k), int(i)], torch.Tensor) else torch.tensor(float(aug[int(k), int(i)]))) > torch.abs(aug[int(max_row), int(i)] if isinstance(aug[int(max_row), int(i)], torch.Tensor) else torch.tensor(float(aug[int(max_row), int(i)]))):
                max_row = k
        if max_row != i:
            lo = min(i, max_row)
            hi = max(i, max_row)
            rows_above = aug[:int(lo), :]
            pivot_row = aug[int(hi):int((hi + 1)), :]
            rows_between = aug[int((lo + 1)):int(hi), :]
            current_row = aug[int(lo):int((lo + 1)), :]
            rows_below = aug[int((hi + 1)):, :]
            aug = torch.cat([rows_above, pivot_row, rows_between, current_row, rows_below])
        pivot_row = aug[int(i):int((i + 1)), :]
        pivot_value = aug[int(i), int(i)]
        rows_below = aug[int((i + 1)):, :]
        elimination_factors = (rows_below[:, int(i):int((i + 1))] / pivot_value)
        eliminated_rows = (rows_below - (elimination_factors * pivot_row))
        aug = torch.cat([aug[:int((i + 1)), :], eliminated_rows])
    x = torch.zeros(int(0), device=DEVICE)
    for i in range(int(0), int(a_row)):
        idx = ((a_col - 1) - i)
        total = aug[int(idx), int(a_col)]
        for j in range(int((idx + 1)), int(a_row)):
            total = (total - (aug[int(idx), int(j)] * x[int(((j - idx) - 1))]))
        val = (total / aug[int(idx), int(idx)])
        val = torch.reshape(val, (1,) if isinstance(1, int) else 1)
        x = torch.cat([val, x])
    return x

def get_2d_array_num_cols(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return get_1d_array_length(x[int(0)])

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def min(x, y):
    if x < y:
        return x
    else:
        return y

def max(x, y):
    if x > y:
        return x
    else:
        return y

# === Classes ===
class LNN(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.b1 = nn.Parameter(torch.as_tensor(b1))
        self.W2 = nn.Parameter(torch.as_tensor(W2))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.learnable_params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float().requires_grad_(True)
        d = (I / 2)
        L = ((self.W2 @ softplus(((self.W1 @ x) + self.b1))) + self.b2)
        dL = compute_grad(L, x)
        hessian = torch.zeros(int(I), int(I), device=DEVICE)
        for i in range(int(0), int(I)):
            row = torch.transpose(compute_grad(dL[int(i)], x), 0, 1)
            hessian[int(i), :] = row
        H_dq_dq = hessian[int(d):int(I), int(d):int(I)]
        H_q_dq = hessian[int(0):int(d), int(d):int(I)]
        dL_dq = torch.squeeze(dL[int(0):int(d)], 1)
        qdot = torch.squeeze(x[int(d):int(I)], 1)
        b = (dL_dq - (torch.transpose(H_q_dq, 0, 1) @ qdot))
        q_ddot = gaussian_solve(H_dq_dq, b)
        return q_ddot

    def train(self, epochs, lr):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        train_dataset_len = len(train_dataset)
        epoch_loss = 0
        for idx in range(int(0), int(epochs)):
            print(idx)
            epoch_loss = 0
            for j in range(int(0), int(train_dataset_len)):
                sample = train_dataset[int(j)]
                x = sample[int(0)]
                true_qddot = sample[int(1)]
                pred = self(x)
                loss = mse_loss(pred, true_qddot)
                epoch_loss = epoch_loss + loss
                learnable_grads = compute_grad(loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
            epoch_loss = (epoch_loss / train_dataset_len)
            print(epoch_loss)
        return epoch_loss

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.W1.copy_((self.W1 - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.b1.copy_((self.b1 - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.W2.copy_((self.W2 - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.b2.copy_((self.b2 - (lr * learnable_grads[int(3)])))

# === Program ===
train_dataset = [[rand_2d_array(2, 1, 1.0), 0.0]]
test_dataset = [[rand_2d_array(2, 1, 1.0), 0.0]]
H = 128
I = 2
W1 = rand_2d_array(H, I, 1.0)
b1 = rand_2d_array(H, 1, 0.5)
W2 = rand_2d_array(1, H, 0.1)
b2 = 0.0
model = LNN(W1, b1, W2, b2).to(DEVICE)
epochs = 1
train_loss = model.train(epochs, 0.001)
print(print(train_loss))
test_sample = test_dataset[int(0)]
initial_state = test_sample[int(0)]
Δt = 0.01
num_steps = 1
trajectory = ode_solver(model, initial_state, Δt, num_steps)