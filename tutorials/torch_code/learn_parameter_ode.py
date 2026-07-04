import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device='cpu')]])
    return results

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def append(x, var):
    new_length = (get_1d_array_length(x) + 1)
    results = zero_1d_array(new_length)
    len_x = get_1d_array_length(x)
    for i in range(int(0), int(new_length)):
        if i < len_x:
            results[int(i)] = x[int(i)]
        else:
            results[int(i)] = var
    return results

def f(y, θ):
    return ((-θ) * y)

def solver(θ):
    y = 1.0
    y_array = torch.tensor([1.0], device='cpu')
    for i in range(int(0), int(timesteps)):
        dy = f(y, θ)
        y = y + (dt * dy)
        y_array = append(y_array, y)
    return y_array

def calculate_loss(θ):
    y_predicted = solver(θ)
    L = 0.0
    m = get_1d_array_length(y_predicted)
    for i in range(int(0), int(m)):
        L = L + ((y_predicted[int(i)] - y_true[int(i)]) ** 2)
    return (L / m)

# === Program ===
timesteps = 10
dt = 0.1
true_theta = 2.0
y_true = solver(true_theta)
θ = torch.tensor(1.0, requires_grad=True)
learning_rate = 0.1
epochs = 1
for i in range(int(0), int(epochs)):
    g = compute_grad(calculate_loss, θ)
    θ = (θ - (learning_rate * g))
y_predicted = solver(θ)