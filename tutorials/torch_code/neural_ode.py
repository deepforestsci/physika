import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def linspace(start, end, n):
    x = torch.zeros(int(n))
    dx = ((end - start) / (n - 1))
    for i in range(int(0), int(n)):
        x[int(i)] = (start + (i * dx))
    return x

def tanh(x):
    return ((torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x))) - torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))) / (torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x))) + torch.exp((0.0 - x) if isinstance((0.0 - x), torch.Tensor) else torch.tensor(float((0.0 - x))))))

def rk4_step(state, t, Δt, ode_func):
    k1 = ode_func(state)
    k2_state = (state + ((0.5 * Δt) * k1))
    k2 = ode_func(k2_state)
    k3_state = (state + ((0.5 * Δt) * k2))
    k3 = ode_func(k3_state)
    k4_state = (state + (Δt * k3))
    k4 = ode_func(k4_state)
    return (state + ((Δt / 6.0) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))

def odesolver(ode_func, y0, Δt, timesteps):
    n_times = len(timesteps)
    trajectory = torch.zeros(int(n_times), int(2), int(1))
    state = y0
    trajectory[int(0)] = state
    for i in range(int(1), int(n_times)):
        current_t = timesteps[int((i - 1))]
        state = rk4_step(state, current_t, Δt, ode_func)
        trajectory[int(i)] = state
    return trajectory

def damped_oscillator(state):
    q, p = state[int(0), int(0)], state[int(1), int(0)]
    gamma = 0.2
    dq = p
    dp = ((0.0 - q) - (gamma * p))
    return torch.tensor([[dq], [dp]], device=DEVICE)

def generate_dataset(y0, Δt, timesteps):
    n_times = len(timesteps)
    trajectory = torch.zeros(int(n_times), int(2), int(1))
    state = y0
    trajectory[int(0)] = state
    for i in range(int(1), int(n_times)):
        current_t = timesteps[int((i - 1))]
        state = rk4_step(state, current_t, Δt, damped_oscillator)
        trajectory[int(i)] = state
    return trajectory

def compute_vjps(z, a, model):
    z = detach_grad(z)
    a = detach(a)
    f_val = model(z)
    scalar = torch.sum((a * f_val) if isinstance((a * f_val), torch.Tensor) else torch.tensor(float((a * f_val))))
    parameter_list = [z, model.W1, model.B1, model.W2, model.B2]
    result = compute_grad(scalar, parameter_list)
    dadt = (-result[int(0)])
    dW1 = (-result[int(1)])
    dB1 = (-result[int(2)])
    dW2 = (-result[int(3)])
    dB2 = (-result[int(4)])
    f_val = detach(f_val)
    vjps_results = [f_val, dadt, dW1, dB1, dW2, dB2]
    return vjps_results

def rk4_step_adjoint(z, a, W1g, B1g, W2g, B2g, dt, model):
    dz1, da1, dW1_1, dB1_1, dW2_1, dB2_1 = compute_vjps(z, a, model)
    z2 = (z + ((0.5 * dt) * dz1))
    a2 = (a + ((0.5 * dt) * da1))
    W1g2 = (W1g + ((0.5 * dt) * dW1_1))
    B1g2 = (B1g + ((0.5 * dt) * dB1_1))
    W2g2 = (W2g + ((0.5 * dt) * dW2_1))
    B2g2 = (B2g + ((0.5 * dt) * dB2_1))
    dz2, da2, dW1_2, dB1_2, dW2_2, dB2_2 = compute_vjps(z2, a2, model)
    z3 = (z + ((0.5 * dt) * dz2))
    a3 = (a + ((0.5 * dt) * da2))
    W1g3 = (W1g + ((0.5 * dt) * dW1_2))
    B1g3 = (B1g + ((0.5 * dt) * dB1_2))
    W2g3 = (W2g + ((0.5 * dt) * dW2_2))
    B2g3 = (B2g + ((0.5 * dt) * dB2_2))
    dz3, da3, dW1_3, dB1_3, dW2_3, dB2_3 = compute_vjps(z3, a3, model)
    z4 = (z + (dt * dz3))
    a4 = (a + (dt * da3))
    W1g4 = (W1g + (dt * dW1_3))
    B1g4 = (B1g + (dt * dB1_3))
    W2g4 = (W2g + (dt * dW2_3))
    B2g4 = (B2g + (dt * dB2_3))
    dz4, da4, dW1_4, dB1_4, dW2_4, dB2_4 = compute_vjps(z4, a4, model)
    new_z = (z + ((dt / 6.0) * (((dz1 + (2 * dz2)) + (2 * dz3)) + dz4)))
    new_a = (a + ((dt / 6.0) * (((da1 + (2 * da2)) + (2 * da3)) + da4)))
    new_W1g = (W1g + ((dt / 6.0) * (((dW1_1 + (2 * dW1_2)) + (2 * dW1_3)) + dW1_4)))
    new_B1g = (B1g + ((dt / 6.0) * (((dB1_1 + (2 * dB1_2)) + (2 * dB1_3)) + dB1_4)))
    new_W2g = (W2g + ((dt / 6.0) * (((dW2_1 + (2 * dW2_2)) + (2 * dW2_3)) + dW2_4)))
    new_B2g = (B2g + ((dt / 6.0) * (((dB2_1 + (2 * dB2_2)) + (2 * dB2_3)) + dB2_4)))
    rk4_results = [new_z, new_a, new_W1g, new_B1g, new_W2g, new_B2g]
    return rk4_results

def adjoint_solver(pred_traj, true_trajectory, Δt, n_steps, model):
    z = pred_traj[int((n_steps - 1))]
    a = (2 * (pred_traj[int((n_steps - 1))] - true_trajectory[int((n_steps - 1))]))
    W1g = torch.zeros(int(n_neurons), int(2))
    B1g = torch.zeros(int(n_neurons), int(1))
    W2g = torch.zeros(int(2), int(n_neurons))
    B2g = torch.zeros(int(2), int(1))
    for i in range(int(0), int((n_steps - 1))):
        aug_state = rk4_step_adjoint(z, a, W1g, B1g, W2g, B2g, (-Δt), model)
        z, a, W1g, B1g, W2g, B2g = aug_state
        idx = ((n_steps - 2) - i)
        a = (a + (2 * (pred_traj[int(idx)] - true_trajectory[int(idx)])))
    a_t0, W1g, B1g, W2g, B2g = aug_state[int(1)], aug_state[int(2)], aug_state[int(3)], aug_state[int(4)], aug_state[int(5)]
    results = [a_t0, W1g, B1g, W2g, B2g]
    return results

# === Classes ===
class ODEFunc(nn.Module):
    def __init__(self, W1, B1, W2, B2):
        super().__init__()
        self.W1 = nn.Parameter(torch.as_tensor(W1))
        self.B1 = nn.Parameter(torch.as_tensor(B1))
        self.W2 = nn.Parameter(torch.as_tensor(W2))
        self.B2 = nn.Parameter(torch.as_tensor(B2))
        self.learnable_params = [self.W1, self.B1, self.W2, self.B2]

    def forward(self, x):
        this = self
        x = torch.as_tensor(x, device=DEVICE).float()
        h1 = tanh(((self.W1 @ x) + self.B1))
        out = ((self.W2 @ h1) + self.B2)
        return out

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class AdamOptimizer(nn.Module):
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

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
t_start, t_end, Δt = 0.0, 15.0, 0.1
n_steps = (int(((t_end - t_start) / Δt)) + 1)
timesteps = linspace(t_start, t_end, n_steps)
y0 = torch.tensor([[1.0], [0.0]], device=DEVICE)
true_trajectory = generate_dataset(y0, Δt, timesteps)
n_neurons = 128
μ, σ = 0.0, 0.01
W1 = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(2),)) for _fi_i in range(int(n_neurons)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
B1 = torch.stack([torch.tensor([0.01], device=DEVICE) for _fi_i in range(int(n_neurons)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
W2 = torch.stack([torch.distributions.Normal(μ, σ).rsample((int(n_neurons),)) for _fi_i in range(int(2)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
B2 = torch.tensor([[0.01], [0.01]], device=DEVICE)
model = ODEFunc(W1, B1, W2, B2).to(DEVICE)
lr = 0.01
adam_W1 = AdamOptimizer(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(n_neurons), int(2)), torch.zeros(int(n_neurons), int(2))).to(DEVICE)
adam_B1 = AdamOptimizer(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(n_neurons), int(1)), torch.zeros(int(n_neurons), int(1))).to(DEVICE)
adam_W2 = AdamOptimizer(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(2), int(n_neurons)), torch.zeros(int(2), int(n_neurons))).to(DEVICE)
adam_B2 = AdamOptimizer(lr, 0.9, 0.999, 1e-08, 0.0, torch.zeros(int(2), int(1)), torch.zeros(int(2), int(1))).to(DEVICE)
epochs = 1
for i in range(int(0), int(epochs)):
    print(i)
    pred_traj = odesolver(model, y0, Δt, timesteps)
    diff = (pred_traj - true_trajectory)
    loss = torch.mean((diff ** 2) if isinstance((diff ** 2), torch.Tensor) else torch.tensor(float((diff ** 2))))
    a_t0, dW1, dB1, dW2, dB2 = adjoint_solver(pred_traj, true_trajectory, Δt, n_steps, model)
    with torch.no_grad():
      model.W1.copy_(adam_W1.step(model.W1, dW1))
    with torch.no_grad():
      model.B1.copy_(adam_B1.step(model.B1, dB1))
    with torch.no_grad():
      model.W2.copy_(adam_W2.step(model.W2, dW2))
    with torch.no_grad():
      model.B2.copy_(adam_B2.step(model.B2, dB2))
    print(loss)
predicted_trajectory = odesolver(model, y0, Δt, timesteps)