import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def minmod(r):
    m1 = (0.5 * ((1.0 + r) - torch.abs((1.0 - r) if isinstance((1.0 - r), torch.Tensor) else torch.tensor(float((1.0 - r))))))
    phi = (0.5 * (m1 + torch.abs(m1 if isinstance(m1, torch.Tensor) else torch.tensor(float(m1)))))
    return phi

def superbee(r):
    s1 = (0.5 * (((2.0 * r) + 1.0) - torch.abs(((2.0 * r) - 1.0) if isinstance(((2.0 * r) - 1.0), torch.Tensor) else torch.tensor(float(((2.0 * r) - 1.0))))))
    s2 = (0.5 * ((r + 2.0) - torch.abs((r - 2.0) if isinstance((r - 2.0), torch.Tensor) else torch.tensor(float((r - 2.0))))))
    s3 = (0.5 * ((s1 + s2) + torch.abs((s1 - s2) if isinstance((s1 - s2), torch.Tensor) else torch.tensor(float((s1 - s2))))))
    phi = (0.5 * (s3 + torch.abs(s3 if isinstance(s3, torch.Tensor) else torch.tensor(float(s3)))))
    return phi

def tanh_act(x):
    y = (1.0 - (2.0 / (torch.exp((2.0 * x) if isinstance((2.0 * x), torch.Tensor) else torch.tensor(float((2.0 * x)))) + 1.0)))
    return y

# === Classes ===
class NeuralFluxLimiter(nn.Module):
    def __init__(self, θ, b2, m_θ, v_θ, m_b, v_b, t_adam):
        super().__init__()
        self.θ = nn.Parameter(torch.as_tensor(θ))
        self.b2 = nn.Parameter(torch.as_tensor(b2))
        self.m_θ = nn.Parameter(torch.as_tensor(m_θ))
        self.v_θ = nn.Parameter(torch.as_tensor(v_θ))
        self.m_b = nn.Parameter(torch.as_tensor(m_b))
        self.v_b = nn.Parameter(torch.as_tensor(v_b))
        self.t_adam = nn.Parameter(torch.as_tensor(t_adam))
        self.learnable_params = [self.θ, self.b2, self.m_θ, self.v_θ, self.m_b, self.v_b, self.t_adam]

    def forward(self, r):
        this = self
        r = torch.as_tensor(r, device=DEVICE).float()
        rc = (0.5 * ((r + 10.0) - torch.abs((r - 10.0) if isinstance((r - 10.0), torch.Tensor) else torch.tensor(float((r - 10.0))))))
        rc = (0.5 * ((rc - 2.0) + torch.abs((rc + 2.0) if isinstance((rc + 2.0), torch.Tensor) else torch.tensor(float((rc + 2.0))))))
        w_in = self.θ[int(0)]
        b_h = self.θ[int(1)]
        w_out = self.θ[int(2)]
        z = ((0.0 * rc) + self.b2)
        for k in range(int(0), int(8)):
            z = (z + (w_out[int(k)] * tanh_act(((w_in[int(k)] * rc) + b_h[int(k)]))))
        s = (1.0 / (1.0 + torch.exp((0.0 - z) if isinstance((0.0 - z), torch.Tensor) else torch.tensor(float((0.0 - z))))))
        phi = (((1.0 - s) * minmod(rc)) + (s * superbee(rc)))
        return phi

    def solve(self, ic):
        this = self
        ic = torch.as_tensor(ic, device=DEVICE).float()
        ρ = zero_1d_array(Nx)
        u0 = zero_1d_array(Nx)
        p0 = zero_1d_array(Nx)
        for i in range(int(0), int(50)):
            ρ[int(i)] = ic[int(0)]
            u0[int(i)] = ic[int(1)]
            p0[int(i)] = ic[int(2)]
        for i in range(int(50), int(100)):
            ρ[int(i)] = ic[int(3)]
            u0[int(i)] = ic[int(4)]
            p0[int(i)] = ic[int(5)]
        mom = (ρ * u0)
        E = ((p0 / (γ - 1.0)) + (((0.5 * ρ) * u0) * u0))
        for n in range(int(0), int(Nt)):
            u = (mom / ρ)
            p = ((γ - 1.0) * (E - ((0.5 * mom) * u)))
            c = torch.sqrt(((γ * p) / ρ) if isinstance(((γ * p) / ρ), torch.Tensor) else torch.tensor(float(((γ * p) / ρ))))
            a = (torch.abs(u if isinstance(u, torch.Tensor) else torch.tensor(float(u))) + c)
            a_face = (0.5 * ((a + torch.roll(a, (-1))) + torch.abs((a - torch.roll(a, (-1))) if isinstance((a - torch.roll(a, (-1))), torch.Tensor) else torch.tensor(float((a - torch.roll(a, (-1))))))))
            lw = ((0.5 * a_face) * (1.0 - ((a_face * dt) / dx)))
            u_face = (0.5 * (u + torch.roll(u, (-1))))
            w_up = (0.5 * (1.0 + (u_face / (torch.abs(u_face if isinstance(u_face, torch.Tensor) else torch.tensor(float(u_face))) + ε))))
            Δρ = (torch.roll(ρ, (-1)) - ρ)
            r_ρ = (((w_up * torch.roll(Δρ, 1)) / (Δρ + ε)) + (((1.0 - w_up) * torch.roll(Δρ, (-1))) / (Δρ + ε)))
            F_ρ = (((0.5 * (mom + torch.roll(mom, (-1)))) - ((0.5 * a_face) * Δρ)) + ((lw * self(r_ρ)) * Δρ))
            f_m = ((mom * u) + p)
            Δm = (torch.roll(mom, (-1)) - mom)
            r_m = (((w_up * torch.roll(Δm, 1)) / (Δm + ε)) + (((1.0 - w_up) * torch.roll(Δm, (-1))) / (Δm + ε)))
            F_m = (((0.5 * (f_m + torch.roll(f_m, (-1)))) - ((0.5 * a_face) * Δm)) + ((lw * self(r_m)) * Δm))
            f_E = (u * (E + p))
            ΔE = (torch.roll(E, (-1)) - E)
            r_E = (((w_up * torch.roll(ΔE, 1)) / (ΔE + ε)) + (((1.0 - w_up) * torch.roll(ΔE, (-1))) / (ΔE + ε)))
            F_E = (((0.5 * (f_E + torch.roll(f_E, (-1)))) - ((0.5 * a_face) * ΔE)) + ((lw * self(r_E)) * ΔE))
            ρ = (ρ - (((dt / dx) * mask) * (F_ρ - torch.roll(F_ρ, 1))))
            mom = (mom - (((dt / dx) * mask) * (F_m - torch.roll(F_m, 1))))
            E = (E - (((dt / dx) * mask) * (F_E - torch.roll(F_E, 1))))
        return ρ

    def loss_ic(self, ic, target):
        this = self
        ic = torch.as_tensor(ic, device=DEVICE).float()
        target = torch.as_tensor(target, device=DEVICE).float()
        pred = self.solve(ic)
        diff = (pred - target)
        l = torch.mean((diff * diff) if isinstance((diff * diff), torch.Tensor) else torch.tensor(float((diff * diff))))
        return l

    def evaluate(self, X, Y):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        Y = torch.as_tensor(Y, device=DEVICE).float()
        l = self.loss_ic(X[int(0)], Y[int(0)])
        return l

    def train(self, X, Y, epochs, lr):
        this = self
        X = torch.as_tensor(X, device=DEVICE).float()
        Y = torch.as_tensor(Y, device=DEVICE).float()
        lr = torch.as_tensor(lr, device=DEVICE).float()
        last = 0
        for e in range(int(0), int(epochs)):
            for j in range(int(0), int(1)):
                current_loss = self.loss_ic(X[int(j)], Y[int(j)])
                learnable_grads = compute_grad(current_loss, self.learnable_params)
                self.update_params(lr, learnable_grads)
                last = current_loss
        return last

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        β1 = 0.9
        β2 = 0.999
        with torch.no_grad():
            self.t_adam.copy_((self.t_adam + 1.0))
        with torch.no_grad():
            self.m_θ.copy_(((β1 * self.m_θ) + ((1.0 - β1) * learnable_grads[int(0)])))
        with torch.no_grad():
            self.v_θ.copy_(((β2 * self.v_θ) + (((1.0 - β2) * learnable_grads[int(0)]) * learnable_grads[int(0)])))
        with torch.no_grad():
            self.θ.copy_((self.θ - ((lr * (self.m_θ / (1.0 - (β1 ** self.t_adam)))) / (torch.sqrt((self.v_θ / (1.0 - (β2 ** self.t_adam))) if isinstance((self.v_θ / (1.0 - (β2 ** self.t_adam))), torch.Tensor) else torch.tensor(float((self.v_θ / (1.0 - (β2 ** self.t_adam)))))) + 1e-08))))
        with torch.no_grad():
            self.m_b.copy_(((β1 * self.m_b) + ((1.0 - β1) * learnable_grads[int(1)])))
        with torch.no_grad():
            self.v_b.copy_(((β2 * self.v_b) + (((1.0 - β2) * learnable_grads[int(1)]) * learnable_grads[int(1)])))
        with torch.no_grad():
            self.b2.copy_((self.b2 - ((lr * (self.m_b / (1.0 - (β1 ** self.t_adam)))) / (torch.sqrt((self.v_b / (1.0 - (β2 ** self.t_adam))) if isinstance((self.v_b / (1.0 - (β2 ** self.t_adam))), torch.Tensor) else torch.tensor(float((self.v_b / (1.0 - (β2 ** self.t_adam)))))) + 1e-08))))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
γ = 1.4
Nx = 100
dx = 0.01
dt = 0.00163934
Nt = 61
ε = 1e-08
initial_states = torch.tensor([[1, 0, 1, 0.125, 0, 0.1]], device=DEVICE)
true_rho = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.976824, 0.909632, 0.84619, 0.786338, 0.729922, 0.676791, 0.6268, 0.579808, 0.535677, 0.494276, 0.455475, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.265574, 0.265574, 0.265574, 0.265574, 0.265574, 0.265574, 0.265574, 0.265574, 0.265574, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]], device=DEVICE)
mask = zero_1d_array(Nx)
for i in range(int(1), int(99)):
    mask[int(i)] = 1.0
θ_zero = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)
minmod_limiter = NeuralFluxLimiter(θ_zero, (-20.0), θ_zero, θ_zero, 0.0, 0.0, 0.0)
superbee_limiter = NeuralFluxLimiter(θ_zero, 20.0, θ_zero, θ_zero, 0.0, 0.0, 0.0)
minmod_loss = minmod_limiter.evaluate(initial_states, true_rho)
physika_print(minmod_loss)
superbee_loss = superbee_limiter.evaluate(initial_states, true_rho)
physika_print(superbee_loss)
θ_0 = torch.tensor([[0.5, (-0.5), 0.3, (-0.3), 0.8, (-0.8), 0.2, (-0.2)], [0.1, (-0.1), 0.4, (-0.4), 0.2, (-0.2), (-0.3), 0.3], [0.3, (-0.2), 0.25, (-0.3), 0.2, 0.15, (-0.25), 0.2]], device=DEVICE)
b2_0 = (-3.0)
adam_m0 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)
adam_v0 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)
net = NeuralFluxLimiter(θ_0, b2_0, adam_m0, adam_v0, 0.0, 0.0, 0.0)
loss_before = net.evaluate(initial_states, true_rho)
physika_print(loss_before)
epochs = 16
lr = 0.1
final_loss = net.train(initial_states, true_rho, epochs, lr)
loss_after = net.evaluate(initial_states, true_rho)
physika_print(loss_after)
r_sample = torch.tensor([(-2.0), (-1.0), (-0.5), 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0], device=DEVICE)
learned_phi = net(r_sample)
physika_print(learned_phi)
minmod_phi = minmod(r_sample)
physika_print(minmod_phi)
superbee_phi = superbee(r_sample)
physika_print(superbee_phi)