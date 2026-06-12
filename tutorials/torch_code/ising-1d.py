import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def tanh(z):
    num = (torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) - torch.exp((-z) if isinstance((-z), torch.Tensor) else torch.tensor(float((-z)))))
    denom = (torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) + torch.exp((-z) if isinstance((-z), torch.Tensor) else torch.tensor(float((-z)))))
    res = (num / denom)
    return res

def H(J, h, spins, n):
    n = (n - 1)
    nn_bulk = torch.stack([(spins[int(i)] * spins[int((i + 1))]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i))]])
    nn_sum = (torch.sum(nn_bulk if isinstance(nn_bulk, torch.Tensor) else torch.tensor(float(nn_bulk))) + (spins[int(n)] * spins[int(0)]))
    field_sum = torch.sum(spins if isinstance(spins, torch.Tensor) else torch.tensor(float(spins)))
    return (((-J) * nn_sum) - (h * field_sum))

def neg_entropy_per_site(p):
    return ((p * torch.log(p if isinstance(p, torch.Tensor) else torch.tensor(float(p)))) + ((1.0 - p) * torch.log((1.0 - p) if isinstance((1.0 - p), torch.Tensor) else torch.tensor(float((1.0 - p))))))

def mean_field_reference(J, h, β, iters):
    m = 0.0
    for it in range(int(0), int(iters)):
        m = tanh((β * (((2.0 * J) * m) + h)))
    return ((1.0 + m) * 0.5)

# === Classes ===
class MeanFieldIsing(nn.Module):
    def __init__(self, logit_p):
        super().__init__()
        self.logit_p = nn.Parameter(torch.as_tensor(logit_p).float())
        self.register_buffer('baseline', torch.tensor(0.0))

    def forward(self, n):
        this = self
        p = (1.0 / (1.0 + torch.exp((-self.logit_p) if isinstance((-self.logit_p), torch.Tensor) else torch.tensor(float((-self.logit_p))))))
        b_s = torch.distributions.Bernoulli(p).sample((int(n),)).detach()
        spins = torch.stack([((2.0 * b_s[int(i)]) - 1.0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i))]])
        return spins

    def loss(self, spins, J, h, β, size):
        this = self
        spins = torch.as_tensor(spins).float()
        J = torch.as_tensor(J).float()
        h = torch.as_tensor(h).float()
        β = torch.as_tensor(β).float()
        size = torch.as_tensor(size).float()
        p = (1.0 / (1.0 + torch.exp((-self.logit_p) if isinstance((-self.logit_p), torch.Tensor) else torch.tensor(float((-self.logit_p))))))
        b = torch.stack([((spins[int(i)] + 1.0) * 0.5) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i))]])
        energy_ps = (H(J, h, spins, size) / n)
        log_prob = torch.sum(torch.stack([((b[int(i)] * torch.log(p if isinstance(p, torch.Tensor) else torch.tensor(float(p)))) + ((1.0 - b[int(i)]) * torch.log((1.0 - p) if isinstance((1.0 - p), torch.Tensor) else torch.tensor(float((1.0 - p)))))) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i))]]) if isinstance(torch.stack([((b[int(i)] * torch.log(p if isinstance(p, torch.Tensor) else torch.tensor(float(p)))) + ((1.0 - b[int(i)]) * torch.log((1.0 - p) if isinstance((1.0 - p), torch.Tensor) else torch.tensor(float((1.0 - p)))))) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i))]]), torch.Tensor) else torch.tensor(float(torch.stack([((b[int(i)] * torch.log(p if isinstance(p, torch.Tensor) else torch.tensor(float(p)))) + ((1.0 - b[int(i)]) * torch.log((1.0 - p) if isinstance((1.0 - p), torch.Tensor) else torch.tensor(float((1.0 - p)))))) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i))]]))))
        energy_term = ((β * (energy_ps - self.baseline)) * log_prob)
        entropy_term = neg_entropy_per_site(p)
        return (energy_term + entropy_term)

    def train(self, n, n_steps, n_batch, lr, J, h, β, size):
        this = self
        lr = torch.as_tensor(lr).float()
        J = torch.as_tensor(J).float()
        h = torch.as_tensor(h).float()
        β = torch.as_tensor(β).float()
        spins_0 = self(n)
        self.baseline = (H(J, h, spins_0, size) / n)
        for step in range(int(0), int(n_steps)):
            batch_loss = 0.0
            mean_energy_ps = 0.0
            for k in range(int(0), int(n_batch)):
                spins = self(n)
                mean_energy_ps = (mean_energy_ps + ((H(J, h, spins, size) / n) / n_batch))
                batch_loss = (batch_loss + (self.loss(spins, J, h, β, size) / n_batch))
            grads = compute_grad(batch_loss, self.params)
            self.baseline = ((0.9 * self.baseline) + (0.1 * mean_energy_ps))
            self.update(lr, grads)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
torch.manual_seed(int(0))
n = 20
J = 1.0
h = 0.5
β = 5.0
steps = 100
batch = 32
logit_init = 0.0
lr = 0.05
ising = MeanFieldIsing(logit_init)
p_before = (1.0 / (1.0 + torch.exp((0.0 - ising.logit_p) if isinstance((0.0 - ising.logit_p), torch.Tensor) else torch.tensor(float((0.0 - ising.logit_p))))))
spins_before = ising(n)
physika_print(ising.train(n, steps, batch, lr, J, h, β, n))
p_after = (1.0 / (1.0 + torch.exp((0.0 - ising.logit_p) if isinstance((0.0 - ising.logit_p), torch.Tensor) else torch.tensor(float((0.0 - ising.logit_p))))))
p_ref = mean_field_reference(J, h, β, 200)
spins_after = ising(n)
physika_print(p_before)
physika_print(p_after)
physika_print(p_ref)