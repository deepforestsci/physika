  ✓ No type errors found

=== Physika generated Pytorch code ===
import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Classes ===
class MatrixMultiply(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = nn.Parameter(torch.as_tensor(x))
        self.learnable_params = [self.x]

    def forward(self, A, B):
        this = self
        A = torch.as_tensor(A, device='cpu').float()
        B = torch.as_tensor(B, device='cpu').float()
        return (A @ B)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
x_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cpu')
n_values = 10
x = torch.stack([torch.stack([(i * 1) for _fi_j in range(int(n_values)) for j in [torch.tensor(float(_fi_j), device='cpu')]]) for _fi_i in range(int(n_values)) for i in [torch.tensor(float(_fi_i), device='cpu')]])
for i in range(int(0), int(100)):
    x = torch.sin(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    x = torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    x = torch.log(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
n_values = 10
A = torch.stack([torch.stack([(i * 1) for _fi_j in range(int(n_values)) for j in [torch.tensor(float(_fi_j), device='cpu')]]) for _fi_i in range(int(n_values)) for i in [torch.tensor(float(_fi_i), device='cpu')]])
B = torch.stack([torch.stack([(i * 1) for _fi_j in range(int(n_values)) for j in [torch.tensor(float(_fi_j), device='cpu')]]) for _fi_i in range(int(n_values)) for i in [torch.tensor(float(_fi_i), device='cpu')]])
obj = MatrixMultiply(1.0).to('cpu')
results = obj(A, B)
=== End Pytorch code ===

