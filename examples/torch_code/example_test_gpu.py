import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
n_values = 1000
x = torch.stack([torch.stack([(i * 1) for _fi_j in range(int(n_values)) for j in [torch.tensor(float(_fi_j), device='cpu')]]) for _fi_i in range(int(n_values)) for i in [torch.tensor(float(_fi_i), device='cpu')]])
for i in range(int(0), int(100)):
    x = torch.nn.functional.relu(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    x = torch.sin(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    x = torch.exp(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    x = torch.log(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))