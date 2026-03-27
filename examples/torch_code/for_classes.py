import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Classes ===
class SumArray(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w).float() if not isinstance(w, torch.Tensor) else w.clone().detach().float())

    def forward(self, x):
        x = torch.as_tensor(x).float()
        total = 0.0
        for i in range(int(len(x))):
            total = (total + x[int(i)])
        return total

# === Program ===
arr = torch.tensor([1.0, 2.0, 3.0, 4.0])
model = SumArray(arr)
physika_print(model(torch.tensor([1.0, 2.0, 3.0, 4.0])))