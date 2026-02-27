import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print

# === Functions ===
def fact(n):
    n = torch.as_tensor(n).float()
    return torch.where(n == 0.0, torch.as_tensor(1.0).float(), torch.as_tensor((n * fact((n - 1.0)))).float())

# === Program ===
physika_print(fact(0.0))
physika_print(fact(1.0))
physika_print(fact(2.0))
physika_print(fact(3.0))
physika_print(fact(4.0))
physika_print(fact(5.0))
physika_print(fact(10.0))