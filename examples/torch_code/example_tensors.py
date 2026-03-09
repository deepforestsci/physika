import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
v = torch.tensor([1.0, 2.0, 3.0])
physika_print(v)
omega = torch.tensor([1.0, 2.0, 3.0])
physika_print(omega)
T = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
v = torch.tensor([1.0, 2.0, 3.0])
w = (T @ v)
physika_print(w)
T = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
v = torch.tensor([1.0, 2.0, 3.0])
w = (T @ v)
physika_print(w)
physika_print(T)