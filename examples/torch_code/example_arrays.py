import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
x = torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
y = (x[int(0.0):int(2.0)+1] + x[int(0.0):int(2.0)+1])
z = (y + torch.tensor([1.0, 3.0, 4.0]))
physika_print(x)
physika_print(y)
physika_print(z)