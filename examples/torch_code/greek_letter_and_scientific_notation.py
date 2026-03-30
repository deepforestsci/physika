import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
α = 1.0
β = 2.0
x = 100000.0
y = 300000.0
results = (α + β)
physika_print(results)
z = (x + y)
physika_print(z)
greek_letters_array = torch.stack([torch.as_tensor(α).float(), torch.as_tensor(β).float()])
physika_print(greek_letters_array)