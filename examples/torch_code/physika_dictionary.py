import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
empty_dict = {}
print(print(empty_dict))
simple_dict = {0: 1.6, 1: 3.2, 2: 5.5}
print(print(simple_dict))
union_dict = {0: 1, 1: 3j, 2: 30, 3: torch.tensor([1, 2, 3], device=DEVICE)}
print(print(union_dict))