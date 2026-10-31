import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def height(t):
    return ((20 * t) - (5 * (t ** 2)))

# === Program ===