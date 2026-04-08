import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
a = 1
b = 0
physika_print((a or b))
physika_print((a and b))
if a or b:
    physika_print('True from or condition')
if a and a:
    physika_print('True from and condition')