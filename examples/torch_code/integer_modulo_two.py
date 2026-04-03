import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.z2 import Z2

# === Program ===
a = Z2(1)
b = Z2(0)
c = (a + b)
physika_print(c)
physika_print((a + a))
d = (a * b)
physika_print(a)
flag = Z2(0)
if flag == 0:
    results = 0.0
else:
    results = 1.0
physika_print(results)