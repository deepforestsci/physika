import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
a = Z2(1)
b = Z2(0)
xor_opposite = Z2((a + b))
xor_same = Z2((a + a))
physika_print(xor_opposite)
physika_print(xor_same)
and_mixed = Z2((a * b))
and_both_ones = Z2((a * a))
physika_print(and_mixed)
physika_print(and_both_ones)
flag = Z2(0)
if flag == 0:
    results = 0.0
else:
    results = 1.0
physika_print(results)