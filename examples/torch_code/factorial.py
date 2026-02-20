import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print

# === Functions ===
def fact(n):
    if n == 0.0:
        return 1.0
    else:
        return (n * fact((n - 1.0)))

# === Program ===
physika_print(fact(0.0))
physika_print(fact(1.0))
physika_print(fact(2.0))
physika_print(fact(3.0))
physika_print(fact(4.0))
physika_print(fact(5.0))
physika_print(fact(10.0))