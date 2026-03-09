import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
total = 0.0
for i in range(len(arr)):
    total = total + arr[int(i)]
physika_print(total)
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
sum_sq = 0.0
for i in range(len(X)):
    sum_sq = sum_sq + (X[int(i)] ** 2.0)
physika_print(sum_sq)
y = torch.tensor([2.0, 4.0, 6.0, 8.0])
mse = 0.0
for i in range(len(X)):
    mse = mse + ((X[int(i)] - y[int(i)]) ** 2.0)
physika_print(mse)