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
u0 = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.75, 0.75, 0.0], [0.0, 0.75, 0.75, 10.0], [0.0, 0.0, 0.0, 0.0]])
u00 = u0[int(0.0), int(0.0)]
u01 = u0[int(0.0), int(1.0)]
u02 = u0[int(0.0), int(2.0)]
u03 = u0[int(0.0), int(3.0)]
u10 = u0[int(1.0), int(0.0)]
u11 = u0[int(1.0), int(1.0)]
u12 = u0[int(1.0), int(2.0)]
u13 = u0[int(1.0), int(3.0)]
u20 = u0[int(2.0), int(0.0)]
u21 = u0[int(2.0), int(1.0)]
u22 = u0[int(2.0), int(2.0)]
u23 = u0[int(2.0), int(3.0)]
u30 = u0[int(3.0), int(0.0)]
u31 = u0[int(3.0), int(1.0)]
u32 = u0[int(3.0), int(2.0)]
u33 = u0[int(3.0), int(3.0)]
A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
A00 = A[int(0.0), int(0.0)]
A11 = A[int(1.0), int(1.0)]
A22 = A[int(2.0), int(2.0)]
T = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]])
T0 = T[int(0.0)]
T12 = T[int(1.0), int(2.0)]
T000 = T[int(0.0), int(0.0), int(0.0)]
T123 = T[int(1.0), int(2.0), int(3.0)]
T012 = T[int(0.0), int(1.0), int(2.0)]