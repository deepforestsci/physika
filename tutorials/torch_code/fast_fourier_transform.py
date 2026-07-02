import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Program ===
Ns = 9
π = 3.141592653589793
center = 4.0
f = torch.stack([(torch.cos((((2 * π) * (n - center)) / Ns) if isinstance((((2 * π) * (n - center)) / Ns), torch.Tensor) else torch.tensor(float((((2 * π) * (n - center)) / Ns)))) + (0.5 * torch.cos(((((2 * π) * 3) * (n - center)) / Ns) if isinstance(((((2 * π) * 3) * (n - center)) / Ns), torch.Tensor) else torch.tensor(float(((((2 * π) * 3) * (n - center)) / Ns)))))) for _fi_n in range(int(Ns)) for n in [torch.tensor(float(_fi_n))]])
F = torch.fft.fft(f)
physika_print(torch.abs(F if isinstance(F, torch.Tensor) else torch.tensor(float(F))))
physika_print(torch.fft.ifft(F))
M = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
G = torch.fft.fft2(M)
physika_print(G)
physika_print(torch.fft.ifft2(G))
V = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
S = torch.fft.fftn(V)
physika_print(S)
physika_print(torch.fft.ifftn(S))
x = torch.as_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0])).requires_grad_(True)
X = torch.fft.fft(x)
y = torch.real(X[int(1)] if isinstance(X[int(1)], torch.Tensor) else torch.tensor(float(X[int(1)])))
physika_print(compute_grad(y, x))
invX = torch.fft.ifft(X)
w = torch.real(invX[int(0)] if isinstance(invX[int(0)], torch.Tensor) else torch.tensor(float(invX[int(0)])))
physika_print(compute_grad(w, x))
x2 = torch.as_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).requires_grad_(True)
X2 = torch.fft.fft(x2)
energy_time = torch.sum((x2 ** 2) if isinstance((x2 ** 2), torch.Tensor) else torch.tensor(float((x2 ** 2))))
energy_freq = (torch.sum((torch.abs(X2 if isinstance(X2, torch.Tensor) else torch.tensor(float(X2))) ** 2) if isinstance((torch.abs(X2 if isinstance(X2, torch.Tensor) else torch.tensor(float(X2))) ** 2), torch.Tensor) else torch.tensor(float((torch.abs(X2 if isinstance(X2, torch.Tensor) else torch.tensor(float(X2))) ** 2)))) / len(x2))
physika_print(energy_time)
physika_print(energy_freq)
physika_print(compute_grad(energy_time, x2))
physika_print(compute_grad(energy_freq, x2))