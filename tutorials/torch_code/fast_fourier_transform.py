import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def spectral_energy(x):
    return torch.sum((torch.abs(torch.fft.fft(x) if isinstance(torch.fft.fft(x), torch.Tensor) else torch.tensor(float(torch.fft.fft(x)))) ** 2) if isinstance((torch.abs(torch.fft.fft(x) if isinstance(torch.fft.fft(x), torch.Tensor) else torch.tensor(float(torch.fft.fft(x)))) ** 2), torch.Tensor) else torch.tensor(float((torch.abs(torch.fft.fft(x) if isinstance(torch.fft.fft(x), torch.Tensor) else torch.tensor(float(torch.fft.fft(x)))) ** 2))))

# === Program ===
Ns = 9
two_pi = 6.283185307179586
center = 4.0
f = torch.stack([(torch.cos(((two_pi * (n - center)) / Ns) if isinstance(((two_pi * (n - center)) / Ns), torch.Tensor) else torch.tensor(float(((two_pi * (n - center)) / Ns)))) + (0.5 * torch.cos((((two_pi * 3) * (n - center)) / Ns) if isinstance((((two_pi * 3) * (n - center)) / Ns), torch.Tensor) else torch.tensor(float((((two_pi * 3) * (n - center)) / Ns)))))) for _fi_n in range(int(Ns)) for n in [torch.tensor(float(_fi_n))]])
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
x = torch.as_tensor(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])).float().requires_grad_(True)
physika_print(compute_grad(lambda _dx: spectral_energy(_dx), x))