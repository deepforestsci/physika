import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def dft(x):
    s = len(x)
    coeffs = torch.stack([torch.sum(torch.stack([(x[int(n)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) - (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n))]]) if isinstance(torch.stack([(x[int(n)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) - (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n))]]), torch.Tensor) else torch.tensor(float(torch.stack([(x[int(n)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) - (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n))]])))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k))]])
    return coeffs

def idft(X):
    s = len(X)
    samples = torch.stack([(torch.sum(torch.stack([(X[int(k)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) + (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k))]]) if isinstance(torch.stack([(X[int(k)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) + (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k))]]), torch.Tensor) else torch.tensor(float(torch.stack([(X[int(k)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) + (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k))]])))) / s) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n))]])
    return samples

def ct_fft(x, s):
    if s == 1.0:
        return x
    even = torch.stack([x[int((2 * a))] for _fi_a in range(int((s / 2))) for a in [torch.tensor(float(_fi_a))]])
    odd = torch.stack([x[int(((2 * a) + 1))] for _fi_a in range(int((s / 2))) for a in [torch.tensor(float(_fi_a))]])
    E = ct_fft(even, (s / 2))
    O = ct_fft(odd, (s / 2))
    twiddle = torch.stack([(torch.cos((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))) - (1j * torch.sin((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))))) for _fi_k in range(int((s / 2))) for k in [torch.tensor(float(_fi_k))]])
    lower = torch.stack([(E[int(k)] + (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int((s / 2))) for k in [torch.tensor(float(_fi_k))]])
    upper = torch.stack([(E[int(k)] - (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int((s / 2))) for k in [torch.tensor(float(_fi_k))]])
    return torch.cat([lower, upper])

def ict_fft(X, s):
    if s == 1.0:
        return X
    even = torch.stack([X[int((2 * a))] for _fi_a in range(int((s / 2))) for a in [torch.tensor(float(_fi_a))]])
    odd = torch.stack([X[int(((2 * a) + 1))] for _fi_a in range(int((s / 2))) for a in [torch.tensor(float(_fi_a))]])
    E = ict_fft(even, (s / 2))
    O = ict_fft(odd, (s / 2))
    twiddle = torch.stack([(torch.cos((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))) + (1j * torch.sin((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))))) for _fi_k in range(int((s / 2))) for k in [torch.tensor(float(_fi_k))]])
    lower = torch.stack([(E[int(k)] + (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int((s / 2))) for k in [torch.tensor(float(_fi_k))]])
    upper = torch.stack([(E[int(k)] - (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int((s / 2))) for k in [torch.tensor(float(_fi_k))]])
    return torch.cat([lower, upper])

# === Program ===
π = 3.141592653589793
signal = torch.stack([torch.cos((((2 * π) * n) / 4) if isinstance((((2 * π) * n) / 4), torch.Tensor) else torch.tensor(float((((2 * π) * n) / 4)))) for _fi_n in range(int(4)) for n in [torch.tensor(float(_fi_n))]])
forward_transform = dft(signal)
physika_print(torch.abs(forward_transform if isinstance(forward_transform, torch.Tensor) else torch.tensor(float(forward_transform))))
physika_print(idft(forward_transform))
fft_sum = ct_fft(signal, 4)
physika_print(torch.abs(fft_sum if isinstance(fft_sum, torch.Tensor) else torch.tensor(float(fft_sum))))
physika_print((ict_fft(fft_sum, 4) / 4))
Ns = 9
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