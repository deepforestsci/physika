import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def dft(x):
    s = len(x)
    coeffs = torch.stack([torch.sum(torch.stack([(x[int(n)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) - (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]]) if isinstance(torch.stack([(x[int(n)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) - (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]]), torch.Tensor) else torch.tensor(float(torch.stack([(x[int(n)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) - (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    return coeffs

def idft(X):
    s = len(X)
    samples = torch.stack([(torch.sum(torch.stack([(X[int(k)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) + (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]) if isinstance(torch.stack([(X[int(k)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) + (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]]), torch.Tensor) else torch.tensor(float(torch.stack([(X[int(k)] * (torch.cos(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s)))) + (1j * torch.sin(((((2 * π) * k) * n) / s) if isinstance(((((2 * π) * k) * n) / s), torch.Tensor) else torch.tensor(float(((((2 * π) * k) * n) / s))))))) for _fi_k in range(int(s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])))) / s) for _fi_n in range(int(s)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
    return samples

def ct_fft(x, s):
    if s == 1.0:
        return x
    half_size = (s / 2)
    even = torch.stack([x[int((2 * a))] for _fi_a in range(int(half_size)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    odd = torch.stack([x[int(((2 * a) + 1))] for _fi_a in range(int(half_size)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    E = ct_fft(even, half_size)
    O = ct_fft(odd, half_size)
    twiddle = torch.stack([(torch.cos((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))) - (1j * torch.sin((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))))) for _fi_k in range(int(half_size)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    lower = torch.stack([(E[int(k)] + (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int(half_size)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    upper = torch.stack([(E[int(k)] - (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int(half_size)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    return torch.cat([lower, upper])

def ict_fft(X, s):
    if s == 1.0:
        return X
    half_size = (s / 2)
    even = torch.stack([X[int((2 * a))] for _fi_a in range(int(half_size)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    odd = torch.stack([X[int(((2 * a) + 1))] for _fi_a in range(int(half_size)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    E = ict_fft(even, half_size)
    O = ict_fft(odd, half_size)
    twiddle = torch.stack([(torch.cos((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))) + (1j * torch.sin((((2 * π) * k) / s) if isinstance((((2 * π) * k) / s), torch.Tensor) else torch.tensor(float((((2 * π) * k) / s)))))) for _fi_k in range(int(half_size)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    lower = torch.stack([(E[int(k)] + (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int(half_size)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    upper = torch.stack([(E[int(k)] - (twiddle[int(k)] * O[int(k)])) for _fi_k in range(int(half_size)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
    return torch.cat([lower, upper])

# === Program ===
π = 3.141592653589793
signal = torch.stack([torch.cos((((2 * π) * n) / 4) if isinstance((((2 * π) * n) / 4), torch.Tensor) else torch.tensor(float((((2 * π) * n) / 4)))) for _fi_n in range(int(4)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
dft_spectrum = dft(signal)
physika_print(dft_spectrum)
physika_print(torch.abs(dft_spectrum if isinstance(dft_spectrum, torch.Tensor) else torch.tensor(float(dft_spectrum))))
physika_print(idft(dft_spectrum))
fft_spectrum = ct_fft(signal, 4)
physika_print(fft_spectrum)
physika_print(torch.abs(fft_spectrum if isinstance(fft_spectrum, torch.Tensor) else torch.tensor(float(fft_spectrum))))
physika_print((ict_fft(fft_spectrum, 4) / 4))
physika_print(dft(signal))
physika_print(ct_fft(signal, 4))
physika_print(torch.fft.fft(signal))
Ns = 9
center = 4.0
signal_1d = torch.stack([(torch.cos((((2 * π) * (n - center)) / Ns) if isinstance((((2 * π) * (n - center)) / Ns), torch.Tensor) else torch.tensor(float((((2 * π) * (n - center)) / Ns)))) + (0.5 * torch.cos(((((2 * π) * 3) * (n - center)) / Ns) if isinstance(((((2 * π) * 3) * (n - center)) / Ns), torch.Tensor) else torch.tensor(float(((((2 * π) * 3) * (n - center)) / Ns)))))) for _fi_n in range(int(Ns)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
spectrum_1d = torch.fft.fft(signal_1d)
physika_print(torch.abs(spectrum_1d if isinstance(spectrum_1d, torch.Tensor) else torch.tensor(float(spectrum_1d))))
physika_print(torch.fft.ifft(spectrum_1d))
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=DEVICE)
spectrum_2d = torch.fft.fft2(matrix)
physika_print(spectrum_2d)
physika_print(torch.fft.ifft2(spectrum_2d))
tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=DEVICE)
spectrum_Nd = torch.fft.fftn(tensor)
physika_print(spectrum_Nd)
physika_print(torch.fft.ifftn(spectrum_Nd))
x = torch.as_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)).requires_grad_(True).to(DEVICE)
X = torch.fft.fft(x)
physika_print(X)
y = torch.real(X[int(1)] if isinstance(X[int(1)], torch.Tensor) else torch.tensor(float(X[int(1)])))
physika_print(y)
physika_print(compute_grad(y, x))
dft_spectrum = dft(x)
dft_coeff = torch.real(dft_spectrum[int(1)] if isinstance(dft_spectrum[int(1)], torch.Tensor) else torch.tensor(float(dft_spectrum[int(1)])))
physika_print(compute_grad(dft_coeff, x))
ct_fft_spectrum = ct_fft(x, 4)
ct_fft_coeff = torch.real(ct_fft_spectrum[int(1)] if isinstance(ct_fft_spectrum[int(1)], torch.Tensor) else torch.tensor(float(ct_fft_spectrum[int(1)])))
physika_print(compute_grad(ct_fft_coeff, x))
inv_signal = torch.fft.ifft(X)
inv_sample = torch.real(inv_signal[int(1)] if isinstance(inv_signal[int(1)], torch.Tensor) else torch.tensor(float(inv_signal[int(1)])))
physika_print(inv_sample)
physika_print(compute_grad(inv_sample, x))
x2 = torch.as_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=DEVICE)).requires_grad_(True).to(DEVICE)
X2 = torch.fft.fft(x2)
energy_time = torch.sum((x2 ** 2) if isinstance((x2 ** 2), torch.Tensor) else torch.tensor(float((x2 ** 2))))
energy_freq = (torch.sum((torch.abs(X2 if isinstance(X2, torch.Tensor) else torch.tensor(float(X2))) ** 2) if isinstance((torch.abs(X2 if isinstance(X2, torch.Tensor) else torch.tensor(float(X2))) ** 2), torch.Tensor) else torch.tensor(float((torch.abs(X2 if isinstance(X2, torch.Tensor) else torch.tensor(float(X2))) ** 2)))) / len(x2))
physika_print(energy_time)
physika_print(energy_freq)
physika_print(compute_grad(energy_time, x2))
physika_print(compute_grad(energy_freq, x2))