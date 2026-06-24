import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
xc = torch.stack([torch.as_tensor((1 + 2j), dtype=torch.complex64), torch.as_tensor((3 + 1j), dtype=torch.complex64), torch.as_tensor((0 + 4j), dtype=torch.complex64), torch.as_tensor((2 + 0j), dtype=torch.complex64)])
Xc = torch.fft.fft(xc)
physika_print(Xc)
xc_rec = torch.fft.ifft(Xc)
physika_print(xc_rec)
x2 = torch.tensor([[2.0, 0.0, 1.0, 3.0], [4.0, 1.0, 0.0, 2.0]])
X2 = torch.fft.fft2(x2)
physika_print(X2)
x2_rec = torch.fft.ifft2(X2)
physika_print(x2_rec)
x3 = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]])
X3 = torch.fft.fftn(x3)
physika_print(X3)
x3_rec = torch.fft.ifftn(X3)
physika_print(x3_rec)
xr = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
Xr = torch.fft.rfft(xr)
physika_print(Xr)
xr_rec = torch.fft.irfft(Xr)
physika_print(xr_rec)
xr2 = torch.tensor([[1.0, 0.0, 2.0, 1.0], [3.0, 1.0, 0.0, 2.0]])
Xr2 = torch.fft.rfft2(xr2)
physika_print(Xr2)
xr2_rec = torch.fft.irfft2(Xr2)
physika_print(xr2_rec)
xr3 = torch.tensor([[[2.0, 1.0, 0.0, 3.0], [1.0, 2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0, 1.0], [3.0, 0.0, 1.0, 2.0]]])
Xr3 = torch.fft.rfftn(xr3)
physika_print(Xr3)
xr3_rec = torch.fft.irfftn(Xr3)
physika_print(xr3_rec)