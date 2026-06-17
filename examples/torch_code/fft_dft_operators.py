import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def op_J(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    cube = torch.reshape(W, (int(s1), int(s2), int(s3),))
    spec = torch.fft.fftn(cube if isinstance(cube, torch.Tensor) else torch.tensor(float(cube)))
    flat = torch.reshape(spec, (int(n),))
    return (flat / n)

def op_I(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    cube = torch.reshape(W, (int(s1), int(s2), int(s3),))
    field = torch.fft.ifftn(cube if isinstance(cube, torch.Tensor) else torch.tensor(float(cube)))
    flat = torch.reshape(field, (int(n),))
    return (flat * n)

# === Program ===
field = torch.stack([torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((2 + 1j), dtype=torch.complex64), torch.as_tensor((0 + 3j), dtype=torch.complex64), torch.as_tensor(((-1) + 0j), dtype=torch.complex64), torch.as_tensor((2 + 0j), dtype=torch.complex64), torch.as_tensor((1 - 1j), dtype=torch.complex64), torch.as_tensor((0 + 0j), dtype=torch.complex64), torch.as_tensor((3 + 2j), dtype=torch.complex64)])
spectrum = op_J(field, 2, 2, 2)
recovered = op_I(spectrum, 2, 2, 2)
physika_print(field)
physika_print(spectrum)
physika_print(recovered)