import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def magnitude(z):
    return torch.abs(z if isinstance(z, torch.Tensor) else torch.tensor(float(z)))

# === Program ===
x = torch.tensor((3 + 1j), dtype=torch.complex64)
y = torch.tensor((5 + 3j), dtype=torch.complex64)
physika_print(x)
physika_print(y)
complex_add = torch.tensor((x + y), dtype=torch.complex64)
physika_print(complex_add)
physika_print((x - y))
complex_mul = torch.tensor((x * y), dtype=torch.complex64)
physika_print(complex_mul)
physika_print(magnitude(x))
array_imag = torch.stack([torch.as_tensor(1j, dtype=torch.complex64), torch.as_tensor(2j, dtype=torch.complex64), torch.as_tensor(3j, dtype=torch.complex64)])
array_complex = torch.stack([torch.as_tensor((1 + 9j), dtype=torch.complex64), torch.as_tensor((7 + 2j), dtype=torch.complex64), torch.as_tensor((3 + 5j), dtype=torch.complex64)])
nested_complex = torch.tensor([[(1 + 2j), (3 + 4j)], [(4 + 9j), (7 + 2j)]], dtype=torch.complex64)
physika_print(array_imag)
physika_print(array_complex)
physika_print(nested_complex)