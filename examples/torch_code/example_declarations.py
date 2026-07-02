import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Program ===
r_unicode = 3.14
r_mathbb = 2.5
r_macro = 1.5
r_ascii = 0.5
physika_print(r_unicode)
physika_print(r_mathbb)
physika_print(r_macro)
physika_print(r_ascii)
r_vector = torch.tensor([1.0, 2.0, 3.0])
r_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
physika_print(r_vector)
physika_print(r_matrix)
z_unicode = int(10)
z_mathbb = int((-4))
z_macro = int(7)
z_ascii = int(42)
physika_print(z_unicode)
physika_print(z_mathbb)
physika_print(z_macro)
physika_print(z_ascii)
z_vector = torch.tensor([1, 2, 3])
physika_print(z_vector)
n_unicode = 5
n_mathbb = 8
n_macro = 3
n_ascii = 1
physika_print(n_unicode)
physika_print(n_mathbb)
physika_print(n_macro)
physika_print(n_ascii)
n_vector = torch.tensor([0, 1, 2, 3])
physika_print(n_vector)
c_unicode = torch.tensor((3 + 1j), dtype=torch.complex64)
c_mathbb = torch.tensor((5 + 3j), dtype=torch.complex64)
physika_print(c_unicode)
physika_print(c_mathbb)
c_vector = torch.stack([torch.as_tensor((1 + 2j), dtype=torch.complex64), torch.as_tensor((3 + 4j), dtype=torch.complex64)])
physika_print(c_vector)