import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def f(x):
    return ((x ** 2) + 1)

# === Program ===
α = 1.0
β = 2.0
x = 100000.0
y = 300000.0
results = (α + β)
physika_print(results)
z = (x + y)
physika_print(z)
greek_letters_array = torch.stack([torch.as_tensor(α), torch.as_tensor(β)])
physika_print(greek_letters_array)
μ = torch.as_tensor(torch.tensor([2])).float().requires_grad_(True)
grad_μ = compute_grad(f, μ)
physika_print(grad_μ)
ℏ = 1.0546e-34
σ = 5.6704e-08
ψ = 0.5
threshold = 1.5
if α < threshold:
    result_if = (α * β)
else:
    result_if = (α + β)
physika_print(result_if)
Ω = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
sum_Ω = 0
for i in range(len(Ω)):
    sum_Ω = sum_Ω + Ω[int(i)]
physika_print(sum_Ω)