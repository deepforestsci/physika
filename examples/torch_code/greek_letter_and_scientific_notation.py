import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
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
print(results)
z = (x + y)
print(z)
greek_letters_array = torch.stack([torch.as_tensor(α), torch.as_tensor(β)])
print(greek_letters_array)
μ = torch.as_tensor(torch.stack([torch.as_tensor(2.0)])).requires_grad_(True).to(DEVICE)
grad_μ = compute_grad(lambda _dμ: f(_dμ), μ)
print(grad_μ)
ℏ = 1.0546e-34
σ = 5.6704e-08
ψ = 0.5
threshold = 1.5
if α < threshold:
    result_if = (α * β)
else:
    result_if = (α + β)
print(result_if)
Ω = torch.stack([torch.as_tensor(0.1), torch.as_tensor(0.2), torch.as_tensor(0.3), torch.as_tensor(0.4), torch.as_tensor(0.5)])
sum_Ω = 0
for i in range(len(Ω)):
    sum_Ω = sum_Ω + Ω[int(i)]
print(sum_Ω)