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
μ = torch.as_tensor(torch.tensor([2.0], device=DEVICE)).requires_grad_(True).to(DEVICE)
grad_μ = compute_grad(f, μ)
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
Ω = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device=DEVICE)
sum_Ω = 0
for i in range(len(Ω)):
    sum_Ω = sum_Ω + Ω[int(i)]
print(sum_Ω)
a = torch.tensor([[1, 2, 3], [4, 5, 6]], device=DEVICE)
a_T = torch.transpose(a, 0, 1)
print(print(a_T))
p = torch.tensor([1, 2, 3], device=DEVICE)
q = torch.tensor([4, 5, 6], device=DEVICE)
sum_p = torch.sum(p)
print(print(sum_p))
dot_p_q = torch.sum(torch.stack([(p[int(i)] * q[int(i)]) for _fi_i in range(int(3)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]))
print(print(dot_p_q))