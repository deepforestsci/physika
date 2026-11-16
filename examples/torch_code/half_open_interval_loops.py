import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def sum_right_open(n):
    right_func_sum = 0.0
    for i in range(int(0), int(n)):
        right_func_sum = right_func_sum + i
    return right_func_sum

def sum_left_open(n):
    left_func_sum = 0.0
    for i in range(int(0) + 1, int(n) + 1):
        left_func_sum = left_func_sum + i
    return left_func_sum

def sum_nested_intervals(n):
    function_nested_sum = 0.0
    for j in range(int(0), int(n)):
        for k in range(int(0) + 1, int(j) + 1):
            function_nested_sum = function_nested_sum + (j + k)
    return function_nested_sum

def dot_with_arr(s):
    a3 = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
    dot_result = 0.0
    for i in range(int(0), int(4)):
        dot_result = dot_result + (s * a3[int(i)])
    return dot_result

# === Classes ===
class IntervalAccumulator(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.as_tensor(value))
        self.learnable_params = list(self.parameters())

    def forward(self, n):
        this = self
        class_interval_sum = 0.0
        for i in range(int(0), int(n)):
            class_interval_sum = class_interval_sum + (i + self.value)
        return class_interval_sum

# === Program ===
sum_right = 0.0
for i in range(int(0), int(4)):
    sum_right = sum_right + i
print(print(sum_right))
sum_left = 0.0
for i in range(int(0) + 1, int(4) + 1):
    sum_left = sum_left + i
print(print(sum_left))
nested_sum = 0.0
for k in range(int(0), int(4)):
    for l in range(int(0) + 1, int(k) + 1):
        nested_sum = nested_sum + (k + l)
print(print(nested_sum))
right_func_result = sum_right_open(4)
left_func_result = sum_left_open(4)
nested_func_result = sum_nested_intervals(4)
print(print(right_func_result))
print(print(left_func_result))
print(print(nested_func_result))
obj = IntervalAccumulator(1.0).to(DEVICE)
class_results = obj(10)
print(print(class_results))
s2 = torch.tensor(1.0, requires_grad=True)
dot_gradient = compute_grad(lambda _ds2: dot_with_arr(_ds2), s2)
print(print(dot_gradient))