import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print
import sympy as sp

# === Program ===
x = sp.Symbol('x')
y = sp.Symbol('y')
u = sp.Function('u')
physika_print(x)
f = ((x ** 2) + (y ** 2))
physika_print(f)
expr = u(x, y)
physika_print(expr)
physika_print(f.subs([(x, 3.0), (y, 4.0)]))
f = (((x ** 3) + ((2 * x) ** 2)) + x)
physika_print(sp.diff(f, x))
expr = ((x ** 2) + (y ** 2))
f = sp.lambdify([x, y], expr, modules={'exp': 'torch.exp', 'log': 'torch.log', 'sin': 'torch.sin', 'cos': 'torch.cos', 'sqrt': 'torch.sqrt', 'abs': 'torch.abs', 'sum': 'torch.sum', 'mean': 'torch.mean', 'real': 'torch.real'})
physika_print(f(3.0, 4.0))
eq = sp.Eq(((2.0 * x) + 3.0), 7.0)
physika_print(sp.solve(eq, x))