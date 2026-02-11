# Physika

A differentiable programming language.

## Workflow

```
example.phyk → Lexer (PLY) → Parser → AST → Type Checker → Runtime Execution (PyTorch)
```

1. The **lexer** tokenizes a given `.phyk` file.
2. **parser** applies grammar rules to produce an Abstract Syntax Tree (AST) containing variables, functions, classes, for-loops, and expressions.
3. The **type checker** validates tensor shapes and type correctness across the AST.
4. The **runtime** interprets the AST datastructure and executes the program, using PyTorch as backend.

## Usage

```bash
python -m execute examples/example_arrays.phyk
```

To inspect the generated PyTorch code:

```bash
python -m execute --print-code examples/example_arrays.phyk
```

## Physika Program Description

Physika uses Unicode math symbols for type annotations. Below is an example of arrays and array operations:

```
x : \mathbb{R}[6] = [1, 2, 3, 5, 6, 7]
y : ℝ[3] = x[0:2] + x[0:2]
z : \R[3] = y + [1, 3, 4]

x
y
z
```

Output:
```
[1.0, 2.0, 3.0, 5.0, 6.0, 7.0] ∈ ℝ[6]
[2.0, 4.0, 6.0] ∈ ℝ[3]
[3.0, 7.0, 10.0] ∈ ℝ[3]
```

The associated AST datastructure:
```
Functions:

Classes:

Program:
  (
    'decl',
    'x',
    (
      'tensor',
      [
        (6, 'invariant'),
      ],
    ),
    (
      'array',
      [
        ('num', 1.0),
        ('num', 2.0),
        ('num', 3.0),
        ('num', 5.0),
        ('num', 6.0),
        ('num', 7.0),
      ],
    ),
    1,
  )
  (
    'decl',
    'y',
    (
      'tensor',
      [
        (3, 'invariant'),
      ],
    ),
    (
      'add',
      (
        'slice',
        'x',
        ('num', 0.0),
        ('num', 2.0),
      ),
      (
        'slice',
        'x',
        ('num', 0.0),
        ('num', 2.0),
      ),
    ),
    2,
  )
  (
    'decl',
    'z',
    (
      'tensor',
      [
        (3, 'invariant'),
      ],
    ),
    (
      'add',
      ('var', 'y'),
      (
        'array',
        [
          ('num', 1.0),
          ('num', 3.0),
          ('num', 4.0),
        ],
      ),
    ),
    3,
  )
  (
    'expr',
    ('var', 'x'),
    0,
  )
  (
    'expr',
    ('var', 'y'),
    0,
  )
  (
    'expr',
    ('var', 'z'),
    0,
  )
```

Finally, Pytorch version for the given Physika program:
```
import torch
import torch.nn as nn
import torch.optim as optim

from runtime import physika_print

# === Program ===
x = torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
y = (x[int(0.0):int(2.0)+1] + x[int(0.0):int(2.0)+1])
z = (y + torch.tensor([1.0, 3.0, 4.0]))
physika_print(x)
physika_print(y)
physika_print(z)
```