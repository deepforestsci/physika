Physika Language Reference
==========================

Physika programs are stored in ``.phyk`` files. Physika uses Unicode math
symbols for type annotations and compiles to PyTorch via a parser built
with PLY.

Types
-----

Scalar
~~~~~~

.. code-block:: text

   x : ℝ = 3.14

1-D array
~~~~~~~~~

.. code-block:: text

   v : ℝ[6] = [1, 2, 3, 5, 6, 7]

2-D array (matrix)
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   A : ℝ[3, 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


Declarations and Expressions
-----------------------------

Variables are declared with a type annotation:

.. code-block:: text

   x : ℝ = [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
   y : ℝ[3] = x[0:2] + x[0:2]
   z : ℝ[3] = y + [1, 3, 4]

Printing a bare variable name outputs its value at runtime:

.. code-block:: text

   x
   y
   z

Output::

   [1.0, 2.0, 3.0, 5.0, 6.0, 7.0] ∈ ℝ[6]
   [2.0, 4.0, 6.0] ∈ ℝ[3]
   [3.0, 7.0, 10.0] ∈ ℝ[3]

Functions
---------

.. code-block:: text

   def f(x : ℝ): ℝ:
       return x * x

   f(3)

Output::

   9.0 ∈ ℝ


Control Flow Operators
----------------------

Conditionals
~~~~~~~~~~~~

.. code-block:: text

   x : ℝ = 0.3
   if x > 0.5:
      y = 3 * (x - 0.75)**2
   else:
      y = x**2 + 2

   y

Output::

   2.09 ∈ ℝ


Output::

   2.0 ∈ ℝ


Gradients
---------

.. code-block:: text

   def f(x: ℝ): ℝ:
    if x > 0.0:
        return x * x
    else:
        return - x

.. code-block:: text

   # positive bracnh
   a : ℝ = 3
   f(a)
   grad(f(a), a)

Output::

   9.0 ∈ ℝ
   6.0 ∈ ℝ

.. code-block:: text

   # negative branch
   b : ℝ = - 2
   f(b)
   grad(f(b), b)

Output::

   2.0 ∈ ℝ
   -1.0 ∈ ℝ


``grad`` calls ``compute_grad`` from the runtime, which differentiates ``f``
with respect to its argument using ``torch.autograd.grad``.

Type Checker
------------

The type checker validates tensor shapes before execution:

- Matrix multiplication must satisfy ``(M×N) @ (N×P)``.
- Adding ``ℝ[2] + ℝ`` is a type error.

Errors include the source line number where the mismatch was detected.
