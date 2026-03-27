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

Differentiable For Loops
------------------------

The four loop forms in Physika are differentiable. ``grad()`` computes a gradient using
Pytorch's autograd.

For-expression
~~~~~~~~~~~~~~~

``for i : ℕ(n) → expr`` constructs an array using ``torch.stack([...])``, which is differentiable:

.. code-block:: text

   def scale_vec(x : ℝ): ℝ[3]:
       return for i : ℕ(3) → x * (i + 1)

   s : ℝ = 2
   scale_vec(s)
   grad(scale_vec(s), s)

Output::

   [2.0, 4.0, 6.0] ∈ ℝ[3]
   [1.0, 2.0, 3.0] ∈ ℝ[3]

The gradient ``[1, 2, 3]`` is the Jacobian ``d(scale_vec)/ds``.

Implicit range for-loop
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   def dot_with_arr(s : ℝ): ℝ:
       a : ℝ[4] = [1, 2, 3, 4]
       result : ℝ = 0
       for i:
           result += s * a[i]
       return result

   s : ℝ = 1
   grad(dot_with_arr(s), s)

Output::

   10.0 ∈ ℝ

Multi-index loop (for i j k:)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-index accumulation loops compile to ``torch.stack`` / ``torch.sum``
and are fully differentiable:

.. code-block:: text

   def matmul_scale(s : ℝ): ℝ:
       A : ℝ[2, 2] = [[1.0, 2.0], [3.0, 4.0]]
       I : ℝ[2, 2] = [[1.0, 0.0], [0.0, 1.0]]
       C : ℝ[2, 2]
       for i j k:
           C[i, j] += s * A[i, k] * I[k, j]
       return sum(C)

   s : ℝ = 1.0
   grad(matmul_scale(s), s)

Output::

   10.0 ∈ ℝ

Jacobian of vector output functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the function returns a vector or tensor, ``grad()`` returns the full
Jacobian matrix instead of a gradient vector:

.. code-block:: text

   # f: ℝ → ℝ[n]
   # grad() returns a vector (df[i]/ds)
   def cos_freqs(x : ℝ): ℝ[4]:
       return for i : ℕ(4) → cos(x * (i + 1.0))

   grad(cos_freqs(x), x)    
   # [-sin(x), -2sin(2x), -3sin(3x), -4sin(4x)]

   # f: ℝ[n] → ℝ[n]
   # calling grad() for f with relation to x returns a matrix (df[i]/dx[j])
   def elementwise_sq(x : ℝ[n]): ℝ[n]:
       return for i → x[i] ** 2

   ev : ℝ[3] = [1.0, 2.0, 3.0]
   grad(elementwise_sq(ev), ev)

Output::

   [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]] ∈ ℝ[3,3]

Type Checker
------------

Physika's type checker runs Hindley-Milner type inference over a given program before
execution and validates scalars (``ℝ``, ``ℕ``, ``ℂ``)
, ``string`` values, arrays and matrices shape compatibility for indexing, slicing, and
element-wise operations. It also checks that function calls and return values
match their declared types. 

Errors are reported with the source line number or the enclosing
function/class name where the mismatch was detected.

Type Representations
~~~~~~~~~~~~~~~~~~~~

Every expression is assigned one of these types:

- ``TScalar`` — A scalar ground type: ``ℝ``, ``ℕ``, ``ℂ``, or ``string``.
- ``TVar`` — An unknown type variable used during unification, (``α0``, ``α1``, etc).
- ``TDim`` — An unknown dimension resolved at unification step (``δ0``, ``δ1``, etc).
- ``TTensor`` — A tensor type ``ℝ[d0, d1, ...]`` whose dimensions are one of:

  - ``int`` — A concrete size from a literal annotation (``ℝ[5]``).
  - ``str`` — A symbolic size from a generic parameter (``ℝ[n]``).
  - ``TDim`` - For an unknown dimension (``ℝ[δ0]``).

- ``TFunc`` — A function type ``(p0, p1, ...) → ret``, where ``pN`` refers to parameters types and ``ret`` refers to the return type.
- ``TInstance`` — the type of a class value (``instance(FullyConnectedNet)``).


``VarCounter`` class
~~~~~~~~~~~~~~~~~~~~

Generates unique placeholder names when running a Physika program which are resolved at unification step.

.. code-block:: text

   VarCounter:
   - new_var() → TVar("α0"), TVar("α1"), etc   (unknown type)
   - new_dim() → TDim("δ0"), TDim("δ1"), etc  (unknown dimension)
   - reset()   → restart from 0, called by run() at session start.

Both ``new_var`` and ``new_dim`` draw from the same counter so
``α2`` and ``δ2`` can never both exist simultaneously.