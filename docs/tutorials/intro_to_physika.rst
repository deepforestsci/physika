Introduction to Physika
=======================

This page is an introduction to Physika. It covers what Physika is, how to
install and run it, a tour of the core language, and one complete program. By
the end you will be able to read any of the other tutorials on this site.

What is Physika?
----------------

Physika is a type-based, differentiable, and probabilistic language for physical and numerical
computation. You write programs in ``.phyk`` files using math-like notation,
and declare every value with its mathematical type: a real ``ℝ``, an integer
``ℤ``, a complex number ``ℂ``, or an array such as ``ℝ[3]``.

Physika compiles to PyTorch. Each value becomes a ``torch.tensor``, so you can
differentiate a function with the ``grad`` operation, where ``grad(expr, x)``
returns the derivative of ``expr`` with respect to an input ``x``, computed by
PyTorch's autograd. Before a program runs, the type checker verifies that every
operation uses compatible types and dimensions. A shape mismatch, such as adding
an ``ℝ[3]`` to an ``ℝ[5]``, is caught before any computation runs. Values can
also be sampled from probability distributions, and you can take gradients with
respect to a distribution's parameters.



How Physika works
-----------------

A ``.phyk`` file passes through a fixed pipeline on its way to a result:

.. code-block:: text

   example.phyk → lexer + parser → AST → type checker → PyTorch code → run

The lexer and parser read the source and build an abstract syntax tree (AST).
The type checker walks that tree and verifies the types and dimensions of every
operation. If it finds a mismatch, compilation stops and the error is reported.
Otherwise the code generator turns the AST into a PyTorch program, which is then
executed. You can inspect the generated PyTorch with the
``--print-code`` flag, which we use in the next section.

Installation
------------

Physika needs Python 3.9 or newer. If you have not set it up yet, follow the
:doc:`/install` guide, which covers the requirements, installing from source,
and running a program.

Your first program
------------------

With Physika installed, let's write and run a small program from start to
finish. We'll start with something simple: the kinetic energy ``½mv²`` of a 3 kg
mass at three different speeds. Create a file ``first.phyk``:

.. code-block:: text

   # kinetic energy of a 3 kg mass at speeds 1, 2 and 3 m/s
   m : ℝ = 3.0
   v : ℝ[3] = [1.0, 2.0, 3.0]
   E : ℝ[3] = 0.5 * m * v * v
   E

Run it:

.. code-block:: bash

   physika first.phyk

You should see::

   ✓ No type errors found
   [1.5, 6.0, 13.5] ∈ ℝ[3]

To print a value, write its name on a line by itself. Physika prints the value
followed by its type. Here the arithmetic ran over the whole array at once, and
the result ``E`` is an ``ℝ[3]``, exactly as declared.

Now look at what Physika actually ran. The ``--print-code`` flag prints the
generated PyTorch instead of executing it:

.. code-block:: bash

   physika first.phyk --print-code

Output::

   === Physika generated Pytorch code ===
   import torch
   import torch.nn as nn
   import torch.optim as optim

   from physika.runtime import physika_print

   # === Program ===
   m = 3.0
   v = torch.tensor([1.0, 2.0, 3.0])
   E = (((0.5 * m) * v) * v)
   physika_print(E)
   === End Pytorch code ===

This is the heart of Physika. A ``.phyk`` file is not interpreted; it is
translated into an ordinary PyTorch program and then run. The array ``v`` became
a ``torch.tensor``, which is why the multiplication applied to every element,
and ``physika_print`` is the runtime helper that prints a value with its type.
Everything you write compiles to readable PyTorch like this, which is what makes
Physika programs differentiable and easy to inspect.

A tour of the language
-----------------------

The rest of this page is a short tour of the core language. Each snippet below
is a complete program you can run on its own, shown with its real output. For
the full detail of any feature, follow the link to the Language Reference at the
end of the section.

1. Types and printing
~~~~~~~~~~~~~~~~~~~~~~~

Every value is declared with a type annotation. The scalar types are the real
``ℝ``, the integer ``ℤ``, and the complex ``ℂ``, and an array also carries its
length, written ``ℝ[3]``:

.. code-block:: text

   a : ℝ = 3.0
   z : ℂ = 2 + 3j
   v : ℝ[3] = [1.0, 2.0, 3.0]
   a
   z
   v

Output::

   3.0 ∈ ℝ
   (2+3j) ∈ ℂ
   [1.0, 2.0, 3.0] ∈ ℝ[3]

For the matrix type (``ℝ[3, 3]``), indexing, and slicing, see :doc:`/language`.

2. Type checking
~~~~~~~~~~~~~~~~

Physika checks the dimensions of every operation before running. Adding two
arrays of different lengths is not a valid operation, so the type checker stops
and reports it:

.. code-block:: text

   a : ℝ[3] = [1, 2, 3]
   b : ℝ[5] = [1, 2, 3, 4, 5]
   a + b

Output::

   Type errors found:
   ✗ Line 0: Shape mismatch in add: ℝ[3] vs ℝ[5]
   1 type error(s) found.

The program never runs. The error is caught at compile time and the offending
operation is named. The same checking covers indexing, slicing, matrix shapes,
and function calls. See the Type Checker section of :doc:`/language`.

3. Functions
~~~~~~~~~~~~

A function declares the type of each parameter and the type it returns. This one
takes a real number and returns its square:

.. code-block:: text

   def square(x : ℝ): ℝ:
       return x * x

   square(5)

Output::

   25 ∈ ℝ

Calling ``square(5)`` runs the compiled function and prints the value it
returns.

4. Conditionals
~~~~~~~~~~~~~~~

A conditional chooses between two branches based on a comparison. Here we set a
flag depending on whether a temperature is above freezing:

.. code-block:: text

   T : ℝ = 285.0
   if T > 273.15:
       state = 1.0
   else:
       state = 0.0
   state

Output::

   1.0 ∈ ℝ

Like the rest of the language, a conditional is differentiable: you can take the
gradient of a result computed through an ``if`` branch. The Differentiable
conditional example in :doc:`/examples` shows this.


5. Loops
~~~~~~~~

Physika has four loop forms. The most compact is the **for-expression**, which
builds an array:

.. code-block:: text

   squares : ℝ[5] = for i : ℕ(5) → i * i
   squares

Output::

   [0.0, 1.0, 4.0, 9.0, 16.0] ∈ ℝ[5]

An **implicit-range loop** ranges over an array you index in the body:

.. code-block:: text

   arr : ℝ[4] = [10, 20, 30, 40]
   total : ℝ = 0
   for i:
       total += arr[i]
   total

Output::

   100 ∈ ℝ

An **explicit-range loop** counts over ``ℕ(start, end)``:

.. code-block:: text

   fact : ℝ = 1
   for k: ℕ(1, 5):
       fact = fact * k
   fact

Output::

   24 ∈ ℝ

A **multi-index loop** accumulates over several indices at once, such as a
matrix multiply written inside a function:

.. code-block:: text

   def matmul(A : ℝ[n, m], B : ℝ[m, o]): ℝ[n, o]:
       C : ℝ[n, o]
       for i j k:
           C[i, j] += A[i, k] * B[k, j]
       return C

   A : ℝ[2, 2] = [[1, 2], [3, 4]]
   B : ℝ[2, 2] = [[1, 0], [0, 1]]
   matmul(A, B)

Output::

   [[1, 2], [3, 4]] ∈ ℝ[2,2]

All four are differentiable. See the Differentiable For Loops section of
:doc:`/language`.

.. note::

   Inside a loop body, use plain assignments (``x = …``). Typed declarations
   (``x : ℝ = …``) and bare-name printing only work at the top level or in a
   function body.

6. Gradients
~~~~~~~~~~~~

``grad`` differentiates a function with respect to one of its inputs. As a
simple example, take the potential energy stored in a spring, ``V(x) = ½kx²``.
Its derivative ``V'(x) = kx`` is the restoring force that pulls the spring back
toward equilibrium, and we can have Physika compute it for us:


.. code-block:: text

   # potential energy of a spring with stiffness k = 4 N/m
   def V(x : ℝ): ℝ:
       return 0.5 * 4.0 * x * x

   x0 : ℝ = 0.5
   V(x0)
   grad(V(x0), x0)

Output::

   0.5 ∈ ℝ
   2.0 ∈ ℝ

The second value is ``V'(0.5)`` with ``k = 4``, which is ``2.0``. We did not
write the derivative anywhere; ``grad`` produced it from the function itself. It
compiles to ``compute_grad`` in the Physika runtime, which calls
``torch.autograd.grad``. Every loop and conditional above is differentiable in
the same way. See the Differentiable For Loops section of :doc:`/language`.

7. Random sampling
~~~~~~~~~~~~~~~~~~~

Physika is probabilistic as well as differentiable. A value can be drawn from a
probability distribution with ``~``, and because the draw stays on the autograd
graph, you can take gradients through it:

.. code-block:: text

   physika.seed(0)
   μ : ℝ = 0.0
   x : ℝ ~ Normal(μ, 1.0)
   grad(x, μ)

Output::

   1.0 ∈ ℝ

Here ``x`` is an ordinary ``ℝ`` sampled from a normal distribution, and ``grad(x, μ)``
differentiates through the draw to give ``1.0``, so you can fit a distribution's 
parameters by gradient descent. ``Normal``, ``Uniform``, ``Beta``, ``Gamma``, and ``Bernoulli`` are 
supported; continuous distributions differentiate through the draw, and discrete 
ones like Bernoulli use a score-function estimator. See the Random sampling section of :doc:`/elf`.

A complete program
------------------

To finish, let's put everything together in one program where a function, a
loop, and ``grad`` work as a unit. We'll find the resting position of a 2 kg
mass hanging from a spring of stiffness ``k = 4 N/m`` under gravity. Its
potential energy is ``U(x) = ½kx² − mgx``, and the mass settles where that
energy is lowest. Rather than solve for that position by hand, we let it relax:
at each step it moves a little way along the force ``−U'(x)``, which ``grad``
provides, until it stops moving. The function defines ``U``, the loop runs the
relaxation steps, and ``grad`` supplies the force:

.. code-block:: text

   # a 2 kg mass on a spring (k = 4 N/m) under gravity
   # potential energy U(x) = ½kx² − mgx
   def U(x : ℝ): ℝ:
       return 0.5 * 4.0 * x * x - 2.0 * 9.8 * x

   # let the mass relax: step downhill along the force −U'(x)
   x : ℝ = 0.0
   for i : ℕ(60):
       x = x - 0.05 * grad(U(x), x)
   x

Output::

   ✓ No type errors found
   4.899992942810059 ∈ ℝ

The mass settles at ``x ≈ 4.9 m``, which is exactly the equilibrium ``mg/k``
found by balancing the spring force against gravity.

Run the program with ``--print-code`` to see the PyTorch it compiles to:

.. code-block:: python

   === Physika generated Pytorch code ===
   import torch
   import torch.nn as nn
   import torch.optim as optim

   from physika.runtime import physika_print
   from physika.runtime import compute_grad

   # === Functions ===
   def U(x):
       return ((((0.5 * 4.0) * x) * x) - ((2.0 * 9.8) * x))

   # === Program ===
   x = torch.tensor(0.0, requires_grad=True)
   for i in range(int(0), int(60)):
       x = (x - (0.05 * compute_grad(lambda _dx: U(_dx), x)))
   physika_print(x)
   === End Pytorch code ===


Where to go next
----------------

A couple of larger features were left out of this introduction. Each is covered
with worked examples in :doc:`/examples`:

- **Classes**: group fields and methods into reusable types.
- **Symbolic math**: ``Symbol``, ``diff``, and ``solve`` via SymPy.

From here:

- The :doc:`/tutorials/index` each take one physical system and learn its
  parameters by gradient descent. You can now read any of them.
- :doc:`/language` documents the full language.
- :doc:`/examples` is a feature-by-feature collection of snippets.
