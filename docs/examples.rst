Examples
========

All examples files are under ``examples/`` directory. Run any of them with:

.. code-block:: bash

   physika examples/<name>.phyk

Add ``--print-ast`` or ``--print-code`` to inspect the generated AST or
PyTorch source.

----

Data Types
----------

Physika supports two core numeric types ℝ for Real number and ℤ for integer
both can be used to declare scalar, arrays, tensors, etc.

Numeric: ℝ and ℤ
~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/example_numeric_types.phyk
   :language: text

Arrays
~~~~~~

.. literalinclude:: ../examples/example_arrays.phyk
   :language: text

Matrices
~~~~~~~~

.. literalinclude:: ../examples/example_matrices.phyk
   :language: text

Tensors (contravariant / covariant)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/example_tensors.phyk
   :language: text

----

Control Flow Operators & Differentiation
----------------------------------------

Conditionals
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/if_else_contexts.phyk
   :language: text

Differentiable conditional
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/diff_ifelse.phyk
   :language: text

Differentiable sin / cos
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/diff_sincos.phyk
   :language: text

Threshold optimisation (gradient descent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/diff_threshold.phyk
   :language: text

For loops and for-expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/for.phyk
   :language: text

Differentiable for loops
~~~~~~~~~~~~~~~~~~~~~~~~~

All loop forms are differentiable. ``grad()`` returns a gradient vector for
scalar-output functions and a full Jacobian matrix for vector-output functions.

.. literalinclude:: ../examples/diff_for.phyk
   :language: text

Factorial (recursive)
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/factorial.phyk
   :language: text

----

Classes
-------


Neural network classes
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/networks.phyk
   :language: text

Training a fully connected network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/train_fully_connected.phyk
   :language: text

----

Physics Simulations
-------------------

.. note::
   The following three examples open a GUI window (matplotlib / PyVista).

Harmonic oscillator
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/harmonic_oscillator.phyk
   :language: text

Simple pendulum (RK4)
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/pendulum_rk4.phyk
   :language: text

Spring pendulum (RK4)
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/spring_pendulum_rk4.phyk
   :language: text

Hamiltonian Neural Network (HNN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/hnn.phyk
   :language: text

Symbolic support
----------------

Physika supports symbolic math via sympy, letting you declare ``Symbol``, ``Function``,
and ``Equation`` types and use built-ins like ``diff``, ``subs``, ``lambdify`` and 
``symbolic_solve`` to derive and solve expressions analytically.

.. note::

   Physika uses ``SymPy`` for symbolic computation.

Core concepts
~~~~~~~~~~~~~

- ``Symbol`` — symbolic variables used in expressions
- ``Function`` — symbolic functions of one or more variables
- ``Equation`` — mathematical relationships between expressions

Operations
~~~~~~~~~~
- ``diff(expr, var)`` — compute derivative w.r.t. a variable
- ``subs(expr, old, new)`` — substitute part of an expression 
- ``lambdify(expr, vars)`` — convert to a numerical function
- ``symbolic_solve(eq, var)`` — solve equations analytically

.. literalinclude:: ../examples/example_symbolic.phyk
   :language: text

Greek Letters and Scientifc Notation
------------------------------------

Physika treats Greek letters as valid symbols and variables, allowing you to write
physics and mathematics in a natural, notation-friendly style. Variables like
``α``, ``β``, ``μ``, ``σ``, ``λ`` and all standard Greek letters are supported.

Scientific notation is also supported natively — values like ``1e5``, ``2.5e-3``,
or ``6.674e-11`` are valid numeric literals.

.. note::

   ``Δ`` is reserved for the Laplacian operator and cannot be used as an symbol/variable.

.. literalinclude:: ../examples/greek_letter_and_scientific_notation.phyk
   :language: text

Z₂ modulo type-2
----------------

Physika supports arithmetic in Z₂ (integers modulo 2), including XOR and AND operations,
making it useful for working with binary logic.

.. literalinclude:: ../examples/integer_modulo_two.phyk
   :language: text
