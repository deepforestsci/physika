Physika
=====================

Physika is a type-based, differentiable programming language for modelling
diverse physical systems and the numerical methods used to solve them.

Programs are written in ``.phyk`` files using notation close to the underlying
mathematics. Every quantity is declared with its type, whether a real number
``ℝ``, an integer ``ℤ``, a complex number ``ℂ``, or an array such as ``ℝ[3]``,
and Physika compiles the program to PyTorch from these declarations.

Programs compile to PyTorch, so every value becomes a ``torch.tensor`` and any
function can be differentiated with ``torch.autograd``. This makes it possible to
fit model parameters to data, optimize a system against an objective, or run
gradient descent through a complete simulation. The type system adds a second
layer of safety: declared shapes are checked for consistency before a program
runs, so dimension errors are reported up front rather than during execution.

A program passes through a fixed pipeline from source to result:

.. code-block:: text

   example.phyk → Lexer → Parser → AST → Type Checker → Runtime (PyTorch)

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   language
   examples
   api
   tutorials/index
   elf
