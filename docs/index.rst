Physika
=====================

Physika is a type-based, differentiable, and probabilistic programming language
for modelling physical systems and the numerical methods used to solve them.  It
lets you describe a system in notation close to its mathematics, in ``.phyk``
files, and run it.

Every program compiles to PyTorch, so you can take the gradient of a result with
respect to its inputs through PyTorch's autograd. Values can also be sampled from
probability distributions, and those samples stay differentiable too. Every value
carries a type, and Physika checks the shapes in an operation before the program
runs, so dimension errors surface up front.


A program passes through a fixed pipeline from source to result:

.. code-block:: text

   example.phyk → Lexer → Parser → AST → Type Checker → Runtime (PyTorch)

New to Physika? Start with the :doc:`/tutorials/intro_to_physika` tutorial. It
covers installing the language, the core syntax, and one complete program. See
:doc:`/language` for the full reference and :doc:`/examples` for short worked
snippets.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   language
   examples
   api
   tutorials/index
   elf
   motivation
