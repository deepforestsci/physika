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

   example.phyk → Lexer → Parser → AST → Type Checker(CIC and HM) → Runtime (PyTorch)

Physika's type checker combines Calculus of Inductive Constructions (CIC)
with Hindley-Milner (HM) type inference algorithm. CIC implementation follows Lean 4's approach of
extending the Calculus of Constructions (a dependently typed λ-calculus) with
inductive types, which elaborates each construct into a CIC term, and checks that term
against a small trusted kernel. This is what lets Physika verify dependently
typed programs, in which a value's type may depend on another value (i.e. using an array's dimension to compute).

Physika CIC is still under development. Constructs that are not supported yet, such as user-defined
inductive types, macro expansion, and quotient types, fall back to HM type
checker. These two paths also differ in code generation. CIC-elaborated terms are
compiled to differentiable PyTorch code through a step we call "torch lowering",
whereas the HM path lowers directly from the AST.

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
   cic
   stdlib