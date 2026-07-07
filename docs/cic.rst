Calculus of Inductive Constructions (CIC)
==========================================

Physika implements a dependently typed kernel following
Calculus of Inductive Constructions that allows writing theorem proofs and dimensional analysis.
Physika CIC support ``Terms``, universes, and dependent function types
(:math:`\Pi`-types), but full inductive-type recursion,
universe cumulativity, and some of the reduction/unification rules for
levels described below are under development. Details of implementation and description of CIC concepts and Physika's kernel will be covered in this section.

First, we describe universe levels, the foundation for building terms, types, :math:`\Pi`-types.
For a typed annotated system, we define a type for each each term. For example, for a variable
``x`` of type ``ℕ``, we write ``x : ℕ``, then giving a value to ``x`` can be written as:

.. code-block:: text

   x: ℕ = 5


Now, what is the type of ``ℕ``? Lets say it is ``ℕ: Type``.
Then, what is the type of ``Type``?

If we say:

.. code-block:: text

   Type: Type
   
Then, we can construct a term of type ``Type`` that should have no terms at all,
which is a contradiction. This is known as Russell's paradox in
type-theoretic form (Girard's paradox). 

To avoid this, CIC introduces an "infinite" hierarchy of universes instead of one ``Type``.

.. code-block:: text

   Prop : Type 0 : Type 1 : Type 2 : ...

``Sort``, or universe, is the general name for a member of this hierarchy. ``Sort 0`` is
``Prop`` (propositions and proofs), ``Sort 1`` is ``Type 0`` (ordinary data
types), ``Sort 2`` is ``Type 1``, and so on.

``Prop`` is special because is the base of the hierarchy. It is impredicative since values are quantifable over every type, including
``Prop`` itself, and the result still lives in ``Prop``:

.. code-block:: text

   ∀ (P : Prop), P ∨ ¬P  # this statement is itself Prop type

This reads as *"for every proposition P, P or not-P holds"*.
``P`` ranges over all propositions, ``Prop`` included. The statement type is ``Prop``.

``Level`` type
--------------

Lean 4's kernel defines universe levels as a inductive type at ``src/Lean/Level.lean`` [Lean4]_:

.. code-block:: text

   inductive Level where
     | zero   : Level
     | succ   : Level → Level
     | max    : Level → Level → Level
     | imax   : Level → Level → Level
     | param  : Name → Level
     | mvar   : LMVarId → Level

Physika implements the same six constructors as six dataclasses, unified under one
``typing.Union`` alias:

.. code-block:: python

   Level = Union[LZero, LSucc, LMax, LIMax, LParam, LMVar]

``Level`` enable type hints throughout Physika's CIC
kernel (``pred: "Level"`` on ``LSucc``, ``l1: "Level"`` / ``l2: "Level"`` on
``LMax`` and ``LIMax``) to reference one of these six node kinds.

Physika implementation
----------------------

``LZero``
~~~~~~~~~

``LZero`` is the base universe level which represents level ``0``. ``LZero`` does not contain data
because there is nothing to store. ``Sort(LZero())`` (``Prop``) is the
universe of propositions and proofs. Every universe level is built up from ``LZero`` being the base of the hierarchy.

``LSucc``
~~~~~~~~~

``Sort(LSucc(LZero()))`` is ``Type 0``, the first predicative universe above
``Prop``. A chain of ``n`` nested ``LSucc`` wrapping a single ``LZero``
represents the concrete level ``n``. So level 1 is ``Type 0``, level 2 is
``Type 1``, and so on.

In CIC, universe's own type is one level higher than itself, written ``Sort u : Sort
(u + 1)``.

``LMax``
~~~~~~~~

CIC hierarchy is cumulative:

.. code-block:: text

    ``Prop``, ``Type 0``, ``Type 1``, ...
     
This means a higher level can stand in front of a lower one.
``max(l1, l2)`` gets the operand needed so the smaller operand
fits inside the bigger.

In Lean 4's kernel, ``max`` is used to compute the level of a product (non artihmetic) type [Lean4]_:

.. code-block:: text

   Prod : Type u → Type v → Type (max u v)

For example, if ``α : Type 2`` and ``β : Type 5``, ``Prod α β`` cannot live in ``Type 2``
(``β`` would not fit inside it). Formation rule picks
``max(2, 5) = 5`` that is big enough for the larger ``Sort``, and since ``5`` is also
``>= 2``, big enough for the smaller one.

``LIMax``
~~~~~~~~~

Unlike ``LMax``, ``LIMmax`` is used when the second operand (codomain)
is ``Prop``:

.. code-block:: text

   Sort(imax(level(domain), Prop))


This is what lets a :math:`\Pi`-type collapse into ``Prop`` regardless of how
big its domain is, for example when returning a ``Prop`` in a theorem.
Because ``imax(l, 0) = 0`` regardless of ``l``, a proposition such as
``∀ x : ℕ, x = x`` stays in ``Prop`` (``imax(1, 0) = 0``). Without this rule, quantifying
over any nontrivial domain would force the whole statement into a large
``Type``, and ``Prop``'s impredicativity would collapse.


``LParam``
~~~~~~~~~~

Lets a Physika function or theorem be generic over its universe instead
of pinned to one concrete ``Type n``. The parameter names live on the defining constant
and will be resolved by substitution at the call site.

In Lean 4, the following polymorphic identity function works for any level [Lean4]_:

.. code-block:: text

   def id.{u} {α : Sort u} (a : α) : α := a

is a single definition that works whether it is applied to an ordinary value
(level 1) or to a type itself (level 2).


``LMVar``
~~~~~~~~~

``LMVar`` represents a universe metavariable, used when `elaborator` doesnt yet know the specific level of a term.
A metavariable placeholder will be created ``(LMVar(id='m1'))`` and later
unified with a concrete level (e.g., ``LSucc(LZero())``) during elaboration.


References
----------

.. [Lean4] de Moura, L. and Ullrich, S. The Lean 4 theorem prover
   and programming language. In *Automated Deduction – CADE 28: 28th International Conference on Automated
   Deduction, Virtual Event, July 12–15, 2021, Proceedings*, pp. 625–635, Berlin, Heidelberg, 2021. Springer-Verlag.
   ISBN 978-3-030-79875-8. doi: `10.1007/978-3-030-79876-5_37 <https://doi.org/10.1007/978-3-030-79876-5_37>`_.