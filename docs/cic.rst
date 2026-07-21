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
which is a contradiction. This is an analogue of Russell's
paradox (based on set theory) related to type theory. Russell's paradox  defines
:math:`R = \{x : x \notin x\}`, the set of all sets that do not contain
themselves. Asking if :math:`R \in R` gives a contradiction. Girard's paradox is the same self-reference problem applied to
type theory. If ``Type : Type``, a type is allowed to quantify over a
collection that includes itself. Girard showed this lets you encode
Russell's construction as a term, deriving a proof of ``False`` in a
truthy calculus.

To avoid this contradiction, CIC introduces an "infinite" hierarchy of universes instead of one ``Type``.

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

Expressions
-----------

In CIC, every object (a program, a data type, a proof, or a type itself) is an expression, also referred as a term.
There is no syntactic distinction between terms and types since they are all components of a common language.
While terms share the same syntax, expressions are categorized by what they represent. An expression is considered
a type if it infers to a ``Sort`` (like ``Prop`` or ``Type_0``), and it is considered a value if it infers to something else (e.g. ``Nat.zero: Nat``).


``Expr``, expressions type, is a central sort used to construct elements that will be manipulated by the kernel and elaborator.
The complete syntax for these expressions includes the following constructors:

* Variables: ``BVar`` (bound variable, de Bruijn index) and ``FVar`` (free variable, identified by ``FVarId``).
* Metavariables: ``MVar``, an unresolved term identified by ``MVarId``, resolved during unification step.
* Constants: ``Const``, a reference to a declaration in the environment, with ``levels`` instantiating its universe parameters.
* Sorts: ``Sort``, wrapping a universe ``Level`` (e.g. ``Sort(LZero())`` is ``Prop``)
* Binders: ``Lam`` (function abstraction), ``ForallE`` (dependent function/Pi type), and ``LetE`` (local definition).
* Operations: ``App`` for single argument function application. Multi-argument calls curry as nested ``App`` and ``Proj`` for structure field projection by index.
* Literals: ``Lit``, wrapping either a ``NatLit`` or ``FloatLit`` value.
* Metadata: ``MData``, attaches extra info to an expression without changing its meaning.

These twelve constructors are unified under one alias:

.. code-block:: python

   Expr = Union[BVar, FVar, MVar, Sort, Const, App, Lam, ForallE, LetE, Lit,
                MData, Proj]

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

``BVar``
~~~~~~~~

Bound variable (``BVar``) are considered placeholders whose values are constrained or "bound" by an operator.
In this case the operators are known as binders (``Lam``
or ``ForallE``). ``BVar``\ s are referenced by position rather than by name (a de Bruijn
index). ``BVar(0)`` refers to the innermost enclosing binder and ``BVar(k)`` skips ``k`` binders outward. This is why
terms like ``fun x => x`` and ``fun y => y`` are
identical CIC terms since no renaming is ever needed to compare
them.

``FVar``
~~~~~~~~

Free variable refers to a local declaration currently in scope. Opposite to ``BVar``, ``FVar`` are identified by
a unique ``FVarId`` rather than by position. Once a binder is opened
(e.g. to inspect a function body during type checking), its ``BVar(0)``
is replaced by a new ``FVar``. Elaboration (type inference,
unification) operates on ``FVar``\ s, not raw de Bruijn indices.

``FVarId``
~~~~~~~~~~

As CIC uses de Bruijn indices to represent bound variables
, the elaborator needs a way to refer to a specific variable in the local context.

``FVarId`` serves this purpose, allowing the elaborator to look
up the type of the variable in the local context. ``FVarId`` usage
is particularly useful when dealing with multiple variable calls and
nested calls.

``MVar``
~~~~~~~~

Represents a metavariable, which is an unknown to be solved by unification.
Created whenever the elaborator needs more information to resolve a type, most commonly for an ``IMPLICIT`` binder's argument (see
``BinderInfo`` below).

``MVarId``
~~~~~~~~~~

Identifier for ``MVar``, similar to of what ``FVarId`` does for
``FVar``. During elaboration a fresh ``MVar`` is emitted for an unresolved type.

``Const``
~~~~~~~~~

Reference to a global declaration. A function defintion (``def``), an inductive type, or a
constructor registered in the ``Environment`` under ``name``. ``Const`` contains
``levels`` fields that instantiates the constant's universe parameters. Empty for
a non-polymorphic constant (``Const("Real", ())``), or one concrete
``Level`` per universe parameter for a polymorphic one such as a
recursor (e.g. ``Const("Nat.rec", (LZero(),))``).

``Sort``
~~~~~~~~

Wraps a universe ``Level``: ``Sort(LZero())`` is ``Prop``,
``Sort(LSucc(LZero()))`` is ``Type 0``, and so on. Every type in CIC
lives in some ``Sort`` — a term is a *type* exactly when its own
inferred type is a ``Sort``; otherwise it's a value.

``BinderInfo``
~~~~~~~~~~~~~~

Tags how binder's argument is supplied at a call site:
``DEFAULT`` (must be given explicitly) or ``IMPLICIT`` (inferred by the
elaborator via unification, which creates a ``MVar``).
Physika mirrors Lean 4's ``BinderInfo`` [Lean4]_ but only implements
these two of its four cases. ``strictImplicit`` and ``instImplicit``
are not implemented.

``Lam``
~~~~~~~

A lambda abstraction can be seen a a basic function definition (``fun (binder_name : binder_type) => body``). This comes from functional programming and λ-calculus.
The type of a ``Lam`` expression must be a ``ForallE``, since it infers to a ``Pi`` type.

``ForallE``
~~~~~~~~~~~

Dependent function type (:math:`\Pi`-type), ``(binder_name :
binder_type) → body``. ``binder_type`` is the domain; ``body`` is the
codomain (output type) and depend on bound variables. When a (:math:`\Pi`-type)
doesn't depend on bound variable, this is just a plain non-dependent arrow type ``Sort(imax(level(binder_type), level(body)))``.

``LetE``
~~~~~~~~

``LetE`` represents local definition binding for a term. ``LetE`` lets users
assign a name and a type to a value. Then, we can reference that name
in the "body" of an expression.

``App``
~~~~~~~

Function application, one argument at a time (curried). CIC has no
native multi-argument application, so ``f(a, b, c)`` is represented as
``App(App(App(f, a), b), c)``.

``Proj``
~~~~~~~~

Structure field projection. ``Proj`` is an expression constructor used to access the fields of an
inductive type registered with just one constructor. ``Proj`` allows
Physika kernel to access and extract a specific field by its index
from constructor arguments, avoiding the overhead of full inductive
elimination.

``Lit``
~~~~~~~

A literal value, currently either a ``Nat`` or a ``Real``, tagged via
``NatLit``/``FloatLit`` (``Literal = Union[NatLit, FloatLit]``).

``NatLit``
~~~~~~~~~~

While CIC represent numbers as inductive types (using zero and successor
constructors and recursors), Lean 4 uses ``NatLit`` to improve performance and Physika applies the same. NatLit allows
Physika CIC to manage large numbers without massive memory overhead
required to unfold nested constructors (like
``Nat.succ (Nat.succ ... Nat.zero)``).

``FloatLit``
~~~~~~~~~~~~

Floating point literal with ``Real`` type (Physika extension from Lean 4).

Lean 4 represents floats via ``Float.mk``. Physika instead treats floats
as first class term literals (``Real``) same way ``NatLit`` tags a ``Nat``
value.

``MData``
~~~~~~~~~

Metadata wrapper that ttaches key-value information (``kvs``) to an
expression without changing its meaning. Physika kernel treats
``MData(kvs, e)`` as identical to ``e``. Exists so the elaborator can
contain extra information alongside a term for error handling.


References
----------

.. [Lean4] de Moura, L. and Ullrich, S. The Lean 4 theorem prover
   and programming language. In *Automated Deduction – CADE 28: 28th International Conference on Automated
   Deduction, Virtual Event, July 12–15, 2021, Proceedings*, pp. 625–635, Berlin, Heidelberg, 2021. Springer-Verlag.
   ISBN 978-3-030-79875-8. doi: `10.1007/978-3-030-79876-5_37 <https://doi.org/10.1007/978-3-030-79876-5_37>`_.