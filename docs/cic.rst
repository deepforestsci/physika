Calculus of Inductive Constructions (CIC)
==========================================

Physika implements a dependently typed kernel following
Calculus of Inductive Constructions (CIC) that allows writing theorem proofs and dimensional analysis.
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


Inductive types
---------------

In previous sections, we covered universe types (Sort), which allow the creation of
an "infinite" hierarchy of universe levels to avoid logical inconsistencies, and dependent
arrow types (ForallE, Lam, App) as CIC expressions (Expr) that represent both universal
quantification in logic (for theorem proofs) and function types in programming (dependent types).
In this section, we introduce inductive types, the third and final component of the type theory
Physika's CIC implements. These three components provide a framework to build Physika's type system,
dependent types, theorem proofs, and dimensional analysis.

An inductive type is defined by a list of constructors, which provide "introduction rules"
for building new objects of that type. An inductive type consists of the objects that can 
be created through these constructors.

Natural numbers (``Nat``) serves as an example of a recursive inductive type. Lean 4 defines ``Nat`` as:

.. code-block:: text

   inductive Nat where
      | zero : Nat
      | succ : Nat → Nat

In this example, we can see ``Nat`` is built from two constructors: ``zero`` and ``succ``. ``zero``
takes no arguments and provides a base starting element. ``succ`` constructor is applied to a previously
constructed ``Nat`` (predecessor), for building the next number. ``Nat`` is the "smallest" type containing
these constructors, meaning it is generated by starting at zero and repeatedly applying succ.

Every inductive type have an elimination rule known as a recursor (for ``Nat``, there is a ``Nat.rec``) which
is derived from inductive declaration. A recursor, as CIC ``Expr`` terms, contains principles of
induction used for both proving theorems about that type and to define recursive functions to compute with.

For instance, based on the two previously defined ``Nat`` constructors, Lean 4 derives ``Nat.rec``
a telescope:

.. code-block:: text

 ``motive → (one minor premise per constructor) → major → conclusion``

Lean 4 generates the following recursor rule:

.. code-block:: text

   Nat.rec.{u} {motive : Nat → Sort u}
  (zero : motive zero)
  (succ : (n : Nat) → motive n → motive n.succ)
  (t : Nat) :
  motive t

``Nat.rec`` lets us compute a ``motive t`` of type ``Nat`` by giving two constructors. First, constructor
``zero`` give us ``motive zero``, that is the base case. Second, ``succ``, for which given any ``n`` and
a value of ``motive n``, ``motive n.succ`` is computed by induction.

Physika generates a similar recursor from an ``InductiveDecl`` via ``derive_recursor``
using  ``FVar``\ s and ``LocalContext.mk_forall``. These recursors contains ι-reduction rules which are verified
when the inductive type is added to ``Environment``.

Inductive types allows computation through ι(iota)-reduction.  When a recursor is applied to a value
of its inductive type, the kernel identifies a major premise (input of the function) and applies
logic defined in minor premises (specify how to compute each constructor).

Before an inductive type is added to Physika's environment (``Environment``), an strict positivity check is applied.
Strict positivity checks that an inductive type is constructed correctly before it is elaborated or the kernel sees it.
A constructor is strictly positive for the inductive type ``T`` if ``T`` appears in positive positions in each field type.
First, inductive type ``T`` must not appear in ``Expr``. Second, an inductive type must not be inside the
domain of an arrow type or function application. This imples anywhere inside
the domain, no matter how deeply nested arrows are within it. In other
words, the domain of a constructor's field cannot contain a self referenced
inductive type, because this can lead to logical inconsistencies. Finally, since
a constructor is a chain of fields (one ``ForallE`` arrow per field), this
domain check is applied separately to each field along that chain.

Environment
-----------
An environment (``Environment``) maps names, including inductive types, to declarations. A global environment keeps track of axioms,
definitions, and inductive types used in a Physika program. ``Environment`` is initialized
before elaboration and type checking. During this step, constant declarations and inductive types are added, and checked for positivity, to be allowed to be used for elaborator
and kernel. Once each CIC term verified and added, ``Environment`` serves as a lookup table for constants and inductive types.

Local context
-------------
When working with dependent types, a binder's type can reference a
previously bound variable. For example:

.. code-block:: text

   def matvec(A : ℝ[m,n], v : ℝ[n]) : ℝ[m]

Here ``v``'s type references ``n``, which comes from ``A``'s type. Terms are
stored with de Bruijn indices. Bound variables are represented as ``BVar`` by position
rather than using a name. Indices can produce errors when working with nested binders.
So, inspired by Lean 4, Physika during elaboration steps under a binder,
``BVar`` are promoted to a new ``FVar`` and sotred in a
``LocalContext``. Elaboration then proceeds with ``FVar``\ s, and
once the binder's body is done being processed, the ``FVar``\ s created for
it are abstracted back into ``BVar``\ s for storage.

Metavariables
-------------
Metavariables (``MVar``) are placeholders for a term whose value isn't known yet. During elaboration,
Physika CIC creates a ``MVar`` when an implicit argument is present, such as the dimension
variable ``n`` in ``ℝ[n]``. Its type (``Nat``) is known immediately, but its value is
determined at unification. Each ``MVar`` is declared within the ``LocalContext`` it was created in,
which guides what it can be solved to.

At unification step, a candidate term is assigned to each unsolved ``MVar`` and checked for a valid assignment
before recording it. A ``MVar`` is assigned a term if it fulfills the following criteria. First, the
candidate must be closed (no loose ``BVar``\ s). Second, it must not reference the ``MVar``
being solved. Then, ``FVar``\'s must already exist in that ``MVar``'s own creation ``LocalContext``. Finally,
the candidate must not reference a metavariable from a deeper scope.

Once every ``MVar`` in a term has been solved, the kernel accepts it. An unresolved ``MVar`` reaching
the kernel is rejected and an error is raised. Since the kernel never has
to infer a ``MVar`` value, it can independently verify the elaborator's output.

Universe levels have their own, parallel metavariable system (``LMVar``), solved the same way but
tracked separately from term-level ``MVar``\ s.

Reduction
---------

While a Physika program is being elaborated to a CIC term, it may still contain
unresolved ``MVar`` placeholders. Even once every placeholder is
solved, two terms can still denote the same object while being written
differently. For example, an inferred return type may need unfolding
before it can be checked against a function's declared one.  Lean 4 implements
``Lean.Meta.whnf`` and ``Lean.Meta.isDefEq`` for solvinf this problem. ``whnf``,
reduces a term to weak-head normal form, and ``is_def_eq``,
decides whether two terms are definitionally equal.

These two functions are used at four points in Physika CIC. During elaboration,
to solve metavariables while a term is being constructed. At
kernel check (``infer_type``), as the final trusted
verification step and at dimensional analysis, to compare and annotate
inferred array/vector shapes.

``whnf`` unfolds only the outermost, head constructor of a term, leaving a
``Lam``'s body, a ``ForallE``'s codomain, and an ``App``'s argument
untouched. ``whnf`` applies the following rules depending the head:

.. code-block:: text

   β (beta)   App(Lam(x, A, b), a)
   δ (delta)  Const("f")
   ζ (zeta)   LetE(x, A, v, b)
   ι (iota)   Recursor(..., ctor args, ...) # rule.rhs[fields, motive, minors]
   Proj       Proj(S, i, S.mk(args))

``whnf`` supports ``Nat.add``/``Nat.mul``/``Nat.sub`` applied to two ``Lit`` operands,
computing the result directly with native arithmetic instead of unfolding
through unary ``Nat.succ`` recursion. This fast path was implemented followin Lean 4's
fast arbitrary-precision arithmetic library.

``is_def_eq(t1, t2, ..., allow_assign)`` instantiates any already
solved ``MVar``\ s on both sides, then checks for plain syntactic equality
before going for reduction. If this fails, both sides are
reduced to ``whnf``. When ``allow_assign=True`` and one side have unassigned ``MVar`` (``?m x1 ... xk``) 
a metavariable applied only to distinct local variables (the Miller pattern)
it's solved directly:

.. code-block:: text

   ?m := λx1 ... xk. t.
   
Miller pattern unification has one possible solution, so it's safe to solve.
Because ``x1 ... xk`` are distinct local variables rather than arbitrary
terms, each one can only correspond to itself inside ``t``. This differs from metavariable's
arguments, which can repeat or be compound terms and different solutions can fit.
If unsolved, a bare unassigned, metavariable on either side is assigned to the other, subject to the same
validity checks described under Metavariables above. A ``Lit`` Nat is paired with an equivalent ``Nat.succ`` chain.
Finally, reduced heads are compared structurally. ``Sort`` levels through
``is_level_def_eq``, ``Const`` by name and level arguments, ``FVar``/``BVar``
by id, ``App`` by parameters on both sides, ``Lam``/``ForallE``
by comparing domains and then bodies opened under a shared new ``FVar``,
and ``Proj`` by ``type_name``/``idx`` plus the inner expression. If exactly
one side is a ``Lam``, the other is applied to a fresh
``FVar`` ( η-expanded) before their bodies are compared. Anything left mismatched is
unequal.

Kernel
------

Following Lean 4's reference, once a Physika program is parsed, then is transformed into
CIC terms, also know as elaboration step. During this step, universe levels, inductive types
with recursors and elimination rules, bound variables (``BVar``), placeholder variables (``MVar``), function types
(``ForallE``), and so on are created. Elaboration is untrusted, since it may produce incorrect ``MVar`` assignments
or unsound inferred types. Physika's trusted kernel re-infers the
program's types and checks them against the declared ones in the elaborated CIC term.
Physika kernel makes CIC terms verifiable with its type inference and type checker
rules, independent from elaboration.

Elaboration
-----------
In CIC, elaboration takes parsed source code, AST parsed from a Physika file, and
converts each AST node into an equivalent CIC expression (``Expr``). Some of these expressions contain
metavariables (``MVar``), placeholders created with a known type but no CIC value yet. Unification
assigns a value to a metavariable. Given two terms that must be equal, it reduces through
the global environment's definitions, the local context's let-bindings, and metavariables already assigned in the
metavariable context, and assigns a value to any unassigned metavariable. If any
metavariable is still unassigned once elaboration finishes, an error is raised. This means the term is incomplete, and the
next step, a small trusted kernel, only accepts fully elaborated terms.

As a brief example, let's consider the following declaration:

.. code-block:: text

   x: ℝ = 1.0 + 2.0

The parser produces an AST node:

.. code-block:: text

   ("body_decl", "x", "ℝ", ("add", ("num", 1.0), ("num", 2.0)))

The goal is to produce a CIC term. That is, an expression built from one of the twelve ``Expr``
constructors introduced earlier. Elaboration recurses into ``add`` node first. Since each operand is a
numeral, each elaborates to a CIC literal, ``Lit(1.0)`` and ``Lit(2.0)``. Both operands
are ``ℝ`` type, so ``add`` elaborates to the registered ``Real.add`` axiom of ``Real`` inductive type, giving the CIC term:

.. code-block:: text

   App(App(Const("Real.add", ()), Lit(1.0)), Lit(2.0))

To simplify this example, no metavariable was needed, since both operands' types were already known.
This term is bound to a local variable ``x``, and the trusted kernel confirms it
have type ``ℝ``.

Elaborating an addition of two real numbers might looks trivial, but the same idea scales. A dependent
Π-type function with implicit parameters elaborates through the same process allowing to compute with dependent type,
and so does a theorem proof. By the Curry–Howard correspondence, a theorem is itself a ``Prop`` type, and its proof is just a
CIC term of that type. So once the proof is elaborated, checking it reduces to ordinary type checking.
Verifying the proof term's type matches the proposition, means proof is complete.

To briefly summarize, elaboration converts an AST into CIC terms for each function, top-level program
statement, and class. Once everything is elaborated, the trusted kernel independently type-checks the whole
program using WHNF and definitional equality as described previously.


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


``Constructor``
~~~~~~~~~~~~~~~

``Constructor`` class is needed to add introduction rules when defining an
inductive type declared in CIC. ``Constructor`` is implemented as a dataclass
that contains rules to build a value of a given ``Type``. For example, in
Physika we define an inductive rule to get a natural
number as a concatenation of successors. First, we give the constructor of
``Nat`` inductive types the name of ``Nat.succ``. Then, its type
goes from one ``Nat`` type to the next closer one (``Nat → Nat``), which means
``∀ n: Nat → Nat``. In CIC, we represent ``∀ n: Nat → Nat`` as an ``Expr`` object
``ForallE("n", nat, nat)``.

``RecursorRule``
~~~~~~~~~~~~~~~~

``RecursorRule`` stores reduction rules for a recursor when applied to an inductive's
constructor. Inductive types are defined by constructors and recursors. Each recursor,
have a group of recursor rules (``RecursorRule``) and each constructor have exactly one recursor rule.
     
``Nat`` inductive type have two constructors (``Nat.zero`` and ``Nat.succ``)
and there is one recursor rule (``RecursorRule``) for each.

``Recursor``
~~~~~~~~~~~~

``Recursor`` is an object that contains recursor information for an inductive
type, such as it recursor rules, number of parameters, motives, minor premises
, etc. Physika's ``Recursor`` class does not run anything by itself, but
``reduction.py`` reads it's information to know which rule to follow.

For example, ``Nat`` has two constructors: ``zero`` and ``succ``. So, ``Recursor``
for ``Nat`` needs two rules; one for what happens at ``Nat.zero``, and one for
what happens at ``Nat.succ``. During elaboration, when there is ``Nat`` type,
``Recursor`` contains the information and rules (as CIC expressions) and compute
with them. Because a proof in CIC is just another term (Curry-Howard correspondence),
proving a theorem using a natural number uses ``Recursor`` too.

``InductiveDecl``
~~~~~~~~~~~~~~~~~

``InductiveDecl`` is what the kernel and elaborator look up for a
given inductive type (``Nat``, ``Vec``, etc.). For example, ``Nat``'s
declaration records its name is ``"Nat"``, that it has two
constructors (``Nat.zero`` and ``Nat.succ``), and that it is recursive (since ``Nat.succ`` mentions
``Nat`` itself).

``mk_builtin_env``
~~~~~~~~~~~~~~~~~~
``Nat`` inductive type is one example of built-in inductive types supported in Lean 4, which are
added by default when running a file ( referenced as "Prelude" see [Mathlib4Prelude]_).

Physika implements a similar approach with ``mk_builtin_env`` , which returns an ``Environment``
pre-populated with built-in inductive types plus reducible inductive operators (e.g. indicates how
to reduce arithmetic operations) and axiom constants. An axiom constant is a ``ConstantInfo`` whose
``value`` is ``None`` but its type is registered in ``Environment``. Hence, the kernel can type check terms
that use these axioms, but it never δ-reduce or compute with them. Physika uses axiom constants for type checking
during elaboration and PyTorch code is generated at run time.

Each inductive declaration is written as an  ``InductiveDecl`` and its recursor is generated by
``derive_recursor``. ``verify_recursor_rules`` checks ι-rules with trusted kernel. 

Current supported inductive types are:

.. list-table::


   * - Inductive Type
     - Level
     - Constructors
   * - ``Nat``
     - ``Type 0``
     - ``Nat.zero``, ``Nat.succ : Nat → Nat``
   * - ``Int``
     - ``Type 0``
     - ``Int.ofNat : Nat → Int``, ``Int.negSucc : Nat → Int``
   * - ``Bool``
     - ``Type 0``
     - ``Bool.false``, ``Bool.true``
   * - ``Fin``
     - ``Nat → Type 0`` (1 index)
     - ``Fin.zero : Fin (succ n)``, ``Fin.succ : Fin n → Fin (succ n)``
   * - ``Vec``
     - ``Type 0 → Nat → Type 0`` (1 param, 1 index)
     - ``Vec.nil : Vec α 0``, ``Vec.cons : α → Vec α n → Vec α (succ n)``
   * - ``Prod``
     - ``Type 0 → Type 0 → Type 0`` (2 params)
     - ``Prod.mk : α → β → Prod α β``

``ConstantInfo``
~~~~~~~~~~~~~~~~
``ConstantInfo`` stores constant declarations
data such as it's name, universe parameters, type, and
value and is used inside ``Environment``.

``InductiveInfo``
~~~~~~~~~~~~~~~~~
``InductiveInfo`` is used in ``Environment`` to store data
about an inductive type. ``InductiveInfo`` keeps track of an
inductive type's declaration, its constructors and recursor
as ``ConstantInfo``, and its ``Recursor``.
   
``Environment``
~~~~~~~~~~~~~~~
``Environment`` keeps track axioms, definitions, and inductive
types used in a Physika program. `Environment`` is initialized
before elaboration and type checking. Once each CIC term (axioms, theorems
and declarations) is added, ``Environment`` serves as a lookup table.

``LocalDeclVar``
~~~~~~~~~~~~~~~~
Represents a binder where a variable is introduced with a specific type
but no assigned value. ``LocalDeclVar`` is used when opening a binder from a local context.

``LocalDeclDef``
~~~~~~~~~~~~~~~~
Represents a local definition (let binding), which includes both a type
and a value (``x: T = t``). ``LocalDeclDef`` is used when opening definition from a local context.

``LocalContext``
~~~~~~~~~~~~~~~~
``LocalContext`` handles two types of declarations:

- ``LocalDeclVar``: A variable introduced by a binder, with a type but no value.

- ``LocalDeclDef``: a let-definition: A variable with both a type and a known value.

``LocalContext`` is immutable: ``push_local`` and ``push_let`` return
a new ``LocalContext`` rather than mutating the existing one. ``push_local`` opens
a binder. Registers a ``FVar`` for a name and type variable (used for ``Lam``/``ForallE``).
``push_let`` does the same for a let-value, additionally recording the bound value (used for ``LetE``).

``mk_lambda`` and ``mk_forall``  abstract one or more ``FVar``\ s back into ``BVar``\ s and wrap the result
in ``Lam``/``ForallE`` expressions, closing a binder once its body has been elaborated.

``LMVarId``
~~~~~~~~~~~

Identifier for a universe-level metavariable. ``LMVar`` is a placeholder node inside a ``Level``
expression (e.g. ``LSucc(LMVar("u.0"))``). Represents the same relationship ``MVarId`` has to ``MVar``. It is used
at ``MetaVarContext.level_assignments``, a dict recording which ``Level`` each level metavariable has been solved to.


``MetaVarKind``
~~~~~~~~~~~~~~~

``MetaVarKind`` indicates how a ``MVar`` might be solved. ``NATURAL`` (implicit args, dimension variables)
is solved freely by the general unifier for regular programs (e.g. dependent types). ``SYNTHETIC``/``SYNTHETIC_OPAQUE``
refers to placeholders for theorem proofs and tactics.


``MetaVarDecl``
~~~~~~~~~~~~~~~

Declaration for one metavariable. Metavariables are declared during elaboration as placeholders for terms
not yet known.

   
``MetaVarContextState``
~~~~~~~~~~~~~~~~~~~~~~~

Saved state of a MetaVarContext for unification. Used when the elaborator tries to unify two terms that might
fail and need to restore ``MetaVarContext``.

``MetaVarContext``
~~~~~~~~~~~~~~~~~~


Mutable context class that tracks all metavariables and their solutions. ``MetaVarContext`` works within a
``LocalContext``. When ``.mk_mvar`` creates an ``MVar``, ``MetaVarContext`` stores the caller's current ``LocalContext`` inside
that ``MVar``'s ``MetaVarDecl`` (``decl.lctx``). Later, at unification, a candidate solution term is assingned for that
``MVar``, ``.is_valid_assignment`` checks every ``FVar`` in the candidate against this stored ``LocalContext``.
A solution may only reference variables that were already in scope when the metavariable was created, which
stops a solution leaking a local variable that would not make sense outside of where the placeholder came from.

Within Physika's CIC, ``MetaVarContext`` is one of the three pieces of state ( including
``Environment`` and ``LocalContext``) that are mutable during elaboration. Expression metavars are stored in 
``expr_assignments`` (``MVarId  → Expr``) and universe level metavars in ``level_assignments`` (``LMVarId → Level``)
The kernel, which never sees a ``MetaVarContext``, only ever accepts terms where every placeholder has already been
resolved.

``Weak-head Normal Form (whnf)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduces a CIC ``expr: Expr`` to weak-head normal form (WHNF). ``whnf`` applies
β, δ, ζ, ι, and Proj reduction rules for checking types at elaboration
time.

Apply reduction rules to an `expr:Expr` until the head is irreducible
(``Sort``, ``Lam``, ``ForallE``, ``Const``, ``FVar``, ``MVar``, or ``BVar``
). First, by  unfolding a let-binding or a let-bound local (ζ(zeta)-
reduction). Then, reducing a defined constant (δ(delta)-reduction). Third
, reducing a lambda application (β(beta)-reduction) and firing a recursor's
ι-rule against a constructor (iota reduction).

``Definitional Equality (is_def_eq)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Checks if two CIC terms ``t1: Expr`` and ``t2: Expr`` are definitionally equal, solving
expression metavariables as needed. ``is_def_eq`` returns
``(True, mctx)`` if equal, ``(False, mctx)`` otherwise.

Instantiates solved MVars, and compare ``t1`` and ``t2`` node by node
checking if they are literally equal, and reduces both sides to WHNF.
If either side is a metavariable applied only to distinct local variables
(Miller pattern), solve it directly. Otherwise falls through ``MVar``
assignment, checking if a ``Nat`` literal resolves to a ``Nat.succ`` chain,
and structural comparison (``App``, binder-opening on ``Lam``/``ForallE``,
one side is a ``Lam`` and the other isn't).

Returns ``mctx: MetaVarContext``, mutated in place if any MVar got assigned
during the call.

``Kernel infer_type``
~~~~~~~~~~~~~~~~~~~~~
One typing rule per ``Expr`` constructor is defined:
* ``Sort u : Sort(u+1)``.
* ``FVar``/``Const`` looked up in ``lctx``/``env``.
* ``App`` checking the argument against the function's.
``ForallE`` domain via ``is_def_eq`` before substituting.
* ``Lam``/``ForallE`` opening their body/codomain under a fresh ``FVar``.
* ``Proj`` delegating to ``proj_field_type``.

Infer typer calls ``whnf``/``is_def_eq`` from ``reduction.py`` with ``allow_assign=False``
so the kernel never assign values for metavariables.

``Kernel check``
~~~~~~~~~~~~~~~~~~~~~
Infers an expression type (``expr: Expr``) and checks if it is equal in defintion to ``expected``, raising
``KernelException`` otherwise. It is the analog of Lean 4's ``Environment.addDecl``, the call an
elaborator makes to verify a declaration's body has its declared type before trusting it.

``Elab``
~~~~~~~~
In Physika, elaboration is driven by ``Elab``, which carries the global environment, local context, and
metavariable context. During elaboration (``.elaborate()``), functions and classes are registered first.
Class signatures are registered as single-constructor inductive types. Function signatures, class
method signatures, and class constructors are all elaborated into ``ForallE`` types. ``ForallE`` terms allows
explicit parameters, implicit dimension parameters, and return type. Once every Π-type is registered, function
bodies are elaborated, then class method bodies, and finally top-level program statements, including
function calls.

At each elaboration step, ``.infer_type`` and ``.unify()`` (both ``Elab`` methods) check terms. For example,
unifying an inferred argument type against a Pi-type's declared binder type. The
final check is not done by ``Elab``, but independently by trusted ``kernel.check``and
``kernel.infer_type``.

Currently in Physika, each AST node tag has its "hardcoded" elaboration rule, one per
statement tag and one per expression tag (``TAG_HANDLERS``/ ``EXPR_TAG_HANDLERS``). This is the reason
why the current implementation is very extense. Every new syntax form requires writing and 
registering a new handler. We are working towards a more robust design following Lean 4's approach. 
This is adding support for macro expansion during elaboration. Core elaboration rules would still live in main Physika,
while users could define their own elaboration rules alongside macros.

References
----------

.. [Lean4] de Moura, L. and Ullrich, S. The Lean 4 theorem prover
   and programming language. In *Automated Deduction – CADE 28: 28th International Conference on Automated
   Deduction, Virtual Event, July 12–15, 2021, Proceedings*, pp. 625–635, Berlin, Heidelberg, 2021. Springer-Verlag.
   ISBN 978-3-030-79875-8. doi: `10.1007/978-3-030-79876-5_37 <https://doi.org/10.1007/978-3-030-79876-5_37>`_.

.. [Mathlib4Prelude] ``Init.Prelude``. Mathlib4 documentation.
   `<https://leanprover-community.github.io/mathlib4_docs/Init/Prelude.html>`_.