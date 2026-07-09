from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class LZero:
    """
    Base universe level, Level 0

    Constructor of Physika's CIC level.

    ``Sort(LZero())`` is ``Prop`` which represents universe of propositions.
    This is the base level of the hierarchy of universes in Physika's CIC
    (``Prop : Type 0 : Type 1 : ...``).
    `LZero` does not contain fields as there is nothing to store.
    Every other ``Level`` is built up from ``LZero`` via ``LSucc``, ``LMax``,
    and ``LIMax``.

    Examples
    --------
    >>> from physika.core.level import LZero
    >>> LZero()
    LZero()
    >>> LZero() == LZero()
    True
    """


@dataclass(frozen=True)
class LSucc:
    """
    Successor level ``pred + 1``. Step out of ``Prop`` into the predicative
    hierarchy.

    ``Sort(LSucc(LZero()))`` is ``Type 0``, the first predicative universe
    above ``Prop``. A Pi-type over ``Type n`` lives in ``Type (n+1)`` or
    higher.
    Chains of ``LSucc`` over ``LZero`` encode finite levels. Level ``n`` is
    ``n`` nested ``LSucc`` wrapping a single ``LZero``. So level ``1`` is
    ``Type 0``, level ``2`` is ``Type 1``, and so on.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc
    >>> type_0 = LSucc(LZero())   # Type 0
    >>> type_1 = LSucc(type_0)    # Type 1
    >>> type_1
    LSucc(pred=LSucc(pred=LZero()))
    """
    pred: "Level"


@dataclass(frozen=True)
class LMax:
    """
    Predicative maximum: ``max(l1, l2)``

    CIC hierarchy is cumulative:

    ``Prop``, ``Type 0``, ``Type 1``, ...

    meaning a higher level can stand in front of a lower one.
    ``max(l1, l2)`` gets the operand needed so the smaller operand
    fits inside the bigger.

    In Lean 4's kernel, ``max`` is used to compute the level of a product
    (non artihmetic) type:
      Prod : Type u → Type v → Type (max u v)

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc, mk_level_max
    >>> mk_level_max(LSucc(LZero()), LZero())
    LSucc(pred=LZero())
    >>> mk_level_max(LSucc(LZero()), LSucc(LSucc(LZero())))
    LMax(l1=LSucc(pred=LZero()), l2=LSucc(pred=LSucc(pred=LZero())))
    """
    l1: "Level"
    l2: "Level"


@dataclass(frozen=True)
class LIMax:
    """
    Impredicative maximum: ``imax(l1, l2)``

    Unlike ``max``, ``imax`` is used when the second operand (codomain)
    is ``Prop``:

      imax(l,  0)  = 0
      imax(0,  l)  = l
      imax(l,  l)  = l
      imax(l1, l2) = LIMax(l1, l2) -> irreducible

    This is what lets a Pi-type collapse into ``Prop`` regardless of how
    big its domain is, for example when returning a ``Prop`` in a theorem.
    Every Pi-type lives in ``Sort(imax(level(domain), level(codomain)))``:

        ∀ x : Nat, x = x   : Prop      -> imax(1, 0) = 0
        Nat → Nat          : Type 0    -> imax(1, 1) = 1

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc, LIMax
    >>> LIMax(LSucc(LZero()), LZero())
    LIMax(l1=LSucc(pred=LZero()), l2=LZero())
    """
    l1: "Level"
    l2: "Level"


@dataclass(frozen=True)
class LParam:
    """
    Universe-polymorphism variable.

    Lets a Physika function or theorem be generic over its universe instead
    of pinned to one concrete ``Type n``.
    The parameter names live on the defining constant
    and will be resolved by substitution at the call site. A
    ``LParam(name)`` is swapped for a concrete ``Level``.


    Examples
    --------
    >>> from physika.core.level import LParam
    >>> LParam("u")
    LParam(name='u')
    >>> LParam("u") == LParam("u")
    True
    """
    name: str


@dataclass(frozen=True)
class LMVar:
    """
    Universe metavariable.

    Used when `elaborator` doesnt yet know the specific level of a term.
    A metavariable placeholder will be created (LMVar(id='m1')) and later
    unified with a concrete level (e.g.,LSucc(LZero())) during elaboration.

    Examples
    --------
    >>> from physika.core.level import LMVar
    >>> LMVar("m1")
    LMVar(id='m1')
    >>> LMVar("m1") == LMVar("m2")
    False
    """
    id: str


Level = Union[LZero, LSucc, LMax, LIMax, LParam, LMVar]
