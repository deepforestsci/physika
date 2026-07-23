from typing import Tuple
from physika.core.expr import Expr


class Constructor:
    """
    Constructor of an inductive type.

    ``Constructor`` is needed as one of introduction rules
    for defining an inductive type that is being declared in CIC.

    This class contains the rules to build a value of a given type
    ``type``.

    For example, in Physika we define the inductive rule to get a natural
    number as a concatenation of successors. First, we give the constructor
    of ``Nat`` inductive types the name of ``Nat.succ``. Then, its type is
    logicaly goes from one ``Nat`` type to the next closer one (``Nat → Nat``)
    ,which means ``∀ n: Nat → Nat``. In CIC, we represent ``∀ n: Nat → Nat``
    as as ``Expr`` object ``ForallE("n", nat, nat)``.

successor

    Parameters
    ----------
    name : str
        Name for a Constructor.
    type : Expr
        The constructor's type. In general, a ``ForallE`` chain, one
        binder per field.


    Examples
    --------
    >>> from physika.core.inductive import Constructor
    >>> from physika.core.expr import ForallE, Const
    >>> nat = Const("Nat", ())
    >>> succ_type = ForallE("n", nat, nat)   # Nat -> Nat
    >>> succ = Constructor("Nat.succ", succ_type)
    >>> succ.name
    'Nat.succ'
    >>> succ.type is succ_type
    True
    """

    def __init__(self, name: str, type: Expr):
        """
        Initializer method for creating a ``Constructor`` object
        for defining a value of an inductive type.
        """
        self.name: str = name
        self.type: Expr = type


class RecursorRule:
    """
    ι-reduction rule for a recursor.

    ``RecursorRule`` stores reduction rules for a recursor when applied to an
    inductive's constructor. Inductive types are defined by constructors and a
    recursors. Each recursor, have a group of recursor rules (``RecursorRule``)
    and each constructor have one recursor rule.

    ``Nat`` inductive type have two constructors (``Nat.zero`` and
    ``Nat.succ``) and there is one recursor rule (``RecursorRule``) for each
    expression.


    Parameters
    ----------
    ctor_name : str
        Constructor name of which this recursor rule will be applied.
    nfields : int
        Number of constructor field arguments.
    rhs : Expr
        ``Expr`` guided by ``BVar`` references.
        Represents a reduced result once the recursor is
        applied to a value built from ``ctor_name``.

    Examples
    --------
    >>> from physika.core.inductive import RecursorRule
    >>> from physika.core.expr import BVar
    >>> zero_rule = RecursorRule("Nat.zero", nfields=0, rhs=BVar(0))
    >>> zero_rule.ctor_name
    'Nat.zero'
    >>> zero_rule.nfields
    0
    """

    def __init__(self, ctor_name: str, nfields: int, rhs: Expr):
        self.ctor_name = ctor_name
        self.nfields = nfields
        self.rhs = rhs


class Recursor:
    """
    ``Recursor`` is an object that contains information for an inductive
    type, such as it recursor rules, number of parameters, motives, minor
    premises, etc. Physika's ``Recursor`` class does not run anything by
    itself, but ``reduction.py`` reads it's information to know which rule
    to follow.

    For example, ``Nat`` has two constructors: ``zero`` and ``succ``. So,
    ``Recursor`` for ``Nat`` needs two rules; one for what happens at ``zero``
    , and one for what happens at ``succ``. During elaboration, when there is
    ``Nat`` type, ``Recursor`` contins the information and rules (as CIC
    expressions) and compute with them. This is how addition (``Nat.add``)
    is defined in Physika. Because a proof in CIC is just another term
    (the Curry-Howard correspondence), proving a theorem using a natural number
    uses ``Recursor`` too.

    Parameters
    ----------
    name : str
        Name of the recursor (e.g. "Nat.rec").
    type : Expr
        Expr Pi-type of the recursor.
    num_params : int
        Number of parameters shared with the parent inductive type.
        (``α`` in ``Vec α n``)
    num_indices : int
        Number of indices that can vary between different applications of
        the inductive type.
        (``n`` in ``Vec α n``)
    num_motives : int
        Number of motives (elimination predicates). This is usually 1 for
        simple inductive types but may be higher for mutual inductive
        declarations (not yet supported)
    num_minors : int
        Number of minor premises, one minor premise
        per constructor.
    rules : Tuple[RecursorRule, ...]
        A collection of ``RecursorRule`` objects (computation rules)
        used during ι-reduction. There is one rule for each
        constructor
    level_params : Tuple[str, ...]
        Universe polymorphism parameters for the recursor.

    Examples
    --------
    >>> from physika.core.inductive import Recursor, RecursorRule
    >>> from physika.core.expr import BVar, Sort
    >>> from physika.core.level import LSucc, LZero
    >>> zero_rule = RecursorRule("Nat.zero", nfields=0, rhs=BVar(0))
    >>> succ_rule = RecursorRule("Nat.succ", nfields=1, rhs=BVar(1))
    >>> nat_rec = Recursor(
    ...     name="Nat.rec",
    ...     type=Sort(LSucc(LZero())),
    ...     num_params=0,
    ...     num_indices=0,
    ...     num_motives=1,
    ...     num_minors=2,
    ...     rules=(zero_rule, succ_rule),
    ... )
    >>> nat_rec.name
    'Nat.rec'
    >>> nat_rec.level_params
    ()
    >>> nat_rec.arity()
    4
    """

    def __init__(
            self,
            name: str,
            type: Expr,
            num_params: int,
            num_indices: int,
            num_motives: int,
            num_minors: int,
            rules: Tuple[RecursorRule, ...],
            level_params: Tuple[str, ...] = (),
    ):
        """
        Init method for ``Recursor``
        """

        self.name = name
        self.type = type
        self.num_params = num_params
        self.num_indices = num_indices
        self.num_motives = num_motives
        self.num_minors = num_minors
        self.rules = rules
        self.level_params = level_params

    def arity(self) -> int:
        """
        Total number of arguments for a ``Recursor``.

        Counts ``num_params``, ``num_motives``, ``num_minors``, and
         ``num_indices``

        Returns
        -------
        int
            Number of total parameters
        """
        # + 1 is the major premise (eliminated at iota reduction)
        return self.num_params + self.num_motives + self.num_minors + self.num_indices + 1  # noqa: E501


class InductiveDecl:
    """
    Declaration of one inductive type.

    ``InductiveDecl`` is what the kernel and elaborator look up for a
    given inductive type (``Nat``, ``Vec``, etc.). For example, ``Nat``'s
    declaration records that its name is ``"Nat"``, that it has two
    constructors (``Nat.zero`` and ``Nat.succ``), that it shares no
    parameters, and that it is recursive (since ``Nat.succ`` mentions
    ``Nat`` itself).

    Parameters
    ----------
    name : str
        Its name (e.g. ``"Nat"``).
    level_params : Tuple[str, ...]
        Universe polymorphism parameters for the type itself; empty
        means the type lives at one fixed level.
    num_params : int
        How many pieces stay the same across every constructor
        (``α`` in ``Vec α n``).
    type : Expr
        The type's own kind — its ``Sort``. ``Nat``, for example, is
        simply a ``Type₀``.
    constructors : Tuple[Constructor, ...]
        Its constructors, in declaration order.
    is_recursive : bool
        True if any constructor mentions the type itself, the way
        ``Nat.succ`` mentions ``Nat``.

    Examples
    --------
    >>> from physika.core.inductive import InductiveDecl, Constructor
    >>> from physika.core.expr import ForallE, Const, TYPE_0
    >>> nat = Const("Nat", ())
    >>> zero = Constructor("Nat.zero", nat)
    >>> succ = Constructor("Nat.succ", ForallE("n", nat, nat))
    >>> nat_decl = InductiveDecl(
    ...     name="Nat",
    ...     level_params=(),
    ...     num_params=0,
    ...     type=TYPE_0,
    ...     constructors=(zero, succ),
    ...     is_recursive=True,
    ... )
    >>> nat_decl.name
    'Nat'
    >>> [c.name for c in nat_decl.constructors]
    ['Nat.zero', 'Nat.succ']
    >>> nat_decl.is_recursive
    True
    """

    def __init__(self, name: str, level_params: Tuple[str, ...],
                 num_params: int, type: Expr,
                 constructors: Tuple[Constructor, ...], is_recursive: bool):

        self.name = name
        self.level_params = level_params
        self.num_params = num_params
        self.type = type
        self.constructors = constructors
        self.is_recursive = is_recursive
