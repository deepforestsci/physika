from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Union

from physika.core.level import (
    Level,
    LZero,
    LSucc,
)


class BinderInfo(Enum):
    """
    When a function call expects an argument, the elaborator looks for an
    explicit argument in the call site. However, in CIC, the elaborator can
    also infer some arguments from context at unification step. Physika mirrors
    Lean 4's ``BinderInfo``, but only implements the ``default``/``implicit``
    cases (see https://github.com/leanprover/lean4/blob/8c9756b28d64dab099da31a4c09229a9e6a2ef35/src/Lean/Expr.lean#L46-L80)

    Whether an argument is explicit or implicit, CIC names these as binders,
    since they are bound to the function type. For example, in the function:
        def f (x : α) {y : β} : T := body
    ``x`` is an explicit binder, while ``y`` is an implicit binder. The
    elaborator will look for an explicit argument for ``x``, but will try to
    infer ``y`` from context.

    The elaborator creates a new ``MVar`` for IMPLICIT binders. DEFAULT
    binders must be supplied explicitly.

    ``BinderInfo`` is used in two places. During elaboration
    telling the elaborator whether to
    expect a user-typed argument or to create a fresh ``MVar``. For printing correct
    brackets to show the binder with: ``(x : T)`` for DEFAULT, ``{x : T}`` for
    IMPLICIT.

    Parameters
    ----------
    DEFAULT: int
        Argument must be supplied explicitly at the call site.
    IMPLICIT: int
        When argument is inferred by the elaborator via unification.

    Examples
    --------
    >>> from physika.core.expr import BinderInfo, Lam, Const
    >>> explicit = Lam("x", Const("Real", ()), Const("x", ()), BinderInfo.DEFAULT)
    >>> implicit = Lam("y", Const("Real", ()), Const("y", ()), BinderInfo.IMPLICIT)  # noqa: E501
    >>> explicit.binder_info
    <BinderInfo.DEFAULT: 1>
    >>> implicit.binder_info
    <BinderInfo.IMPLICIT: 2>
    >>> explicit.binder_info == implicit.binder_info
    False
    """
    DEFAULT = auto()  # explicitly
    IMPLICIT = auto()  # inferred by unification


@dataclass(frozen=True)
class FVarId:
    """
    Identifier for a free variable.

    As CIC uses de Bruijn indices to represent bound variables
    (ocurring in lambda or pi expressions), the elaborator needs
    a way to refer to a specific variable in the local context.

    ``FVarId`` serves this purpose, allowing the elaborator to look
    up the type of the variable in the local context. ``FVarId`` usage
    is particularly useful when dealing with multiple variable calls and
    nested calls.

    For example, when elaborating ``fun (x : Nat) => x + 5`` the elaborator
    will:

    1. Encode the body ``x + 5`` as ``App(App(Const("Nat.add", ()),
        BVar(0)), NatLit(5))``. Here ``BVar(0)`` refers to the innermost
        binder, which is ``x``.
    2. Elaborate with local context ``("x" : Nat)`` producing a
        ``FVarId("x.0")`` and records ``x.0 → Nat`` in ``LocalContext``.
    3. Substitute ``BVar(0)`` with ``FVar(FVarId("x.0"))``. Now we can keep
        track of ``x``  rather than just encoding a position relative to
        the binder (``BVar(0)``).
    4. Infer the type of function's body (``Nat.add(FVar(x.0), 5)``), which
        requires knowing FVar(x.0) type. This is done by looking ``x.0``
        keyword in ``LocalContext`` giving ``Nat`` type.
    5. Close *that inferred type* (``Nat``, from step 4 — not the body
        value itself) back over ``x`` by abstracting ``FVar("x.0")`` back
        to ``BVar(0)``, producing the final Pi type
        ``ForallE("x", Const("Nat", ()), Const("Nat", ()))``, i.e.
        ``∀ x : Nat, Nat``.

    Parameters
    ----------
    id : str
        Unique identifier string, e.g. ``"x.0"``.

    Examples
    --------
    >>> from physika.core.expr import FVarId, FVar, Const
    >>> x1 = FVar(FVarId("x.0"))
    >>> x2 = FVar(FVarId("x.1"))
    >>> x1 == x2
    False
    >>> types = {x1.id: Const("Nat", ()), x2.id: Const("Real", ())}
    >>> types[x1.id]
    Const(name='Nat', levels=())
    >>> types[x2.id]
    Const(name='Real', levels=())
    """
    id: str

    def __str__(self) -> str:
        """
        Return the string representation of the FVarId.
        """
        return self.id


@dataclass(frozen=True)
class MVarId:
    """
    Identifier for a metavariable.

    As the elaborator processes a Physika file to produce terms for the kernel,
    it often encounters implicit arguments that
    are not resolved.

    ``MVarId`` allows the elaborator to create a unique metavariable
    (``MVar``) identifier for these unknown terms. While ``FVarId``
    refers to a known variable in the local context, ``MVarId`` allow
    the elaborator to track and refer to a specific "unknown" term during
    the unification and elaboration steps.

    This is useful when one unknown part of a proof depends on another,
    allowing the system to keep track of these unknowns until types are
    inferred and the metavariables can be resolved.

    Parameters
    ----------
    id : str
        Unique identifier string, e.g. ``"m.0"``.

    Examples
    --------
    >>> from physika.core.expr import MVarId
    >>> m0 = MVarId("m.0")
    >>> m1 = MVarId("m.1")
    >>> m0 == m1
    False
    >>> str(m0)
    'm.0'
    """
    id: str

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class NatLit:
    """
    Non-negative integer literal, of ``Nat`` type.

    While CIC represent numbers as inductive types (using zero and successor
    constructors), Lean 4 uses ``NatLit`` to improve performance. NatLit allows
    Physika CIC to manage large numbers without massive memory overhead
    required to unfold nested constructors (like
    ``Nat.succ (Nat.succ ... Nat.zero)``).

    Any expression constructed as a NatLit is automatically inferred to be of
    type Nat, which is also useful for definitional equality.

    Parameters
    ----------
    val : int
        Literal's non-negative integer value.

    Examples
    --------
    >>> from physika.core.expr import NatLit, Lit
    >>> Lit(NatLit(3)).val
    NatLit(val=3)
    """
    val: int


@dataclass(frozen=True)
class FloatLit:
    """
    Floating point literal with ``Real`` type (Physika extension from Lean 4).

    Lean 4 represents floats via ``Float.mk``. Physika instead treats floats
    as first class term literals (``Real``) same way ``NatLit`` tags a ``Nat``
    value.

    Parameters
    ----------
    val : float
        Literal's floating-point value.

    Examples
    --------
    >>> from physika.core.expr import FloatLit, Lit
    >>> Lit(FloatLit(1.5)).val
    FloatLit(val=1.5)
    """
    val: float


Literal = Union[NatLit, FloatLit]


@dataclass(frozen=True)
class BVar:
    """
    Bound variable expression node.

    ``BVar`, referred as bound variable, is an expression to
    represent a variable that is in the scope of a binder, such
    as a lambda, pi, or let expression.

    Bound variables are implemented using Bruijn indices (natural numbers)
    A de Bruijn index represents a bound variable from the bottom binder
    (function argument) to the top binder (outermost function argument).
    So``BVar(0)`` will always refer to innermost binder. For example, in
    the expression ``fun (a b c d e) => bvar(0)``, ``bvar(0)`` refers to
    ``e``, ``bvar(1)`` is ``d`` and so on.


    CIC uses de Bruijn indices to avoid name collision issues. However,
    because indices are relative, their meaning changes as you move
    through an expression. When Physika kernel enters the body of a lambda
    or pi expression, it typically replaces ``BVar`` expressions with
    ``FVar``. When exiting the body, function abstraction, Physika's kernel
    replaces ``FVar`` keywordss with the correct BVar indices

    If there are any ``BVar`` expressions after type inference, is considered
    a type error. This is because CIC substitute all bound variables with free
    variables before inspecting the body of a term. If a BVar is still present
    , it means a binder was not properly opened or the term is ill-formed.

    Parameters
    ----------
    idx : int
        De Bruijn index, how many enclosing binders to skip
        . ``0`` means innermost bound variable.

    Examples
    --------
    >>> from physika.core.expr import BVar, Lam, Const
    >>> identity = Lam("x", Const("Real", ()), BVar(0))
    >>> identity.body
    BVar(idx=0)
    >>> identity.body == BVar(0)
    True
    """
    idx: int


@dataclass(frozen=True)
class FVar:
    """
    Free variable. A local declaration, identified by a
    unique ``FVarId``.

    Once a binder is "opened" (to inspect a function body during
    type checking), its ``BVar(0)`` is replaced by a fresh ``FVar``.
    Elaboration work like type inference and unification uses
    ``FVar``s, not raw de Bruijn indices.

    Parameters
    ----------
    id : FVarId
        Unique identifier of this free variable.

    Examples
    --------
    >>> from physika.core.expr import FVar, FVarId
    >>> x = FVar(FVarId("x.0"))
    >>> x.id
    FVarId(id='x.0')
    >>> str(x.id)
    'x.0'
    """
    id: FVarId


@dataclass(frozen=True)
class MVar:
    """
    Metavariable represents an unknown to be solved by unification,
    identified by a unique ``MVarId``.

    Created whenever the elaborator needs to pause before resolving a type.
    Most commonly for an ``IMPLICIT`` binder's argument (``BinderInfo``),
    which the elaborator fills in via unification rather than requiring
    it explicitly.

    Parameters
    ----------
    id : MVarId
        Unique identifier of this metavariable.

    Examples
    --------
    >>> from physika.core.expr import MVar, MVarId
    >>> hole = MVar(MVarId("m.0"))
    >>> hole.id
    MVarId(id='m.0')
    >>> str(hole.id)
    'm.0'
    """
    id: MVarId


@dataclass(frozen=True)
class Sort:
    """
    A universe that represent the type of types.

    ``Sort(LZero())`` is ``Prop``. ``Sort(LSucc(LZero()))`` is ``Type
    0`` and so on up the hierarchy (``Prop : Type 0 : Type 1 : ...``).
    Every type in CIC lives in some ``Sort``, and the hierarchy is what
    prevents Russell's paradox: ``Type n : Type (n+1)``, never ``Type n
    : Type n``.

    In CIC, a term is a type if and only if is of type Sort. For example,
    ``Nat`` is a type because ``Nat : Sort(LSucc(LZero()))``. Else, these are
    values, like ``3: Nat``.

    Parameters
    ----------
    level : Level
        The universe level. For example, ``LZero()`` for ``Prop``.

    Examples
    --------
    >>> from physika.core.expr import Sort
    >>> from physika.core.level import LZero, LSucc
    >>> Sort(LZero()) == Sort(LZero())
    True
    >>> Sort(LZero()) == Sort(LSucc(LZero()))
    False
    """
    level: Level


@dataclass(frozen=True)
class Const:
    """
    Reference to a global declaration.

    A function defintion ``def``, an inductive type, or
    a constructor registered in the ``Environment`` under ``name``.

    ``levels`` instantiates the constant's universe parameters.

    Parameters
    ----------
    name : str
        Declaration's name in the ``Environment``.

    levels : Tuple[Level, ...]
        Universe level arguments instantiating the constant's universe
        parameters.

    Examples
    --------
    >>> from physika.core.expr import Const
    >>> real = Const("Real", ())
    >>> real.name
    'Real'
    >>> real.levels
    ()
    """
    name: str
    levels: Tuple[Level, ...]


@dataclass(frozen=True)
class App:
    """
    Function application, one argument at a time (currying).
    CIC has no native multi argument application, so ``f(a, b, c)``
    is represented as ``App(App(App(f, a), b), c)``.

    Parameters
    ----------
    func : Expr
        Function being applied.
    arg : Expr
        Single argument it's applied to.

    Examples
    --------
    >>> from physika.core.expr import App, Const
    >>> term = App(App(Const("Real.add", ()), Const("x", ())), Const("y", ()))
    >>> term.func
    App(func=Const(name='Real.add', levels=()), arg=Const(name='x', levels=()))
    >>> term.arg
    Const(name='y', levels=())
    """
    func: "Expr"
    arg: "Expr"


@dataclass(frozen=True)
class Lam:
    """
    Lambda abstraction.

    In a lambda expression (``fun (x : Nat) => x + 5``), a BVar(0) in
    funciton body uses a de Bruijn index to point to a specific innermost binder.
    ``BinderInfo`` associate to ``Lam`` determines how the argument is handled
    at call sites (``DEFAULT`` vs ``IMPLICIT``). DEFAULT binders must be explicitly
    supplied and IMPLICIT binders are inferred by the elaborator.
    While this metadata is ised by the elaborator and allows the pretty printer to
    display specifc arguments, it does not alter core kernel procedures like type
    inference or reduction

    Parameters
    ----------
    binder_name : str
        Display name only.
    binder_type : Expr
        Type of the bound argument.
    body : Expr
        Function body.
    binder_info : BinderInfo
        Whether the argument is ``DEFAULT`` or ``IMPLICIT``.

    Examples
    --------
    >>> from physika.core.expr import Lam, Const, BinderInfo
    >>> identity = Lam("x", Const("Real", ()), Const("x", ()), BinderInfo.DEFAULT)  # noqa: E501
    >>> identity.binder_name, identity.binder_type, identity.body
    ('x', Const(name='Real', levels=()), Const(name='x', levels=()))
    """
    binder_name: str
    binder_type: "Expr"
    body: "Expr"
    binder_info: BinderInfo = BinderInfo.DEFAULT


@dataclass(frozen=True)
class ForallE:
    """
    Dependent function type (Pi type).

    ``ForallE`` defines dependent function type or proposition. Represents
    the type of a function by associating an object or proof of ``P(x)`` with
    every input ``x`` of type ``T`` (``∀x:T. U``). ``Lam`` is a value
    expression whose inferred type is a ``ForallE``, ensuring that every
    program has a correct type signature. ``ForallE`` is a type expression
    because its inference results in a Sort.

    Parameters
    ----------
    binder_name : str
        Display name only.
    binder_type : Expr
        Type of the bound argument (the domain).
    body : Expr
        Codomain.
    binder_info : BinderInfo
        Whether the argument is ``DEFAULT`` or ``IMPLICIT``.

    Examples
    --------
    >>> from physika.core.expr import ForallE, Const
    >>> arrow = ForallE("_", Const("Real", ()), Const("Real", ()))
    >>> arrow.binder_type == arrow.body
    True
    >>> arrow.binder_info
    <BinderInfo.DEFAULT: 1>
    """
    binder_name: str
    binder_type: "Expr"
    body: "Expr"
    binder_info: BinderInfo = BinderInfo.DEFAULT


@dataclass(frozen=True)
class LetE:
    """
    Let expression.

    ``LetE`` represents a local definition of a term. ``LetE`` lets users
    assign a name and a type to a value. Then, we can reference that name
    in the "body" of a expression.

    Parameters
    ----------
    binder_name : str
        Name of the local definition.
    type : Expr
        Type of the local definition.
    value : Expr
        Value of the local definition.
    body : Expr
        Expression that uses the local definition.
    non_dep : bool
        If True, the body does not depend on the value of the local definition.

    Examples
    --------
    >>> from physika.core.expr import LetE, Const, BVar
    >>> term = LetE("x", Const("Real", ()), Const("one", ()), BVar(0))
    >>> term.value
    Const(name='one', levels=())
    >>> term.body
    BVar(idx=0)
    >>> term.non_dep
    False
    """
    binder_name: str
    type: "Expr"
    value: "Expr"
    body: "Expr"
    non_dep: bool = False


@dataclass(frozen=True)
class Lit:
    """
    ``Lit`` serves as a container for basic values (Nat and Float) to improve
    performance.

    Without literals, a large number would be represented as a big
    chain of inductive constructors (multiple of Nat.succ applications),
    which is expensive in terms of memory and computation.
    Physika kernel allows basic arithmetic operation (addition, multiplication,
    division, etc.) and comparisons directly on ``NatLit``/``FloatLit`` values
    during reduction, rather than looking inductive constructors and recursor.

    Parameters
    ----------
    val : Literal
        Literal value, which can be ``NatLit`` or ``FloatLit``.

    Examples
    --------
    >>> from physika.core.expr import Lit, NatLit, FloatLit
    >>> Lit(NatLit(3)).val
    NatLit(val=3)
    >>> Lit(FloatLit(1.5)).val
    FloatLit(val=1.5)
    """
    val: Literal


@dataclass(frozen=True)
class MData:
    """
    Metadata wrapper.

    Expression constructor used to attach auxiliary information to a term

    ``MData`` serves as a wrapper that allows Physika to give details with
    an expression without changing the expression's meaning.
    This metadata is typically used by the elaborator, pretty printer,
    to provide hints for error messages.

    Parameters
    ----------
    kvs : Tuple[Tuple[str, Union[str, int, float, bool]], ...]
        Key-value pairs of metadata information.
    expr : Expr
        Expression to which the metadata is attached.

    Examples
    --------
    >>> from physika.core.expr import MData, Const
    >>> term = MData((("line", 12),), Const("x", ()))
    >>> term.kvs
    (('line', 12),)
    >>> term.expr
    Const(name='x', levels=())
    """
    kvs: Tuple[Tuple[str, Union[str, int, float, bool]], ...]
    expr: "Expr"


@dataclass(frozen=True)
class Proj:
    """
    Structure field projection.

    ``Proj`` is an expression constructor used to access the fields of an
    inductive type registered with just one constructor. ``Proj`` allows
    Physika kernel to access and extract a specific field by its index
    from constructor arguments, avoiding the overhead of full inductive
    elimination.

    Parameters
    ----------
    type_name : str
        Name of the structure type.
    idx : int
        Index of the field to be projected (0-based).
    expr : Expr
        Eexpression representing the structure instance from which the field
        is being projected.

    Examples
    --------
    >>> from physika.core.expr import Proj, Const
    >>> ray = Const("ray", ())
    >>> proj = Proj("Ray", 0, ray)
    >>> proj.type_name, proj.idx, proj.expr
    ('Ray', 0, Const(name='ray', levels=()))
    """
    type_name: str
    idx: int
    expr: "Expr"


Expr = Union[BVar, FVar, MVar, Sort, Const, App, Lam, ForallE, LetE, Lit,
             MData, Proj]

PROP = Sort(LZero())  # Sort 0
TYPE_0 = Sort(LSucc(LZero()))  # Sort 1
TYPE_1 = Sort(LSucc(LSucc(LZero())))  # Sort 2
