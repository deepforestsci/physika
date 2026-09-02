from typing import Dict, Optional, Tuple

from physika.core.expr import Expr
from physika.core.inductive import InductiveDecl, Recursor


class ConstantInfo:
    """
    ``ConstantInfo`` stores constant declarations
    data such as it's name, universe parameters, type, and
    value and is used inside ``Environment``.

    Parameters
    ----------
    name : str
        Unique identifier (``Const(name, levels)``)
    level_params : Tuple[str, ...]
        Universe polymorphism parameter names.
    type : Expr
        Pi-type (chain of ``ForallE`` nodes)
    value : Optional[Expr]
        None for axioms. ``body`` Expr for contant
        declarations.

    Examples
    --------
    >>> from physika.core.environment import ConstantInfo
    >>> from physika.core.expr import ForallE, Const
    >>> nat = Const("Nat", ())
    >>> succ_ci = ConstantInfo(
    ...     name="Nat.succ",
    ...     level_params=(),
    ...     type=ForallE("n", nat, nat),
    ...     value=None,
    ... )
    >>> succ_ci.name
    'Nat.succ'
    >>> succ_ci.value is None
    True
    """

    def __init__(self, name: str, level_params: Tuple[str, ...], type: Expr,
                 value: Optional[Expr]):
        self.name = name
        self.level_params = level_params
        self.type = type
        self.value = value


class InductiveInfo:
    """
    ``InductiveInfo`` is used in ``Environment`` to store data
    about an inductive type. ``InductiveInfo`` keeps track of an
    inductive type's declaration, its constructors and recursor
    as ``ConstantInfo``, and its ``Recursor``.

    Parameters
    ----------
    decl : InductiveDecl
        Inductive declaration
    ctors : Dict[str, ConstantInfo]
        Dictionary with constructor name as keys and ConstantInfo as values
    recursor : ConstantInfo
        Recursor as a ConstantInfo for lookup
    rec_info : Optional[Recursor]
        Recursor with iota reduction rules.

    Examples
    --------
    >>> from physika.core.environment import ConstantInfo, InductiveInfo
    >>> from physika.core.inductive import InductiveDecl, Constructor
    >>> from physika.core.expr import ForallE, Const, TYPE_0
    >>> nat = Const("Nat", ())
    >>> zero_ci = ConstantInfo("Nat.zero", (), nat, None)
    >>> succ_ci = ConstantInfo("Nat.succ", (), ForallE("n", nat, nat), None)
    >>> rec_ci = ConstantInfo("Nat.rec", ("u",), nat, None)
    >>> nat_decl = InductiveDecl(
    ...     name="Nat", level_params=(), num_params=0, type=TYPE_0,
    ...     constructors=(Constructor("Nat.zero", nat),
    ...                   Constructor("Nat.succ", ForallE("n", nat, nat))),
    ...     is_recursive=True,
    ... )
    >>> nat_info = InductiveInfo(
    ...     decl=nat_decl,
    ...     ctors={"Nat.zero": zero_ci, "Nat.succ": succ_ci},
    ...     recursor=rec_ci,
    ... )
    >>> nat_info.decl.name
    'Nat'
    >>> sorted(nat_info.ctors)
    ['Nat.succ', 'Nat.zero']
    >>> nat_info.rec_info is None
    True
    """

    def __init__(self,
                 decl: InductiveDecl,
                 ctors: Dict[str, ConstantInfo],
                 recursor: ConstantInfo,
                 rec_info: Optional[Recursor] = None):

        self.decl = decl
        self.ctors = ctors
        self.recursor = recursor
        self.rec_info = rec_info


class Environment:
    """
    ``Environment`` keeps track axioms, definitions, and inductive
    types used in a Physika program. `Environment`` is initialized
    before elaboration and type checking. Once each CIC term (axioms, theorems
    and declarations) is added, ``Environment`` serves as a lookup table.

    Examples
    --------
    >>> from physika.core.environment import Environment, ConstantInfo
    >>> from physika.core.expr import Const
    >>> env = Environment()
    >>> env.add_constant(ConstantInfo("Nat.zero", (), Const("Nat", ()), None))
    >>> "Nat.zero" in env.constants
    True
    >>> env.constants["Nat.zero"].name
    'Nat.zero'
    """

    def __init__(self) -> None:
        """
        Create an empty ``Environment``. No constants and no inductive
        types registered yet.

        Examples
        --------
        >>> from physika.core.environment import Environment
        >>> env = Environment()
        >>> list(env.constants.values())
        []
        """
        self.constants: Dict[str, ConstantInfo] = {}
        self.inductives: Dict[str, InductiveInfo] = {}

    def add_constant(self, c: ConstantInfo) -> None:
        """
        Register a constant by its name.

        Parameters
        ----------
        c : ConstantInfo
            Constant declaration to be registered.

        Examples
        --------
        >>> from physika.core.environment import Environment, ConstantInfo  # noqa: E501
        >>> from physika.core.expr import Const
        >>> env = Environment()
        >>> env.add_constant(ConstantInfo("Nat.zero", (), Const("Nat", ()), None))
        >>> "Nat.zero" in env.constants
        True
        """
        if c.name in self.constants:
            #  every name in ``Environment`` must be unique
            raise ValueError(
                f"Environment: constant '{c.name}' already registered")
        self.constants[c.name] = c

    def add_inductive(self, i: InductiveInfo) -> None:
        """
        Register an inductive type.

        Every constructor and recursor in ``i`` are registered as
        ``ConstantInfo``. Strict positivity checks ``i.decl``
        constructors are well defined. If ``i.rec_info`` is given, its
        ``RecursorRule``\\ s is re-verified
        with the trusted kernel.

        Parameters
        ----------
        i : InductiveInfo
            Inductive type to be registered.

        Examples
        --------
        >>> from physika.core.environment import Environment, ConstantInfo, InductiveInfo  # noqa: E501
        >>> from physika.core.inductive import InductiveDecl, Constructor
        >>> from physika.core.expr import ForallE, Const, TYPE_0
        >>> env = Environment()
        >>> nat = Const("Nat", ())
        >>> nat_decl = InductiveDecl(
        ...     name="Nat", level_params=(), num_params=0, type=TYPE_0,
        ...     constructors=(Constructor("Nat.zero", nat),
        ...                   Constructor("Nat.succ", ForallE("n", nat, nat))),
        ...     is_recursive=True,
        ... )
        >>> nat_info = InductiveInfo(
        ...     decl=nat_decl,
        ...     ctors={"Nat.zero": ConstantInfo("Nat.zero", (), nat, None),
        ...            "Nat.succ": ConstantInfo("Nat.succ", (), ForallE("n", nat, nat), None)},
        ...     recursor=ConstantInfo("Nat.rec", ("u",), nat, None),
        ... )
        >>> env.add_inductive(nat_info)
        >>> env.inductives["Nat"].decl.name
        'Nat'
        >>> "Nat.zero" in env.constants, "Nat.rec" in env.constants
        (True, True)
        """
        from physika.utils.cic_utils.inductive_utils import (
            check_positivity_for_inductive,
            verify_recursor_rules,
        )
        type_name = i.decl.name
        if type_name in self.inductives:
            raise ValueError(
                f"Environment: inductive '{type_name}' already registered")

        is_inductive_former = lambda name, _t=type_name: name == _t  # noqa: E731,E501
        err = check_positivity_for_inductive(i.decl, is_inductive_former)
        if err is not None:
            raise ValueError(f"Environment: inductive '{type_name}': {err}")

        if i.rec_info is not None:

            scratch = Environment()
            scratch.constants = dict(self.constants)
            scratch.inductives = dict(self.inductives)
            for ctor_ci in i.ctors.values():
                scratch.add_constant(ctor_ci)
            scratch.add_constant(i.recursor)
            scratch.inductives[type_name] = i
            rec_err = verify_recursor_rules(i.decl, i.ctors, i.recursor,
                                            i.rec_info, scratch)
            if rec_err is not None:
                raise ValueError(
                    f"Environment: inductive '{type_name}': {rec_err}")

        self.inductives[type_name] = i
        for ctor_ci in i.ctors.values():
            self.add_constant(ctor_ci)
        self.add_constant(i.recursor)
