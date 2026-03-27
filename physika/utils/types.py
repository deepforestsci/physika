from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union
import itertools


@dataclass(frozen=True)
class TVar:
    """
    A type unification variable (α0, α1, ...).

    Created by ``new_var()`` whenever the checker needs a placeholder
    for an unknown type.  Resolved to a concrete type when ``unify``
    adds a binding ``{αN: <some type>}`` at the substitution step.

    Parameters
    ----------
    name : str
        Unique identifier. For example``"α0"``.

    Examples
    --------
    >>> from physika.utils.types import TVar
    >>> TVar("α0")
    α0
    """
    name: str

    def __repr__(self) -> str:
        """Return the variable name as its string representation.

        Returns
        -------
        str
            The string name

        Examples
        --------
        >>> from physika.utils.types import TVar
        >>> repr(TVar("α0"))
        'α0'
        """
        return self.name


@dataclass(frozen=True)
class TDim:
    """A dimension unification variable (δ0, δ1, ...).

    Used inside ``TTensor.dims`` when the size of an axis is not yet
    known.  Resolved to a concrete integer or symbolic string
    when ``_unify_dim`` adds a binding to the substitution.

    Parameters
    ----------
    name : str
        Unique identifier for dimension variable (``"δ2"``).

    Examples
    --------
    >>> from physika.utils.types import TDim
    >>> TDim("δ0")
    δ0
    """
    name: str

    def __repr__(self) -> str:
        """Return the dimension variable name as its string representation.

        Returns
        -------
        str
            The string name (``"δ2"``).

        Examples
        --------
        >>> from physika.utils.types import TDim
        >>> repr(TDim("δ2"))
        'δ2'
        """
        return self.name


@dataclass(frozen=True)
class TScalar:
    """
    A ground scalar type.

    Supports the four built-in scalars singletons
    ``T_REAL`` (ℝ), ``T_NAT`` (ℕ), ``T_COMPLEX`` (ℂ), and
    ``T_STRING``.  ``TScalar`` is never created with a fresh name
    during inference.

    Parameters
    ----------
    name : str
        Unicode symbol (one of ``"ℝ"``, ``"ℕ"``, ``"ℂ"``, ``"string"``).

    Examples
    --------
    >>> from physika.utils.types import TScalar
    >>> TScalar("ℝ")
    ℝ
    >>> TScalar("ℝ") == TScalar("ℕ")
    False
    """
    name: str

    def __repr__(self) -> str:
        """Return the scalar's unicode symbol as its string representation.

        Returns
        -------
        str
            One of ``"ℝ"``, ``"ℕ"``, ``"ℂ"``, or ``"string"``.

        Examples
        --------
        >>> from physika.utils.types import TScalar
        >>> repr(TScalar("ℝ"))
        'ℝ'
        """
        return self.name


@dataclass(frozen=True)
class TTensor:
    """
    A tensor type whose shape is a sequence of (dimension, variance) pairs.

    Each dimension entry is one of:

    * ``int`` — a concrete size known at inference time.
    * ``str`` — a symbolic size from a type annotation.
    * ``TDim`` — a fresh unknown dimension, resolved by ``_unify_dim``.

    Passing a ``TVar`` as a dimension raises ``TypeError``: ``TVar`` is a
    type level unknown and is not handled by the dimension unification path
    , so it would go unresolved.

    Parameters
    ----------
    dims : tuple
        Sequence of ``(dim, variance)`` pairs where each ``dim`` is an
        ``int``, ``str``, or ``TDim``.

    Raises
    ------
    TypeError
        If any dimension entry is a ``TVar``.

    Examples
    --------
    >>> from physika.utils.types import TTensor, TDim
    >>> # int literal size from a concrete annotation:
    >>> TTensor(((5, "invariant"),))         # arr : ℝ[5]
    ℝ[5]
    >>> TTensor(((3, "invariant"), (4, "invariant")))   # mat : ℝ[3, 4]
    ℝ[3,4]
    >>> # str symbolic size from a generic parameter annotation:
    >>> TTensor((("n", "invariant"),))       # u : ℝ[n]
    ℝ[n]
    >>> TTensor((("n", "invariant"), ("m", "invariant")))  # A : ℝ[n, m]
    ℝ[n,m]
    >>> # TDim unknown dimension, resolved at unification step:
    >>> TTensor(((TDim("δ0"), "invariant"),))
    ℝ[δ0]
    >>> TTensor(((TDim("δ0"), "invariant"), (TDim("δ1"), "invariant")))
    ℝ[δ0,δ1]
    """
    dims: tuple

    def __post_init__(self) -> None:
        """
        Reject TVar entries in dims for TTensor.

        Raises
        ------
        TypeError
            If any dimension entry is a ``TVar``.

        Examples
        --------
        >>> from physika.utils.types import TTensor, TVar, TDim
        >>> TTensor(((TVar("α0"), "invariant"),))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: TTensor.dims entry α0 is a TVar; ...
        """
        for dim, _ in self.dims:
            if isinstance(dim, TVar):
                raise TypeError(f"TTensor.dims entry {dim!r} is a TVar; "
                                "use TDim for unknown dimensions.")

    def __repr__(self) -> str:
        """Return the tensor type in Physika notation ``ℝ[d0,d1,...,dN]``.

        Each dimension is rendered via ``str()``, so concrete integers
        appear as numbers (e.g. ``3``), symbolic dims as their name
        (e.g. ``n``), and unresolved ``TDim`` variables as ``δN``.

        Returns
        -------
        str
            Unicode string tensor like ``"ℝ[3,4]"`` or ``"ℝ[n,m]"``.

        Examples
        --------
        >>> from physika.utils.types import TTensor
        >>> repr(TTensor(((3, "invariant"), (4, "invariant"))))
        'ℝ[3,4]'
        >>> repr(TTensor((("n", "invariant"),)))
        'ℝ[n]'
        """
        ds = [str(d) for d, _ in self.dims]
        return f"ℝ[{','.join(ds)}]"


@dataclass(frozen=True)
class TFunc:
    """
    A function type ``(p0, p1, ...): return_type``.

    Used in ``func_env`` to store registered function signatures and
    in ``infer_expr`` to check call argument types against the
    declared parameter types.

    Parameters
    ----------
    params : tuple
        Function parameter types like: ``(T_REAL, TTensor(...))``
    ret : Type
        Return type, or ``None`` if the function has no declared return.

    Examples
    --------
    >>> from physika.utils.types import TFunc, TScalar
    >>> TFunc((TScalar("ℝ"),), TScalar("ℝ"))
    (ℝ) → ℝ
    """
    params: tuple
    ret: Any

    def __repr__(self) -> str:
        """
        Return the function type in arrow notation ``(p0, p1, ...) → ret``.

        Parameters are comma separated and wrapped in parentheses. The
        return type follows the ``→`` symbol.

        Returns
        -------
        str
            Arrow notation, e.g. ``"(ℝ, ℝ[n]) → ℝ"``.

        Examples
        --------
        >>> from physika.utils.types import TFunc, TScalar
        >>> repr(TFunc((TScalar("ℝ"),), TScalar("ℝ")))
        '(ℝ) → ℝ'
        """
        ps = ", ".join(repr(p) for p in self.params)
        return f"({ps}) → {self.ret!r}"


@dataclass(frozen=True)
class TInstance:
    """
    The type of a value produced by instantiating a Physika class.

    Two instances types unify only if they share the same ``class_name``.

    Parameters
    ----------
    class_name : str
        Name of the Physika class.

    Examples
    --------
    >>> from physika.utils.types import TInstance
    >>> TInstance("FullyConnectedNet")
    instance(FullyConnectedNet)
    >>> TInstance("FullyConnectedNet") == TInstance("FullyConnectedNet")
    True
    """
    class_name: str

    def __repr__(self) -> str:
        """
        Returns the instance type as ``instance(ClassName)``.

        Returns
        -------
        str
            Instance label.

        Examples
        --------
        >>> from physika.utils.types import TInstance
        >>> repr(TInstance("FullyConnectedNet"))
        'instance(FullyConnectedNet)'
        """
        return f"instance({self.class_name})"


Type = Union[TVar, TDim, TScalar, TTensor, TFunc, TInstance]

# Ground scalar types
T_REAL = TScalar("ℝ")
T_NAT = TScalar("ℕ")
T_COMPLEX = TScalar("ℂ")
T_STRING = TScalar("string")


class VarCounter:
    """
    Shared counter for generating unique type and dimension variable names.

    ``new_var()`` and ``new_dim()`` draw  from the same counter, so
    every ``α`` for TVar and ``δ`` for ``TDim`` produced during type checker
    is unique.
    Call ``reset()``restart numbering from zero (for unit testing).

    Examples
    --------
    >>> from physika.utils.types import VarCounter, TVar, TDim
    >>> c = VarCounter()
    >>> c.new_var()
    α0
    >>> c.new_var()
    α1
    >>> c.new_dim()
    δ2
    >>> c.reset()
    >>> c.new_var()
    α0
    """

    def __init__(self) -> None:
        """
        Initializes a counter to generate TVar and TDims to be resolved at
        unification step.
        """
        self.c = itertools.count()

    def new_var(self) -> TVar:
        """
        Return a new type variable.

        Each call advances the counter, so the returned name is
        never repeated.

        Returns
        -------
        TVar
            A new type variable (``α0``, ``α1``, etc)

        Examples
        --------
        >>> from physika.utils.types import VarCounter, TVar
        >>> c = VarCounter()
        >>> c.new_var()
        α0
        >>> c.new_var()
        α1
        """
        return TVar(f"α{next(self.c)}")

    def new_dim(self) -> TDim:
        """
        Return a new unique dimension variable.

        Dimension variables (``TDim``) share the same counter as type variables
        (``TVar``).

        Returns
        -------
        TDim
            A new dimension variable ()``δ0``, ``δ1``, etc).

        Examples
        --------
        >>> from physika.utils.types import VarCounter, TDim, TVar
        >>> c = VarCounter()
        >>> c.new_var()
        α0
        >>> c.new_dim()
        δ1
        """
        return TDim(f"δ{next(self.c)}")

    def reset(self) -> None:
        """
        Reset the counter to zero.

        After calling this, the next ``new_var()`` call returns ``α0`` again.
        Intended for use when initializing a physika program and in tests.
        """
        self.c = itertools.count()


counter = VarCounter()
new_var = counter.new_var
new_dim = counter.new_dim
