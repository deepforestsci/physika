from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union, Callable
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


class Substitution(dict):
    """
    Mapping from type and dimension variable names to types.

    Represents the solution of type inference accumulated during unification.
    Each entry pair ``"name":  Type`` means that variable *name* has been
    unified to *Type*. ``Substitution`` chains are applied recursively, so
    ``apply`` method always returns a resolved type with no remaining
    bound variables.

    Examples
    --------
    >>> from physika.utils.types import Substitution, TVar, TDim, T_REAL, TTensor  # noqa: E501
    >>> s = Substitution({"α0": T_REAL, "δ0": 3})
    >>> s.apply(TVar("α0"))
    ℝ
    >>> s.apply(TVar("α1"))
    α1
    >>> s.apply(TTensor(((TDim("δ0"), "invariant"),)))
    ℝ[3]
    """

    def apply(self, t: Type) -> Type:
        """
        Recursively apply ``Substitution`` to a given type.

        For a type variable or dimension variable whose name is bound in
        this substitution, follows the chain until a concrete type or
        unbound variable is reached.

        Structured types (``TTensor``, ``TFunc``) are reconstructed with all
        sub-types resolved.

        Parameters
        ----------
        t : Type
            Any HM type: ``TVar``, ``TDim``, ``TScalar``, ``TTensor``,
            ``TFunc``, or ``TInstance``.

        Returns
        -------
        Type
            Resolved type.

        Examples
        --------
        >>> from physika.utils.types import Substitution, TVar, T_REAL
        >>> s = Substitution({"α0": T_REAL, "α1": TVar("α0")})
        >>> s.apply(TVar("α1"))   # chain: α1 -> α0 -> ℝ
        ℝ
        """
        if isinstance(t, (TVar, TDim)):
            if t.name in self:
                return self.apply(self[t.name])
            return t
        if isinstance(t, TScalar):
            return t
        if isinstance(t, TTensor):
            return TTensor(tuple((self.apply_dim(d), v) for d, v in t.dims))
        if isinstance(t, TFunc):
            return TFunc(tuple(self.apply(p) for p in t.params),
                         self.apply(t.ret))
        if isinstance(t, TInstance):
            return t
        return t

    def apply_dim(self, d: Any) -> Any:
        """
        Resolve a single tensor dimension entry.

        Dimension entries may be ``TVar``/``TDim`` during inference or
        plain integers for concrete sizes.  Chains of variable bindings are
        followed until a concrete value or unbound variable is found.

        Parameters
        ----------
        d : Any
            A dimension entry (``TVar``or ``TDim``), or a concrete
            integer/value.

        Returns
        -------
        Any
            A concrete integer or an unbound ``TVar``/``TDim``.

        Examples
        --------
        >>> from physika.utils.types import Substitution, TDim
        >>> s = Substitution({"δ0": TDim("δ1"), "δ1": 4})
        >>> s.apply_dim(TDim("δ0"))   # chain: δ0 -> δ1 -> 4
        4
        >>> s.apply_dim(TDim("δ2"))
        δ2
        >>> s.apply_dim(3)
        3
        """
        if isinstance(d, (TVar, TDim)):
            if d.name in self:
                resolved = self[d.name]
                if isinstance(resolved, (TVar, TDim)):
                    return self.apply_dim(resolved)
                return resolved
        return d

    def apply_env(self, env: dict) -> dict:
        """
        Apply this ``Substitution`` to every type in a type environment.

        Parameters
        ----------
        env : dict
            A mapping (``{name: Type}``).

        Returns
        -------
        dict
            A new environment with all types resolved via ``apply``.

        Examples
        --------
        >>> from physika.utils.types import Substitution, TVar, T_REAL
        >>> s = Substitution({"α0": T_REAL})
        >>> s.apply_env({"x": TVar("α0"), "y": None})
        {'x': ℝ, 'y': None}
        """
        return {
            k: self.apply(v) if v is not None else None
            for k, v in env.items()
        }

    def compose(self, other: Substitution) -> Substitution:
        """
        Return the composition following the structure:

        ``self ∘ other``

        Applying the result is equivalent to first applying *other*, then
        applying *self*. In other words, apply *self* to every value in
        *other*, then add any bindings from *self* not already present.

        Parameters
        ----------
        other : Substitution
            The substitution to compose with.

        Returns
        -------
        Substitution
            A new substitution object.

        Examples
        --------
        >>> from physika.utils.types import Substitution, TVar, T_REAL
        >>> s1 = Substitution({"α0": T_REAL})
        >>> s2 = Substitution({"α1": TVar("α0")})
        >>> composed = s1.compose(s2)
        >>> composed
        {'α1': ℝ, 'α0': ℝ}
        >>> composed.apply(TVar("α1"))   # α1 -> α0 -> ℝ
        ℝ
        """
        result = Substitution({k: self.apply(v) for k, v in other.items()})
        for k, v in self.items():
            if k not in result:
                result[k] = v
        return result


def check_function(
    name: str,
    fdef: dict,
    func_env: dict,
    class_env: dict,
    add_error: Callable[[str], None],
) -> None:
    """
    Checks that a function defined in Physika is well typed.

    ``check_function`` creates a local environment from the body statements,
    declared parameter and return types, and runs ``infer_stmts`` over the
    function body statements. Then, at unification step, the return expression
    is type checked against the declared return type.

    Class defintions (name, class constructors/fields, TInstance) are stored in
    ``func_env`` for checking class instances used in a function.

    Parameters
    ----------
    name : str
        Function name, used only for error messages.
    fdef : dict
        Function definition dict from ``unified_ast["functions"]``, with
        keys ``"params"``, ``"statements"``, ``"body"``, ``"return_type"``.
    func_env : dict
        Function signature registry ``{name: ([param_types], ret_type)}``.
    class_env : dict
        Global class definition registry, used by ``infer_expr`` to resolve
        constructor calls.
    add_error : Callable[[str], None]
        Callback that receives the type error, if any, as string.

    Examples
    --------
    >>> from physika.utils.types import check_function
    >>> errors = []
    >>> fdef = {"params": [("x", "ℝ")], "statements": [], "body": ("var", "x"), "return_type": "ℝ"}  # noqa: E501
    >>> check_function("identity", fdef, {}, {}, errors.append)
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import from_typespec, unify, type_to_str  # noqa: E501
    from physika.utils.infer_expr import infer_expr
    from physika.utils.infer_stmts import infer_stmts

    params = fdef["params"]
    stmts = fdef.get("statements", [])
    body = fdef.get("body")
    return_type = from_typespec(fdef.get("return_type"))

    local_env = {pn: (from_typespec(pt) or new_var()) for pn, pt in params}

    s = Substitution()

    # infers types of right hand side statements (body statements), unifies
    # with declared types and add/resolved bindings from Substitution s.
    final_env, s = infer_stmts(
        stmts, local_env, s, func_env, class_env,
        lambda msg: add_error(f"In function '{name}': {msg}"), name,
        return_type)

    # Infers and checks types of function's return expressions
    if body is not None:

        def _add(msg):
            add_error(f"In function '{name}': {msg}")

        if (isinstance(body, tuple) and body[0] == "tuple_return"
                and isinstance(return_type, tuple)
                and return_type[0] == "tuple_type"):
            for i, (ret_expr,
                    decl_t) in enumerate(zip(body[1:], return_type[1])):
                t, s = infer_expr(ret_expr, final_env, s, func_env, class_env,
                                  _add)
                if t is not None and decl_t is not None:
                    try:
                        s = unify(decl_t, s.apply(t), s)
                    except TypeError as e:
                        _add(f"return type mismatch at position {i}: "
                             f"declared {type_to_str(decl_t)}, "
                             f"got {type_to_str(s.apply(t))}: {e}")
        else:
            body_t, s = infer_expr(body, final_env, s, func_env, class_env,
                                   _add)
            body_t = s.apply(body_t) if body_t is not None else None
            if return_type is not None and body_t is not None:
                try:
                    s = unify(return_type, body_t, s)
                except TypeError as e:
                    _add(f"return type mismatch: "
                         f"declared {type_to_str(return_type)}, "
                         f"got {type_to_str(body_t)}: {e}")


def check_class(
    name: str,
    cdef: dict,
    func_env: dict,
    class_env: dict,
    add_error: Callable[[str], None],
) -> None:
    """
    Check types of a Physika class object.

    Adds class parameters, fields, methods and body statements into a local
    environment, runs ``infer_stmts``, and unifies the return expression type
    against the declared return type.

    Parameters
    ----------
    name : str
        Class name for adding error messages.
    cdef : dict
        Class definition dict from ``unified_ast["classes"]``, with keys
        ``"class_params"``, ``"statements"``, ``"body"``, ``"return_type"``.
    func_env : dict
        Function signature registry.
    class_env : dict
        Class definition registry.
    add_error : Callable[[str], None]
        Callback that receives error string.

    Examples
    --------
    >>> from physika.utils.types import check_class
    >>> errors = []
    >>> cdef = {
    ...     "class_params": [("w", "ℝ")],
    ...     "statements": [],
    ...     "body": ("mul", ("var", "w"), ("var", "x")),
    ... }
    >>> check_class("Linear", cdef, {}, {}, errors.append)
    >>> errors
    []
    """
    from physika.utils.type_checker_utils import from_typespec, unify, type_to_str  # noqa: E501
    from physika.utils.infer_expr import infer_expr
    from physika.utils.infer_stmts import infer_stmts

    methods = cdef.get("methods", [])
    class_params = list(cdef.get("class_params", []))
    fields = class_params + list(cdef.get("fields", []))

    # add constructor details to func_env
    # (only class_params, not declared fields)
    func_env[name] = (
        [from_typespec(pt) or new_var() for _, pt in class_params],
        TInstance(name),
    )

    # Add class details to class_env
    class_env[name] = {
        "fields": fields,
        "methods": {
            m["name"]: {
                "params": m.get("params", []),
                "return_type": m.get("return_type")
            }
            for m in methods
        },
    }

    local_env = {pn: (from_typespec(pt) or new_var()) for pn, pt in fields}
    local_env["this"] = TInstance(name)

    for method in methods:

        method_name = method.get("name")
        method_params = method.get("params", [])
        method_stmts = method.get("statements", [])
        body = method.get("body")
        return_type = from_typespec(method.get("return_type"))

        # register method params in local_env
        for pn, pt in method_params:
            local_env[pn] = from_typespec(pt) or new_var()

        # check body statements
        s = Substitution()
        final_env, s = infer_stmts(
            method_stmts,
            local_env,
            s,
            func_env,
            class_env,
            lambda msg: add_error(
                f"In class '{name}', method '{method_name}': {msg}"),
            name,
            return_type,
        )

        # Body are return expressions
        if body is not None:
            err_prefix = f"In class '{name}', method '{method_name}'"

            def _add(msg):
                add_error(f"{err_prefix}: {msg}")

            if (isinstance(body, tuple) and body[0] == "tuple_return"
                    and isinstance(return_type, tuple)
                    and return_type[0] == "tuple_type"):
                # Tuple return type → ℝ, ℝ: check each component separately
                for i, (ret_expr,
                        decl_t) in enumerate(zip(body[1:], return_type[1])):
                    t, s = infer_expr(ret_expr, final_env, s, func_env,
                                      class_env, _add)
                    if t is not None and decl_t is not None:
                        try:
                            s = unify(decl_t, s.apply(t), s)
                        except TypeError as e:
                            _add(f"return type mismatch at position {i}: "
                                 f"declared {type_to_str(decl_t)}, "
                                 f"got {type_to_str(s.apply(t))}: {e}")
            else:
                body_t, s = infer_expr(body, final_env, s, func_env, class_env,
                                       _add)
                if return_type is not None and body_t is not None:
                    try:
                        s = unify(return_type, s.apply(body_t), s)
                    except TypeError as e:
                        _add(f"return type mismatch: "
                             f"declared {type_to_str(return_type)}, "
                             f"got {type_to_str(s.apply(body_t))}: {e}")


def check_statement(
    stmt: Any,
    type_env: dict,
    func_env: dict,
    class_env: dict,
    add_error: Callable[[str], None],
) -> None:
    """
    Check types of a program statement.

    Calls ``infer_stmts`` to check types of Physika valid statements
    and adds error message, if any.

    Parameters
    ----------
    stmt : Any
        ASTNode tuple for statements. For example:
        ``("decl", "x", "ℝ", ("num", 3.0), 1)``.
    type_env : dict
        Mutable type environment that updates as declarations are processed.
    func_env : dict
        Global function signature registry.
    class_env : dict
        Global class definition registry.
    add_error : Callable[[str], None]
        Callback that receives a error string.

    Examples
    --------
    >>> from physika.utils.types import check_statement
    >>> from physika.utils.types import T_REAL
    >>> errors = []
    >>> env = {}
    >>> check_statement(("decl", "x", "ℝ", ("num", 3.0), 1), env, {}, {}, errors.append)  # noqa: E501
    >>> errors
    []
    >>> env["x"] == T_REAL
    True
    """
    from physika.utils.infer_stmts import infer_stmts

    if stmt is None:
        return
    line = stmt[-1] if len(stmt) > 1 and isinstance(stmt[-1], int) else None

    prefix_line_msg = f"Line {line}: " if line is not None else ""
    updated_env, _ = infer_stmts(
        [stmt], type_env, Substitution(), func_env, class_env,
        lambda msg: add_error(f"{prefix_line_msg}{msg}"))
    type_env.update(updated_env)
