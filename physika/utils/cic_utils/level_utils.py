from physika.core.level import (Level, LZero, LSucc, LMax, LIMax, LParam,
                                LMVar)
from typing import Optional, List


def mk_level_max(l1: Level, l2: Level) -> Level:
    """
    Compares two universe levels (``l1``, ``l2``) and returns the maximum
    level between them.

    Parameters
    ----------
    l1 : Level
        The first universe level to compare.
    l2 : Level
        The second universe level to compare.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc
    >>> from physika.utils.cic_utils.level_utils import mk_level_max
    >>> mk_level_max(LSucc(LZero()), LZero())
    LSucc(pred=LZero())
    >>> mk_level_max(LSucc(LZero()), LSucc(LSucc(LZero())))
    LMax(l1=LSucc(pred=LZero()), l2=LSucc(pred=LSucc(pred=LZero())))
    """
    if isinstance(l1, LZero):
        return l2
    if isinstance(l2, LZero):
        return l1
    if l1 == l2:
        return l1
    return LMax(l1, l2)


def mk_level_imax(l1: Level, l2: Level) -> Level:
    """
    Compares two universe levels (``l1``, ``l2``) and returns the maximum
    level between them, with the special case that if ``l2`` is 0, the result
    is 0.

    Parameters
    ----------
    l1 : Level
        The first universe level to compare.
    l2 : Level
        The second universe level to compare.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc
    >>> from physika.utils.cic_utils.level_utils import mk_level_imax
    >>> mk_level_imax(LSucc(LZero()), LZero())
    LZero()
    >>> mk_level_imax(LSucc(LZero()), LSucc(LZero()))
    LSucc(pred=LZero())
    """
    if isinstance(l2, LZero):
        return LZero()
    if isinstance(l1, LZero):
        return l2
    if l1 == l2:
        return l1
    return LIMax(l1, l2)


def level_has_mvar(lvl: Level, mvar_id: Optional[str] = None) -> bool:
    """
    Occurs-check for universe-level metavariables. Check if ``lvl``contains an
    unsolved LMVar placeholder inside it.

    If ``mvar_id`` is None, ``level_has_mvar`` checks if any level is resolved
    before evaluating. When ``mvar_id`` is provided, ``level_has_mvar`` checks
    if ``LMVar`` with the specified id is present.

    Parameters
    ----------
    lvl : Level
        Level to search for a metavariable node.
    mvar_id : Optional[str], default None
        If given, only match an LMVar with this specific id; otherwise
        match any LMVar.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc, LMVar, LParam
    >>> level_has_mvar(LMVar("m1"))
    True
    >>> level_has_mvar(LSucc(LMVar("m1")))
    True
    >>> level_has_mvar(LParam("u"))
    False
    >>> level_has_mvar(LMVar("m1"), "m1")
    True
    >>> level_has_mvar(LMVar("m1"), "m2")
    False
    """
    if isinstance(lvl, LMVar):
        return mvar_id is None or lvl.id == mvar_id
    elif isinstance(lvl, LSucc):
        return level_has_mvar(lvl.pred, mvar_id)
    elif isinstance(lvl, (LMax, LIMax)):
        return level_has_mvar(lvl.l1, mvar_id) or level_has_mvar(
            lvl.l2, mvar_id)
    return False


def collect_params(lvl: Level) -> set:
    """
    Collects every ``LParam`` name and ``LMVar`` that appears in ``lvl:Level``,
    used by ``levels_equal``.

    Parameters
    ----------
    lvl : Level
        Level expression to look for ``LParam`` and ``LMVar`.

    Examples
    --------
    >>> from physika.core.level import LParam, LMVar, LSucc
    >>> from physika.utils.cic_utils.level_utils import collect_params
    >>> collect_params(LParam("u"))
    {'u'}
    >>> collect_params(LSucc(LMVar("m1")))
    {'?m1'}
    """
    if isinstance(lvl, (LZero, )):
        return set()
    if isinstance(lvl, LParam):
        return {lvl.name}
    if isinstance(lvl, LMVar):
        return {f"?{lvl.id}"}
    if isinstance(lvl, LSucc):
        return collect_params(lvl.pred)
    if isinstance(lvl, (LMax, LIMax)):
        return collect_params(lvl.l1) | collect_params(lvl.l2)
    return set()


def eval_level(lvl: Level, subst: dict) -> int:
    """
    Evaluates ``lvl`` to an integer in substitution dictionary ``subst``.

    Parameters
    ----------
    lvl : Level
        Level expression to evaluate.
    subst : dict
        Mapping from ``LParam`` and ``LMVar` to their substituted value,
        ``0`` or ``1``.

    Examples
    --------
    >>> from physika.core.level import LParam, LSucc, LMax
    >>> from physika.utils.cic_utils.level_utils import eval_level
    >>> eval_level(LSucc(LParam("u")), {"u": 1})
    2
    >>> eval_level(LMax(LParam("u"), LParam("v")), {"u": 0, "v": 1})
    1
    """
    if isinstance(lvl, LZero):
        return 0
    if isinstance(lvl, LParam):
        return subst.get(lvl.name, 0)
    if isinstance(lvl, LMVar):
        return subst.get(f"?{lvl.id}", 0)
    if isinstance(lvl, LSucc):
        return eval_level(lvl.pred, subst) + 1
    if isinstance(lvl, LMax):
        return max(eval_level(lvl.l1, subst), eval_level(lvl.l2, subst))
    if isinstance(lvl, LIMax):
        n2 = eval_level(lvl.l2, subst)
        if n2 == 0:
            return 0
        return max(eval_level(lvl.l1, subst), n2)
    return 0


def levels_equal(l1: Level, l2: Level) -> bool:
    """
    Two universe levels are equal if they agree on every substitution of
    their parameters to {0, 1}.

    First, collect all parameter names from both levels

    Parameters
    ----------
    l1 : Level
        The first universe level to compare.
    l2 : Level
        The second universe level to compare.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc, LParam, LMax, LIMax  # noqa: E501
    >>> from physika.utils.cic_utils.level_utils import levels_equal
    >>> levels_equal(LSucc(LZero()), LSucc(LZero()))
    True
    >>> levels_equal(LMax(LParam("u"), LParam("v")), LMax(LParam("v"), LParam("u")))
    True
    >>> levels_equal(LIMax(LParam("u"), LZero()), LZero())
    True
    """
    n1, n2 = level_to_nat(l1), level_to_nat(l2)
    if n1 is not None and n2 is not None:
        return n1 == n2
    params = sorted(collect_params(l1) | collect_params(l2))
    n = len(params)
    # checks 2^n
    for bits in range(1 << n):
        # parameters takes values {0, 1} during level evaluation
        subst = {params[i]: (bits >> i) & 1 for i in range(n)}
        if eval_level(l1, subst) != eval_level(l2, subst):
            return False
    return True


def level_to_nat(lvl: Level) -> Optional[int]:
    """
    Evaluates ``lvl`` to a concrete ``int``. Returns ``None`` if any ``LParam``
    or ``LMVar`` is present.

    Parameters
    ----------
    lvl : Level
        Level expression to evaluate.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc, LMax, LParam
    >>> from physika.utils.cic_utils.level_utils import level_to_nat
    >>> level_to_nat(LSucc(LSucc(LZero())))
    2
    >>> level_to_nat(LMax(LSucc(LZero()), LZero()))
    1
    >>> level_to_nat(LParam("u")) is None
    True
    """
    if isinstance(lvl, LZero):
        return 0
    elif isinstance(lvl, LSucc):
        n = level_to_nat(lvl.pred)
        return None if n is None else n + 1
    elif isinstance(lvl, LMax):
        n1, n2 = level_to_nat(lvl.l1), level_to_nat(lvl.l2)
        return None if n1 is None or n2 is None else max(n1, n2)
    elif isinstance(lvl, LIMax):
        n1, n2 = level_to_nat(lvl.l1), level_to_nat(lvl.l2)
        if n1 is None or n2 is None:
            return None
        return 0 if n2 == 0 else max(n1, n2)
    else:  # LParam, LMVar
        return None


# TODO: Comment from here
def instantiate_level_params(lvl: Level, params: List[str],
                             levels: List[Level]) -> Level:
    """
    Walks a ``Level`` expression tree and replaces ``LParam[i]`` with a concret
    level (``level[i]``) if ``params[i]`` is the same as `lvl: Level` name.

    Parameters
    ----------
    lvl: Level
        Level expression that is being checked for LParam nodes.
    params: list
        List of parameters strings
    levels: list
        List of levels where params[i] matches with ``lvl`` name.

    Examples
    --------
    >>> from physika.core.level import LZero, LSucc, LParam
    >>> instantiate_level_params(LParam("u"), ["v", "u"],
    ...                          [LZero(), LSucc(LZero())])
    LSucc(pred=LZero())
    >>> instantiate_level_params(LParam("v"), ["u"], [LSucc(LZero())])
    LParam(name='v')
    """
    if isinstance(lvl, LZero):
        return lvl
    elif isinstance(lvl, LSucc):
        return LSucc(instantiate_level_params(lvl.pred, params, levels))
    elif isinstance(lvl, LMax):
        return mk_level_max(
            instantiate_level_params(lvl.l1, params, levels),
            instantiate_level_params(lvl.l2, params, levels),
        )
    elif isinstance(lvl, LIMax):
        return mk_level_imax(
            instantiate_level_params(lvl.l1, params, levels),
            instantiate_level_params(lvl.l2, params, levels),
        )
    elif isinstance(lvl, LParam):
        try:
            return levels[params.index(lvl.name)]
        except ValueError:
            return lvl  # unbound parameter — leave as-is
    else:  # LMVar
        return lvl
