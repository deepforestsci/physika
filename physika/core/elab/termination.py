from typing import List, Tuple
from physika.utils.cic_utils.elab_utils import (
    arg_decreases,
    collect_calls_to,
)


def check_recursive_termination(func_name: str, params: List[Tuple[str, str]],
                                body_node: object, statements: list,
                                errors: List[str], context_label: str) -> None:
    """
    Find a parameter position that changes on recursive calls and accept the
    function if it exists. Appends a error to list of errors if nothing is
    returned or if no position exists.

    Parameters
    ----------
    func_name : str
        Function's name.
    params : List[Tuple[str, object]]
        Function's ``(name, declared_type)`` parameter list.
    body_node : object
        Return expression of the function in AST format.
    statements : list
        Function's statement list.
    errors : List[str]
        List to append to on failure.
    context_label : str
        Location prefix for any appended diagnostic.

    Example
    -------
    >>> from physika.core.elab.termination import check_recursive_termination
    >>> errors = []
    >>> check_recursive_termination(
    ...     "fact", [("n", "ℕ")],
    ...     ("call", "fact", [("sub", ("var", "n"), ("num", 1.0))]),
    ...     [], errors, "In function 'fact'",
    ... )
    >>> errors
    []
    >>> errors = []
    >>> check_recursive_termination(
    ...     "bad", [("n", "ℕ")],
    ...     ("call", "bad", [("add", ("var", "n"), ("num", 1.0))]),
    ...     [], errors, "In function 'bad'",
    ... )
    >>> len(errors)
    1
    """
    if not params:
        return
    calls = collect_calls_to({func_name}, body_node, statements)
    if not calls:
        return

    param_names = [p[0] for p in params]
    param_types = [p[1] for p in params]

    def decreases_at(pos: int) -> bool:
        return all(
            pos < len(args)
            and arg_decreases(args[pos], param_names[pos], param_types[pos])
            for _, args in calls)

    if any(decreases_at(pos) for pos in range(len(params))):
        return

    errors.append(
        f"{context_label}: recursive call to '{func_name}' does not decrease"
        f" at any parameter position")


def check_mutual_recursion_termination(functions: dict,
                                       errors: List[str]) -> set:
    """
    Mutual recursion cycles during elaboration pass, and verify in each cycle
    a parameter that decreases by a recognized pattern on every call in its
    own cycle.

    Parameters
    ----------
    functions : dict
        Every function in this elaboration pass.
    errors : List[str]
        List to append errors for any unverifiable cycle member.


    Example
    -------
    >>> from physika.core.elab.termination import check_mutual_recursion_termination  # noqa:E501
    >>> functions = {
    ...     "is_even": {"params": [("n", "ℕ")], "statements": [],
    ...                 "body": ("call", "is_odd",
    ...                          [("sub", ("var", "n"), ("num", 1.0))])},
    ...     "is_odd": {"params": [("n", "ℕ")], "statements": [],
    ...                "body": ("call", "is_even",
    ...                         [("sub", ("var", "n"), ("num", 1.0))])},
    ... }
    >>> errors = []
    >>> sorted(check_mutual_recursion_termination(functions, errors))
    ['is_even', 'is_odd']
    >>> errors
    []
    """
    names = set(functions.keys())
    call_graph: dict = {
        name: collect_calls_to(names, fd.get("body"), fd.get("statements", []))
        for name, fd in functions.items()
    }

    def in_cycle(start: str) -> bool:
        stack = [
            callee for callee, _ in call_graph.get(start, [])
            if callee != start
        ]
        seen: set = set()
        while stack:
            node = stack.pop()
            if node == start:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(callee for callee, _ in call_graph.get(node, []))
        return False

    cycle_members = {name for name in names if in_cycle(name)}

    for name in functions:
        if name not in cycle_members:
            continue
        func_def = functions[name]
        params = func_def.get("params", [])
        context_label = f"In function '{name}'"
        if not params:
            errors.append(
                f"{context_label}: takes no parameters, cannot verify termination"  # noqa: E501
            )
            continue
        measure_name, measure_type = params[0]
        calls = [(callee, args) for callee, args in call_graph[name]
                 if callee in cycle_members]
        if calls and all(
                args and arg_decreases(args[0], measure_name, measure_type)
                for _, args in calls):
            continue
        errors.append(
            f"{context_label}: part of a mutual recursion cycle, but its "
            f"first parameter '{measure_name}' does not decrease "
            f"on every call to another function in "
            f"the cycle. Cannot verify this mutual recursion terminates")

    return cycle_members
