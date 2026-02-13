from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn as nn


def print_type_check_results(type_errors: list[str]) -> None:
    """Print type-checking results and exit on errors.

    If *type_errors* is non-empty, prints each error prefixed with a
    cross mark, shows a summary count, and terminates the process with
    ``sys.exit(1)``.  If no errors are present, prints a success
    message.

    Parameters
    ----------
    type_errors : list[str]
        A list of human-readable error messages returned by
        ``TypeChecker.run()`` (or the ``type_check`` wrapper).

    Examples
    --------
    >>> from utils.print_utils import print_type_check_results
    >>> print_type_check_results([])
      ✓ No type errors found
    """
    if type_errors:
        print("Type errors found:")
        for error in type_errors:
            print(f"  ✗ {error}")
        print(f"{len(type_errors)} type error(s) found.")

        sys.exit(1)
    else:
        print("  ✓ No type errors found")


def _pformat(value: Any, indent: int = 0) -> str:
    """Pretty-format an AST value with indentation.

    Recursively formats tuples, lists, dicts, and scalars into readable,
    indented string. Short collections that fit within 80 columns are
    kept on a single line and longer ones are expanded vertically.

    Parameters
    ----------
    value : Any
        The AST value to format — a tuple, list, dict, ``int``,
        ``float``, ``str``, or ``None``.
    indent : int, default 0
        Current indentation level (each level = 2 spaces).

    Returns
    -------
    str
        A formatted, possibly multi-line string.

    Examples
    --------
    >>> from utils.print_utils import _pformat
    >>> _pformat(("num", 3.0))
    "('num', 3.0)"
    >>> _pformat({"x": 1}, indent=1)
    '  {\\n    x: 1\\n  }'
    """
    prefix = "  " * indent

    if value is None:
        return f"{prefix}None"

    if isinstance(value, (int, float)):
        return f"{prefix}{value}"

    if isinstance(value, str):
        return f"{prefix}{repr(value)}"

    if isinstance(value, dict):
        if not value:
            return f"{prefix}{{}}"
        lines = [f"{prefix}{{"]
        for k, v in value.items():
            v_str = _pformat(v, indent + 2)
            # If the formatted value is a single line, put key: value on one line
            v_stripped = v_str.strip()
            if "\n" not in v_stripped:
                lines.append(f"{prefix}  {k}: {v_stripped}")
            else:
                lines.append(f"{prefix}  {k}:")
                lines.append(v_str)
        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        # Check if all items are simple scalars
        if all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in value):
            items = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in value)
            oneline = f"{prefix}[{items}]"
            if len(oneline) <= 80:
                return oneline
        lines = [f"{prefix}["]
        for item in value:
            lines.append(f"{_pformat(item, indent + 1)},")
        lines.append(f"{prefix}]")
        return "\n".join(lines)

    if isinstance(value, tuple):
        if not value:
            return f"{prefix}()"
        # Check if all items are simple scalars
        if all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in value):
            items = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in value)
            oneline = f"{prefix}({items})"
            if len(oneline) <= 80:
                return oneline
        lines = [f"{prefix}("]
        for item in value:
            lines.append(f"{_pformat(item, indent + 1)},")
        lines.append(f"{prefix})")
        return "\n".join(lines)

    return f"{prefix}{repr(value)}"


def print_unified_ast(unified_ast: dict[str, Any]) -> str:
    """Return a pretty-printed string of the unified AST.

    Formats the three sections — ``"functions"``, ``"classes"``, and
    ``"program"`` — with indentation for easy reading.  Used for
    debugging with the ``print_ast=True`` flag in
    ``build_unified_ast``.

    Parameters
    ----------
    unified_ast : dict[str, Any]
        The unified AST dict produced by ``build_unified_ast()``,
        with keys ``"functions"``, ``"classes"``, and ``"program"``.

    Returns
    -------
    str
        A multi-line, indented string representation of the entire
        unified AST.

    Examples
    --------
    >>> from utils.ast_utils import build_unified_ast
    >>> ast = {"functions": {}, "classes": {}, "program": [("expr", ("num", 1.0), 1)]}
    >>> print(print_unified_ast(ast))
    Functions:
    <BLANKLINE>
    Classes:
    <BLANKLINE>
    Program:
      (
        'expr',
        ('num', 1.0),
        1,
      )
    """
    lines = []

    # Functions
    lines.append("Functions:")
    for name, func_def in unified_ast["functions"].items():
        lines.append(f"  {name}:")
        lines.append(f"    params: {func_def['params']}")
        if func_def.get('statements'):
            lines.append("    statements:")
            for stmt in func_def['statements']:
                lines.append(_pformat(stmt, 3))
        lines.append("    body:")
        lines.append(_pformat(func_def['body'], 3))

    # Classes
    lines.append("\nClasses:")
    for name, class_def in unified_ast["classes"].items():
        lines.append(f"  {name}:")
        lines.append(f"    class_params: {class_def['class_params']}")
        lines.append(f"    lambda_params: {class_def['lambda_params']}")
        if class_def.get('has_loop'):
            lines.append(f"    loop_var: {class_def['loop_var']}")
            lines.append("    loop_body:")
            for stmt in class_def['loop_body']:
                lines.append(_pformat(stmt, 3))
        lines.append("    body:")
        lines.append(_pformat(class_def['body'], 3))
        if class_def.get('has_loss'):
            lines.append("    loss_body:")
            lines.append(_pformat(class_def['loss_body'], 3))

    # Program
    lines.append("\nProgram:")
    for stmt in unified_ast["program"]:
        lines.append(_pformat(stmt, 1))

    return "\n".join(lines)


def _from_torch(v: Any) -> Any:
    """Convert a torch value to a plain Python value for display.

    Scalars are unwrapped via ``.item()``, tensors are converted to
    nested Python lists via ``.tolist()``, and complex values whose
    imaginary part is negligible (``< 1e-10``) are reduced to their
    real part.

    Parameters
    ----------
    v : Any
        The value to convert. A ``torch.Tensor``, ``complex``,
        or any other Python value (returned as-is).

    Returns
    -------
    Any
        A plain Python scalar (``int``, ``float``, ``complex``) or
        nested ``list``.

    Examples
    --------
    >>> from utils.print_utils import _from_torch
    >>> _from_torch(torch.tensor(3.0))
    3.0
    >>> _from_torch(torch.tensor([1.0, 2.0]))
    [1.0, 2.0]
    >>> _from_torch(complex(2.0, 0.0))
    2.0
    """
    if not isinstance(v, torch.Tensor):
        if isinstance(v, complex):
            if abs(v.imag) < 1e-10:
                return v.real
            return v
        return v
    if v.numel() == 1:
        val = v.item()
        if isinstance(val, complex) and abs(val.imag) < 1e-10:
            return val.real
        return val
    return v.detach().tolist()


def _infer_type(v: Any) -> str:
    """Infer the Physika type string for a value.

    Maps Python / PyTorch values to their Physika type notation:
    ``"ℝ"`` for real scalars, ``"ℂ"`` for complex, ``"ℝ[n]"`` for
    vectors, ``"ℝ[m,n]"`` for matrices, and the class name for
    ``nn.Module`` subclasses.

    Parameters
    ----------
    v : Any
        The value whose type to infer — ``torch.Tensor``, ``int``,
        ``float``, ``complex``, ``list``, or ``nn.Module``.

    Returns
    -------
    str
        A Physika type string (e.g. ``"ℝ"``, ``"ℝ[3]"``,
        ``"ℝ[2,3]"``, ``"ℂ"``).

    Examples
    --------
    >>> from utils.print_utils import _infer_type
    >>> _infer_type(3.0)
    'ℝ'
    >>> _infer_type(torch.tensor([1.0, 2.0, 3.0]))
    'ℝ[3]'
    >>> _infer_type(complex(1, 2))
    'ℂ'
    """
    if isinstance(v, complex):
        if v.imag == 0:
            return "ℝ"
        return "ℂ"
    if isinstance(v, torch.Tensor) and v.is_complex():
        if v.imag.abs().max() < 1e-10:
            return "ℝ"
        return "ℂ"
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return "ℝ"
        if v.dim() == 1:
            return f"ℝ[{v.shape[0]}]"
        dims = ",".join(str(d) for d in v.shape)
        return f"ℝ[{dims}]"
    if isinstance(v, (int, float)):
        return "ℝ"
    if isinstance(v, list):
        shape = []
        current = v
        while isinstance(current, list) and len(current) > 0:
            shape.append(len(current))
            current = current[0]
        return f"ℝ[{','.join(str(d) for d in shape)}]"
    if isinstance(v, nn.Module):
        return type(v).__name__
    return str(type(v).__name__)
