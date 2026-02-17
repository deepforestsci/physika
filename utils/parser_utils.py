from __future__ import annotations

from utils.ast_utils import ASTNode


def find_indexed_arrays(ast: ASTNode, loop_var: str) -> list[str]:
    """Collect names of arrays indexed by a given loop variable.

    Recursively walks an AST subtree looking for ``("index", name, idx)``
    nodes where *idx* resolves to *loop_var*.  This function is called during parse
    to infer the iteration count of ``for`` loops: the generated code
    iterates ``range(len(arr))`` where *arr* is the first array found.

    The index expression is matched against three representations the
    parser may produce for the same loop variable:

    * ``("var", loop_var)`` — standard variable reference.
    * bare ``loop_var`` string — legacy / simplified form.
    * ``("imaginary",)`` — when *loop_var* is ``"i"`` (the lexer emits
      the ``IMAGINARY`` token for the identifier ``i``).

    Parameters
    ----------
    ast : ASTNode
        The AST subtree to search (typically a loop body statement or
        a list of statements).
    loop_var : str
        The loop variable name to look for in index positions
        (e.g. ``"i"``, ``"k"``).

    Returns
    -------
    list[str]
        Array name strings indexed by *loop_var*, in encounter order
        (may contain duplicates).

    Examples
    --------
    >>> from utils.parser_utils import find_indexed_arrays
    >>> stmt = ("loop_assign", "total",
    ...         ("add", ("var", "total"),
    ...                 ("index", "arr", ("var", "i"))))
    >>> find_indexed_arrays(stmt, "i")
    ['arr']
    >>> stmt2 = ("add", ("index", "X", ("imaginary",)),
    ...                 ("index", "y", ("imaginary",)))
    >>> find_indexed_arrays(stmt2, "i")
    ['X', 'y']
    """
    arrays: list[str] = []

    def visit(node: ASTNode) -> None:
        if node is None:
            return
        if isinstance(node, tuple):
            if len(node) >= 3 and node[0] == "index":
                array_name = node[1]
                index_expr = node[2]
                is_loop = (
                    index_expr == ("var", loop_var) or
                    index_expr == loop_var or
                    (loop_var == "i" and index_expr == ("imaginary",))
                )
                if is_loop:
                    if isinstance(array_name, str):
                        arrays.append(array_name)
                    elif isinstance(array_name, tuple) and array_name[0] == "var":
                        arrays.append(array_name[1])
            for item in node:
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(ast)
    return arrays
