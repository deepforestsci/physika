from physika.parser import parser
from physika.lexer import lexer
from pathlib import Path
from physika.parser import symbol_table


def find_module(module_name: str, source_file_path: Path) -> Path:
    """
    Helper function to locate a Physika module file.

    Parameters
    ----------
    module_name : str
        Name of the Physika module
    source_file_path : Path
        Absolute path of the current Physika source file performing the import.

    Returns
    -------
    Path
        Absolute or relative path to the resolved ``.phyk`` module file.

    Examples
    --------
    >>> from pathlib import Path
    >>> from physika.utils.import_manager import find_module
    >>> source_file_path = Path(
    ...     "examples/example_import_statement.phyk"
    ... )
    >>> module_path = find_module(
    ...     "factorial",
    ...     source_file_path
    ... )
    >>> module_path.name
    'factorial.phyk'
    """
    module_path = Path(module_name.replace(".", "/"))
    module_file = module_path.with_suffix(".phyk")
    search_paths = [source_file_path.parent, *source_file_path.parent.parents]
    for path in search_paths:
        file_name = path / module_file
        if file_name.exists():
            return file_name
    raise ImportError(f"module '{module_name}' not found")


def collect_func_dependencies(func_name: str, module_symbols: dict,
                              visited: set) -> set:
    """
    Recursively collects dependent functions.

    Parameters
    ----------
    func_name : str
        The function name to collect depdendencies for.
    module_symbols : dict
        Symbol table of the imported module
    visited : set
        Set of already-visited function names

    Returns
    -------
    set
        Set of all function names that ``func_name`` depends on
        (also include ``func_name`` itself).

    Examples
    --------
    >>> from physika.utils.import_manager import collect_func_dependencies
    >>> module_symbols = {
    ...     "dependent_f": {
    ...         "type": "function",
    ...         "value": {
    ...             "params": [("x", "ℝ")],
    ...             "return_type": "ℝ",
    ...             "body": ("pow", ("var", "x"), ("num", 2)),
    ...             "has_loop": False,
    ...             "statements": []
    ...         }
    ...     },
    ...     "square_f": {
    ...         "type": "function",
    ...         "value": {
    ...             "params": [("x", "ℝ")],
    ...             "return_type": "ℝ",
    ...             "body": ("call", "dependent_f", [("var", "x")]),
    ...             "has_loop": False,
    ...             "statements": []
    ...         }
    ...     }
    ... }
    >>> collect_func_dependencies("square_f", module_symbols, set()) == {
    ...     "square_f", "dependent_f"
    ... }
    True
    """
    if func_name in visited or func_name not in module_symbols:
        return visited
    visited.add(func_name)
    # AST node of function
    func_def = module_symbols[func_name]["value"]
    nodes_to_check = [func_def.get("body")] + func_def.get("statements", [])

    while nodes_to_check:
        node = nodes_to_check.pop()
        if isinstance(node, tuple):
            # find "call" AST
            if node[0] == "call":
                collect_func_dependencies(node[1], module_symbols, visited)
            nodes_to_check.extend(node[1:])
        elif isinstance(node, list):
            nodes_to_check.extend(node)

    return visited


def resolve_imports(local_program_ast: list, source_file_path: Path) -> list:
    """
    Resolve imported Physika symbols from external ``.phyk`` modules.

    Parameters
    ----------
    local_program_ast: list
        Parsed AST nodes of the current Physika source file.
    source_file_path: Path
        current file path

    Returns
    -------
    list
        Updated AST with imported definitions resolved and appended.

    Examples
    --------
    >>> from pathlib import Path
    >>> from physika.utils.import_manager import resolve_imports
    >>> local_program_ast = [
    ...     ('import', 'factorial', ['fact']),
    ...     ('expr', ('call', 'fact', [('num', 1.0)]), 0)
    ... ]
    >>> source_file_path = Path(
    ...     "examples/example_import_statement.phyk"
    ... )
    >>> resolved = resolve_imports(
    ...     local_program_ast,
    ...     source_file_path
    ... )
    >>> resolved[0][0]
    'func_def'
    >>> resolved[0][1]
    'fact'
    """
    extra_nodes: list[tuple] = []
    for node in local_program_ast:
        if isinstance(node, tuple) and node[0] == "import":
            module_name = node[1]
            symbols = node[2]
            # load and parse the module
            module_path = find_module(module_name, source_file_path)
            with open(module_path, "r", encoding="utf-8") as f:
                source = f.read()

            saved_symbols = symbol_table.copy()
            module_ast = parser.parse(source, lexer=lexer)
            module_symbols = symbol_table.copy()
            # restore original symbol table
            symbol_table.clear()
            symbol_table.update(saved_symbols)
            for symbol in symbols:
                # collect dependencies if it's a function
                deps = collect_func_dependencies(symbol, module_symbols, set())

                found = False
                for module_node in module_ast:
                    if (isinstance(module_node, tuple) and len(module_node) > 1
                            and isinstance(module_node[1], str)
                            and (module_node[1] == symbol
                                 or module_node[1] in deps)):  # noqa
                        name = module_node[1]

                        if name == symbol:
                            found = True

                        if module_node in extra_nodes:
                            continue

                        # Replace an earlier declaration of the same name.
                        existing_index = next(
                            (i for i, node in enumerate(extra_nodes)
                             if isinstance(node, tuple) and len(node) > 1
                             and node[1] == name), None)
                        if existing_index is None:
                            extra_nodes.append(module_node)
                        else:
                            extra_nodes[existing_index] = module_node

                        if name in module_symbols:
                            symbol_table[name] = module_symbols[name]

                if not found:
                    raise ImportError(
                        f"cannot import '{symbol}' from '{module_name}'")
        else:
            extra_nodes.append(node)
    return extra_nodes
