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
    >>> from physika.import_manager import find_module
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
    search_paths = [source_file_path.parent, Path(".")]
    for path in search_paths:
        file_name = path / f"{module_name}.phyk"
        if file_name.exists():
            return file_name
    raise ImportError(f"module '{module_name}' not found")


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
    >>> from physika.import_manager import resolve_imports
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
    extra_nodes = []
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
                # find the requested symbol in module AST
                found = False
                for module_node in module_ast:
                    if (isinstance(module_node, tuple) and len(module_node) > 1
                            and module_node[1] == symbol):
                        extra_nodes.append(module_node)
                        if symbol in module_symbols:
                            symbol_table[symbol] = module_symbols[symbol]
                        found = True
                        break
                if not found:
                    raise ImportError(
                        f"cannot import '{symbol}' from '{module_name}'")
        else:
            extra_nodes.append(node)
    return extra_nodes
