import sys
import importlib.util
import physika.runtime
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.type_checker import TypeChecker
from physika.codegen import from_ast_to_torch
from physika.utils.print_utils import print_type_check_results
from physika.utils.ast_utils import build_unified_ast
from physika.utils.import_manager import resolve_imports
from pathlib import Path


def inject_sibling_python_module(source_file_path: Path) -> None:
    """Give a ``.phyk`` file access to Python code defined next to it.

    If ``<name>.py`` exists alongside ``<name>.phyk``, load it and merge its
    top-level names into ``physika.runtime`` so the generated torch code
    (which executes with ``vars(physika.runtime)`` as its globals) can call
    them directly, e.g. plain functions doing file I/O, ASE, numpy, etc.
    that physika's own DSL/type system can't express.

    Parameters
    ----------
    source_file_path : Path
        Absolute path of the ``.phyk`` file being run.
    """
    sibling_path = source_file_path.with_suffix(".py")
    if not sibling_path.exists():
        return

    sibling_dir = str(sibling_path.parent)
    if sibling_dir not in sys.path:
        sys.path.insert(0, sibling_dir)

    spec = importlib.util.spec_from_file_location(sibling_path.stem, sibling_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Python module '{sibling_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, value in vars(module).items():
        if not name.startswith("__"):
            setattr(physika.runtime, name, value)


def main():
    print_code = "--print-code" in sys.argv
    print_ast = "--print-ast" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    source_file_path = Path(args[0]).resolve()
    with open(args[0], "r", encoding="utf-8") as f:
        source = f.read()

    inject_sibling_python_module(source_file_path)

    # Parse tokens to AST
    local_program_ast = parser.parse(source, lexer=lexer)

    local_program_ast = resolve_imports(local_program_ast, source_file_path)

    # Build unified AST (I think this can be done in parser)
    unified_ast = build_unified_ast(local_program_ast,
                                    symbol_table,
                                    print_ast=print_ast)

    # Type checking
    type_status = TypeChecker(unified_ast).run()
    print_type_check_results(type_status)

    # Generate PyTorch code and execute it
    generated_code = from_ast_to_torch(unified_ast, print_code=print_code)
    exec(generated_code, vars(physika.runtime))


if __name__ == "__main__":
    main()
