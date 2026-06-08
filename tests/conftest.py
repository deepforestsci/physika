from pathlib import Path
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast
from physika.codegen import from_ast_to_torch
from physika.utils.import_manager import resolve_imports

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def exec_phyk(stem: str) -> dict:
    """
    Helper function to execute a .phyk file and return the resulting namespace
    ``ns`` dict.
    """
    phyk_file = (EXAMPLES_DIR / f"{stem}.phyk")
    source = phyk_file.read_text()

    symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = parser.parse(source, lexer=lexer)

    if any(
            isinstance(node, tuple) and node[0] == "import"
            for node in program_ast):
        program_ast = resolve_imports(program_ast, phyk_file.resolve())
    unified = build_unified_ast(program_ast, symbol_table)
    code = from_ast_to_torch(unified, print_code=False)
    ns: dict = {}
    exec(code, ns)
    return ns
