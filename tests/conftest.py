from pathlib import Path
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast
from physika.codegen import from_ast_to_torch
from physika.utils.import_manager import resolve_imports
from io import StringIO
from contextlib import redirect_stdout
from typing import Optional

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


def type_errors(src: str, phyk_file: Optional[str] = None) -> list:
    """
    Parse Physika source string, run the type checker and return the list of
    error strings if any.
    """
    import physika.parser as pm
    from physika.lexer import lexer
    from physika.type_checker import TypeChecker
    pm.symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = pm.parser.parse(src, lexer=lexer)
    if phyk_file is not None:
        program_ast = resolve_imports(
            program_ast,
            Path(phyk_file).resolve(),
        )
    ast = build_unified_ast(program_ast, pm.symbol_table)
    return TypeChecker(ast).run()


def run_phyk(src: str, phyk_file: Optional[str] = None) -> dict:
    """
    Helper function to parse, emits codegen, and exec a Physika source
    string.
    """
    import physika.parser as pm
    from physika.lexer import lexer
    pm.symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = pm.parser.parse(src, lexer=lexer)
    if phyk_file is not None:
        program_ast = resolve_imports(
            program_ast,
            Path(phyk_file).resolve(),
        )
    ast = build_unified_ast(
        program_ast,
        pm.symbol_table,
    )
    code = from_ast_to_torch(ast, print_code=False)
    ns: dict = {}
    exec(code, ns)
    return ns


def capture_output(src):
    """Helper function to capture the full output of Physika program."""
    buf = StringIO()
    with redirect_stdout(buf):
        run_phyk(src)
    return buf.getvalue()
