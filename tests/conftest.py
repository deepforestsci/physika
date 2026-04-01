from pathlib import Path
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast
from physika.codegen import from_ast_to_torch

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def exec_phyk(stem: str) -> dict:
    """
    Helper function to execute a .phyk file and return the resulting namespace
    ``ns`` dict.
    """
    source = (EXAMPLES_DIR / f"{stem}.phyk").read_text()
    symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = parser.parse(source, lexer=lexer)
    unified = build_unified_ast(program_ast, symbol_table)
    code = from_ast_to_torch(unified, print_code=False)
    ns: dict = {}
    exec(code, ns)
    return ns
