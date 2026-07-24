"""Regenerate ast/ and torch_code/ reference files for .phyk files.

Usage: python gen_refs.py <stem> [<stem> ...]
e.g.:  python gen_refs.py dft_operators

Finds <stem>.phyk in examples/ first, then tutorials/, and writes the
two reference files next to it (<dir>/ast/<stem>.py and
<dir>/torch_code/<stem>.py).
"""
import pprint
import sys
from pathlib import Path

from physika.codegen import from_ast_to_torch
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast
from physika.utils.import_manager import resolve_imports

ROOT = Path(__file__).parent


def gen(stem: str) -> None:
    for base in (ROOT / "examples", ROOT / "tutorials"):
        phyk = base / f"{stem}.phyk"
        if phyk.exists():
            break
    else:
        sys.exit(f"error: {stem}.phyk not found in examples/ or tutorials/")

    source = phyk.read_text(encoding="utf-8")
    symbol_table.clear()
    lexer.lexer.lineno = 1  # reset PLY line counter, like the tests do
    program_ast = parser.parse(source, lexer=lexer)
    if any(isinstance(n, tuple) and n[0] == "import" for n in program_ast):
        program_ast = resolve_imports(program_ast, phyk.resolve())
    unified = build_unified_ast(program_ast, symbol_table)
    code = from_ast_to_torch(unified, print_code=False)

    ast_file = base / "ast" / f"{stem}.py"
    torch_file = base / "torch_code" / f"{stem}.py"
    ast_file.write_text(
        "EXPECTED = " + pprint.pformat(unified, width=79, sort_dicts=False)
        + "\n", encoding="utf-8")
    torch_file.write_text(code, encoding="utf-8")
    print(f"wrote {ast_file}")
    print(f"wrote {torch_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for stem_arg in sys.argv[1:]:
        gen(stem_arg)
