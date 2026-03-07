"""Unit tests for symbolic"""

from codegen import from_ast_to_torch
from utils.ast_utils import build_unified_ast
from parser import parser, symbol_table
from lexer import lexer
from pathlib import Path
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
TORCH_CODE_DIR = EXAMPLES_DIR / "torch_code"


def parse_source_to_ast(source: str) -> dict:
    symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = parser.parse(source, lexer=lexer)
    return build_unified_ast(program_ast, symbol_table)

def test_symbolic_example():
    """test symbolic example file matches reference torch code"""
    phyk_file = EXAMPLES_DIR / "example_symbolic.phyk"
    src = phyk_file.read_text()
    code = from_ast_to_torch(parse_source_to_ast(src), print_code=False)
    torch_file = TORCH_CODE_DIR / "example_symbolic.py"
    expected = torch_file.read_text()
    assert code == expected