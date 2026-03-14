from physika.utils.ast_utils import build_unified_ast
from tests.test_codegen import parse_source_to_ast
import pytest

def test_greek_and_scientific():
    """"""
    src = """
α: R = 1.0
β: R = 2.0
γ: R = 1e-7
"""
    ast = parse_source_to_ast(src)
    program_str = str(ast["program"])
    assert "α" in program_str
    assert "β" in program_str
    assert "γ" in program_str
    assert "1e-07" in program_str