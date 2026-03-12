from physika.codegen import from_ast_to_torch
from physika.utils.ast_utils import build_unified_ast, ast_uses_func
from physika.parser import parser, symbol_table
from physika.lexer import lexer
from pathlib import Path
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

def parse_source_to_ast(source: str) -> dict:
    """Run lexer/parser and build_unified_ast on a Physika source string."""
    symbol_table.clear()
    lexer.lexer.lineno = 1          # reset PLY line counter for deterministic output
    program_ast = parser.parse(source, lexer=lexer)
    return build_unified_ast(program_ast, symbol_table)

def test_grad_calls_in_function_statements():
    """compute_grad must be imported when grad is used in function statements"""
    phyk_file = EXAMPLES_DIR / "example_check_gradients.phyk"
    src = phyk_file.read_text()
    ast = parse_source_to_ast(src)
    code_phyk = from_ast_to_torch(ast, print_code=False)
    assert "from physika.runtime import compute_grad" in code_phyk

    # check grad calls inside function statements
    func_section = code_phyk.split("# === Functions ===")[1].split("# === Program ===")[0]
    assert "compute_grad" in func_section, "compute grad not found in generated torch code"

    grad_in_ast = any(ast_uses_func(stmt, "grad") for stmt in ast["functions"]["f"]["statements"])
    assert grad_in_ast, "grad call not found in generated ast code"

def test_grad_correctness():
    """test the correctness of gradients"""
    import torch
    phyk_file = EXAMPLES_DIR / "example_check_gradients.phyk"
    src = phyk_file.read_text()
    ast = parse_source_to_ast(src)
    code = from_ast_to_torch(ast, print_code=False)
        
    local = {}
    exec(code, local)
    result = local["f"](torch.tensor([1.0, 2.0], requires_grad=True))
    assert torch.allclose(result, torch.tensor([2.0, 4.0]))