from pathlib import Path
from typing import get_args

import pytest

from lexer import lexer
from parser import parser, symbol_table
from utils.ast_utils import build_unified_ast, ExprTag, StmtTag, BodyStmtTag, TypeTag


VALID_TAGS = set(
    get_args(ExprTag) + get_args(StmtTag) + get_args(BodyStmtTag) + get_args(TypeTag)
)

DATA_DIR = Path(__file__).parent / "data"
AST_DIR = DATA_DIR / "ast"
PHYK_FILES = sorted(DATA_DIR.glob("*.phyk"))
PHYK_IDS = [f.stem for f in PHYK_FILES]


def parse_source(source: str):
    """Helper function that runs lexer/parser on a Physika source string."""
    symbol_table.clear()
    lexer.lexer.lineno = 1          # reset PLY line counter for deterministic output
    program_ast = parser.parse(source, lexer=lexer)
    return program_ast, symbol_table


def load_expected_ast(stem: str) -> dict:
    """Helper function that load the expected AST dict from ``data/ast/<expected>.py``."""
    ns = {}
    exec((AST_DIR / f"{stem}.py").read_text(), ns)
    return ns["EXPECTED"]


def is_ast_node(node) -> bool:
    """Helper function that return True if *node* is a valid ``ASTNode`` leaf or composite.
    """
    if node is None or isinstance(node, (str, int, float, bool)):
        return True
    if isinstance(node, list):
        return all(is_ast_node(child) for child in node)
    if isinstance(node, tuple):
        return all(is_ast_node(child) for child in node)
    return False


def is_unified_ast(ast) -> bool:
    """Validate ``dict[str, Union[dict[str, ASTNode], list[ASTNode]]]``.

    Checks
    ------
    1. Top-level is a dict with keys ``{"functions", "classes", "program"}``.
    2. ``"functions"`` and ``"classes"`` are ``dict[str, dict]`` whose nested
       field values are valid ASTNodes.
    3. ``"program"`` is a ``list`` of valid ASTNodes.
    """
    if not isinstance(ast, dict):
        return False
    if set(ast.keys()) != {"functions", "classes", "program"}:
        return False

    for key in ("functions", "classes"):
        section = ast[key]
        if not isinstance(section, dict):
            return False
        for name, definition in section.items():
            if not isinstance(name, str):
                return False
            if not isinstance(definition, dict):
                return False
            for field_val in definition.values():
                if not is_ast_node(field_val):
                    return False

    if not isinstance(ast["program"], list):
        return False
    return all(is_ast_node(stmt) for stmt in ast["program"])




@pytest.mark.parametrize("phyk_file", PHYK_FILES, ids=PHYK_IDS)
def test_build_unified_ast(phyk_file):
    """Verify build_unified_ast returns dict[str, Union[dict[str, ASTNode], list[ASTNode]]]."""
    src = phyk_file.read_text()
    program_ast, sym_tb = parse_source(src)
    ast = build_unified_ast(program_ast, sym_tb)
    assert is_unified_ast(ast)


@pytest.mark.parametrize("phyk_file", PHYK_FILES, ids=PHYK_IDS)
def test_program_stmts_have_valid_tags(phyk_file):
    """Verify every program statement tuple starts with a known tag."""
    src = phyk_file.read_text()
    program_ast, sym_tb = parse_source(src)
    ast = build_unified_ast(program_ast, sym_tb)
    for stmt in ast["program"]:
        assert isinstance(stmt, tuple)


@pytest.mark.parametrize("phyk_file", PHYK_FILES, ids=PHYK_IDS)
def test_function_bodies_are_ast_nodes(phyk_file):
    """Verify every function body is a valid ASTNode."""
    src = phyk_file.read_text()
    program_ast, sym_tb = parse_source(src)
    ast = build_unified_ast(program_ast, sym_tb)
    for name, func_def in ast["functions"].items():
        assert is_ast_node(func_def["body"]), f"Function {name!r} body is not a valid ASTNode"


@pytest.mark.parametrize("phyk_file", PHYK_FILES, ids=PHYK_IDS)
def test_class_bodies_are_ast_nodes(phyk_file):
    """Verify every class forward body is a valid ASTNode."""
    src = phyk_file.read_text()
    program_ast, sym_tb = parse_source(src)
    ast = build_unified_ast(program_ast, sym_tb)
    for name, class_def in ast["classes"].items():
        assert is_ast_node(class_def["body"])
        if class_def.get("has_loss"):
            assert is_ast_node(class_def["loss_body"])


@pytest.mark.parametrize("phyk_file", PHYK_FILES, ids=PHYK_IDS)
def test_ast_matches_expected(phyk_file):
    """Verify build_unified_ast output matches the expected AST in data/ast/."""
    src = phyk_file.read_text()
    program_ast, sym_tb = parse_source(src)
    actual = build_unified_ast(program_ast, sym_tb)
    expected = load_expected_ast(phyk_file.stem)
    assert actual == expected
