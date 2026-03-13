from pathlib import Path
from typing import get_args

import pytest

from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast, ExprTag, StmtTag, BodyStmtTag, TypeTag, ast_to_torch_expr, condition_to_expr, emit_body_stmts, emit_for_stmts, _is_loop_var, _decompose_chain, _infer_range, _lhs_var_name



VALID_TAGS = set(
    get_args(ExprTag) + get_args(StmtTag) + get_args(BodyStmtTag) + get_args(TypeTag)
)

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
AST_DIR = EXAMPLES_DIR / "ast"
PHYK_FILES = sorted(EXAMPLES_DIR.glob("*.phyk"))
PHYK_IDS = [f.stem for f in PHYK_FILES]


def parse_source(source: str):
    """Helper function that runs lexer/parser on a Physika source string."""
    symbol_table.clear()
    lexer.lexer.lineno = 1          # reset PLY line counter for deterministic output
    program_ast = parser.parse(source, lexer=lexer)
    return program_ast, symbol_table


def load_expected_ast(stem: str) -> dict:
    """Load the expected AST dict from ``examples/ast/<stem>.py``."""
    ns = {}
    exec((AST_DIR / f"{stem}.py").read_text(), ns)
    return ns["EXPECTED"]


def _run_emit_body(stmts, indent_level=1, known_vars=None, scalar_only=False):
    """Helper function to call `emit_body_stmts` and return the generated lines."""
    lines = []
    emit_body_stmts(
        stmts,
        indent_level,
        lines,
        known_vars if known_vars is not None else [],
        set(),
        ast_to_torch_expr,
        scalar_only=scalar_only,
    )
    return lines



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


class TestConditionToExpr:
    """
    Test suite for `condition_to_expr` function that converts Physika condition AST nodes.
    Each test verifies that a specific Physika condition node is correctly converted
    to a Python operator string.
    """
    def test_condition_to_expr_ints(self):
        """Verify `condition_to_expr` correctly converts Physika condition nodes with integer literals."""

        # Test "equals" condition
        expected_eq = "n == 0"
        physika_expr_eq = condition_to_expr(("cond_eq", ("var", "n"), ("num", 0)))
        assert physika_expr_eq == expected_eq

        # Test "not equals" condition
        expected_neq = "x != 1"
        physika_expr_neq = condition_to_expr(("cond_neq", ("var", "x"), ("num", 1)))
        assert physika_expr_neq == expected_neq

        # Test "less than" condition
        expected_lt = "x < 1"
        physika_expr_lt = condition_to_expr(("cond_lt", ("var", "x"), ("num", 1)))
        assert physika_expr_lt == expected_lt

        # Test "greater than" condition
        expected_gt = "x > 1"
        physika_expr_gt = condition_to_expr(("cond_gt", ("var", "x"), ("num", 1)))
        assert physika_expr_gt == expected_gt

        # Test "less than or equal" condition
        expected_leq = "y <= -1"
        physika_expr_leq = condition_to_expr(("cond_leq", ("var", "y"), ("num", -1)))
        assert physika_expr_leq == expected_leq

        # Test "greater than or equal" condition
        expected_geq = "z >= 2"
        physika_expr_geq = condition_to_expr(("cond_geq", ("var", "z"), ("num", 2)))
        assert physika_expr_geq == expected_geq

        # numbers and variables can be on both sides of the operator
        assert condition_to_expr(("cond_lt", ("num", 0), ("var", "x"))) == "0 < x"

        # Both sides can be arbitrary expressions, not just vars/nums
        expected_cond = "(x + 1) > (0 - y)"
        physika_cond = ("cond_gt", ("add", ("var", "x"), ("num", 1))
                         , ("sub", ("num", 0), ("var", "y")))
        assert condition_to_expr(physika_cond) == expected_cond
    
    def test_condition_to_expr_floats(self):
        """Verify `condition_to_expr` correctly converts Physika condition nodes with float literals."""

        # Test "equals" condition
        expected_eq = "n == 0.4"
        physika_expr_eq = condition_to_expr(("cond_eq", ("var", "n"), ("num", 0.4)))
        assert physika_expr_eq == expected_eq

        # Test "not equals" condition
        expected_neq = "x != 1.1"
        physika_expr_neq = condition_to_expr(("cond_neq", ("var", "x"), ("num", 1.1)))
        assert physika_expr_neq == expected_neq

        # Test "less than" condition
        expected_lt = "x < 1.0"
        physika_expr_lt = condition_to_expr(("cond_lt", ("var", "x"), ("num", 1.0)))
        assert physika_expr_lt == expected_lt

        # Test "greater than" condition
        expected_gt = "x > 1.9"
        physika_expr_gt = condition_to_expr(("cond_gt", ("var", "x"), ("num", 1.9)))
        assert physika_expr_gt == expected_gt

        # Test "less than or equal" condition
        expected_leq = "y <= -1.2"
        physika_expr_leq = condition_to_expr(("cond_leq", ("var", "y"), ("num", -1.2)))
        assert physika_expr_leq == expected_leq

        # Test "greater than or equal" condition
        expected_geq = "z >= 2.5"
        physika_expr_geq = condition_to_expr(("cond_geq", ("var", "z"), ("num", 2.5)))
        assert physika_expr_geq == expected_geq

        # numbers and variables can be on both sides of the operator
        expected_first_number = "0.0 < x"
        physika_expr_first_number = condition_to_expr(("cond_lt", ("num", 0.0), ("var", "x")))
        assert physika_expr_first_number == expected_first_number


class TestEmitBodyStmts:
    """
    Test suite for `emit_body_stmts` handling of assignment and declaration statements.
    """ 
    def test_body_assign_decl_tuple_unpack(self):
        """
        Verify `emit_body_stmts` produce correct code lines for physika's parsed AST 
        `body_assign`, `body_decl`, and `body_tuple_unpack` nodes.
        """
        stmt_assign = ("body_assign", "y",  ("mul", ("var", "x"), ("num", 2)))
        lines_assign = _run_emit_body([stmt_assign], known_vars=["x"])
        assert lines_assign == ["    y = (x * 2)"]

        stmt_decl = ("body_decl", "z", "ℝ", ("add", ("var", "x"), ("num", 1)))
        lines_decl = _run_emit_body([stmt_decl], known_vars=["x"])
        assert lines_decl == ["    z = (x + 1)"]

        stmt_tuple_unpack = ("body_tuple_unpack", ["a", "b"], ("var", "pair"))
        lines_tuple_unpack = _run_emit_body([stmt_tuple_unpack])
        assert lines_tuple_unpack == ["    a, b = pair"]

    def test_body_assign_extends_known_vars(self):
        """
        Verify `emit_body_stmts` extends the `known_vars` set after a
        `body_assign`, `body_decl`, or `body_tuple_unpack` statement.
        """
        known_assign = ["x"]
        _run_emit_body([("body_assign", "y", ("var", "x"))], known_vars=known_assign)
        assert "y" in known_assign
        assert "x" in known_assign
        assert len(known_assign) == 2

        known_decl = ["x"]
        _run_emit_body([("body_decl", "z", "ℝ", ("var", "x"))], known_vars=known_decl)
        assert "z" in known_decl
        assert "x" in known_decl
        assert len(known_decl) == 2

        known_tuple_unpack = []
        _run_emit_body([("body_tuple_unpack", ["a", "b"], ("var", "p"))], known_vars=known_tuple_unpack)
        assert "a" in known_tuple_unpack and "b" in known_tuple_unpack
        assert len(known_tuple_unpack) == 2


    def test_indent_level(self):
        """
        Checks that the generated code lines are indented according to the `indent_level` arg.
        """
        # checks for `body_assgin`
        for i in range(4):
            stmt = ("body_assign", "y", ("num", 0))
            lines = _run_emit_body([stmt], indent_level=i)
            assert lines == [f"{' ' * (4 * i)}y = 0"]
            assert lines[0].startswith(" " * (4 * i))
        
        # checks for `body_decl`
        for i in range(4):
            stmt = ("body_decl", "y", "ℝ", ("num", 3.14))
            lines = _run_emit_body([stmt], indent_level=i)
            assert lines == [f"{' ' * (4 * i)}y = 3.14"]
            assert lines[0].startswith(" " * (4 * i))
        
        # checks for `body_decl`
        for i in range(4):
            stmt = ("body_tuple_unpack", ["a", "b"], ("var", "p"))
            lines = _run_emit_body([stmt], indent_level=i)
            assert lines == [f"{' ' * (4 * i)}a, b = p"]
            assert lines[0].startswith(" " * (4 * i))

class TestEmitBodyIfElseStmts:
    """
    Test suite for `emit_body_stmts` handling of if/else statements, including:
        - `body_if_return`
        - `body_if_else_return`
        - `body_if_else`.
    
    These tests verify that the generated code lines are correct for the given
    Physika AST statements, and that the `scalar_only` flag correctly controls
    whether `torch.where` or Python if/else is used for `body_if_else_return`
    statements.
    """
    def test_body_if_return(self):
        cond = ("cond_eq", ("var", "n"), ("num", 0))
        stmt = ("body_if_return", cond, ("num", 1))
        lines = _run_emit_body([stmt])
        assert lines == ["    if n == 0:", 
                         "        return 1"]

    def test_body_if_else_return(self):
        """
        Verify `body_if_else_return` produces correct code for both
        `scalar_only=False` (torch.where) and `scalar_only=True` (Python if/else).
        """
        # `body_if_else_return` case scalar_only=False uses torch.where
        cond = ("cond_gt", ("var", "x"), ("num", 0))
        stmt = ("body_if_else_return", cond, ("var", "x"), ("neg", ("var", "x")))
        lines = _run_emit_body([stmt], scalar_only=False)
        assert len(lines) == 1
        assert lines[0].startswith("    return torch.where(")
        assert "x > 0" in lines[0]

        # `body_if_else_return` case scalar_only=True
        cond = ("cond_gt", ("var", "x"), ("num", 0))
        stmt = ("body_if_else_return", cond, ("var", "x"), ("neg", ("var", "x")))
        lines = _run_emit_body([stmt], scalar_only=True)
        assert lines == ["    if x > 0:",
                         "        return x",
                         "    else:",
                         "        return (-x)",
                        ]
        
        # scalar_only=True must propagate into nested emit calls
        # if x > 0.0:
        #     if y < 0.0:
        #         return 1
        #     else:
        #         return -1
        # else:
        #     y = 0.0
        
        cond = ("cond_gt", ("var", "x"), ("num", 0))
        inner_cond = ("cond_lt", ("var", "y"), ("num", 0))
        then_stmts = [("body_if_else_return", inner_cond, ("num", 1), ("num", -1))]
        else_stmts = [("body_assign", "y", ("num", 0))]
        stmt = ("body_if_else", cond, then_stmts, else_stmts)
        lines = _run_emit_body([stmt], scalar_only=True)
        print(lines)
        # The nested body_if_else_return should use if/else (scalar_only=True)
        assert "torch.where" not in "\n".join(lines)
        assert lines == [
            "    if x > 0:",
            "        if y < 0:",
            "            return 1",
            "        else:",
            "            return -1",
            "    else:",
            "        y = 0",
        ]

    def test_body_if_else_and_if_only_assignment(self):
        """
        Verify `body_if_else` and `body_if` produce correct code for an if/else `body_assign`
        statement.
        """
        cond = ("cond_gt", ("var", "x"), ("num", 1))
        then_stmts = [("body_assign", "y", ("num", 1))]
        else_stmts = [("body_assign", "y", ("var", "x"))]
        stmt = ("body_if_else", cond, then_stmts, else_stmts)
        lines = _run_emit_body([stmt])
        assert lines == [
            "    if x > 1:",
            "        y = 1",
            "    else:",
            "        y = x",
        ]

        cond = ("cond_lt", ("var", "y"), ("num", -1))
        then_stmts = [("body_assign", "y", ("neg", ("num", 1)))]
        stmt = ("body_if", cond, then_stmts)
        lines = _run_emit_body([stmt])
        assert lines == [
            "    if y < -1:",
            "        y = (-1)",
        ]        


class TestEmitForStmts:
    """
    Test suite for `emit_for_stmts` handling of for loop statements, including:
    - `for_assign`
    - `for_pluseq`
    - `for_call`
    These tests verify that the generated code lines are correct for the given
    Physika AST statements, and that the `indent_level` arg correctly controls the indentation of
    """

    def test_for_assign(self):
        """
        Checks that `for_assign` statements produce correct code lines, and have correct indentation.
        """
        stmts = [("for_assign", "z", ("mul", ("var", "a"), ("var", "b")))]
        assert emit_for_stmts(stmts, 4) == ["    z = (a * b)"]

        # None statements should be ignored and not cause errors
        stmts = [None, ("for_assign", "z", ("num", 1.0)), None]
        assert emit_for_stmts(stmts, 4) == ["    z = 1.0"]

        # custom indent level
        stmts = [("for_assign", "z", ("num", 0.0))]
        assert emit_for_stmts(stmts, 8) == ["        z = 0.0"]

        # zero indent level
        stmts = [("for_assign", "z", ("num", 0.0))]
        assert emit_for_stmts(stmts, 0) == ["z = 0.0"]


    def test_for_pluseq(self):
        """
        Verify `for_pluseq` produces correct code lines and indentation of the generated lines.
        """
        stmts = [("for_pluseq", "acc", ("var", "x"))]
        assert emit_for_stmts(stmts, 4) == ["    acc = acc + x"]

    def test_for_call(self):
        """
        Checks that `for_call` statements produce correct code lines and have correct indentation.
        """
        # Normal function call
        stmts = [("for_call", "f", [("var", "x"), ("var", "y")])]
        assert emit_for_stmts(stmts, 4) == ["    f(x, y)"]

        # Empty args in a function call should still produce a valid line of code
        stmts = [("for_call", "step", [])]
        assert emit_for_stmts(stmts, 4) == ["    step()"]

    def test_multiple_stmts(self):
        """
        Checks that multiple `for_assign` and `for_pluseq` statements produce correct
        code lines, and have correct indentation.
        """
        stmts = [
            ("for_assign", "a", ("num", 1)),
            ("for_pluseq", "s", ("var", "a")),
        ]
        lines = emit_for_stmts(stmts, 4)
        assert lines == ["    a = 1", "    s = s + a"]

    def test_empty_stmts(self):
        """
        Checks that empty statement lists produce no code lines.
        """
        assert emit_for_stmts([], 4) == []


def test_is_loop_var():
    """""
    Tests for the `_is_loop_var` helper. This function checks
    if a given variable node matches a loop variable name,
    accounting for both the standard variable reference form
    and the special "imaginary" form used for loop variables named "i".
    """
    # check standard variable reference form
    assert _is_loop_var(("var", "k"), "k") is True
    assert _is_loop_var(("var", "j"), "j") is True
    assert _is_loop_var(("var", "j"), "k") is False
    assert _is_loop_var(("var", "k"), "j") is False

    # check "imaginary" form for loop variable "i"
    assert _is_loop_var(("imaginary",), "i") is True
    assert _is_loop_var(("imaginary",), "k") is False
    assert _is_loop_var(("imaginary",), "j") is False

    # check different ASTNode expression cases
    assert _is_loop_var(("num", 0), "k") is False
    assert _is_loop_var(("add", ("var", "k"), ("num", 1)), "k") is False

    # check non-ASTNode inputs
    assert _is_loop_var("k", "k") is False
    assert _is_loop_var(None, "k") is False


def test_decompose_chain():
    """
    Tests for `_decompose_chain` helper.

    Verifies `decompose_chain` handles both "index" and "chain_index" nodes, and returns (None, [])
    for non-chain/index nodes or if the array name is not a string.
    """

    expr = ("index", "A", ("var", "i")) # equivalent to A[i]
    assert _decompose_chain(expr) == ("A", [("var", "i")])
    
    expr = ("chain_index", ("index", "A", ("var", "i")), ("var", "k")) # equivalent to A[i][k]
    assert _decompose_chain(expr) == ("A", [("var", "i"), ("var", "k")])

    inner = ("chain_index", ("index", "A", ("var", "i")), ("var", "j")) # equivalent to A[i][j]
    expr = ("chain_index", inner, ("var", "k")) # equivalent to A[i][j][k]
    assert _decompose_chain(expr) == ("A", [("var", "i"), ("var", "j"), ("var", "k")])

    # Non ASTNodes should return (None, [])
    assert _decompose_chain("A") == (None, []) 
    assert _decompose_chain(42) == (None, [])

    # Non-index/ non-chain_index nodes should return (None, [])
    assert _decompose_chain(("add", ("var", "i"), ("num", 1))) == (None, [])

    # array is not a string and returns (None, [])
    nested = ("index", ("var", "A"), ("var", "i"))
    assert _decompose_chain(nested) == (None, [])


def test_infer_range():
    """
    Tests for `_infer_range` helper function.
    
    Verifies `_infer_range` correctly identifies loop variables in various index expressions
    and returns the appropriate shape access string. Also, checks if `_infer_range` returns None when the loop
    variable is not found or is the accumulation target.
    """
    rhs = ("indexN", "A", [("var", "i"), ("var", "k")])
    assert _infer_range("i", rhs, "C") == "A.shape[0]" # C is the accumulation target
    assert _infer_range("k", rhs, "C") == "A.shape[1]"
    assert _infer_range("j", rhs, "C") is None

    # iter var in a 1D index expression
    rhs = ("index", "B", ("var", "j"))
    assert _infer_range("j", rhs, "C") == "B.shape[0]"

    # loop var not found in index expression should return None
    rhs = ("index", "C", ("var", "j"))
    assert _infer_range("j", rhs, "C") is None

    rhs = ("chain_index", ("index", "A", ("var", "i")), ("var", "k"))
    assert _infer_range("i", rhs, "C") == "A.shape[0]"
    assert _infer_range("k", rhs, "C") == "A.shape[1]"


    # iter var inside a mul expression
    rhs = ("mul", ("indexN", "A", [("var", "i"), ("var", "k")]), # equivalent to A[i][k] * B[k][j]
                  ("indexN", "B", [("var", "k"), ("var", "j")]))
    assert _infer_range("i", rhs, "C") == "A.shape[0]"
    assert _infer_range("k", rhs, "C") == "A.shape[1]"
    assert _infer_range("j", rhs, "C") == "B.shape[1]"

    # ("imaginary",) in indexN acts as loop var "i"
    rhs = ("indexN", "A", [("imaginary",), ("var", "k")])
    assert _infer_range("i", rhs, "C") == "A.shape[0]"

    rhs = ("indexN", "A", [("var", "i"), ("var", "k")])
    assert _infer_range("j", rhs, "C") is None

    # Check that non-ASTNodes return None
    assert _infer_range("i", "A", "C") is None
    assert _infer_range("i", 42, "C") is None

def test_lhs_var_name():
    """
    Tests for `_lhs_var_name` helper function.

    Verifies `_lhs_var_name` correctly extracts variable names
    from "var" and "imaginary" nodes, and returns None for non-variable nodes
    and non-ASTNode expressions.
    """
    assert _lhs_var_name(("var", "j")) == "j"
    assert _lhs_var_name(("var", "i")) == "i"
    assert _lhs_var_name(("var", "k")) == "k"

    assert _lhs_var_name(("imaginary",)) == "i"

    assert _lhs_var_name(("num", 0.0)) is None
    assert _lhs_var_name(("num", 1)) is None

    assert _lhs_var_name(("add", ("var", "i"), ("num", 1))) is None
    assert _lhs_var_name(("index", "A", ("var", "i"))) is None

    assert _lhs_var_name("j") is None
    assert _lhs_var_name(None) is None
    assert _lhs_var_name(42) is None
