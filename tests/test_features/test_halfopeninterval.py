from physika.features.halfopeninterval import HalfOpenIntervalLoop
from physika.utils.ast_utils import ast_to_torch_expr, build_unified_ast
import pytest
from tests.conftest import exec_phyk

fwd_rules = HalfOpenIntervalLoop().forward_rules()
emit_loop_right_open = fwd_rules["for_loop_right_open"]
emit_loop_left_open = fwd_rules["for_loop_left_open"]


def parse_physika(src):
    """Parse a Physika source string and return the unified AST."""
    import physika.parser as pm
    from physika.lexer import lexer
    pm.symbol_table.clear()
    lexer.lexer.lineno = 1  # reset PLY line counter for deterministic output
    program_ast = pm.parser.parse(src, lexer=lexer)
    return build_unified_ast(program_ast, pm.symbol_table)


@pytest.fixture(scope="module")
def ns():
    """
    Execute half_open_interval_loops.phyk, build unified AST,
    execute; return namespace.
    """
    return exec_phyk("half_open_interval_loops")


class TestHalfOpenIntervalLoop:
    """
    Checks parser is emitting the correct AST nodes,
    that the forward rules have the correct keys and type
    rules catches proper errors.
    """

    def test_elf_name(self):
        """
        ELF is registered under the name 'half_open_inteval_loop'.
        """
        assert HalfOpenIntervalLoop.name == "half_open_inteval_loop"

    def test_parser_rules(self):
        """
        parser_rules should return two handlers.
        """
        rules = HalfOpenIntervalLoop().parser_rules()
        assert len(rules) == 8
        names = [r.__name__ for r in rules]
        assert "p_right_open_for_range" in names
        assert "p_for_statement_right_open_range" in names
        assert "p_left_open_for_range" in names
        assert "p_for_statement_left_open_for_range" in names
        assert "p_func_body_right_open_for_range" in names
        assert "p_func_body_left_open_for_range" in names
        assert "p_func_body_stmt_right_open_for_range" in names
        assert "p_func_body_stmt_left_open_for_range" in names

    def test_forward_rules_keys(self):
        """
        forward rules should return two rules.
        """
        rules = HalfOpenIntervalLoop().forward_rules()
        assert len(rules) == 2
        assert set(
            rules.keys()) == {"for_loop_right_open", "for_loop_left_open"}


class TestHalfOpenLoopForwardRules:
    """
    Tests for HalfOpenIntervalLoop forward_rules code generation.
    """

    def test_right_open_loop_emit(self):
        """
        Check that a right-open interval loop emits the correct Python code.
        """
        ast = parse_physika("for i ∈ [0, 4):\n"
                            "    total += i")
        node = ast["program"][0]
        assert node[0] == "for_loop_right_open"
        result = emit_loop_right_open(node, ast_to_torch_expr)
        assert result == ("for i in range(int(0), int(4)):\n"
                          "    total = total + i")

    def test_left_open_loop_emit(self):
        """
        Check that a left-open interval loop emits the correct Python code.
        """
        ast = parse_physika("for i ∈ (0, 4]:\n"
                            "    total += i\n")
        node = ast["program"][0]
        assert node[0] == "for_loop_left_open"
        result = emit_loop_left_open(node, ast_to_torch_expr)

        assert result == ("for i in range(int(0) + 1, int(4) + 1):\n"
                          "    total = total + i")


class TestHalfOpenPhysikaCode:
    """
    Tests for ``examples/half_open_interval_loops.phyk`` correctness
    """

    def test_program_level(self, ns):
        # Tests for program level code
        sum_right = ns["sum_right"]
        assert sum_right == 6

        sum_left = ns["sum_left"]
        assert sum_left == 10

        nested_sum = ns["nested_sum"]
        assert nested_sum == 24

    def test_function_level(self, ns):
        # Tests for function level code
        right_func_result = ns["right_func_result"]
        assert right_func_result == 6

        left_func_result = ns["left_func_result"]
        assert left_func_result == 10

        nested_func_result = ns["nested_func_result"]
        assert nested_func_result == 24

    def test_class_level(self, ns):
        # Tests for class level code
        class_results = ns["class_results"]
        assert class_results == 55

    def test_differentiability(self, ns):
        # Tests for differentiability
        dot_gradient = ns["dot_gradient"]
        assert dot_gradient == 10
