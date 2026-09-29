from physika.features.dictionary import DictionaryFeature
from physika.utils.ast_utils import ast_to_torch_expr
import torch
from tests.conftest import exec_phyk
import pytest

forward_rules = DictionaryFeature().forward_rules()
emit_dict = forward_rules["dict"]


@pytest.fixture(scope="module")
def dict_ns():
    """
    Namespace dict that contains variables, functions and classes
    when example/physika_dictionary.phyk is executed.
    """
    return exec_phyk("physika_dictionary")


class TestDictionaryFeature:
    """
    Checks parser is emitting the correct AST nodes,
    that the forward rules have the correct keys and type
    rules catches proper errors.
    """

    def test_elf_name(self):
        """
        ELF is registered under the name 'dictionary'.
        """
        assert DictionaryFeature.name == "dictionary"

    def test_parser_rules(self):
        """
        parser rules should return nine handlers.
        """
        rules = DictionaryFeature().parser_rules()
        assert len(rules) == 9
        names = [r.__name__ for r in rules]
        assert "p_type_dict" in names
        assert "p_factor_dict_empty" in names
        assert "p_factor_dict" in names
        assert "p_func_factor_dict_empty" in names
        assert "p_func_factor_dict" in names
        assert "p_dict_items_single" in names
        assert "p_dict_items_multi" in names
        assert "p_dict_items_newline" in names

    def test_forward_rules_keys(self):
        """
        forward rules should return one handler.
        """
        rules = DictionaryFeature().forward_rules()
        assert len(rules) == 1
        assert set(rules.keys()) == {"dict"}
        assert all(callable(v) for v in rules.values())


class TestEmitDict:
    """
    Test cases for ``emit_dict`` method.
    """

    def test_empty_dict_node(self):
        """
        Test case for empty dictionary node.
        """
        node = ("dict", [])
        result = emit_dict(node, ast_to_torch_expr)
        assert result == "{}"

    def test_simple_dict_node(self):
        """
        Test case for simple dictionary node.
        """
        node = (
            "dict",
            [
                (("num", 1), ("num", 5)),
                (("num", 2), ("num", 10)),
                (("num", 3), ("complex", 1j)),
            ],
        )

        result = emit_dict(node, ast_to_torch_expr)

        assert result == "{1: 5, 2: 10, 3: 1j}"

    def test_array_dict_node(self):
        """
        Test case of tensor dictionary node.
        """
        node = (
            "dict",
            [
                (
                    ("num", 0),
                    ("array", [("num", 1), ("num", 2), ("num", 3)]),
                ),
                (
                    ("num", 1),
                    ("array", [("num", 4), ("num", 5), ("num", 6)]),
                ),
            ],
        )

        result = emit_dict(node, ast_to_torch_expr)

        assert result == ("{0: torch.tensor([1, 2, 3], device=DEVICE), "
                          "1: torch.tensor([4, 5, 6], device=DEVICE)}")


class TestExampleDict:
    """
    Test cases for ``examples/physika_dictionary.phyk`` file.
    """

    def test_declarations(self, dict_ns):
        """
        Tests for dictionary declarations.
        """
        empty_dict = dict_ns["empty_dict"]
        assert empty_dict == {}

        simple_dict = dict_ns["simple_dict"]
        assert simple_dict == {0: 1.6, 1: 3.2, 2: 5.5}

        union_dict = dict_ns["union_dict"]
        assert union_dict[0] == 1
        assert union_dict[1] == 3j
        assert union_dict[2] == 30
        assert torch.equal(
            union_dict[3],
            torch.tensor([1, 2, 3]),
        )

        func_dict = dict_ns["func_dict"]
        assert func_dict == {0: 1.0, 1: 2.0}

        class_dict = dict_ns["class_dict"]
        assert class_dict == {0: 1.6, 1: 3.2, 2: 5.5}

    def test_indexing(self, dict_ns):
        """
        Tests for dictionary indexing.
        """
        first_value = dict_ns["first_value"]
        assert first_value == 1

        last_value = dict_ns["last_value"]
        assert last_value.tolist() == [1, 2, 3]

    def test_update(self, dict_ns):
        """
        Tests for dictionary update.
        """
        example_dict = dict_ns["example_dict"]
        assert example_dict[0] == 1.6
        assert example_dict[1] == 3.2
        assert torch.equal(example_dict[2], torch.tensor([1, 2, 3]))

    def test_differentiability(self, dict_ns):
        """
        Tests for dictionary differentiability.
        """
        scalar_diff = dict_ns["scalar_diff"]
        assert scalar_diff == 2

        array_diff = dict_ns["array_diff"]
        assert array_diff.tolist() == [2, 4, 6]
