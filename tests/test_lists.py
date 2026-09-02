import pytest
from tests.conftest import exec_phyk
import torch


@pytest.fixture(scope="module")
def lists_ns():
    """
    Execute example_lists.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_lists")


class TestLists:
    """Test suites for ``examples/example_lists.phyk file"""

    def test_basic_declarations(self, lists_ns):
        # Tests for basic lists declarations.
        x = lists_ns["x"]
        y = lists_ns["y"]
        complex_list = lists_ns["complex_list"]
        simple_nested_list = lists_ns["simple_nested_list"]
        nested_list = lists_ns["nested_list"]
        mixed_list = lists_ns["mixed_list"]

        assert x.tolist() == [1.0, 2.0, 3.0]
        assert y.tolist() == [9.0, 3.0, 5.0, 1.0, 4.0]
        assert complex_list.tolist() == [1j, 2j, 3j]
        assert torch.equal(simple_nested_list[0], torch.tensor([1.0, 2.0,
                                                                3.0]))
        assert torch.equal(simple_nested_list[1],
                           torch.tensor([9.0, 3.0, 5.0, 1.0, 4.0]))
        assert torch.equal(nested_list[2], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(nested_list[3][0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(nested_list[3][1],
                           torch.tensor([9.0, 3.0, 5.0, 1.0, 4.0]))
        assert torch.equal(mixed_list[1], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(mixed_list[2], torch.tensor([1j, 2j, 3j]))
        assert torch.equal(mixed_list[3][0],
                           torch.tensor([9.0, 3.0, 5.0, 1.0, 4.0]))
        assert torch.equal(mixed_list[3][1], torch.tensor([1j, 2j, 3j]))

    def test_lists_indexing(self, lists_ns):
        # Tests for lists indexing.
        simple_index_x = lists_ns["simple_index_x"]
        simple_complex_index = lists_ns["simple_complex_index"]
        index_simple_nested_list = lists_ns["index_simple_nested_list"]
        index_nested_list = lists_ns["index_nested_list"]
        index_mixed_list = lists_ns["index_mixed_list"]
        diff_list_first_index = lists_ns["diff_list_first_index"]
        diff_list_second_index = lists_ns["diff_list_second_index"]

        assert simple_index_x == 1.0
        assert simple_complex_index == 2j
        assert torch.equal(index_simple_nested_list,
                           torch.tensor([1.0, 2.0, 3.0]))
        assert len(index_nested_list) == 2
        assert torch.equal(index_nested_list[0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(index_nested_list[1],
                           torch.tensor([9.0, 3.0, 5.0, 1.0, 4.0]))
        assert len(index_mixed_list) == 2
        assert torch.equal(index_mixed_list[0],
                           torch.tensor([9.0, 3.0, 5.0, 1.0, 4.0]))
        assert torch.equal(index_mixed_list[1], torch.tensor([1j, 2j, 3j]))
        assert torch.equal(diff_list_first_index, torch.tensor([[1, 2], [3,
                                                                         4]]))
        assert torch.equal(diff_list_second_index, torch.tensor([5, 6]))

    def test_lists_differentiation(self, lists_ns):
        # Tests for lists differentiability
        scalar_grad = lists_ns["scalar_grad"]
        tensor_grad = lists_ns["tensor_grad"]
        nested_grad = lists_ns["nested_grad"]

        assert scalar_grad == 2.0
        assert tensor_grad.tolist() == [18.0, 6.0, 10.0, 2.0, 8.0]
        assert nested_grad.tolist() == [2.0, 4.0, 6.0]

    def test_lists_functions(self, lists_ns):
        # Tests for using lists in Physika functions.
        squared_tensor = lists_ns["squared_tensor"]
        assert squared_tensor.tolist() == [1.0, 4.0, 9.0]

        f_results = lists_ns["f_results"]
        assert f_results == [1, 2, 3]

    def test_lists_class(self, lists_ns):
        # Tests for using lists in Physika classes.
        obj_list = lists_ns["obj_list"]

        assert torch.equal(obj_list[2], torch.tensor([1.0, 2.0, 3.0]))
        assert len(obj_list[3]) == 2
        assert torch.equal(obj_list[3][0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(obj_list[3][1],
                           torch.tensor([9.0, 3.0, 5.0, 1.0, 4.0]))

        obj_value_results = lists_ns["obj_value_results"]
        assert obj_value_results == [1, 2]
