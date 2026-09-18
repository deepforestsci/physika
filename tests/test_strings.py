import pytest
from tests.conftest import exec_phyk


@pytest.fixture(scope="module")
def strings_ns():
    """
    Execute example_strings.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_strings")


class TestStrings:
    """
    Test suite for ``examples/example_strings.phyk`` file.
    """

    def test_string_dtype(self, strings_ns):
        """
        Tests for string data type,
        * declarations
        * concatenation
        * indexing
        * slicing
        """

        # declarations
        empty_string = strings_ns["empty_string"]
        assert empty_string == ""

        name = strings_ns["name"]
        assert name == "Physika"

        msg = strings_ns["msg"]
        assert msg == "Hello, World!"

        # concatenation
        full_name = strings_ns["full_name"]
        assert full_name == "Physika Language"

        # indexing
        first_character = strings_ns["first_character"]
        assert first_character == "P"

        third_character = strings_ns["third_character"]
        assert third_character == "y"

        # slicing
        first_part = strings_ns["first_part"]
        assert first_part == "Physika"

        last_part = strings_ns["last_part"]
        assert last_part == "Language"
