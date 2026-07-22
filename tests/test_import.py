from pathlib import Path
from physika.utils.import_manager import find_module, resolve_imports
from physika.parser import symbol_table
from physika.lexer import lexer

import pytest
from tests.conftest import exec_phyk

r_tol = 1e-02


@pytest.fixture(scope="module")
def numeric_ns():
    """
    Execute example_import_statement.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_import_statement")


class TestExampleImportFile:
    """Tests for ``example_import_statement.phyk`` file"""

    def test_single_import(self, numeric_ns):
        """Test single import statements"""
        assert "fact_results" in numeric_ns
        assert numeric_ns["fact_results"] == 1.0

    def test_multiple_import(self, numeric_ns):
        """Test multiple import statements"""
        assert "torch_funcs_results" in numeric_ns
        assert "f_results" in numeric_ns

        torch_results = numeric_ns["torch_funcs_results"]

        assert len(torch_results) == 6

        assert abs(numeric_ns["f_results"].item() - 0.5403) < r_tol

    def test_import_class(self, numeric_ns):
        """Test import class from another file"""
        class_value = numeric_ns["class_value"]
        assert class_value == 1

    def test_import_array(self, numeric_ns):
        """Test import array and perform operations"""
        v = numeric_ns["v"]
        grad_f_x = numeric_ns["grad_f_x"]
        assert v.tolist() == [1.0, 2.0, 6.0]
        assert grad_f_x == 12

    def test_import_from_different_directory(self, numeric_ns):
        """Test import from another directory"""
        assert "superbee" in numeric_ns

        phi = numeric_ns["φ"]
        assert len(phi) == 5
        assert phi.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]


class TestImportManager:
    """Tests for physika/import_manager.py file"""

    def test_find_module(self):
        """
        find_module should correctly locate a Physika module file.
        """
        source_file_path = Path(
            "examples/example_import_statement.phyk").resolve()
        module_path = find_module("factorial", source_file_path)

        assert module_path.name == "factorial.phyk"
        assert module_path.exists()

    def test_resolve_imports(self):
        """
        resolve_imports should append only explicitly imported
        symbols to the program AST.
        """

        symbol_table.clear()
        lexer.lexer.lineno = 1

        local_program_ast = [("import", "factorial", ["fact"]),
                             ("expr", ("call", "fact", [("num", 1.0)]), 0)]

        source_file_path = Path(
            "examples/example_import_statement.phyk").resolve()

        resolved = resolve_imports(local_program_ast, source_file_path)

        # imported function definition added
        assert resolved[0][0] == "func_def"
        assert resolved[0][1] == "fact"

        # original program expression preserved
        assert resolved[1] == ("expr", ("call", "fact", [("num", 1.0)]), 0)
