from physika.core.elab.termination import (
    check_mutual_recursion_termination,
    check_recursive_termination,
)


class TestCheckRecursiveTermination:
    """
    Tests for ``check_recursive_termination``.
    """

    def test_recursive_termination(self):
        """
        Tests self-recursion termination and nat argument decreases.
        """
        errors = []
        check_recursive_termination(
            "fact",
            [("n", "ℕ")],
            ("call", "fact", [("sub", ("var", "n"), ("num", 1.0))]),
            [],
            errors,
            "In function 'fact'",
        )
        assert errors == []

        # non decreasing nat arguments should report error
        errors = []
        check_recursive_termination(
            "bad",
            [("n", "ℕ")],
            ("call", "bad", [("add", ("var", "n"), ("num", 1.0))]),
            [],
            errors,
            "In function 'bad'",
        )

        assert len(errors) == 1
        assert errors[
            0] == "In function 'bad': recursive call to 'bad' does not decrease at any parameter position"  # noqa: E501

        # check recursive call in body statemtnts is correct in structure.
        errors = []
        check_recursive_termination(
            "loopy",
            [("n", "ℕ")],
            ("var", "acc"),
            [("body_assign", "acc", ("call", "loopy", [("sub", ("var", "n"),
                                                        ("num", 1.0))]))],
            errors,
            "In function 'loopy'",
        )
        assert errors == []


class TestCheckMutualRecursionTermination:
    """
    Tests for ``check_mutual_recursion_termination``.
    """

    def is_even_odd_functions(self) -> dict:
        """
        AST mutual recursive functions to test against
        ``check_mutual_recursion_termination``.
        """
        return {
            "is_even": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body":
                ("call", "is_odd", [("sub", ("var", "n"), ("num", 1.0))])
            },
            "is_odd": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body":
                ("call", "is_even", [("sub", ("var", "n"), ("num", 1.0))])
            },
        }

    def test_detects_two_function_cycle(self):
        """
        Checks a mutual recursive call is found
        """
        # base case: No mutualrecursive functions, no errors
        functions = {
            "f": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body": ("num", 1.0)
            },
            "g": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body": ("num", 2.0)
            },
        }
        errors = []
        cycle = check_mutual_recursion_termination(functions, errors)
        assert cycle == set()
        assert errors == []
        # checks mutuial recursive functions: odd and even
        errors = []
        cycle = check_mutual_recursion_termination(
            self.is_even_odd_functions(), errors)
        assert cycle == {"is_even", "is_odd"}
        assert errors == []

        # self recursive function should not be catched neither errors
        functions = {
            "fact": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body": ("call", "fact", [("sub", ("var", "n"), ("num", 1.0))])
            },
        }
        errors = []
        cycle = check_mutual_recursion_termination(functions, errors)
        assert cycle == set()
        assert errors == []

        # mutual recursion whith non decreasing arg should error.
        functions = {
            "is_even": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body":
                ("call", "is_odd", [("add", ("var", "n"), ("num", 1.0))])
            },
            "is_odd": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body":
                ("call", "is_even", [("add", ("var", "n"), ("num", 1.0))])
            },
        }
        errors = []
        cycle = check_mutual_recursion_termination(functions, errors)
        assert cycle == {"is_even", "is_odd"}
        assert len(errors) == 2
        assert errors[
            0] == "In function 'is_even': part of a mutual recursion cycle, but its first parameter 'n' does not decrease on every call to another function in the cycle. Cannot verify this mutual recursion terminates"  # noqa: E501
        assert errors[
            1] == "In function 'is_odd': part of a mutual recursion cycle, but its first parameter 'n' does not decrease on every call to another function in the cycle. Cannot verify this mutual recursion terminates"  # noqa: E501

        # test three mutual recursice functions cycle
        functions = {
            "a": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body": ("call", "b", [("sub", ("var", "n"), ("num", 1.0))])
            },
            "b": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body": ("call", "c", [("sub", ("var", "n"), ("num", 1.0))])
            },
            "c": {
                "params": [("n", "ℕ")],
                "statements": [],
                "body": ("call", "a", [("sub", ("var", "n"), ("num", 1.0))])
            },
        }
        errors = []
        cycle = check_mutual_recursion_termination(functions, errors)
        assert cycle == {"a", "b", "c"}
        assert errors == []
