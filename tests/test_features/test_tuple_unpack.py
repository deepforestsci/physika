import pytest
from physika.features.tuple_unpack import TupleUnpackFeature
from physika.utils.ast_utils import ast_to_torch_expr, build_unified_ast
from physika.codegen import from_ast_to_torch
from tests.test_features.test_classes import parse_physika


def type_errors(src: str) -> list:
    """
    Parse Physika source string, run the type checker and return the list of
    error strings if any.
    """
    import physika.parser as pm
    from physika.lexer import lexer
    from physika.type_checker import TypeChecker
    pm.symbol_table.clear()
    lexer.lexer.lineno = 1
    ast = build_unified_ast(pm.parser.parse(src, lexer=lexer), pm.symbol_table)
    return TypeChecker(ast).run()


def run_phyk(src: str) -> dict:
    """
    Helper function to parse, emits codegen, and exec a Physika source
    string.
    """
    import physika.parser as pm
    from physika.lexer import lexer
    pm.symbol_table.clear()
    lexer.lexer.lineno = 1
    ast = build_unified_ast(pm.parser.parse(src, lexer=lexer), pm.symbol_table)
    code = from_ast_to_torch(ast, print_code=False)
    ns: dict = {}
    exec(code, ns)
    return ns


simple_src = """
class Simple(v: ℝ):
    def get() → ℝ:
        return this.v, this.v
    def sum_pairs(n: ℕ) → ℝ:
        total : ℝ = 0.0
        for k : ℕ(n):
            a, b = this.get()
            total = total + a + b
        return total
"""

pair_src = """
class Pair(a: ℝ, b: ℝ):
    def get() → ℝ:
        return this.a, this.b
    def sum() → ℝ:
        x, y = this.get()
        return x + y
"""

model_src = """
class Model(a: ℝ, b: ℝ):
    def pair() → ℝ:
        return this.a, this.b
    def run(steps: ℕ) → ℝ:
        x, y = this.pair()
        total : ℝ = x + y
        for k : ℕ(steps):
            p, q = this.pair()
            total = total + p + q
        return total
"""


class TestTupleUnpack:
    """
    Checks parser is emitting the correct AST nodes,
    that the forward rules have the correct keys and type
    rules catches proper errors.
    """

    def test_elf_name(self):
        """
        ELF is registered under the name 'tuple_unpack'.
        """
        assert TupleUnpackFeature.name == "tuple_unpack"

    def test_parser_rules(self):
        """
        parser_rules should return nine handlers.
        """
        rules = TupleUnpackFeature().parser_rules()
        assert len(rules) == 9
        names = [r.__name__ for r in rules]
        assert "p_return_type_single" in names
        assert "p_return_type_tuple" in names
        assert "p_typed_id_list" in names
        assert "p_return_expr_list" in names
        assert "p_top_level_expr_list" in names
        assert "p_func_body_stmt_tuple_unpack" in names
        assert "p_func_loop_stmt_tuple_unpack" in names
        assert "p_statement_tuple_unpack" in names
        assert "p_for_statement_tuple_unpack" in names

        # each rule doc must mention both alternatives
        combined = " ".join(r.__doc__ for r in rules)
        assert "func_body_stmt" in combined
        assert "func_loop_stmt" in combined
        assert "statement" in combined
        assert "for_statement" in combined
        assert "id_list EQUALS" in combined
        assert "typed_id_list EQUALS" in combined
        assert "typed_id_list" in combined

    def test_forward_rules_keys(self):
        """
        forward rules contains 'loop_tuple_unpack' and 'stmt_tuple_unpack'.
        """
        rules = TupleUnpackFeature().forward_rules()
        assert set(rules.keys()) == {
            "expr_list", "loop_tuple_unpack", "stmt_tuple_unpack"
        }
        assert callable(rules["loop_tuple_unpack"])
        assert callable(rules["stmt_tuple_unpack"])

    def test_type_rules_keys(self):
        """type_rules() exposes handlers for all tuple-related node tags."""
        rules = TupleUnpackFeature().type_rules()
        assert set(rules.keys()) == {
            "tuple_return",
            "expr_list",
            "body_tuple_unpack",
            "loop_tuple_unpack",
            "stmt_tuple_unpack",
        }
        assert all(callable(v) for v in rules.values())


class TestForwardRules:
    """
    Tests for both forward rule handlers. for-loop bodies inside
    functions and class methods, program level and for-loop bodies.
    """

    def test_emit_function_call(self):
        """Unpack two values from a plain function call."""
        rules = TupleUnpackFeature().forward_rules()
        node = ("loop_tuple_unpack", ["a", "b"], ("call", "f", [("var", "n")]))
        assert rules["loop_tuple_unpack"](node,
                                          ast_to_torch_expr) == "a, b = f(n)"

    def test_self_method_call(self):
        """Unpack two values from a self.method() call"""
        rules = TupleUnpackFeature().forward_rules()
        node = ("loop_tuple_unpack", ["spins", "log_prob"], ("call", "self",
                                                             [("var", "n")]))
        result = rules["loop_tuple_unpack"](node, ast_to_torch_expr)
        assert result == "spins, log_prob = self(n)"

    def test_tuple_unpack_top_level(self):
        """Top-level unpack of a function call"""
        rules = TupleUnpackFeature().forward_rules()
        node = ("stmt_tuple_unpack", ["a", "b"], ("call", "g", [("var", "x")]))
        result = rules["stmt_tuple_unpack"](node, ast_to_torch_expr)
        assert result == "a, b = g(x)"


class TestTypeRules:
    """
    Tests for type rules in TupleUnpackFeature.

    Physika source strings are first parsed and the type checked
    catching errors if any.
    """

    def test_type_rules_keys(self):
        """type_rules() contains all tuple-related node tags."""
        rules = TupleUnpackFeature().type_rules()
        assert set(rules.keys()) == {
            "tuple_return",
            "expr_list",
            "body_tuple_unpack",
            "loop_tuple_unpack",
            "stmt_tuple_unpack",
        }

    def test_tuple_return_shape_mismatch(self):
        """
        shape mismatch inside the first return expression is reported.
        """
        src = ("class T(a: ℝ[3], b: ℝ[2]):\n"
               "    def bad() → ℝ:\n"
               "        return this.a / this.b, 1.0\n")
        errors = type_errors(src)

        assert len(errors) == 1
        assert errors[
            0] == "In class 'T', method 'bad': Shape mismatch in div: ℝ[3] vs ℝ[2]"  # noqa: E501

    def test_tuple_return_no_errorsr(self):
        """``return this.x, this.y`` with learnable parameters x:ℝ, y:ℝ"""
        src = ("class T(x: ℝ, y: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.x, this.y\n")
        errors = type_errors(src)
        assert errors == []

        # test unpacking the returned pair at program level
        src = ("class T(x: ℝ, y: ℝ):\n"
               "    def get() → ℝ, ℝ:\n"
               "        return this.x, this.y\n"
               "\n"
               "t : T = T(1.0, 2.0)\n"
               "a: ℝ, b: ℝ = t.get()\n"
               "result : ℝ = a + b\n")
        assert type_errors(src) == []

    def test_body_tuple_unpack_vars(self):
        """
        Unpacked tuple variables are usable at rutime
        """
        src = ("class T(v: ℝ, w: ℝ):\n"
               "    def f() → ℝ:\n"
               "        x: ℝ, y: ℝ = this.v, this.w\n"
               "        return x + y\n")
        assert type_errors(src) == []

    def test_body_tuple_unpack_shape_mismatch(self):
        """
        check type errors are catched when declaring wrong type variables.
        """
        src = ("class T(a: ℝ[3], b: ℝ[2]):\n"
               "    def bad() → ℝ:\n"
               "        x: ℝ, y: ℝ = this.a, this.b\n"
               "        return x + y\n")
        errors = type_errors(src)

        assert len(errors) == 4
        assert errors[
            0] == "In class 'T', method 'bad': Type mismatch in tuple unpack: 'x' declared as ℝ but got ℝ[3]"  # noqa: E501
        assert errors[
            1] == "In class 'T', method 'bad': Type mismatch in tuple unpack: 'y' declared as ℝ but got ℝ[2]"  # noqa: E501
        assert errors[
            2] == "In class 'T', method 'bad': Shape mismatch in add: ℝ[3] vs ℝ[2]"  # noqa: E501
        assert errors[
            3] == "In class 'T', method 'bad': return type mismatch: declared ℝ, got ℝ[3]: Cannot unify scalar ℝ with tensor ℝ[3]"  # noqa: E501

    def test_loop_tuple_unpack_usable(self):
        """
        Unpacked variable names inside a for loop body are
        usable inside iteration.
        """
        src = ("class T(v: ℝ):\n"
               "    def f(n: ℕ) → ℝ:\n"
               "        total : ℝ = 0.0\n"
               "        for k : ℕ(n):\n"
               "            p, q = this.v\n"
               "            total = total + p + q\n"
               "        return total\n")
        assert type_errors(src) == []

    def test_loop_tuple_unpack_type_errors(self):
        """
        A type errir should be catch for wrong type operations, but correct
        tuple unpack
        """
        src = ("class T(a: ℝ[3], b: ℝ[2]):\n"
               "    def bad(n: ℕ) → ℝ:\n"
               "        total : ℝ = 0.0\n"
               "        for k : ℕ(n):\n"
               "            p, q = this.a / this.b\n"
               "            total = total + p + q\n"
               "        return total\n")
        errors = type_errors(src)
        assert len(errors) == 1
        assert errors[
            0] == "In class 'T', method 'bad': Shape mismatch in div: ℝ[3] vs ℝ[2]"  # noqa: E501

    def test_stmt_tuple_unpack_names_usable(self):
        """
        Test that tuple unpack works at program level.
        """
        src = ("class T(u: ℝ, v: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.u, this.v\n"
               "\n"
               "t : T = T(1.0, 2.0)\n"
               "u: ℝ, v : ℝ= t.get()\n"
               "result : ℝ = u + v\n")
        assert type_errors(src) == []

    def test_stmt_tuple_unpack_shape_mismatch_is_caught(self):
        """
        ``u, v = arr1 / arr2`` at program level where arr1:ℝ[3] and
        arr2:ℝ[2] — the RHS shape mismatch is reported.
        """
        src = ("arr1 : ℝ[3] = for i : ℕ(3) → 1.0\n"
               "arr2 : ℝ[2] = for i : ℕ(2) → 1.0\n"
               "u, v = arr1 / arr2\n")
        errors = type_errors(src)
        assert any("mismatch" in e.lower() for e in errors)

    def test_body_tuple_unpack_no_error(self):
        """
        Test multiple (more than two) tuple unpack values are usable
        in runtime.
        """
        src = ("class T(v: ℝ):\n"
               "    def compute(n: ℕ) → ℝ:\n"
               "        arr : ℝ[3] = for i : ℕ(3) → this.v\n"
               "        total : ℝ = 0.0\n"
               "        for k : ℕ(n):\n"
               "            a: ℝ, b: ℝ, c: ℝ = arr\n"
               "            total = total + a + b + c\n"
               "        return total\n")
        assert type_errors(src) == []

    def test_body_tuple_unpack_wrong_type_errors(self):
        """
        Wrong type declaration inside for-loop of a class method.
        """
        src = ("class T(v: ℝ):\n"
               "    def compute(n: ℕ) → ℝ:\n"
               "        arr : ℝ[3] = for i : ℕ(3) → this.v\n"
               "        total : ℝ = 0.0\n"
               "        for k : ℕ(n):\n"
               "            a: ℝ[3], b: ℝ, c: ℝ = arr\n"
               "            total = total + a + b + c\n"
               "        return total\n")
        errors = type_errors(src)
        assert len(errors) == 1
        assert errors[
            0] == "In class 'T', method 'compute': Type mismatch in tuple unpack: 'a' declared as ℝ[3] but element type is ℝ"  # noqa: E501


class TestParserRules:
    """
    Tests that grammar rules produce correct AST nodes.
    """

    def test_loop_tuple_unpack_class_method(self):
        """
        tuple unpack inside a class method for-loop is parsed as
        ``("loop_tuple_unpack", [value1, value2, ..., valuen], expr)``.
        """
        ast = parse_physika(simple_src)
        method = next(m for m in ast["classes"]["Simple"]["methods"]
                      if m["name"] == "sum_pairs")
        for_stmt = next(s for s in method["statements"]
                        if s[0] == "body_for_range")
        loop_body = for_stmt[4]  # (tag, var, start, end, body_list)
        unpack_nodes = [s for s in loop_body if s[0] == "loop_tuple_unpack"]
        assert len(unpack_nodes) == 1

    def test_loop_tuple_unpack_names(self):
        """Variables ``a``, ``b`` are present after parsing"""
        ast = parse_physika(simple_src)
        method = next(m for m in ast["classes"]["Simple"]["methods"]
                      if m["name"] == "sum_pairs")
        for_stmt = next(s for s in method["statements"]
                        if s[0] == "body_for_range")
        loop_body = for_stmt[4]
        unpack = next(s for s in loop_body if s[0] == "loop_tuple_unpack")
        assert unpack[1] == ["a", "b"]

    def test_body_tuple_unpack_class_method(self):
        """
        ``a, b = expr`` directly in a class method body
        is parsed as ``("body_tuple_unpack", [value_1, value_2, ..., value_n],
        expr)``.
        """
        ast = parse_physika(pair_src)
        method = next(m for m in ast["classes"]["Pair"]["methods"]
                      if m["name"] == "sum")
        stmts = method["statements"]
        unpack_stmts = [s for s in stmts if s[0] == "body_tuple_unpack"]
        assert len(unpack_stmts) == 1
        assert unpack_stmts[0][1] == ["x", "y"]

    def test_multiple_tuple_unpack_in_function(self):
        """
        Test multiple value tuple unpack works properly.
        """
        src = ("class T(v: ℝ):\n"
               "    def compute() → ℝ:\n"
               "        arr : ℝ[3] = for i : ℕ(3) → this.v\n"
               "        a, b, c = arr\n"
               "        return a + b + c\n")
        ast = parse_physika(src)
        method = next(m for m in ast["classes"]["T"]["methods"]
                      if m["name"] == "compute")
        stmts = method["statements"]
        three_name = next(s for s in stmts
                          if s[0] == "body_tuple_unpack" and len(s[1]) == 3)
        assert three_name[1] == ["a", "b", "c"]

    def test_tuple_return_node_in_class_method(self):
        """
        ``return x, y`` in a class method body is stored as
        ``("tuple_return", e1, e2)`` after ``unwrap_return``.
        """
        ast = parse_physika(pair_src)
        method = next(m for m in ast["classes"]["Pair"]["methods"]
                      if m["name"] == "get")
        assert method["body"] == ('tuple_return', ('field_access',
                                                   ('var', 'this'), 'a'),
                                  ('field_access', ('var', 'this'), 'b'))

    def test_body_tuple_unpack_in_top_level_function(self):
        """
        tuple unpack in a top-level function body is parsed as
        "body_tuple_unpack" ASTNode.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.a, this.b\n"
               "def use_pair(s: Src): ℝ:\n"
               "    x, y = s.get()\n"
               "    return x + y\n")
        ast = parse_physika(src)
        func = ast["functions"]["use_pair"]
        stmts = func["statements"]
        unpack = next(s for s in stmts if s[0] == "body_tuple_unpack")
        assert unpack[1] == ["x", "y"]

    def test_tuple_unpack_top_level_function_for_loop(self):
        """
        tuple unpack inside a for loop in a top-level
        function is parsed as "loop_tuple_unpack" ASTNode.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ, ℝ:\n"
               "        return this.a, this.b\n"
               "def accumulate(s: Src, n: ℕ): ℝ:\n"
               "    total : ℝ = 0.0\n"
               "    for k : ℕ(n):\n"
               "        x, y = s.get()\n"
               "        total = total + x + y\n"
               "    return total\n")
        ast = parse_physika(src)
        func = ast["functions"]["accumulate"]
        for_stmt = next(s for s in func["statements"]
                        if s[0] == "body_for_range")
        loop_body = for_stmt[4]
        unpack = next(s for s in loop_body if s[0] == "loop_tuple_unpack")
        assert unpack[1] == ["x", "y"]

    def test_four_name_body_tuple_unpack(self):
        """
        Test tuple unpack grammar rule supports any N ≥ 2.
        """
        src = ("class T(v: ℝ):\n"
               "    def compute() → ℝ:\n"
               "        arr : ℝ[4] = for i : ℕ(4) → this.v\n"
               "        a, b, c, d = arr\n"
               "        return a + b + c + d\n")
        ast = parse_physika(src)
        method = next(m for m in ast["classes"]["T"]["methods"]
                      if m["name"] == "compute")
        unpack = next(s for s in method["statements"]
                      if s[0] == "body_tuple_unpack")
        assert unpack[1] == ["a", "b", "c", "d"]

    def test_n_name_stmt_tuple_unpack_at_program_level(self):
        """
        ``a, b, c, d = arr`` at program level produces
        ``("stmt_tuple_unpack", ["a","b","c","d"], arr_expr)``.
        """
        src = ("class T(v: ℝ):\n"
               "    def val() → ℝ, ℝ:\n"
               "        return this.v, this.v\n"
               "t : T = T(1.0)\n"
               "arr : ℝ[4] = for i : ℕ(4) → t.v\n"
               "a: ℝ, b: ℝ, c:ℝ, d: ℝ = arr\n")
        ast = parse_physika(src)
        unpack = next(s for s in ast["program"]
                      if isinstance(s, tuple) and s[0] == "stmt_tuple_unpack")
        assert unpack[1] == [('a', 'ℝ'), ('b', 'ℝ'), ('c', 'ℝ'), ('d', 'ℝ')]

    def test_literal_comma_rhs_produces_expr_list(self):
        """
        ``a: ℝ, b: ℝ = 1, 2`` at program level produces
        ``("stmt_tuple_unpack", [...], ("expr_list", [num(1), num(2)]))``.
        """
        src = "a: ℝ, b: ℝ = 1, 2\n"
        ast = parse_physika(src)
        unpack = next(s for s in ast["program"]
                      if isinstance(s, tuple) and s[0] == "stmt_tuple_unpack")
        assert unpack[1] == [('a', 'ℝ'), ('b', 'ℝ')]
        rhs = unpack[2]
        assert isinstance(rhs, tuple) and rhs[0] == "expr_list"
        assert len(rhs[1]) == 2

    def test_three_literal_comma_rhs_produces_expr_list(self):
        """
        ``a: ℝ, b: ℝ, c: ℝ = 10, 20, 30`` produces expr_list with 3 elements.
        """
        src = "a: ℝ, b: ℝ, c: ℝ = 10, 20, 30\n"
        ast = parse_physika(src)
        unpack = next(s for s in ast["program"]
                      if isinstance(s, tuple) and s[0] == "stmt_tuple_unpack")
        rhs = unpack[2]
        assert rhs[0] == "expr_list"
        assert len(rhs[1]) == 3


class TestTypeRulesLiteralComma:
    """Type-checker behaviour for the new literal-comma unpack forms."""

    def test_typed_literal_comma_no_error(self):
        """``a: ℝ, b: ℝ = 1.0, 2.0`` passes the type checker."""
        src = "a: ℝ, b: ℝ = 1.0, 2.0\n"
        assert type_errors(src) == []

    def test_four_typed_literals_no_error(self):
        """``a: ℝ, b: ℝ, c: ℝ, d: ℝ = 1, 2, 3, 4`` passes the type checker."""
        src = "a: ℝ, b: ℝ, c: ℝ, d: ℝ = 1, 2, 3, 4\n"
        assert type_errors(src) == []

    def test_typed_literal_comma_wrong_type_caught(self):
        """
        ``a: ℝ[3], b: ℝ = 1.0, 2.0`` — 'a' declared as ℝ[3] but literal is ℝ.
        """
        src = "a: ℝ[3], b: ℝ = 1.0, 2.0\n"
        errors = type_errors(src)
        assert len(errors) >= 1
        assert any("'a'" in e and "ℝ[3]" in e for e in errors)

    def test_untyped_literal_comma_no_error(self):
        """``a, b = 1, 2`` (no type annotations) passes the type checker."""
        src = "a, b = 1, 2\n"
        assert type_errors(src) == []


class TestTupleUnpackIntegration:
    """
    Check an end-to-end integration of TupleUnpack feature at parser, codegen
    and type rules.
    """

    def test_tuple_unpack_in_class_method_for_loop(self):
        """
        tuple unpack inside a for-loop body
        in a class method.
        """
        ns = run_phyk(simple_src + """
s : Simple = Simple(2.0)
result : ℝ = s.sum_pairs(3)
""")
        # 3 iterations x (2 + 2) = 12
        assert ns["result"].item() == pytest.approx(12.0)

    def test_tuple_unpack_in_class_method_body(self):
        """
        Test tuple unpack at the class method body level.
        """
        ns = run_phyk(pair_src + """
p : Pair = Pair(5.0, 3.0)
result : ℝ = p.sum()
""")
        assert ns["result"].item() == pytest.approx(8.0)

    def test_tuple_unpack_method_for_loop(self):
        """
        Both body-level and loop-level tuple unpack used in the same method.
        """
        ns = run_phyk(model_src + """
m : Model = Model(1.0, 2.0)
result : ℝ = m.run(3)
""")
        assert ns["result"].item() == pytest.approx(12.0)

    def test_tuple_unpack_multi_value_method(self):
        """
        Tests three value tuple unpack in a class method body works correctly.
        """
        src = ("class T(v: ℝ):\n"
               "    def compute() → ℝ:\n"
               "        arr : ℝ[3] = for i : ℕ(3) → this.v\n"
               "        a, b, c = arr\n"
               "        return a + b + c\n"
               "\n"
               "t : T = T(2.0)\n"
               "result : ℝ = t.compute()\n")
        ns = run_phyk(src)
        # arr = [2, 2, 2]
        # a+b+c = 6
        assert ns["result"].item() == pytest.approx(6.0)

    def test_tuple_unpack_in_nested_for_loops(self):
        """
        Tuple unpack in the inner body of two nested for-loops.
        """
        src = ("class Src(v: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.v, this.v\n"
               "    def run(n: ℕ, m: ℕ) → ℝ:\n"
               "        total : ℝ = 0.0\n"
               "        for i : ℕ(n):\n"
               "            for k : ℕ(m):\n"
               "                a: ℝ, b: ℝ = this.get()\n"
               "                total = total + a + b\n"
               "        return total\n"
               "\n"
               "s : Src = Src(1.0)\n"
               "result : ℝ = s.run(3, 2)\n")
        ns = run_phyk(src)
        assert ns["result"].item() == pytest.approx(12.0)

    def test_tuple_unpack_values_computable(self):
        """
        Tuple-unpacked variables used as arguments inside a for-expression
        in the same method body.
        """
        src = ("class Src(lo: ℝ, hi: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.lo, this.hi\n"
               "    def spread(n: ℕ) → ℝ:\n"
               "        a, b = this.get()\n"
               "        arr : ℝ[n] = for i : ℕ(n) → a + b\n"
               "        return sum(arr)\n"
               "\n"
               "s : Src = Src(2.0, 3.0)\n"
               "result : ℝ = s.spread(4)\n")
        ns = run_phyk(src)

        assert ns["result"].item() == pytest.approx(20.0)

    def test_tuple_unpack_at_program_level(self):
        """
        `stmt` tuple unpack at program level.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.a, this.b\n"
               "s : Src = Src(3.0, 4.0)\n"
               "x, y = s.get()\n"
               "result : ℝ = x + y\n")
        ns = run_phyk(src)
        assert ns["result"].item() == pytest.approx(7.0)

    def test_tuple_unpack_top_level_for_loop(self):
        """
        tuple unpack inside a top-level for loop body.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.a, this.b\n"
               "s : Src = Src(2.0, 3.0)\n"
               "total : ℝ = 0.0\n"
               "for k : ℕ(4):\n"
               "    p: ℝ, q: ℝ = s.get()\n"
               "    total = total + p + q\n")
        ns = run_phyk(src)
        assert ns["total"].item() == pytest.approx(20.0)

    def test_tuple_unpack_top_level_if_else_for_loop(self):
        """
        Test tuple unpack with different control flow operations.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ, ℝ:\n"
               "        return this.a, this.b\n"
               "s : Src = Src(5.0, 2.0)\n"
               "x: ℝ, y: ℝ = s.get()\n"
               "best : ℝ = 0.0\n"
               "if x > y:\n"
               "    best = x\n"
               "else:\n"
               "    best = y\n"
               "total : ℝ = 0.0\n"
               "for k : ℕ(3):\n"
               "    p: ℝ, q: ℝ = s.get()\n"
               "    total = total + p\n")
        ns = run_phyk(src)
        assert ns["best"].item() == pytest.approx(5.0)
        assert ns["total"].item() == pytest.approx(15.0)

    def test_tuple_unpack_in_top_level_function_body(self):
        """
        tuple unpack in a top-level function.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.a, this.b\n"
               "def use_pair(s: Src): ℝ:\n"
               "    x: ℝ, y: ℝ = s.get()\n"
               "    return x + y\n"
               "s : Src = Src(3.0, 4.0)\n"
               "result : ℝ = use_pair(s)\n")
        ns = run_phyk(src)
        assert ns["result"].item() == pytest.approx(7.0)

    def test_loop_tuple_unpack_top_level_function(self):
        """
        tuple unpack inside a for-loop in a
        top-level function.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.a, this.b\n"
               "def accumulate(s: Src, n: ℕ): ℝ:\n"
               "    total : ℝ = 0.0\n"
               "    for k : ℕ(n):\n"
               "        x, y = s.get()\n"
               "        total = total + x + y\n"
               "    return total\n"
               "s : Src = Src(2.0, 3.0)\n"
               "result : ℝ = accumulate(s, 4)\n")
        ns = run_phyk(src)

        assert ns["result"].item() == pytest.approx(20.0)

    def test_tuple_unpack_if_else_top_level_function(self):
        """
        tuple unpack followed by an if-else in a top-level function.
        The unpacked names are used as condition operands.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.a, this.b\n"
               "def larger(s: Src): ℝ:\n"
               "    x, y = s.get()\n"
               "    result : ℝ = 0.0\n"
               "    if x > y:\n"
               "        result = x\n"
               "    else:\n"
               "        result = y\n"
               "    return result\n"
               "s : Src = Src(5.0, 3.0)\n"
               "out : ℝ = larger(s)\n")
        ns = run_phyk(src)
        assert ns["out"].item() == pytest.approx(5.0)

    def test_tuple_unpack_body_for_loop_if_else_top_level_function(self):
        """
        Three Physika control flow operations combined in one top-level
        function.
        """
        src = ("class Src(a: ℝ, b: ℝ):\n"
               "    def get() → ℝ:\n"
               "        return this.a, this.b\n"
               "def cond_sum(s: Src, n: ℕ): ℝ:\n"
               "    x, y = s.get()\n"
               "    total : ℝ = 0.0\n"
               "    for k : ℕ(n):\n"
               "        p: ℝ, q: ℝ = s.get()\n"
               "        if p > q:\n"
               "            total = total + p\n"
               "        else:\n"
               "            total = total + q\n"
               "    return total\n"
               "s : Src = Src(5.0, 3.0)\n"
               "result : ℝ = cond_sum(s, 4)\n")
        ns = run_phyk(src)

        assert ns["result"].item() == pytest.approx(20.0)

    def test_tuple_unpack_elf_integration(self):
        """
        TupleUnpackFeature and ClassFeature integration.
        """
        src = ("class Accum(base: ℝ):\n"
               "    baseline : ℝ\n"
               "    def get_pair() → ℝ, ℝ:\n"
               "        return this.base, this.baseline\n"
               "    def accumulate(steps: ℕ) → ℝ:\n"
               "        this.baseline = this.base * 0.0\n"
               "        total : ℝ = 0.0\n"
               "        for k : ℕ(steps):\n"
               "            a: ℝ, b: ℝ = this.get_pair()\n"
               "            this.baseline = this.baseline + a\n"
               "            total = total + a + b\n"
               "        return total\n"
               "\n"
               "acc : Accum = Accum(1.0)\n"
               "result : ℝ = acc.accumulate(3)\n")
        ns = run_phyk(src)
        assert ns["result"].item() == pytest.approx(6.0)

    def test_multiple_comma_unpack(self):
        """
        Tests comma separated literal at program level.
        """
        src = "a: ℝ, b: ℝ, c: ℝ, d: ℝ = 1, 2, 3, 4\n"
        ns = run_phyk(src)
        assert ns["a"] == pytest.approx(1.0)
        assert ns["b"] == pytest.approx(2.0)
        assert ns["c"] == pytest.approx(3.0)
        assert ns["d"] == pytest.approx(4.0)

    def test_comma_unpack_top_level_for_loop(self):
        """
        tuple unpack inside a top-level for-loop body.
        """
        src = ("total : ℝ = 0.0\n"
               "for k : ℕ(3):\n"
               "    a: ℝ, b: ℝ = 10, 20\n"
               "    total = total + a + b\n")
        ns = run_phyk(src)
        assert ns["total"] == pytest.approx(90.0)

    def test_untyped_literal_comma_unpack_at_program_level(self):
        """
        tuple unpack without type annotations at program level.
        """
        src = "a, b = 1, 2\n"
        ns = run_phyk(src)
        assert ns["a"] == pytest.approx(1.0)
        assert ns["b"] == pytest.approx(2.0)

    def test_arithmetic_comma_unpack_at_program_level(self):
        """
        ``a: ℝ, b: ℝ = 1 + 2, 3 * 4`` — arithmetic expressions in each slot.
        """
        src = "a: ℝ, b: ℝ = 1 + 2, 3 * 4\n"
        ns = run_phyk(src)
        assert ns["a"] == pytest.approx(3.0)
        assert ns["b"] == pytest.approx(12.0)

    def test_three_value_literal_comma_unpack(self):
        """``a: ℝ, b: ℝ, c: ℝ = 10, 20, 30`` at program level."""
        src = "a: ℝ, b: ℝ, c: ℝ = 10, 20, 30\n"
        ns = run_phyk(src)
        assert ns["a"] == pytest.approx(10.0)
        assert ns["b"] == pytest.approx(20.0)
        assert ns["c"] == pytest.approx(30.0)
