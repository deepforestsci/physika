from physika.features.classes import (is_learnable, replace_class_params,
                                      unwrap_return, build_class, emit_method,
                                      generate_class, make_parser_rules,
                                      ClassFeature)
import re
import pytest
from physika.utils.ast_utils import build_unified_ast
from physika.utils.types import TScalar, TTensor, TInstance, Substitution
import sys
import os
from tests.conftest import exec_phyk
import torch


def make_method(name="ke", params=None, body=None, statements=None):
    """
    Helper function to create a basic method for testing.
    """
    return {
        "name": name,
        "params": params or [],
        "return_type": "ℝ",
        "statements": statements or [],
        "body": body,
    }


def particle_class_def():
    """
    Return the class_def produced by ``build_class`` for:

        class Particle:
            mass : ℝ
            def ke() : ℝ:
                return 0.5 * this.mass
    """
    kinetic_energy_body = ("mul", ("num", 0.5), ("field_access",
                                                 ("var", "this"), "mass"))
    ke = make_method(body=kinetic_energy_body)
    body_items = [
        ("field_decl", "mass", "ℝ"),
        ("method_def", ke),
    ]
    return build_class(constructor_params=None, body_items=body_items)


def parse_physika(src):
    """Parse a Physika source string and return the populated symbol_table."""
    import physika.parser as pm
    from physika.lexer import lexer
    pm.symbol_table.clear()
    program_ast = pm.parser.parse(src, lexer=lexer)
    return build_unified_ast(program_ast, pm.symbol_table)


def ns():
    """
    Namespace dict that contains variables, functions and classes
    when example/physika_class.phyk is executed.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    return exec_phyk("physika_class")


def make_env():
    rules = ClassFeature().type_rules()
    check_field = rules["field_access"]
    check_method = rules["method_call"]
    s = Substitution()

    physika_source = ("class Particle:\n"
                      "    x : ℝ\n"
                      "    y : ℝ[2]\n"
                      "    def ke() : ℝ:\n"
                      "        return x\n"
                      "    def step(dt: ℝ) -> ℝ:\n"
                      "        return dt\n")
    ast = parse_physika(physika_source)
    particle_def = ast["classes"]["Particle"]
    class_env = {
        "Particle": {
            **particle_def,
            "methods": {
                m["name"]: m
                for m in particle_def["methods"]
            },
        }
    }

    def infer_instance(*args):
        """
        Infers class TInsance type for Particle
        """
        return TInstance("Particle"), s

    def infer_none(*args):
        """
        Looks for class name directly, which produces an error.
        Method calls should be on class instances.
        """
        return None, s

    return check_field, check_method, s, class_env, infer_instance, infer_none


class TestIsLearnable:
    """
    Tests for ``is_learnable`` helper function to determine
    which physika variables require an ``nn.Parameter`` wrapper.
    """

    def test_valid_learnables(self):
        """
        Valid learnable types
        """
        assert is_learnable("ℝ") is True
        assert is_learnable("R") is True
        assert is_learnable(("tensor", [(3, "invariant")])) is True
        assert is_learnable(("tensor", [(3, "invariant"),
                                        (4, "invariant")])) is True

    def test_not_valid(self):
        """
        Checks invalid Physika types are not learnable
        """
        # class instances type are not learnable parameters.
        assert is_learnable(("struct_type", "Particle")) is False
        # string annotations are not learnable
        for t in ("int", "bool", "str", "ℕ", "ℂ", ""):
            assert is_learnable(t) is False, f"expected False for {t!r}"


class TestReplaceClassParams:
    """
    Tests for ``replace_class_params`` helper function to add 'self.' prefix.
    """

    def test_params_replacement(self):
        """A bare param name in an expression becomes self.param."""
        assert replace_class_params("0.5 * mass",
                                    [("mass", "ℝ")]) == "0.5 * self.mass"

        # self.param should never have double prefix
        assert replace_class_params("self.mass * 2",
                                    [("mass", "ℝ")]) == "self.mass * 2"
        # multiple replacement
        result = replace_class_params(
            "pos + vel * mass",
            [("pos", ("tensor", [2])), ("vel", ("tensor", [2])),
             ("mass", "ℝ")],
        )
        assert result == "self.pos + self.vel * self.mass"

        # no matching params returns code no changes
        assert replace_class_params("x + y * z",
                                    [("mass", "ℝ")]) == "x + y * z"
        # empty param list also dont change code
        assert replace_class_params("mass * 0.5", []) == "mass * 0.5"


class TestUnwrapReturn:
    """
    Tests for ``unwrap_return`` helper fucntion.

    Converts return nodes into a body expression used by ``emit_method``.
    """

    def test_return_single(self):
        """An AST node type inside return_single passes through unchanged."""
        for expr in [("var", "h"), ("num", 42),
                     ("add", ("var", "x"), ("var", "y"))]:
            assert unwrap_return(("return_single", expr)) == expr

    def test_return_tuple(self):
        """return_tuple is retagged as ("tuple_return", e1, e2)."""
        e1, e2 = ("var", "new_pos"), ("var", "new_vel")
        assert unwrap_return(
            ("return_tuple", e1, e2)) == ("tuple_return", e1, e2)


class TestBuildClass:
    """
    Constructs a class dict used by ``generate_class``.
    """

    def test_class_fields_become_constructor_params(self):
        """
        When no constructor params are defined, field declarations in the
        body are promoted to constructor_params.

            class Particle:
                mass : ℝ
        """
        result = build_class(None, [("field_decl", "mass", "ℝ")])
        assert result["class_params"] == [("mass", "ℝ")]

        # fields is always an empty list
        assert result["fields"] == []

        # multiple fields
        body = [
            ("field_decl", "pos", ("tensor", [2])),
            ("field_decl", "vel", ("tensor", [2])),
            ("field_decl", "mass", "ℝ"),
        ]
        result = build_class(None, body)
        names = [p[0] for p in result["class_params"]]
        assert names == ["pos", "vel", "mass"]
        assert result["fields"] == []

    def test_constructor_params(self):
        """
        Parameters list is stored as is defined

            class HNN(W: ℝ[M,N], b: ℝ[M]):
                ...
        """
        params = [("W", ("tensor", [("M", "invariant"), ("N", "invariant")])),
                  ("b", ("tensor", [("M", "invariant")]))]
        result = build_class(params, [])
        assert result["class_params"] == params
        assert result["fields"] == []

    def test_methods(self):
        """method_def body items are collected into the methods list."""
        ke = make_method()
        result = build_class(None, [("field_decl", "mass", "ℝ"),
                                    ("method_def", ke)])
        assert len(result["methods"]) == 1
        assert result["methods"][0]["name"] == "ke"

        # check for multiple methods
        m1 = make_method("ke")
        m2 = make_method("step", params=[("dt", "ℝ")])
        body = [("field_decl", "mass", "ℝ"), ("method_def", m1),
                ("method_def", m2)]
        result = build_class(None, body)
        assert [m["name"] for m in result["methods"]] == ["ke", "step"]


class TestEmitMethod:
    """
    Emits the Python method lines for a single Physika class method.
    """

    def test_no_params(self):
        """
        Method with no zero params with a scalar body:

            def ke() : ℝ:
                return 0.5 * this.mass
        """
        method = make_method(body=("var", "mass"))
        lines = emit_method(method, [("mass", "ℝ")], lambda _: "this.mass",
                            True)
        assert lines[1] == "    def ke(self):"
        assert lines[2] == "        this = self"
        assert lines[-1] == "        return self.mass"

    def test_lambda_maps_to_forward(self):
        """λ method name is emitted as Python 'foward'."""
        method = make_method(name="λ", params=[("x", "ℝ")], body=("var", "x"))
        lines = emit_method(method, [], lambda node: node[1], True)
        assert "def forward(self, x):" in lines[1]

    def test_named_method(self):
        """method names pass through unchanged."""
        method = make_method(name="step", body=("var", "mass"))
        lines = emit_method(method, [("mass", "ℝ")], lambda _: "this.mass",
                            True)
        assert "def step(self):" in lines[1]

    def test_learnable_param(self):
        """ℝ method param gets converted to torch.tensor."""
        method = make_method(name="scale",
                             params=[("s", "ℝ")],
                             body=("var", "s"))
        lines = emit_method(method, [], lambda node: node[1], True)
        # conversion line must appear before the return
        assert any("torch.as_tensor(s).float()" in ln for ln in lines)

    def test_tuple_return(self):
        """tuple_return body emits two value return statement."""
        body = ("tuple_return", ("var", "a"), ("var", "b"))
        method = make_method(name="split", body=body)
        lines = emit_method(method, [], lambda node: node[1], True)
        assert lines[-1] == "        return (a, b)"

    def test_this_replaced_with_self(self):
        """this.field become self.field."""
        method = make_method(body=("var", "x"))
        lines = emit_method(method, [("mass", "ℝ")],
                            lambda _: "0.5 * this.mass", True)
        # skip lines[2] which is the intentional "this = self" alias
        for line in lines[3:]:
            assert "this" not in line


class TestGenerateClass:
    """
    Verifies the full PyTorch ``nn.Module`` source generated from a Physika
    class definition.
    """

    def test_class_generation(self):
        code = generate_class("Particle", particle_class_def())
        # new python code should contain __init__ and __super__
        assert "def __init__(self, mass):" in code
        assert "super().__init__()" in code

        # check code inheriths from nn.Module
        assert "class Particle(nn.Module):" in code

        # mass field should be learnable
        assert "torch.as_tensor(mass).float()" in code
        assert "nn.Parameter" not in code

        # check method
        assert "def ke(self):" in code

        # params to be updated in backprop
        assert "@property" in code
        assert "def params(self):" in code
        assert "return list(self.parameters())" in code
        assert "def update(self, lr, grads):" in code
        assert "p -= lr * g" in code

    def test_constructur_params_and_forward(self):
        """
        class with forward method treat learnable params as ``nn.Parameter``
        so the optimizer can update them.
        """
        fwd = make_method(name="λ", params=[("x", "ℝ")], body=("var", "x"))
        class_def = build_class([("w", "ℝ")], [("method_def", fwd)])
        code = generate_class("Linear", class_def)
        # equovalent to:
        # class Linear(w: ℝ):
        #   def λ(x: ℝ) → ℝ:
        assert "nn.Parameter(torch.as_tensor(w).float())" in code


class TestMakeParserRules:
    """
    Verifies the PLY grammar functions for the class syntax.
    """

    def test_make_parser(self):
        # make_parser_rules returns a plain list
        assert isinstance(make_parser_rules(), list)

        # Exactly 16 grammar rules
        assert len(make_parser_rules()) == 16

        # every item should be a callable p_ functino
        for rule in make_parser_rules():
            assert callable(rule)

            # with docstrings for PLY
            assert rule.__doc__ is not None
            assert rule.__doc__.strip() != ""

            # functions should be named with p_ prefix."""
            assert rule.__name__.startswith("p_")


class TestLexerParserRules:
    """
    One test per PLY grammar rule in ``make_parser_rules``.

    Each test parses a Physika program and test proper structure of produce
    ASTNode.
    """

    def test_lexer_rules(self):
        """
        Test for new lexer rules added to Physika classes.
        """
        # lexer rules have CLASS and DOT tokens
        rules = ClassFeature().lexer_rules()
        assert "CLASS" in rules["tokens"]
        assert "DOT" in rules["tokens"]

        # 'class' keyword is mapped to CLASS token
        assert rules["reserved"]["class"] == "CLASS"

        # t_DOT regex pattern matches a literal dot
        t_dot = rules["token_funcs"][0]
        pattern = t_dot.__doc__
        assert re.fullmatch(pattern, ".") is not None
        assert re.fullmatch(pattern, "a") is None

    def test_p_statement_class(self):
        """Checks class defintions with and without constructor parameters"""

        # classo typo in keyword raises SyntaxError
        with pytest.raises(SyntaxError):
            parse_physika("classo Particle:\n    mass : ℝ\n")

        # class with no constructor params
        class_name = 'Particle'
        ast = parse_physika("class Particle:\n    mass : ℝ\n")
        # (mass is a field promoted to class_params)
        assert ast["classes"][class_name]['class_params'] == [("mass", "ℝ")]

        # class with constructor params
        class_name = 'HNN'
        ast = parse_physika(
            "class HNN(w: ℝ):\n    def ke() : ℝ:\n        return w\n")

        assert ast["classes"][class_name]['class_params'] == [("w", "ℝ")]

        # class name used as a type annotation in a function
        ast = parse_physika("def run(p: Particle) : ℝ:\n    return 0.0\n")
        params = ast['functions']["run"]["params"]
        assert ("p", ("struct_type", "Particle")) in params

    def test_p_class_fields(self):
        """Checks that class fields are registered."""

        # 3 fields pos, vel and mass
        ast = parse_physika(
            "class P:\n    pos : ℝ\n    vel : ℝ\n    mass : ℝ\n")
        # promoted to class_params
        class_P = ast['classes']['P']
        assert [p[0]
                for p in class_P["class_params"]] == ["pos", "vel", "mass"]

        assert class_P["fields"] == []
        assert class_P["methods"] == []

        # field declaration stores (var_name, type) values in class_params
        assert ("mass", "ℝ") in class_P["class_params"]

        # field access via 'this.*' in method body
        ast = parse_physika(
            "class P:\n    mass : ℝ\n    def ke() : ℝ:\n        return 0.5 * this.mass\n"  # noqa: E501
        )

        class_P = ast['classes']['P']

        body = next(m for m in class_P["methods"] if m["name"] == "ke")["body"]

        assert ("field_access", ("var", "this"), "mass") in body
        # in this case, body have just a return statement
        assert body == ('mul', ('num', 0.5), ('field_access', ('var', 'this'),
                                              'mass'))

    def test_p_class_method(self):
        """Method declaration is registered in the methods list."""

        # check ke method exists in class's P ASTNode (empty params)
        ast = parse_physika(
            "class P:\n    mass : ℝ\n    def ke() : ℝ:\n        return mass\n")

        class_P = ast['classes']['P']
        assert any(m["name"] == "ke" for m in class_P["methods"])

        # Method with params, body statements and return
        ast = parse_physika(
            "class P:\n    x : ℝ\n    def step(dt: ℝ) -> ℝ:\n        v : ℝ = 2.0\n        return dt * v\n"  # noqa: E501
        )
        class_P = ast['classes']['P']

        m = next(m for m in class_P["methods"] if m["name"] == "step")
        assert set(m.keys()) == {
            'name', 'params', 'return_type', 'statements', 'body'
        }
        assert m["params"] == [("dt", "ℝ")]
        assert m['return_type'] == 'ℝ'
        assert m['statements'] == [('body_decl', 'v', 'ℝ', ('num', 2.0))]
        assert m['body'] == ('mul', ('var', 'dt'), ('var', 'v'))

        # Method with params and return (body) leaves statements empty
        ast = parse_physika(
            "class P:\n    w : ℝ\n    def scale(s: ℝ) -> ℝ:\n        return s\n"  # noqa: E501
        )
        class_P = ast['classes']['P']

        m = next(m for m in class_P["methods"] if m["name"] == "scale")
        assert m["params"] == [("s", "ℝ")]
        assert m["statements"] == []
        assert m["body"] == ('var', 's')

        # Method with no params, no stmts and return only
        ast = parse_physika(
            "class P:\n    mass : ℝ\n    def ke() : ℝ:\n        return mass\n")
        class_P = ast['classes']['P']

        m = next(m for m in class_P["methods"] if m["name"] == "ke")
        assert m["params"] == []
        assert m["statements"] == []
        assert m["body"] == ("var", "mass")

        # return is an expr
        ast = parse_physika(
            "class P:\n    mass : ℝ\n    def ke() : ℝ:\n        return mass\n")
        class_P = ast['classes']['P']

        body = next(m for m in class_P["methods"] if m["name"] == "ke")['body']
        assert body == ('var', 'mass')

        # return two expr, expr is a tuple_return node
        ast = parse_physika(
            "class P:\n    pos : ℝ\n    vel : ℝ\n    def step() : ℝ:\n        return pos, vel\n"  # noqa: E501
        )
        class_P = ast['classes']['P']
        body = next(m for m in class_P["methods"]
                    if m["name"] == "step")["body"]
        assert isinstance(body, tuple)
        assert body[0] == "tuple_return"
        assert body[1] == ("var", "pos")
        assert body[2] == ("var", "vel")
        assert body == ("tuple_return", ("var", "pos"), ("var", "vel"))

        # PhysikaClass.method(args) should produce ("method_call", obj_expr,
        # method_name, args) ASTNode
        ast = parse_physika("def run(p: P) : ℝ:\n    return p.ke()\n")
        body = ast['functions']["run"]["body"]
        function_name, function_args = "ke", []
        assert body == ("method_call", ("var", "p"), function_name,
                        function_args)

        # PhysikaClass.method() present in statements
        ast = parse_physika("def run(p: P) : ℝ:\n    p.step()\n")
        function_name, function_args = "step", []

        stmts = ast['functions']["run"]["statements"]
        assert ("body_expr", ("method_call", ("var", "p"), function_name,
                              function_args)) in stmts


class TestTypeRules:

    def test_field_and_access(self):
        """
        Tests for ``check_field_access`` type rule
        """
        # checks correct TScalar type infernec for a valid field on an instance

        check_field, _, s, class_env, infer_instance, _ = make_env()
        node = ("field_access", ("var", "v"), "x")
        t, _ = check_field(node, {}, s, {}, class_env, lambda _: None,
                           infer_instance)
        assert isinstance(t, TScalar)

        # y is declared as ℝ[2]
        check_field, _, s, class_env, infer_instance, _ = make_env()
        node = ("field_access", ("var", "v"), "y")
        t, _ = check_field(node, {}, s, {}, class_env, lambda _: None,
                           infer_instance)
        assert isinstance(t, TTensor)

        # registers an error and returns None for a field not declared
        # on the class
        check_field, _, s, class_env, infer_instance, _ = make_env()
        errors = []
        node = ("field_access", ("var", "v"), "z")
        t, _ = check_field(node, {}, s, {}, class_env, errors.append,
                           infer_instance)
        assert t is None
        assert any("has no field 'z'" in e for e in errors)

        # Registers an error when the class name is used directly instead
        # of an instance
        check_field, _, s, class_env, _, infer_none = make_env()
        errors = []
        node = ("field_access", ("var", "Particle"), "x")
        t, _ = check_field(node, {}, s, {}, class_env, errors.append,
                           infer_none)
        assert t is None
        assert any("class constructor, not an instance" in e for e in errors)

    def test_method_call(self):
        """
        Tests for the ``check_method_call`` type rule
        """
        # Returns the declared TScalar return type
        _, check_method, s, class_env, infer_instance, _ = make_env()
        node = ("method_call", ("var", "p"), "ke", [])
        t, _ = check_method(node, {}, s, {}, class_env, lambda _: None,
                            infer_instance)
        assert isinstance(t, TScalar)

        # registers an error and returns None for a non declared method
        _, check_method, s, class_env, infer_instance, _ = make_env()
        errors = []
        node = ("method_call", ("var", "p"), "potential_energy", [])
        t, _ = check_method(node, {}, s, {}, class_env, errors.append,
                            infer_instance)
        assert t is None
        assert any("has no method 'potential_energy'" in e for e in errors)

        # registers an error when the argument count does not match
        _, check_method, s, class_env, infer_instance, _ = make_env()
        errors = []
        # step expects 1 arg (dt: R)
        # called with 0
        node = ("method_call", ("var", "p"), "step", [])
        check_method(node, {}, s, {}, class_env, errors.append, infer_instance)
        assert any("expects 1 argument" in e for e in errors)

        # registers an error when inferred argument type does not match
        # the declared parameter type
        _, check_method, s, class_env, infer_instance, _ = make_env()
        errors = []
        # step expects dt: ℝ (TScalar)
        # infer_instance returns TInstance for every expr
        node = ("method_call", ("var", "p"), "step", [("var", "p")])
        check_method(node, {}, s, {}, class_env, errors.append, infer_instance)
        print(errors)
        errors == 3
        assert any("expected" in e and "got" in e for e in errors)

        # error when the class name is used directly instead of an instance
        _, check_method, s, class_env, _, infer_none = make_env()
        errors = []
        node = ("method_call", ("var", "Particle"), "ke", [])
        t, _ = check_method(node, {}, s, {}, class_env, errors.append,
                            infer_none)
        assert t is None
        assert any("class constructor, not an instance" in e for e in errors)


class TestClassForwardRules:
    """
    Verifies code generated from forward rules is valid taking
    ASTNodes as inputs.
    """

    def test_forward_rules(self):
        """
        Takes ASTNodes and produce correct Pytorch code.
        """
        # forward_rules for field_access, method_call, and class_def
        rules = ClassFeature().forward_rules()
        assert set(
            rules.keys()) == {"field_access", "method_call", "class_def"}
        assert all(callable(h) for h in rules.values())

        # field_access emits '*class.field' string."""
        emit = ClassFeature().forward_rules()["field_access"]
        node = ("field_access", ("var", "p"), "mass")
        assert emit(node, lambda n: n[1]) == "p.mass"

        # method_call emits '*class.method(args)' call string.
        emit = ClassFeature().forward_rules()["method_call"]
        assert emit(("method_call", ("var", "p"), "ke", []),
                    lambda n: n[1]) == "p.ke()"
        assert emit(
            ("method_call", ("var", "p"), "step", [("var", "dt")]),
            lambda n: n[1],
        ) == "p.step(dt)"

        # class_def emits a complete nn.Module subclass
        emit = ClassFeature().forward_rules()["class_def"]
        class_def = build_class([("mass", "ℝ")], [])
        code = emit(("class_def", "Particle", class_def))
        assert "class Particle(nn.Module):" in code
        assert "def __init__(self, mass):" in code
        assert "torch.as_tensor(mass).float()" in code


class TestPhysikaClass:
    """
    Integration tests for ``ClassFeature`` using example from
    ``examples/physika_class.phyk``.

    Executes the full pipeline (lexer/parser, type check, and codegen) and
    checks numeric outputs match expected values.
    """

    def test_name(self):
        """ELF plugin name is 'physika-class'."""
        assert ClassFeature.name == "physika-class"

    def test_physika_class_example(self):
        """
        Verifies physika_class.phyk example runs correctly and checks
        output values.
        """
        a = ns()["a"]
        # checks field access
        assert float(a.x) == pytest.approx(3.0)
        assert float(a.y) == pytest.approx(4.0)
        # a.dot(b) == 3*1 + 4*0 == 3.0
        assert float(ns()["dot_ab"]) == pytest.approx(3.0)

        # c = a.scale(4) in examples returns a new Vec
        # with both components multiplied by 4.
        c = ns()["c"]
        assert float(c.x) == pytest.approx(a.x * 4)
        assert float(c.y) == pytest.approx(a.y * 4)
        # kinetic energy method call
        assert float(ns()["ke0"]) == pytest.approx(0.5 * 9 * (sum([1, 0]))**2)

        assert torch.allclose(ns()["p1"].pos.float(),
                              torch.tensor([0.5, 10.0]),
                              atol=1e-4)
        # takes correct gradient
        assert torch.allclose(ns()["dKE_dv"].float(),
                              torch.tensor([2.0, 3.4]),
                              atol=1e-4)
