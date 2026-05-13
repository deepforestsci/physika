from physika.features.classes import (
    is_learnable,
    replace_class_params,
    unwrap_return,
    build_class,
    emit_method,
    generate_class,
    make_parser_rules,
)


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
        assert result["constructor_params"] == [("mass", "ℝ")]

        # fields is always an empty list
        assert result["fields"] == []

        # multiple fields
        body = [
            ("field_decl", "pos", ("tensor", [2])),
            ("field_decl", "vel", ("tensor", [2])),
            ("field_decl", "mass", "ℝ"),
        ]
        result = build_class(None, body)
        names = [p[0] for p in result["constructor_params"]]
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
        assert result["constructor_params"] == params
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
