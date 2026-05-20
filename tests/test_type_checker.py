from physika.type_checker import TypeChecker
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.utils.ast_utils import build_unified_ast

PHYK_PROGRAM = """\
x : ℝ[3] = [1.0, 2.0, 3.0]
y : ℝ[3] = [4.0, 5.0, 6.0]

for k:
    x[k] = x[k] * 2.0
flag : ℝ = 1.0

if flag > 0.5:
    s = 2.0
else:
    s = 1.0
sum_s : ℝ = 10 + s

def dot(a : ℝ[3], b : ℝ[3]) : ℝ:
    acc : ℝ
    for k:
        acc += a[k] * b[k]
    return acc

def nested_range(s : ℝ) : ℝ:
    result : ℝ
    for i: ℕ(3):
        for j: ℕ(i, 3):
            result += s * i * 1.0 + s * j * 2.0
    return result

def scale_vec(a : ℝ[3], c : ℝ) : ℝ[3]:
    return for i : ℕ(3) → a[i] * c

def relu(v : ℝ) : ℝ:
    if v > 0.0:
        return v
    else:
        return 0.0

class Vec:
    x : ℝ
    y : ℝ
    def norm_sq() : ℝ:
        return this.x * this.x + this.y * this.y
    def dot_with(other : Vec) : ℝ:
        return this.x * other.x + this.y * other.y

r : ℝ = dot(x, y)
v: Vec = Vec(3.0, 4.0)
w: Vec = Vec(1.0, 0.0)
relu(r)
"""


def typecheck_phyk(source: str) -> list[str]:
    """
    Helper function that parse a Physika source program and return TypeChecker
    errors.
    """
    symbol_table.clear()
    lexer.lexer.lineno = 1
    program_ast = parser.parse(source, lexer=lexer)
    unified = build_unified_ast(program_ast, symbol_table)
    return TypeChecker(unified).run()


class TestTypeCheckerCleanPrograms:
    """Run TypeChecker for well typed programs."""

    def test_basic_stmts(self):
        """
        Checks that physika programs with basic statements are well typed.
        Basic statements include `decl`, `assing`, `for-loop`, `if-else` nodes
        over scalar (ℝ), array (ℝ[n]), matrices (ℝ[n, m]), tensors (ℝ[n, m, o])
        , etc types.
        """
        # scalar decl
        assert typecheck_phyk("x : ℝ = 1.0\n") == []

        # 1d array decl
        assert typecheck_phyk("v : ℝ[3] = [1.0, 2.0, 3.0]\n") == []
        # `assign` node
        assert typecheck_phyk("x = 42.0\n") == []

        # operations over 1d arrays using for-loops
        assert typecheck_phyk("x : ℝ[3] = [1.0, 2.0, 3.0]\n"
                              "for k:\n"
                              "    x[k] = x[k] * 2.0\n") == []

        # if-else blocks scalar check
        assert typecheck_phyk("x : ℝ = 0.3\n"
                              "if x > 0.5:\n"
                              "    y = 3.0\n"
                              "else:\n"
                              "    y = 0.0\n") == []

        assert typecheck_phyk("x : ℝ = 1.0\n"
                              "if x > 0.0:\n"
                              "    x = x + 1.0\n") == []

    def test_valid_function(self):
        """
        Verifies functions args and return types matches declared
        params types and number. Also checks inner body statements are
        well typed.
        """

        # declared return type matches inferred type ℝ
        assert typecheck_phyk("def f(x : ℝ) : ℝ:\n"
                              "    return x\n") == []

        # function calls
        assert typecheck_phyk("def sq(x : ℝ) : ℝ:\n"
                              "    return x ** 2\n"
                              "def apply_sq(x : ℝ) : ℝ:\n"
                              "    return sq(x)\n") == []

        # function calls works properly with variables at top-level program
        assert typecheck_phyk("def f(x : ℝ) : ℝ:\n"
                              "    return x\n"
                              "x : ℝ = 1.0\n"
                              "f(x)\n") == []

        # for loop inside a function
        assert typecheck_phyk("def dot(x : ℝ[3], y : ℝ[3]) : ℝ:\n"
                              "    s : ℝ = 0.0\n"
                              "    for k:\n"
                              "        s += x[k] * y[k]\n"
                              "    return s\n") == []

        # Explicit range for-loop
        assert typecheck_phyk(
            "def nested_sum(s : ℝ) : ℝ:\n"
            "    result : ℝ = 0.0\n"
            "    for i: ℕ(4):\n"
            "        for j: ℕ(i, 4):\n"
            "            result += s * i * 1.0 + s * j * 1.0\n"
            "    return result\n") == []

        # implicit for loops
        assert typecheck_phyk(
            "def scale_vec(x : ℝ) : ℝ[3]:\n"
            "    return for i : ℕ(3) → x * (i + 1.0)\n") == []

        # Function if-else return branches
        assert typecheck_phyk("def relu(x : ℝ) : ℝ:\n"
                              "    if x > 0.0:\n"
                              "        return x\n"
                              "    else:\n"
                              "        return 0.0\n") == []

        # function with if-only early return
        assert typecheck_phyk("def f(x : ℝ) : ℝ:\n"
                              "    if x > 0.0:\n"
                              "        return x\n"
                              "    return 0.0\n") == []

        # if-else for branch assignment (no return)
        assert typecheck_phyk("def clamp(x : ℝ) : ℝ:\n"
                              "    if x > 1.0:\n"
                              "        y = 1.0\n"
                              "    else:\n"
                              "        y = x\n"
                              "    return y\n") == []

    def test_valid_class(self):
        """
        Verifies class params, fields, methods and return types matches
        declared params types and number. Also checks inner method body
        statements are well typed.
        """

        # tests fieds, methods and returns are well typed
        assert typecheck_phyk(
            "class Vec:\n"
            "    x : ℝ\n"
            "    y : ℝ\n"
            "    def norm_sq() : ℝ:\n"
            "        return this.x * this.x + this.y * this.y\n") == []

        # tests class instantiation
        assert typecheck_phyk("class Vec:\n"
                              "    x : ℝ\n"
                              "    y : ℝ\n"
                              "Vec(1.0, 2.0)\n") == []

    def test_program_with_multiple_operations(self):
        """
        Physika program that exercises most of type checkable objects.
        This test verifies type checking for top-level decls, for loops
        + if/else, functions with for loop / nested range for loop / if-else
        return branches, class with fields and multiple methods (including a
        method that takes another instance as a parameter), and top level
        calls to functions.
        """

        assert typecheck_phyk(PHYK_PROGRAM) == []


class TestTypeCheckerErrors:
    """TypeChecker reports errors on bad typed physika programs."""

    def test_top_level_type_mismatch(self):
        # declared ℝ[3], inferred ℝ
        errors = typecheck_phyk("v : ℝ[3] = 0.0\n")
        assert len(errors) == 1
        assert errors[
            0] == "Line 1: Type mismatch for 'v': declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # multiple type error indicates different lines source
        errors = typecheck_phyk("a : ℝ[2] = 0.0\n"  # declared ℝ[2], inferred ℝ
                                "b : ℝ[3] = 0.0\n"  # declared ℝ[3], inferred ℝ
                                )
        assert len(errors) == 2
        assert errors[
            0] == "Line 1: Type mismatch for 'a': declared ℝ[2], got ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501
        assert errors[
            1] == "Line 2: Type mismatch for 'b': declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

    def test_functions_type_mismatch(self):
        """
        Functions are checked for parameters and return types and number. A
        mismatch reports an error. Inner function statements, body statements,
        are also type checked using loca enviroment.
        """
        errors = typecheck_phyk("def f(x : ℝ, y : ℝ) : ℝ:\n"  # recieves 2 args
                                "    return x\n"
                                "f(1.0)\n"  # error, given 1
                                )
        assert len(errors) == 1
        assert "Line 0: Function 'f' expects 2 args, got 1" == errors[0]

        # declared return type is ℝ[3] but infers ℝ
        errors = typecheck_phyk("def f(x : ℝ) : ℝ[3]:\n"
                                "    return x\n")
        assert len(errors) == 1
        assert errors[
            0] == "In function 'f': return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # v declared in body statements is ℝ[3] but infers ℝ
        errors = typecheck_phyk("def f(x : ℝ) : ℝ:\n"
                                "    v : ℝ[3] = 0.0\n"
                                "    return x\n")
        assert len(errors) == 1
        assert errors[
            0] == "In function 'f': In 'f': 'v' declared ℝ[3], inferred ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # errors in separate functions are reported independently
        errors = typecheck_phyk("def f1(x : ℝ) : ℝ[2]:\n"
                                "    return x\n"
                                "def f2(x : ℝ) : ℝ[3]:\n"
                                "    return x\n")
        assert len(errors) == 2
        assert errors[
            0] == "In function 'f1': return type mismatch: declared ℝ[2], got ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501
        assert errors[
            1] == "In function 'f2': return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # implicit for loop with a mismatch declared return type
        errors = typecheck_phyk("def scale_vec(x : ℝ) : ℝ:\n"
                                "    return for i : ℕ(3) → x * (i + 1.0)\n")
        assert errors
        assert len(errors) == 1
        assert errors[
            0] == "In function 'scale_vec': return type mismatch: declared ℝ, got ℝ[3]: Cannot unify scalar ℝ with tensor ℝ[3]"  # noqa: E501

        # if-else returning the wrong type
        errors = typecheck_phyk("def f(x : ℝ) : ℝ[3]:\n"
                                "    if x > 0.0:\n"
                                "        return x\n"
                                "    else:\n"
                                "        return 0.0\n")
        assert len(errors) == 1
        assert errors[
            0] == "In function 'f': if/else return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # if branch returning the wrong type
        errors = typecheck_phyk("def f(x : ℝ) : ℝ[3]:\n"
                                "    if x > 0.0:\n"
                                "        return x\n"
                                "    return 0.0\n")
        assert len(errors) == 2
        assert errors[
            0] == "In function 'f': if-return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501
        assert errors[
            1] == "In function 'f': return type mismatch: declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

    def test_class_type_mismatch(self):
        """
        Verifies that errors are reported correctly when type checking
        Physika classes. Type checker looks for correct number and types
        of constructor parameters. Also, stores classes's fields and methods
        where types are checked as functions having the context of the class
        env.
        """

        errors = typecheck_phyk("class Vec:\n"
                                "    x : ℝ\n"
                                "    y : ℝ\n"
                                "    def bad() : ℝ[2]:\n"  # declared ℝ[2]
                                "        return this.x\n"  # returning ℝ
                                )
        assert len(errors) == 1
        assert errors[
            0] == "In class 'Vec', method 'bad': return type mismatch: declared ℝ[2], got ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501

        # method calls a function with wrong param number
        errors = typecheck_phyk(
            "def sq(x : ℝ) : ℝ:\n"
            "    return x * x\n"
            "class Vec:\n"
            "    x : ℝ\n"
            "    y : ℝ\n"
            "    def m() : ℝ:\n"
            "        return sq(this.x, this.y)\n"  # sq recieves one param
        )
        assert len(errors) == 1
        assert errors[
            0] == "In class 'Vec', method 'm': Function 'sq' expects 1 args, got 2"  # noqa: E501

        # a class constructor with the wrong number of args
        errors = typecheck_phyk("class Vec:\n"
                                "    x : ℝ\n"
                                "    y : ℝ\n"
                                "Vec(1.0)\n")
        assert len(errors) == 1
        assert errors[0] == "Line 0: Function 'Vec' expects 2 args, got 1"

        # a class constructor with the wrong types
        errors = typecheck_phyk("class Vec:\n"
                                "    x : ℝ\n"
                                "    y : ℝ\n"
                                "Vec([1.0, 0.0], [0.0, 1.0])\n")
        assert len(errors) == 2
        assert "Line 0: Arg 0 of 'Vec': Cannot unify scalar ℝ with tensor ℝ[2]" == errors[  # noqa: E501
            0]
        assert "Line 0: Arg 1 of 'Vec': Cannot unify scalar ℝ with tensor ℝ[2]" == errors[  # noqa: E501
            1]

    def test_program_with_multiple_operations(self):
        """
        Uses PHYK_PROGRAM and introduce type errors with
        string replacement. Each case modifies one declaration or return type
        and asserts the error message produced.
        """
        # function return type declared ℝ but returns ℝ[3]
        modified = PHYK_PROGRAM.replace(
            "def scale_vec(a : ℝ[3], c : ℝ) : ℝ[3]:",
            "def scale_vec(a : ℝ[3], c : ℝ) : ℝ:",
        )
        errors = typecheck_phyk(modified)
        assert len(errors) == 1
        assert errors[
            0] == "In function 'scale_vec': return type mismatch: declared ℝ, got ℝ[3]: Cannot unify scalar ℝ with tensor ℝ[3]"  # noqa: E501

        # if/else branches in `relu` return ℝ
        # but declared return type is ℝ[2]
        modified = modified.replace(
            "def relu(v : ℝ) : ℝ:",
            "def relu(v : ℝ) : ℝ[2]:",
        )
        errors = typecheck_phyk(modified)
        assert len(errors) == 2
        assert errors[
            1] == "In function 'relu': if/else return type mismatch: declared ℝ[2], got ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501

        # `Vec` `norm_sq` method return type declared ℝ[2] but returns ℝ
        modified = modified.replace(
            "    def norm_sq() : ℝ:",
            "    def norm_sq() : ℝ[2]:",
        )
        errors = typecheck_phyk(modified)
        assert len(errors) == 3
        assert errors[
            2] == "In class 'Vec', method 'norm_sq': return type mismatch: declared ℝ[2], got ℝ: Cannot unify tensor ℝ[2] with scalar ℝ"  # noqa: E501

        # top level decl declared ℝ[3] but `dot()`` returns ℝ
        modified = modified.replace(
            "r : ℝ = dot(x, y)",
            "r : ℝ[3] = dot(x, y)",
        )
        errors = typecheck_phyk(modified)
        assert len(errors) == 4
        assert errors[
            3] == "Line 44: Type mismatch for 'r': declared ℝ[3], got ℝ: Cannot unify tensor ℝ[3] with scalar ℝ"  # noqa: E501

        # class constructor called with wrong types
        modified = modified.replace(
            "v: Vec = Vec(3.0, 4.0)",
            "v: Vec = Vec([3.0, 4.0], [4.0, 3.0])",
        )
        errors = typecheck_phyk(modified)

        assert len(errors) == 6
        assert errors[
            4] == "Line 45: Arg 0 of 'Vec': Cannot unify scalar ℝ with tensor ℝ[2]"  # noqa: E501
        assert errors[
            5] == "Line 45: Arg 1 of 'Vec': Cannot unify scalar ℝ with tensor ℝ[2]"  # noqa: E501

        # class constructor called with wrong args number
        modified = modified.replace(
            "w: Vec = Vec(1.0, 0.0)",
            "w: Vec = Vec(1.0)",
        )
        errors = typecheck_phyk(modified)
        assert len(errors) == 7
        assert errors[6] == "Line 46: Function 'Vec' expects 2 args, got 1"
