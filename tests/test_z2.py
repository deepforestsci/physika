import pytest
from physika.z2 import Z2


class TestZ2:
    """Tests XOR operations"""

    @pytest.mark.parametrize("a,b,result", [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ])
    def test_xor(self, a, b, result):
        assert Z2(a) + Z2(b) == Z2(result)

    """Tests AND operations"""

    @pytest.mark.parametrize("a,b,result", [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
    ])
    def test_and(self, a, b, result):
        assert Z2(a) * Z2(b) == Z2(result)

    """Test commutativity of addition and multiplication in Z2"""

    def test_commutativity(self):
        a, b = Z2(1), Z2(0)
        assert a + b == b + a
        assert a * b == b * a

    """Test associativity of addition and multiplication in Z2"""

    def test_associativity(self):
        a, b, c = Z2(1), Z2(1), Z2(0)
        assert (a + b) + c == a + (b + c)
        assert (a * b) * c == a * (b * c)

    """Test equality and inequality between Z2 elements"""

    def test_equality(self):
        assert Z2(1) == Z2(1)
        assert Z2(0) != Z2(1)
