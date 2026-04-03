# Draft version class


class Z2:

    def __init__(self, val):
        self.val = int(val) % 2

    def __add__(self, other):
        other_val = other.val if isinstance(other, Z2) else int(other)
        return Z2(self.val + other_val)

    def __mul__(self, other):
        other_val = other.val if isinstance(other, Z2) else int(other)
        return Z2(self.val * other_val)

    def __repr__(self):
        return f"{self.val}"

    def __bool__(self):
        return bool(self.val)

    def __int__(self):
        return self.val

    def __eq__(self, other):
        other_val = other.val if isinstance(other, Z2) else int(other)
        return self.val == other_val
