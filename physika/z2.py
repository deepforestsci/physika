from typing import Union

class Z2:
    """
    A class representing elements of the finite field Z2 (integers modulo 2).

    Parameters
    ----------
    val : int
        The integer value to be reduced modulo 2.

    Examples
    --------
    >>> a = Z2(1)
    >>> b = Z2(0)
    >>> # XOR operation
    >>> a + b
    1
    >>> # AND operation
    >>> a * b
    0
    """

    def __init__(self, val: Union[int, "Z2"]) -> "Z2":
        """
        Initialize Z2 element.

        Parameters
        ----------
        val : int
            The integer value to be reduced modulo 2.
        
        Returns
        -------
        None
        """
        self.val = int(val) % 2

    def __add__(self, other: Union[int, "Z2"]) -> "Z2":
        """
        Add two Z2 elements.

        Parameters
        ----------
        other : Z2 or int
            The element to add.
        
        Returns
        -------
        Z2
            A new Z2 element.
        """
        other_val = other.val if isinstance(other, Z2) else int(other)
        return Z2(self.val + other_val)

    def __mul__(self, other: Union[int, "Z2"]) -> "Z2":
        """
        Multiply two Z2 elements.

        Parameters
        ----------
        other : Z2 or int
            The element to multiply.
        
        Returns
        -------
        Z2
            A new Z2 element.
        """
        other_val = other.val if isinstance(other, Z2) else int(other)
        return Z2(self.val * other_val)

    def __repr__(self) -> str:
        """
        Return the string representation of the Z2 element.

        Returns
        -------
        str
            The string representation ("0" or "1").
        """
        return f"{self.val}"

    def __bool__(self) -> bool:
        """
        Convert the Z2 element to a boolean.

        Returns
        -------
        bool
            False if value is 0, True if value is 1.
        """
        return bool(self.val)

    def __int__(self) -> int:
        """
        Convert the Z2 element to an integer.

        Returns
        -------
        int
            The integer value (0 or 1).
        """
        return self.val

    def __eq__(self, other: object) -> bool:
        """
        Check equality between two Z2 elements.

        Parameters
        ----------
        other : Z2 or int
            The element to compare with.

        Returns
        -------
        bool
            True if values are equal modulo 2, False otherwise.
        """
        other_val = other.val if isinstance(other, Z2) else int(other)
        return self.val == other_val
