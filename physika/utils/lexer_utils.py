_MANGLE = {
    "\u2202": "partial_",
    "\u00B2": "_sq",
    "\u2044": "_over_",
}


def mangle_identifier(name: str) -> str:
    """
    Replace characters that Python identifiers cannot contain with ASCII
    equivalents.

    Parameters
    ----------
    name : str
        The identifier text as matched by the lexer
        (e.g. ``"∂𝘓⁄∂x"``).

    Returns
    -------
    str
        The identifier with every character found in ``_MANGLE`` replaced
        by its mapped string (e.g. ``"partial_𝘓_over_partial_x"``).

    Examples
    --------
    >>> mangle_identifier("∂x")
    'partial_x'
    >>> mangle_identifier("x²")
    'x_sq'
    >>> mangle_identifier("∂𝘓⁄∂x")
    'partial_𝘓_over_partial_x'
    >>> mangle_identifier("alpha")
    'alpha'
    """
    return "".join(_MANGLE.get(ch, ch) for ch in name)
