EXPECTED = {
    "functions": {},
    "classes": {},
    "program": [
        ("decl", "arr", ("tensor", [(3, "invariant")]),
         ("array", [("num", 1.0), ("num", 2.0), ("num", 3.0)]), 1),
        ("decl", "total", "ℝ", ("num", 0.0), 2),
        ("for_loop", "i",
         [("for_pluseq", "total", ("index", "arr", ("imaginary",)))],
         ["arr"], 3),
    ],
}
