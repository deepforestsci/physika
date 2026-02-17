EXPECTED = {
    "functions": {
        "f": {
            "params": [("x", "ℝ")],
            "return_type": "ℝ",
            "body": ("var", "x"),
            "has_loop": False,
            "statements": [],
        }
    },
    "classes": {},
    "program": [
        ("func_def", "f"),
        ("expr", ("call", "f", [("num", 1.0)]), 0),
    ],
}
