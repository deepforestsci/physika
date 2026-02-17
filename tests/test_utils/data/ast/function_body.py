EXPECTED = {
    "functions": {
        "g": {
            "params": [("a", "ℝ"), ("b", "ℝ")],
            "return_type": "ℝ",
            "body": ("var", "c"),
            "has_loop": False,
            "statements": [
                ("body_assign", "c", ("add", ("var", "a"), ("var", "b"))),
            ],
        }
    },
    "classes": {},
    "program": [("func_def", "g")],
}
