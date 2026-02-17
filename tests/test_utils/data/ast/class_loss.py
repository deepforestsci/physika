EXPECTED = {
    "functions": {},
    "classes": {
        "M": {
            "class_params": [("w", "ℝ")],
            "lambda_params": [("x", "ℝ")],
            "return_type": "ℝ",
            "body": ("mul", ("var", "w"), ("var", "x")),
            "has_loop": False,
            "has_loss": True,
            "loss_params": [("y", "ℝ"), ("t", "ℝ")],
            "loss_body": ("pow", ("sub", ("var", "y"), ("var", "t")), ("num", 2.0)),
        }
    },
    "program": [("class_def", "M")],
}
