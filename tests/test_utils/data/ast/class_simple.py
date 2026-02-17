EXPECTED = {
    "functions": {},
    "classes": {
        "Net": {
            "class_params": [("w", "ℝ")],
            "lambda_params": [("x", "ℝ")],
            "return_type": "ℝ",
            "body": ("mul", ("var", "w"), ("var", "x")),
            "has_loop": False,
            "has_loss": False,
        }
    },
    "program": [("class_def", "Net")],
}
