from physika.elf import ELF
from typing import Callable
from physika.utils.types import TTensor, Substitution


def make_parser_rules():

    def p_factor_transpose(p):
        """factor : ID TRANSPOSE"""
        p[0] = ("transpose", ("var", p[1]))

    def p_func_factor_transpose(p):
        """func_factor : ID TRANSPOSE"""
        p[0] = ("transpose", ("var", p[1]))

    return [p_factor_transpose, p_func_factor_transpose]


class GreekLetters(ELF):

    name = "greek_letters"

    def lexer_rules(self) -> dict:

        def t_TRANSPOSE(t):
            r"ᵀ"
            return t        

        return {
            "reserved": {
                "physika": "PHYSIKA"
            },
            "tokens": ["TRANSPOSE", "PARTIAL"],
            "token_funcs": [
                t_TRANSPOSE
                ]
        }

    def parser_rules(self) -> list:
        return make_parser_rules()

    def type_rules(self) -> dict:

        def expr_transpose(node: tuple, env: dict, s: Substitution,
                           func_env: dict, class_env: dict,
                           add_error: Callable, infer_expr: Callable):
            # AST: ("transpose", ("var", "x"))
            operand = node[1]

            operand_type, s = infer_expr(
                operand,
                env,
                s,
                func_env,
                class_env,
                add_error,
            )
            if operand_type is None:
                return None, s

            if len(operand_type.dims) != 2:
                add_error(f"Transpose requires a 2D tensor, "
                          f"got {len(operand_type.dims)} dimensions")
                return None, s

            operand_type = s.apply(operand_type)
            return TTensor((operand_type.dims[1], operand_type.dims[0])), s

        return {"transpose": expr_transpose}

    def forward_rules(self) -> dict:

        def emit_transpose(node: tuple, to_expr: Callable, **ctx) -> str:
            var = node[1][1]
            return f"torch.transpose({var}, 0, 1)"

        return {"transpose": emit_transpose}
