import sys

from lexer import lexer
from parser import parser, symbol_table
from type_checker import type_check
from codegen import from_ast_to_torch
from utils.print_utils import print_type_check_results
from utils.ast_utils import build_unified_ast

if __name__ == "__main__":
    print_code = "--print-code" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    with open(args[0], "r", encoding="utf-8") as f:
        source = f.read()

    # Parse tokens to AST
    local_program_ast = parser.parse(source, lexer=lexer)
    # Build unified AST (I think this can be done in parser)
    unified_ast = build_unified_ast(local_program_ast, symbol_table, print_ast=False)
    
    # Type checking
    type_status = type_check(unified_ast)
    print_type_check_results(type_status)

    # Generate PyTorch code and execute it
    generated_code = from_ast_to_torch(unified_ast, print_code=print_code)
    exec(generated_code)
