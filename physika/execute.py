import sys
import physika.runtime
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.type_checker import type_check
from physika.codegen import from_ast_to_torch
from physika.utils.print_utils import print_type_check_results
from physika.utils.ast_utils import build_unified_ast
from physika.import_manager import resolve_imports
from pathlib import Path


def main():
    print_code = "--print-code" in sys.argv
    print_ast = "--print-ast" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    source_file_path = Path(args[0]).resolve()
    print(source_file_path)
    with open(args[0], "r", encoding="utf-8") as f:
        source = f.read()

    # Parse tokens to AST
    local_program_ast = parser.parse(source, lexer=lexer)

    local_program_ast = resolve_imports(local_program_ast, source_file_path)

    # Build unified AST (I think this can be done in parser)
    unified_ast = build_unified_ast(local_program_ast,
                                    symbol_table,
                                    print_ast=print_ast)

    # Type checking
    type_status = type_check(unified_ast)
    print_type_check_results(type_status)

    # Generate PyTorch code and execute it
    generated_code = from_ast_to_torch(unified_ast, print_code=print_code)
    exec(generated_code, vars(physika.runtime))


if __name__ == "__main__":
    main()
