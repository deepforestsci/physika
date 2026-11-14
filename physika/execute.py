import sys
from pathlib import Path

import physika.runtime
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.codegen import from_ast_to_torch
from physika.type_checker import TypeChecker
from physika.utils.print_utils import print_type_check_results
from physika.utils.ast_utils import build_unified_ast
from physika.utils.import_manager import resolve_imports

from physika.core.elab.elab import Elab
from physika.core.inductive import mk_builtin_env
from physika.units import dim_analysis
from physika.solver.solver import solve_equations, solve_bindings


def main():
    print_code = "--print-code" in sys.argv
    print_ast = "--print-ast" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    source_file_path = Path(args[0]).resolve()
    with open(args[0], "r", encoding="utf-8") as f:
        source = f.read()

    # Parse tokens to AST
    local_program_ast = parser.parse(source, lexer=lexer)
    local_program_ast = resolve_imports(local_program_ast, source_file_path)

    physika_nodes = [s for s in local_program_ast]

    # Build CIC env with user defined functions
    cic_elab = Elab(mk_builtin_env())

    unified_ast = None
    func_sigs: dict = {}  # name keys, (param units, return units) values
    solved_bindings: dict = {}  # Solve-derived closed forms, name -> AST
    if physika_nodes:
        unified_ast = build_unified_ast(physika_nodes,
                                        symbol_table,
                                        print_ast=print_ast)

        # 1. CIC elaboration and kernel check
        cic_result = cic_elab.elaborate(unified_ast)
        cic_errors = list(cic_result.get("errors", []))

        resolved_functions = set(cic_result.get("resolved_bodies") or {})
        resolved_methods = set(cic_result.get("resolved_methods") or {})

        # Dimensional analysis check
        unit_errors = dim_analysis(unified_ast, cic_elab.state.env, func_sigs)

        if unit_errors:
            print(f"{len(unit_errors)} dimensional analysis error(s) found.")
            for e in unit_errors:
                print(f"  ✗ {e}")

        # Kernel-verified equation solving
        solve_report = solve_equations(unified_ast, cic_elab.state.env)
        if solve_report:
            print("\nEquation solver:")
            for line in solve_report:
                print(f"  {line}")

        # solve closed forms as runtime bindings
        solved_bindings = solve_bindings(unified_ast, cic_elab.state.env)

        if cic_errors:
            print(f"(CIC: {len(cic_errors)} construct(s) not fully verified "
                  "yet; switching to HM type checking and standard codegen "
                  "for those)")
            for e in cic_errors:
                print(f"    - {e}")

        # 2. Fall back to the Hindley-Milner for terms that failed elaboration.
        type_status = TypeChecker(
            unified_ast,
            skip_functions=resolved_functions,
            skip_methods=resolved_methods,
        ).run()
        print_type_check_results(type_status)

    if not physika_nodes:
        return

    generated_code = from_ast_to_torch(
        unified_ast,
        print_code=print_code,
        resolved_bodies=cic_result.get("resolved_bodies"),
        resolved_methods=cic_result.get("resolved_methods"),
        resolved_program=cic_result.get("resolved_program"),
        resolved_program_fvar_names=cic_result.get(
            "resolved_program_fvar_names"),
        cic_env=cic_elab.state.env,
        func_sigs=func_sigs,
        solve_bindings=solved_bindings,
    )
    exec(generated_code, vars(physika.runtime))


if __name__ == "__main__":
    main()
