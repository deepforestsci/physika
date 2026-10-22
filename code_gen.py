import pytest

from pathlib import Path
from tests.test_tutorials import (
        parse_source_to_ast, Elab, mk_builtin_env, from_ast_to_torch,
        dim_analysis, TORCH_CODE_DIR, HEADER
    )

TUTORIAL = Path(__file__).parent / "tutorials" / "circular_fingerprints.phyk"

def codegen(phyk_file):
    src = phyk_file.read_text()
    unified_ast = parse_source_to_ast(
        src,
        phyk_file.resolve(),
    )
    cic_elab = Elab(mk_builtin_env())
    cic_result = cic_elab.elaborate(unified_ast)
    func_sigs: dict = {}
    dim_analysis(unified_ast, cic_elab.state.env, func_sigs)

    code_phyk = from_ast_to_torch(
        unified_ast,
        print_code=False,
        resolved_bodies=cic_result.get("resolved_bodies"),
        resolved_methods=cic_result.get("resolved_methods"),
        resolved_program=cic_result.get("resolved_program"),
        resolved_program_fvar_names=cic_result.get(
            "resolved_program_fvar_names"),
        cic_env=cic_elab.state.env,
        func_sigs=func_sigs,
    )
    code_torch = (TORCH_CODE_DIR / f"{phyk_file.stem}.py").read_text()

    return code_phyk

print(codegen(TUTORIAL))
