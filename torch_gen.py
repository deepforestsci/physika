from tests.test_tutorials import mk_builtin_env, dim_analysis, parse_source_to_ast, Elab, from_ast_to_torch
from pathlib import Path

TORCH_CODE_DIR = Path('tutorials/torch_code')

def test_codegen_matches_reference(phyk_file):
    """
    Code generated from each .phyk file must exactly match its reference
    python file in torch_code dir
    """
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
    #code_torch = (TORCH_CODE_DIR / f"{phyk_file.stem}.py").read_text()
    
    return code_phyk


res = test_codegen_matches_reference(Path('tutorials/electrostatics_and_circuit_dynamics.phyk'))
print(res)

