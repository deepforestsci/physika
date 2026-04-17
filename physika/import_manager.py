from physika.parser import parser
from physika.lexer import lexer
from pathlib import Path


def find_module(module_name: str, source_file_path: Path) -> Path:
    search_paths = [
        source_file_path.parent,
        Path(".")
    ]
    for path in search_paths:
        file_name = path / f"{module_name}.phyk"
        if file_name.exists():
            return file_name
    raise ImportError(f"module '{module_name}' not found")

def resolve_imports(local_program_ast, source_file_path):
    extra_nodes = []
    for node in local_program_ast:
        if isinstance(node, tuple) and node[0] == "import":
            module_name = node[1]
            symbol = node[2]

            # load and parse the module
            module_path = find_module(module_name, source_file_path)
            with open(module_path, "r", encoding="utf-8") as f:
                source = f.read()

            module_ast = parser.parse(source, lexer=lexer)

            # find the requested symbol in module AST
            found = False
            for module_node in module_ast:
                if (isinstance(module_node, tuple) and module_node[1] == symbol):
                    extra_nodes.append(module_node)
                    found = True
                    break
            if not found:
                raise ImportError(
                    f"cannot import '{symbol}' from '{module_name}'"
                )
        else:
            extra_nodes.append(node)
    return extra_nodes
