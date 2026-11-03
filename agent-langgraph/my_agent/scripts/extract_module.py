import ast
import argparse
from typing import List, Dict, Optional


class ClassFunctionExtractor(ast.NodeVisitor):
    def __init__(self, class_names: List[str], func_filter: Optional[Dict[str, List[str]]] = None):
        self.class_names = set(class_names)
        self.func_filter = func_filter or {}
        self.class_nodes = []

    def visit_ClassDef(self, node):
        if node.name in self.class_names:
            if node.name in self.func_filter:
                # Keep only specific methods
                filtered_body = [
                    item for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name in self.func_filter[node.name]
                ]
                node.body = filtered_body
            self.class_nodes.append(node)
        self.generic_visit(node)


def extract_filtered_code(source_path: str, class_names: List[str], func_filter: Optional[Dict[str, List[str]]] = None) -> str:
    with open(source_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    extractor = ClassFunctionExtractor(class_names, func_filter)
    extractor.visit(tree)

    lines = source_code.splitlines()
    output_lines = []

    for node in extractor.class_nodes:
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', None)
        if end is None:
            end = len(lines)

        # Slice the class block
        class_block = lines[start:end]

        # If filtering functions, we recompile a new class block from AST
        if node.name in extractor.func_filter:
            new_code = ast.unparse(node)  # Python 3.9+
            output_lines.append(new_code)
        else:
            output_lines.extend(class_block)

        output_lines.append("")  # blank line between classes

    return "\n".join(output_lines)


def parse_func_filter(arg: List[str]) -> Dict[str, List[str]]:
    """
    Parse format like: ClassA:func1,func2 ClassB:func3
    """
    result = {}
    for entry in arg:
        if ":" not in entry:
            continue
        cls, funcs = entry.split(":")
        result[cls] = funcs.split(",")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specified class (and optionally function) definitions from a Python file.")
    parser.add_argument("source", help="Path to the source Python file")
    parser.add_argument("classes", nargs="+", help="Class names to extract")
    parser.add_argument("--functions", nargs="*", help='Optional: specify functions in format ClassA:func1,func2 ClassB:func3', default=[])
    parser.add_argument("--output", help="Optional output file")

    args = parser.parse_args()
    func_filter = parse_func_filter(args.functions)

    print(func_filter)
    result = extract_filtered_code(args.source, args.classes, func_filter)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)
