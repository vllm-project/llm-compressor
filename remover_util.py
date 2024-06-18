import ast
import os
import sys
from typing import Set, List
import argparse

class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)

def get_imports_from_file(file_path: str) -> Set[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)
    analyzer = ImportAnalyzer()
    analyzer.visit(tree)
    return analyzer.imports

def get_all_imports(files_to_check: List[str]) -> Set[str]:
    all_imports = set()
    checked_imports = set()

    def _check_imports(imports: Set[str]):
        for imp in imports:
            if imp not in checked_imports:
                checked_imports.add(imp)
                imp_path = imp.replace('.', os.sep) + '.py'
                if os.path.exists(imp_path):
                    new_imports = get_imports_from_file(imp_path)
                    all_imports.update(new_imports)
                    _check_imports(new_imports)

    for file_path in files_to_check:
        imports = get_imports_from_file(file_path)
        all_imports.update(imports)
        _check_imports(imports)

    return all_imports

def find_files_to_keep(files_to_keep: List[str]) -> Set[str]:
    all_imports = get_all_imports(files_to_keep)
    all_imports.update(files_to_keep)
    return all_imports

def list_all_files_in_repo(repo_path: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                all_files.append(os.path.join(root, file))
    return all_files

def find_unused_files(repo_path: str, files_to_keep: List[str]) -> List[str]:
    all_files = list_all_files_in_repo(repo_path)
    files_to_keep = set(files_to_keep)
    used_files = find_files_to_keep(files_to_keep)
    unused_files = [f for f in all_files if f not in used_files]
    return unused_files

def collect_files_from_args(args: List[str]) -> List[str]:
    files_to_keep = []
    for path in args:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        files_to_keep.append(os.path.join(root, file))
        elif os.path.isfile(path) and path.endswith('.py'):
            files_to_keep.append(path)
        else:
            print(f"Warning: {path} is not a valid Python file or directory")
    return files_to_keep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze imports and find unused files in a Python repository.')
    parser.add_argument('repo_path', type=str, help='Path to the Python repository')
    parser.add_argument('files_to_keep', nargs='+', help='Files and directories to keep')

    args = parser.parse_args()

    repo_path = args.repo_path
    files_to_keep = collect_files_from_args(args.files_to_keep)

    unused_files = find_unused_files(repo_path, files_to_keep)
    print("Unused files:")
    for file in unused_files:
        print(file)
