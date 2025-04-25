#!/usr/bin/env python3
"""
generate_md_docs.py

Script to generate markdown documentation for each .py file in the code base.
For each .py file, creates a .md file beside it with a summary of the module,
its classes, and functions including their docstrings.
Usage:
    python generate_md_docs.py
"""
import os
import ast


def summarize_file(filepath):
    """
    Summarize the given Python file by extracting its module docstring,
    classes, and functions with their docstrings.
    Returns the markdown content as a string.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return f"# {os.path.basename(filepath)}\n\nUnable to parse file."

    lines = []
    lines.append(f"# {os.path.basename(filepath)}")
    module_doc = ast.get_docstring(tree)
    if module_doc:
        lines.append(module_doc)

    lines.append("## Contents")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"### Class `{node.name}`")
            doc = ast.get_docstring(node) or ''
            if doc:
                lines.append(doc)
        elif isinstance(node, ast.FunctionDef):
            lines.append(f"### Function `{node.name}`")
            doc = ast.get_docstring(node) or ''
            if doc:
                lines.append(doc)

    return "\n\n".join(lines)


def main():
    """
    Walk the repository, find all .py files, and generate corresponding .md files.
    """
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        # skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fname in filenames:
            if not fname.endswith('.py'):
                continue
            py_path = os.path.join(dirpath, fname)
            md_path = os.path.splitext(py_path)[0] + '.md'
            try:
                content = summarize_file(py_path)
                with open(md_path, 'w', encoding='utf-8') as md_file:
                    md_file.write(content)
                print(f"Generated: {md_path}")
            except Exception as e:
                print(f"Failed to generate {md_path}: {e}")


if __name__ == '__main__':  # noqa: C0114
    main()