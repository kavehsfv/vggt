# generate_md_docs.py

generate_md_docs.py

Script to generate markdown documentation for each .py file in the code base.
For each .py file, creates a .md file beside it with a summary of the module,
its classes, and functions including their docstrings.
Usage:
    python generate_md_docs.py

## Contents

### Function `summarize_file`

Summarize the given Python file by extracting its module docstring,
classes, and functions with their docstrings.
Returns the markdown content as a string.

### Function `main`

Walk the repository, find all .py files, and generate corresponding .md files.