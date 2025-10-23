"""
Module for processing and cleaning source code before feeding it into a model.

This file contains functions to remove comments, extra spaces, and other elements
that may distort the vector representation of the code. These functions can be
extended or replaced with more sophisticated preprocessing methods if needed.
"""

import re
from typing import Iterable


def strip_comments_and_whitespace(code: str) -> str:
    """Removes single-line comments and empty lines from the code.

    This function demonstrates a simple approach to cleaning Python code:
    it removes comments that start with the `#` symbol and skips empty lines.
    For other programming languages, additional regular expressions may be required.

    Args:
        code: Source code as a string.

    Returns:
        Cleaned code without single-line comments and empty lines.
    """
    # Видаляємо Python‑стильні однорядкові коментарі
    code_no_comments = re.sub(r"#.*", "", code)
    # Розбиваємо на рядки, обрізаючи праві пробіли та пропускаючи порожні
    cleaned_lines = [line.rstrip() for line in code_no_comments.splitlines() if line.strip()]
    return "\n".join(cleaned_lines)


def remove_duplicate_imports(lines: Iterable[str]) -> list[str]:
    """Example function that removes duplicate imports from the beginning of the file.

    Args:
        lines: Iterable sequence of code lines.

    Returns:
        List of lines without repeated import statements.
    """
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line.startswith("import") or line.startswith("from"):
            if line not in seen:
                seen.add(line)
                result.append(line)
        else:
            result.append(line)
    return result


__all__ = [
    "strip_comments_and_whitespace",
    "remove_duplicate_imports",
]