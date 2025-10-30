"""
Module for processing and cleaning source code before feeding it into a model.

This file contains functions to remove comments, extra spaces, and other elements
that may distort the vector representation of the code. These functions can be
extended or replaced with more sophisticated preprocessing methods if needed.
"""

import re


def remove_single_line_comments(code: str) -> str:
    """
    Removes single-line comments of the form // ...
    """
    return re.sub(r"//.*", "", code)


def remove_multi_line_comments(code: str) -> str:
    """
    Removes multi-line comments of the form /* ...  */
    """
    return re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)


def remove_empty_lines(code: str) -> str:
    """
    Removes empty lines and lines with only whitespace
    """
    return "\n".join([line for line in code.splitlines() if line.strip()])


def normalize_whitespace(code: str) -> str:
    """
    Cleans up whitespace: trims leading/trailing spaces and replaces tabs with 4 spaces
    """
    return "\n".join(line.strip().replace("\t", "    ") for line in code.splitlines())


def clean_code(code: str) -> str:
    """
    Full code cleaning: comments, whitespace, empty lines
    Args:
        code: Raw source code as a string.
    """
    code = remove_single_line_comments(code)
    code = remove_multi_line_comments(code)
    code = normalize_whitespace(code)
    code = remove_empty_lines(code)
    return code