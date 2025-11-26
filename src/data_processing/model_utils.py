import os
import javalang
import pandas as pd
import numpy as np
from radon.raw import analyze as raw_analyze
from radon.complexity import cc_visit


def load_java_code_from_file(filepath: str) -> str:
    """Reads a Java source file and returns its content as a string."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def extract_metrics_from_java(code: str) -> pd.DataFrame:
    """
    Extracts basic code metrics from Java source code.
    Returns a DataFrame with columns matching those in CSQA dataset.
    """
    # --- Raw metrics via radon ---
    raw_metrics = raw_analyze(code)

    # --- Cyclomatic complexity metrics ---
    complexity = cc_visit(code)

    if complexity:
        avg_complexity = np.mean([c.complexity for c in complexity])
        max_complexity = np.max([c.complexity for c in complexity])
        num_methods = len(complexity)
    else:
        avg_complexity = 0
        max_complexity = 0
        num_methods = 0

    # --- AST parsing via javalang (optional for later) ---
    try:
        tree = javalang.parse.parse(code)
        num_classes = sum(1 for path, node in tree if isinstance(node, javalang.tree.ClassDeclaration))
        num_fields = sum(len(node.fields) for path, node in tree if isinstance(node, javalang.tree.ClassDeclaration))
    except:
        num_classes = 0
        num_fields = 0

    # --- Construct features ---
    features = {
        'loc': raw_metrics.loc,  # lines of code
        'lloc': raw_metrics.lloc,
        'sloc': raw_metrics.sloc,
        'comments': raw_metrics.comments,
        'multi': raw_metrics.multi,
        'blank': raw_metrics.blank,
        'num_classes': num_classes,
        'num_fields': num_fields,
        'num_methods': num_methods,
        'avg_cyclomatic_complexity': avg_complexity,
        'max_cyclomatic_complexity': max_complexity,
    }

    return pd.DataFrame([features])