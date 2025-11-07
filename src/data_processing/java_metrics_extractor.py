import lizard
import re
import math
import numpy as np
import pandas as pd
from typing import Dict


def extract_metrics_from_java(code: str) -> pd.DataFrame:
    """
    Extracts CSQA-compatible metrics from raw Java source code string.

    Returns:
        pd.DataFrame: DataFrame with a single row containing the metrics.
    """

    # Normalize line endings
    if not isinstance(code, str):
        code = str(code)
    code_lines = code.splitlines()

    # Analyze code with lizard (wrap call to be robust)
    try:
        analysis = lizard.analyze_file.analyze_source_code("Temp.java", code)
        functions = analysis.function_list
    except Exception:
        functions = []

    num_functions = len(functions)

    # Collect per-function metrics if available
    cyclo_list = []
    nloc_list = []
    param_list = []
    token_count = 0
    func_identifiers = []  # for TCC estimation

    for f in functions:
        # lizard's FunctionInfo has attributes: cyclomatic_complexity, length, parameter_count, token_count
        cyclo = getattr(f, 'cyclomatic_complexity', 0)
        length = getattr(f, 'length', 0)
        params = getattr(f, 'parameter_count', 0)
        tokens = getattr(f, 'token_count', 0) if hasattr(f, 'token_count') else 0

        cyclo_list.append(cyclo)
        nloc_list.append(length)
        param_list.append(params)
        token_count += tokens

        # extract function body lines using start_line/end_line if available
        start = getattr(f, 'start_line', None)
        end = getattr(f, 'end_line', None)
        body = ''
        if start is not None and end is not None and 1 <= start <= end:
            # slice lines (lizard lines are 1-based)
            body = '\n'.join(code_lines[start - 1:end])
        else:
            # fallback: try to find function name and extract a small window
            body = ''
        ids = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', body)
        func_identifiers.append(set(ids))

    # Compute statistical aggregates
    sum_cyclomatic = sum(cyclo_list) if cyclo_list else 0
    avg_cyclomatic = float(np.mean(cyclo_list)) if cyclo_list else 0.0
    max_cyclomatic = max(cyclo_list) if cyclo_list else 0
    min_cyclomatic = min(cyclo_list) if cyclo_list else 0

    avg_nloc = float(np.mean(nloc_list)) if nloc_list else 0.0
    avg_params = float(np.mean(param_list)) if param_list else 0.0

    # Comment lines (// or /* ... */ occurrences count lines starting with comment markers)
    comment_lines = 0
    in_block = False
    for line in code_lines:
        s = line.strip()
        if s.startswith('/*'):
            in_block = True
            comment_lines += 1
            if s.endswith('*/') and len(s) > 2:
                in_block = False
        elif in_block:
            comment_lines += 1
            if '*/' in s:
                in_block = False
        elif s.startswith('//'):
            comment_lines += 1

    # Statement count estimate (very rough)
    stmt_count = len(re.findall(r';', code))
    # declarations (rough match for common Java types and generics)
    decl_stmts = len(re.findall(r'\b(?:int|float|double|long|short|byte|boolean|char|String|List|Map|Set)\b\s+[a-zA-Z_][a-zA-Z0-9_]*', code))
    return_stmts = len(re.findall(r'\breturn\b', code))
    if_stmts = len(re.findall(r'\bif\b', code))
    new_stmts = len(re.findall(r'\bnew\s+[A-Za-z_][A-Za-z0-9_<>]*', code))
    assign_stmts = len(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=]', code))

    # Keywords (simplified approximation)
    keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch', 'return', 'break', 'continue', 'new']
    keyword_count = sum(len(re.findall(r'\b' + re.escape(k) + r'\b', code)) for k in keywords)

    # Operators (approximate)
    ops_wo_assign = ['+', '-', '*', '/', '%', '++', '--', '&&', '||', '!', '==', '!=', '>=', '<=', '>', '<']
    op_wo_assign_count = sum(code.count(op) for op in ops_wo_assign)

    # Identifiers
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    unique_identifiers = set(identifiers)

    # Lines
    non_blank_lines = [line for line in code_lines if line.strip()]
    line_count = len(non_blank_lines)
    blank_lines = len([line for line in code_lines if not line.strip()])

    # Estimate MAXNESTING by analyzing braces depth
    def estimate_max_nesting_local(code_str: str) -> int:
        max_depth = 0
        current_depth = 0
        for line in code_str.split('\n'):
            opening = line.count('{')
            closing = line.count('}')
            current_depth += opening - closing
            if current_depth > max_depth:
                max_depth = current_depth
        return max_depth if max_depth >= 0 else 0

    # RFC (Response For a Class) estimation: number of methods + number of external calls
    def estimate_rfc_local(code_str: str) -> int:
        method_defs = re.findall(r'\b(?:public|protected|private)?\s+[A-Za-z0-9_<>, \[\]]+\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*\{', code_str)
        method_calls = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\s*\(', code_str)
        return len(set(method_defs)) + len(set(method_calls))

    # ATFD (Access to Foreign Data) estimation: count of object.field occurrences (excluding this.)
    def estimate_atfd_local(code_str: str) -> int:
        accesses = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]*\b', code_str)
        accesses = [a for a in accesses if a not in ('this',)]
        return len(accesses)

    # Count method calls
    def count_method_calls_local(code_str: str) -> int:
        return len(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\s*\(', code_str))

    # FANOUT: number of distinct external qualifiers before dot (approximate)
    fanout_set = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\.', code))
    fanout_set = {x for x in fanout_set if x not in ('this',)}
    fanout = len(fanout_set)

    # CountClassBase: number of 'extends' occurrences (approximate)
    class_bases = re.findall(r'\bclass\s+[A-Za-z0-9_]+\s+extends\s+([A-Za-z0-9_]+)', code)
    count_class_base = len(class_bases)

    # CountDeclClassVariable: rough count of class-level field declarations
    class_field_decls = re.findall(r'\b(?:public|protected|private)\s+[A-Za-z0-9_<>, \[\]]+\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:=|;)', code)
    count_decl_class_variable = len(class_field_decls)

    # CountClassCoupled: number of distinct capitalized identifiers (heuristic for types referenced)
    class_coupled_set = set(re.findall(r'\b([A-Z][A-Za-z0-9_]*)\b', code))
    count_class_coupled = len(class_coupled_set)

    # WMCNAMM approximation: sum of cyclomatic complexities
    wmcnamm = sum_cyclomatic

    # TCC approximation: fraction of method pairs that share at least one identifier
    def estimate_tcc(func_id_sets: list) -> float:
        n = len(func_id_sets)
        if n <= 1:
            return 0.0
        pairs = 0
        connected = 0
        for i in range(n):
            for j in range(i + 1, n):
                pairs += 1
                if func_id_sets[i] & func_id_sets[j]:
                    connected += 1
        return connected / pairs if pairs > 0 else 0.0

    tcc = estimate_tcc(func_identifiers)

    # Final dictionary
    metrics = {
        "LOC": line_count,
        "CYCLO": sum_cyclomatic,
        "MAXNESTING": estimate_max_nesting_local(code),
        "CC": avg_cyclomatic,
        "CM": comment_lines,
        "FANOUT": fanout,
        "CountInput": placeholder if (placeholder := 0) is not None else 0,
        "CountLineBlank": blank_lines,
        "CountLineCode": max(0, line_count - blank_lines),
        "CountLineCodeDecl": decl_stmts,
        "CountLineCodeExe": max(0, stmt_count - decl_stmts),
        "CountLineComment": comment_lines,
        "CountOutput": return_stmts,
        "CountPath": placeholder,
        "CountPathLog": placeholder,
        "CountSemicolon": stmt_count,
        "CountStmt": stmt_count,
        "CountStmtDecl": decl_stmts,
        "CountStmtExe": max(0, stmt_count - decl_stmts),
        "CyclomaticModified": sum_cyclomatic,
        "CyclomaticStrict": sum_cyclomatic,
        "Essential": placeholder,
        "Knots": placeholder,
        "MaxEssentialKnots": placeholder,
        "MinEssentialKnots": placeholder,
        "RatioCommentToCode": comment_lines / (line_count + 1e-5),
        "SumCyclomatic": sum_cyclomatic,
        "SumCyclomaticModified": sum_cyclomatic,
        "SumCyclomaticStrict": sum_cyclomatic,
        "SumEssential": placeholder,
        "NOPL": placeholder,
        "NOPA": avg_params,
        "AvgCyclomatic": avg_cyclomatic,
        "AvgCyclomaticModified": avg_cyclomatic,
        "AvgCyclomaticStrict": avg_cyclomatic,
        "AvgEssential": placeholder,
        "AvgLine": avg_nloc,
        "AvgLineBlank": (blank_lines / (num_functions + 1e-5)) if num_functions > 0 else 0.0,
        "AvgLineCode": avg_nloc,
        "AvgLineComment": (comment_lines / (num_functions + 1e-5)) if num_functions > 0 else 0.0,
        "CountClassBase": count_class_base,
        "CountClassCoupled": count_class_coupled,
        "CountClassDerived": placeholder,
        "CountDeclClassMethod": num_functions,
        "CountDeclClassVariable": count_decl_class_variable,
        "CountDeclInstanceMethod": num_functions,
        "CountDeclInstanceVariable": placeholder,
        "CountDeclMethod": num_functions,
        "CountDeclMethodAll": num_functions,
        "CountDeclMethodDefault": placeholder,
        "CountDeclMethodPrivate": placeholder,
        "CountDeclMethodProtected": placeholder,
        "CountDeclMethodPublic": placeholder,
        "MaxCyclomatic": max_cyclomatic,
        "MaxCyclomaticModified": max_cyclomatic,
        "MaxCyclomaticStrict": max_cyclomatic,
        "MaxEssential": placeholder,
        "MaxInheritanceTree": placeholder,
        "MaxNesting": estimate_max_nesting_local(code),
        "PercentLackOfCohesion": placeholder,
        "WOC": placeholder,
        "WMCNAMM": wmcnamm,
        "LOCNAMM": placeholder,
        "NOMNAMM": num_functions,
        "NOAM": avg_params,
        "TCC": tcc,
        "PK_CountDeclClassMethod": num_functions,
        "NumberOfTokens": token_count,
        "NumberOfIdentifies": len(identifiers),
        "NumberOfReturnAndPrintStatements": return_stmts,
        "NumberOfConditionalJumpStatements": if_stmts,
        "NumberOfKeywords": keyword_count,
        "NumberOfAssignments": assign_stmts,
        "NumberOfOperatorsWithoutAssignments": op_wo_assign_count,
        "NumberOfUniqueIdentifiers": len(unique_identifiers),
        "NumberOfDots": code.count('.'),
        "NumberOfNewStatements": new_stmts,
        "MinLineCode": min(nloc_list) if nloc_list else 0,
        "CountLineCodeNAMM": max(0, line_count - blank_lines),
        "LogStmtDecl": math.log1p(decl_stmts),
        "CSNOMNAMM": num_functions,
        "NIM": num_functions,
        "RFC": estimate_rfc_local(code),
        "CFNAMM": placeholder,
        "DAC": placeholder,
        "NMO": placeholder,
        "NOII": placeholder,
        "LogCyclomaticStrict": math.log1p(sum_cyclomatic),
        "FANIN": placeholder,
        "ATFD": estimate_atfd_local(code),
        "NumberOfMethodCalls": count_method_calls_local(code),
        "SumCountPath": placeholder,
        "NumberOfClassConstructors": placeholder,
        "AvgLineCodeExe": (stmt_count - decl_stmts) / (num_functions + 1e-5),
        "AvgStmtDecl": decl_stmts / (num_functions + 1e-5) if num_functions > 0 else 0.0,
        "NumberOfDepends": placeholder,
    }

    return pd.DataFrame([metrics])