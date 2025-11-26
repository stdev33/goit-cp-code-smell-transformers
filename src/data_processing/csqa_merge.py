import pandas as pd

def load_csqa_file(file_path: str, smell_name: str, is_class_level: bool) -> pd.DataFrame:
    """
    Loads a CSQA file and adds a 'smell' column with the smell name.
    Removes the longname / Classlongname / etc. column.

    Args:
        file_path (str): Path to the CSV file
        smell_name (str): Name of the code smell
        is_class_level (bool): Indicates if the smell is at class level or method level

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = pd.read_csv(file_path)

    name_cols = ['longname', 'Classlongname']
    for col in name_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df['smell'] = smell_name

    df['level'] = 'class' if is_class_level else 'method'

    return df


def merge_all_csqa_files(file_map: dict) -> pd.DataFrame:
    """
    Merges all files into a single DataFrame.

    Args:
        file_map (dict): dictionary mapping smell names to (file_path, is_class_level) tuples

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    dfs = []
    for smell_name, (path, is_class_level) in file_map.items():
        df = load_csqa_file(path, smell_name, is_class_level)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged