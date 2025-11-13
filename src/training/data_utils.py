import pandas as pd

def load_and_prepare_dataset(csv_path: str):
    """
    Load the dataset, filter SmellyCode++ entries, and prepare labels for multi-label classification.

    Args:
        csv_path (str): Path to the merged CSV file

    Returns:
        df (pd.DataFrame): Filtered and labeled dataframe
        label_cols (List[str]): List of target columns
    """
    # Load the merged dataset
    df = pd.read_csv(
        csv_path,
        dtype={"File": str, "Project": str, "Class": str, "Code": str}
    )

    # Filter only SmellyCode++ records with non-empty code
    df = df[(df["Source"] == "SmellyCode++") & (df["Code"].notnull()) &(df["Code"].str.strip() != "")]

    # Define labels to be used for multi-label classification
    label_cols = ["Long Method", "God Class", "Large Class", "Feature Envy", "Data Class", "Clean"]

    # Combine 'God Class' and 'Large Class' into a single column
    if "God Class" in df.columns and "Large Class" in df.columns:
        df["God/Large Class"] = ((df["God Class"] == 1) | (df["Large Class"] == 1)).astype(int)
        label_cols = ["Long Method", "God/Large Class", "Feature Envy", "Data Class", "Clean"]

    return df, label_cols