import pandas as pd
import numpy as np
from typing import Tuple, List


def load_and_prepare_evaluation_data(path: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load and preprocess the evaluation dataset for transformer inference.

    This function filters the dataset to match the training setup:
    - Uses only rows with Source == "SmellyCode++"
    - Removes empty or null code cells
    - Combines "God Class" and "Large Class" into a unified "God/Large Class"

    Args:
        path (str): Path to the evaluation CSV file.

    Returns:
        texts (List[str]): List of code snippets.
        y_true (np.ndarray): Ground truth label matrix.
        label_cols (List[str]): List of label column names.
    """
    df = pd.read_csv(path)

    # Filter same as training
    df = df[(df["Source"] == "SmellyCode++") & (df["Code"].notnull()) & (df["Code"].str.strip() != "")]

    # Combine God + Large Class
    df["God/Large Class"] = ((df["God Class"] == 1) | (df["Large Class"] == 1)).astype(int)

    label_cols = ["Long Method", "God/Large Class", "Feature Envy", "Data Class", "Clean"]
    texts = df["Code"].tolist()
    y_true = df[label_cols].values

    return texts, y_true, label_cols