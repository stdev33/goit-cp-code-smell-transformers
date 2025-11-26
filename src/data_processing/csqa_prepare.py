from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def split_and_scale(df: pd.DataFrame, label_col: str = 'label', test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training and testing sets, applies standard scaling to features.
    Handles missing labels safely before conversion.

    Args:
        df (pd.DataFrame): Merged CSQA dataframe with features and label column.
        label_col (str): Name of the target column.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        Tuple of scaled X_train, X_test, y_train, y_test, and the scaler object.
    """
    # Drop label and non-feature columns if present (safe: ignore missing columns)
    X = df.drop(columns=[label_col, 'smell', 'level'], errors='ignore')

    # Extract target and remove rows with missing labels
    y = df[label_col] if label_col in df.columns else pd.Series([None] * len(df), index=df.index)
    mask = y.notna()
    X = X[mask]
    y = y[mask].astype(int)

    # Handle missing values in features: fill numeric columns with median
    # (median is robust to outliers and preserves feature scale)
    numeric_cols = X.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    else:
        # fallback: fill any remaining NaNs with zeros
        X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler