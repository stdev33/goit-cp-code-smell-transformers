import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.constants.smell_mapping import SMELL_MAPPING


def prepare_multiclass_dataset(input_path: str = "../data/processed/csqa_merged_metrics.csv",
                               output_path: str = "../data/processed/csqa_multiclass_balanced.csv",
                               test_size: float = 0.2,
                               random_state: int = 42):
    """
    Prepares a multiclass dataset from CSQA merged metrics, mapping 'smell' values
    to specific code smell types (Long Method, Large/God Class, Feature Envy, Data Class),
    balances the dataset using SMOTE, and saves the result.

    Args:
        input_path (str): Path to the merged CSQA dataset.
        output_path (str): Where to save the processed dataset.
        test_size (float): Proportion of the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """

    df = pd.read_csv(input_path)

    # Normalize smell names and create label mapping
    df['smell'] = df['smell'].str.strip().str.lower()

    df['label'] = df['smell'].map(SMELL_MAPPING)
    df['label'] = df['label'].fillna(0).astype(int)  # 0 = clean

    # Drop non-feature columns
    X = df.drop(columns=['label', 'smell', 'level'], errors='ignore')
    y = df['label']

    # Fill NaN values with median
    X = X.fillna(X.median())

    # Split train/test before balancing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Apply SMOTE only on training data (to prevent data leakage)
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Save balanced dataset
    balanced_df = pd.DataFrame(X_train_resampled, columns=X.columns)
    balanced_df['label'] = y_train_resampled
    balanced_df.to_csv(output_path, index=False)

    print(f"✅ Saved balanced multiclass dataset: {output_path}")
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler