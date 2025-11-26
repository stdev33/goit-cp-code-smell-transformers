import joblib
import pandas as pd
from src.utils.metric_extraction import extract_metrics_from_code
from src.constants.smell_mapping import SMELL_MAPPING

def batch_infer_on_dataframe(df: pd.DataFrame,
                              model_path: str,
                              scaler_path: str,
                              code_column: str = "Cleaned_Code") -> pd.DataFrame:
    """
    Extract metrics from code and perform inference using the specified model and scaler.

    Args:
        df (pd.DataFrame): Input dataframe with code.
        model_path (str): Path to saved model (.pkl).
        scaler_path (str): Path to saved scaler (.pkl).
        code_column (str): Name of column containing code.

    Returns:
        pd.DataFrame: DataFrame with predicted code smell label.
    """
    # Extract metrics
    df_metrics = df[code_column].apply(extract_metrics_from_code)
    metrics_df = pd.DataFrame(list(df_metrics))

    # Handle missing values
    metrics_df = metrics_df.fillna(metrics_df.median())

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scale metrics
    X_scaled = scaler.transform(metrics_df)

    # Predict
    y_pred = model.predict(X_scaled)

    # Map prediction to smell name (inverse of SMELL_MAPPING)
    inverse_mapping = {v: k.title() for k, v in SMELL_MAPPING.items()}
    df['predicted_smell'] = [inverse_mapping.get(lbl, "Clean") for lbl in y_pred]

    return df