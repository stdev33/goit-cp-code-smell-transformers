import pandas as pd
import joblib
from typing import List, Literal
from src.data_processing.java_metrics_extractor import extract_metrics_from_java
from sklearn.preprocessing import StandardScaler

ModelName = Literal["random_forest", "xgboost"]

MODEL_PATHS = {
    "random_forest": {
        "model": "../models/random_forest_smell_model.pkl",
        "scaler": "../models/standard_scaler.pkl"
    },
    "xgboost": {
        "model": "../models/xgb_smell_model.pkl",
        "scaler": "../models/standard_scaler_xgb.pkl"
    }
}

CLASS_LABELS = {
    1: "Long Method",
    2: "God/Large Class",
    3: "Feature Envy",
    4: "Data Class",
    5: "Clean"
}


def run_inference(model_name: ModelName, df_input: pd.DataFrame, limit_per_class: int = 100) -> pd.DataFrame:
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")

    if "Cleaned_Code" not in df_input.columns or "Long method" not in df_input.columns:
        raise ValueError("Input DataFrame must contain 'Cleaned_Code' and label columns.")

    df_input = df_input.copy()
    df_input["clean"] = (
        (df_input["Long method"] == 0) &
        (df_input["God class"] == 0) &
        (df_input["Feature envy"] == 0) &
        (df_input["Data class"] == 0)
    ).astype(int)

    label_columns = ["Long method", "God class", "Feature envy", "Data class", "clean"]
    selected_df = pd.DataFrame()

    for label_idx, col in enumerate(label_columns, start=1):
        temp_df = df_input[df_input[col] == 1]
        if not temp_df.empty:
            temp_df = temp_df.sample(n=min(limit_per_class, len(temp_df)), random_state=42).copy()
            temp_df["true_label"] = label_idx
            selected_df = pd.concat([selected_df, temp_df], ignore_index=True)

    code_list = selected_df["Cleaned_Code"].tolist()
    true_labels = selected_df["true_label"].tolist()

    # Load model and scaler
    model = joblib.load(MODEL_PATHS[model_name]["model"])
    scaler: StandardScaler = joblib.load(MODEL_PATHS[model_name]["scaler"])

    # Extract metrics from each code snippet individually
    metrics_list = []
    for code in code_list:
        metrics_df = extract_metrics_from_java(code)
        metrics_list.append(metrics_df)

    df_metrics = pd.concat(metrics_list, ignore_index=True)

    # Scale features
    X_scaled = scaler.transform(df_metrics)

    # Predict
    y_pred = model.predict(X_scaled)

    # If XGBoost — shift labels back from 0–3 → 1–4
    if model_name == "xgboost":
        y_pred = y_pred + 1

    return pd.DataFrame({
        "Code": code_list,
        "True_Label": true_labels,
        "Predicted_Label": y_pred,
        "Predicted_Smell": [CLASS_LABELS[label] for label in y_pred]
    })