from typing import List, Literal
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

ModelName = Literal["random_forest", "xgboost"]

MODEL_PATHS = {
    "random_forest": {
        "model": "../models/random_forest_multilabel.pkl",
        "scaler": "../models/standard_scaler_multilabel.pkl"
    },
    "xgboost": {
        "model": "../models/xgboost_multilabel.pkl",
        "scaler": "../models/standard_scaler_multilabel.pkl"
    }
}

CLASS_LABELS = {
    1: "Long Method",
    2: "God/Large Class",
    3: "Feature Envy",
    4: "Data Class",
    5: "Clean"
}


def run_inference_pipeline(df: pd.DataFrame, model_type: ModelName, save_path: str = None):
    feature_cols = [
        'NumberOfOperatorsWithoutAssignments', 'LOC', 'NumberOfUniqueIdentifiers',
        'NumberOfIdentifies', 'NumberOfAssignments', 'CYCLO', 'NumberOfTokens'
    ]
    label_cols = ["Long Method", "God Class", "Feature Envy", "Data Class", "Clean"]

    X_eval = df[feature_cols]
    y_eval = df[label_cols]

    # Load the model
    model = joblib.load(MODEL_PATHS[model_type]["model"])
    scaler: StandardScaler = joblib.load(MODEL_PATHS[model_type]["scaler"])

    # Predict
    X_eval_scaled = scaler.transform(X_eval)
    y_pred = model.predict(X_eval_scaled)

    # Evaluation
    print(f"\n=== {model_type.upper()} Inference Evaluation ===")
    for idx, label in enumerate(label_cols):
        print(f"\n--- Label: {label} ---")
        print(classification_report(y_eval[label], y_pred[:, idx]))

    print(f"Hamming Loss: {hamming_loss(y_eval, y_pred):.4f}")
    subset_acc = accuracy_score(y_eval, y_pred)
    print(f"Subset Accuracy (Exact Match): {subset_acc:.4f}")

    # Save to CSV if requested
    if save_path:
        pred_df = df.copy()
        for i, label in enumerate(label_cols):
            pred_df[f"Predicted_{label}"] = y_pred[:, i]
        pred_df.to_csv(save_path, index=False)
        print(f"✔️ Predictions saved to: {save_path}")
