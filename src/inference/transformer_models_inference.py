import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from sklearn.metrics import classification_report, hamming_loss, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
import time


def load_sequence_classification_model(model_path: str):
    """
    Load a sequence classification model and its tokenizer (e.g., CodeBERT or GraphCodeBERT).

    Args:
        model_path (str): Path to the pretrained or fine-tuned model directory.

    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded sequence classification model in evaluation mode.
    """
    print("[INFO] Starting load_sequence_classification_model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("[INFO] Finished load_sequence_classification_model.")
    return tokenizer, model


def load_seq2seq_model(model_path: str):
    """
    Load a seq2seq model and its tokenizer (e.g., CodeT5).

    Args:
        model_path (str): Path to the pretrained or fine-tuned model directory.

    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded seq2seq model in evaluation mode.
    """
    print("[INFO] Starting load_seq2seq_model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    print("[INFO] Finished load_seq2seq_model.")
    return tokenizer, model


def predict_multilabel_classification(texts: List[str], tokenizer, model, threshold=0.5, batch_size=16):
    start_time = time.time()
    print("[INFO] Starting predict_multilabel_classification...")
    model.eval()

    preds = []
    probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**encodings)
            logits = output.logits
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            batch_preds = (batch_probs >= threshold).astype(int)

            probs.append(batch_probs)
            preds.append(batch_preds)

    preds = np.vstack(preds)
    probs = np.vstack(probs)
    end_time = time.time()
    print(f"[INFO] Finished predict_multilabel_classification. Time taken: {end_time - start_time:.2f} seconds.")
    return preds, probs


def predict_codet5(texts: List[str], tokenizer, model, batch_size: int = 16):
    """
    Perform batched seq2seq inference using CodeT5 and convert generated text to multilabel vector.

    Args:
        texts (List[str]): List of input texts for generation.
        tokenizer: Tokenizer compatible with the seq2seq model.
        model: Seq2seq model (e.g., CodeT5).
        batch_size (int): Number of samples per batch.

    Returns:
        decoded_all (List[str]): List of generated label strings.
        text_to_vector (function): Function to convert label string to multilabel vector.
    """
    start_time = time.time()
    print("[INFO] Starting predict_codet5...")
    model.eval()
    decoded_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64
            )

        decoded_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoded_all.extend(decoded_batch)

    def text_to_vector(t: str, classes: List[str]):
        labels = [c.strip().lower() for c in t.split(",")]
        return [1 if c.lower() in labels else 0 for c in classes]

    end_time = time.time()
    print(f"[INFO] Finished predict_codet5. Time taken: {end_time - start_time:.2f} seconds.")
    return decoded_all, text_to_vector


def evaluate_predictions(y_true, y_pred, label_cols: List[str]):
    """
    Evaluate multilabel classification predictions using various metrics.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        label_cols (List[str]): List of label names.

    Returns:
        cls_report (str): Classification report as a string.
        ham_loss (float): Hamming loss score.
        subset_acc (float): Subset accuracy score.
    """
    start_time = time.time()
    print("[INFO] Starting evaluate_predictions...")
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=label_cols,
        zero_division=0,
        output_dict=False
    )

    ham_loss = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)
    end_time = time.time()
    print(f"[INFO] Finished evaluate_predictions. Time taken: {end_time - start_time:.2f} seconds.")
    return cls_report, ham_loss, subset_acc


def plot_pr_curves(y_true, probs, label_cols, model_name, output_dir):
    """
    Plot precision-recall curves for each label and save the plots.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        probs (np.ndarray): Predicted probabilities for each label.
        label_cols (List[str]): List of label names.
        model_name (str): Name of the model (used in plot titles and filenames).
        output_dir (str): Directory to save the plots.
    """
    start_time = time.time()
    print("[INFO] Starting plot_pr_curves...")
    os.makedirs(output_dir, exist_ok=True)

    for i, label in enumerate(label_cols):
        precision, recall, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} — PR Curve ({label})")
        path = os.path.join(output_dir, f"{model_name}_pr_{label.replace('/', '_')}.png")
        plt.savefig(path, dpi=150)
        plt.close()
    end_time = time.time()
    print(f"[INFO] Finished plot_pr_curves. Time taken: {end_time - start_time:.2f} seconds.")


def save_predictions_csv(texts, y_true, y_pred, probs, label_cols, output_path):
    """
    Save predictions, true labels, and probabilities to a CSV file.

    Args:
        texts (List[str]): List of input texts.
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        probs (np.ndarray): Predicted probabilities for each label.
        label_cols (List[str]): List of label names.
        output_path (str): File path to save the CSV.
    """
    start_time = time.time()
    print("[INFO] Starting save_predictions_csv...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({
        "Code": texts
    })

    for i, col in enumerate(label_cols):
        df[f"true_{col}"] = y_true[:, i]
        df[f"pred_{col}"] = y_pred[:, i]
        df[f"prob_{col}"] = probs[:, i]

    df.to_csv(output_path, index=False)
    print(f"✔️ Saved predictions to: {output_path}")
    end_time = time.time()
    print(f"[INFO] Finished save_predictions_csv. Time taken: {end_time - start_time:.2f} seconds.")