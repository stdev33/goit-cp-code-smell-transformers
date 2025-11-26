import os
import numpy as np
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import f1_score


def create_text_labels(df, label_cols):
    """
    Convert multilabel columns to a comma-separated text string (e.g. "Long Method, Data Class").
    """
    return df[label_cols].apply(
        lambda row: ", ".join([label for label, val in row.items() if val == 1]) or "Clean",
        axis=1
    )


def compute_metrics_text(pred):
    preds = pred.predictions
    labels = pred.label_ids
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def str_to_set(s):
        return set(map(str.strip, s.lower().split(",")))

    f1_micro_list = []
    f1_macro_list = []

    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        y_pred = str_to_set(pred_str)
        y_true = str_to_set(label_str)
        all_labels = y_pred.union(y_true)

        y_pred_bin = [1 if label in y_pred else 0 for label in all_labels]
        y_true_bin = [1 if label in y_true else 0 for label in all_labels]

        f1_micro_list.append(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0))
        f1_macro_list.append(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))

    return {
        "micro_f1": np.mean(f1_micro_list),
        "macro_f1": np.mean(f1_macro_list)
    }


def train_codet5_multilabel(model_name: str, df, label_cols: List[str], output_dir: str):
    global tokenizer  # Needed for compute_metrics_text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Convert labels to text strings
    df["target_text"] = create_text_labels(df, label_cols)

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Tokenize inputs and targets
    def preprocess(example):
        model_inputs = tokenizer(
            example["Code"],
            max_length=512,
            padding="max_length",
            truncation=True
        )
        labels = tokenizer(
            example["target_text"],
            max_length=128,
            padding="max_length",
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_pandas(train_df).map(preprocess, remove_columns=list(train_df.columns))
    val_dataset = Dataset.from_pandas(val_df).map(preprocess, remove_columns=list(val_df.columns))

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch",
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_text
    )

    trainer.train()

    # Save the model and tokenizer
    model_path = os.path.join(output_dir, f"{model_name.split('/')[-1]}_multilabel_finetuned")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✔️ Model and tokenizer saved to: {model_path}")