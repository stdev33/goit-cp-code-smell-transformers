import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os

def compute_multilabel_metrics(pred):
    logits, labels = pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds)
    }

def train_multilabel_transformer(model_name: str, df, label_cols, output_dir: str):
    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_cols), problem_type="multi_label_classification"
    )

    # Prepare data
    texts = df["Code"].tolist()
    labels = df[label_cols].values

    # Split
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Tokenize
    def tokenize_function(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512
        )

    train_encodings = tokenize_function(texts_train)
    val_encodings = tokenize_function(texts_val)

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": np.array(labels_train, dtype=np.float32)
    })
    val_dataset = Dataset.from_dict({
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": np.array(labels_val, dtype=np.float32)
    })

    print(train_dataset.features)

    # Training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_multilabel_metrics
    )

    trainer.train()

    # Save fine-tuned model
    model_path = os.path.join(output_dir, f"{model_name.split('/')[-1]}_multilabel_finetuned")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✔️ Model and tokenizer saved to: {model_path}")