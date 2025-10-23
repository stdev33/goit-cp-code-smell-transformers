"""
Script for fine-tuning a transformer model for code smell detection.

This file provides a minimal example of how to load a pre-trained model,
prepare a dataset, and run the training process. You can modify the
architecture, metrics, task, or model as needed.

Run this module as:

    python -m src.training.train_codebert

or from the command line at the repository root:

    python src/training/train_codebert.py
"""

from __future__ import annotations

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from typing import Callable, Dict, Any
import torch


def load_data(name: str, subset: str | None = None) -> DatasetDict:
    """Loads a dataset for training.

    This is a simple example. You can replace the default dataset
    with your own CSV/JSON or a preprocessed set of files with code smells.

    Args:
        name: Name of the dataset on Hugging Face Datasets.
        subset: Optional key for sub-dataset (e.g., "python").

    Returns:
        A DatasetDict object with train/validation/test splits.
    """
    if subset is not None:
        dataset = load_dataset(name, subset)
    else:
        dataset = load_dataset(name)
    return dataset


def tokenize_function(tokenizer: AutoTokenizer) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Returns a function for tokenizing code examples.

    Args:
        tokenizer: Initialized tokenizer.

    Returns:
        A function that takes a dictionary with a "code" field
        and returns prepared tensors for the model.
    """
    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["code"],
            padding=False,
            truncation=True,
            max_length=512,
        )

    return tokenize


def main() -> None:
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load demonstration dataset. Replace with your real dataset with
    # code and "label" field, where 1 = smell, 0 = clean.
    dataset = load_data("code_search_net", "python")

    # Use train/test split as train/validation for this example
    # ("train", "test", and "validation" exist, but "test" may not have labels)
    data = DatasetDict(
        train=dataset["train"],
        validation=dataset["validation"],
    )

    # Tokenize all code examples
    tokenize = tokenize_function(tokenizer)
    tokenized = data.map(tokenize, batched=True, remove_columns=data["train"].column_names)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Training parameters (adjust to your task)
    training_args = TrainingArguments(
        output_dir="models/codebert-finetuned",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model("models/codebert-finetuned")


if __name__ == "__main__":
    main()