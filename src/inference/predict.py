"""
Module for performing inference using a trained transformer model
to detect the presence of code smells in code fragments. This file
demonstrates a basic class that can be adapted for different models and configurations.
"""

from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple


class CodeSmellDetector:
    """Class for inference with a trained code smell classification model.

    Args:
        model_path: Path to the saved model (folder with config and weights).
        tokenizer_name: Name of the tokenizer (by default — the same as used during training).
    """

    def __init__(self, model_path: str, tokenizer_name: str = "microsoft/codebert-base") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, code: str) -> List[float]:
        """Returns class probabilities for the given code fragment.

        Args:
            code: Source code to analyze.

        Returns:
            A list of probabilities for each class (first — probability of no smell, second — probability of presence).
        """
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = outputs.logits.softmax(dim=-1)[0].tolist()
        return probabilities


def main() -> None:
    """Simple example of using CodeSmellDetector for a single code fragment."""
    detector = CodeSmellDetector(model_path="models/codebert-finetuned")
    code_sample = """def example(a, b):\n    # TODO: improve this function\n    result = a + b\n    return result\n"""
    probs = detector.predict(code_sample)
    print(f"Probability of no smell: {probs[0]:.2f}, probability of smell: {probs[1]:.2f}")


if __name__ == "__main__":
    main()