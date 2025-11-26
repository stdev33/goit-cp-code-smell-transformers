import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.cleaner import clean_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeSmellDetector:
    """
    A simple wrapper around a fine-tuned CodeT5 model to detect code smells
    in source code snippets.
    """

    def __init__(self, model_path: str = "models/transformers/codet5/codet5-base_multilabel_finetuned"):
        """
        Initialize the model and tokenizer from a local fine-tuned checkpoint.
        """
        logger.info("Loading CodeT5 model from: %s", model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def predict_smells(self, code: str, max_length: int = 512) -> str:
        """
        Predict the code smell label(s) for a given source code snippet.
        """
        logger.info("Predicting smells for code snippet...")

        # Clean the input code before prediction
        code = clean_code(code)

        input_text = f"detect code smell: {code.strip()}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=32,
                num_beams=4,
                early_stopping=True
            )
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Prediction: %s", prediction)
        return prediction


def main():
    parser = argparse.ArgumentParser(description="Detect code smells using a fine-tuned CodeT5 model.")
    parser.add_argument("--code", type=str, help="Path to the source code file to analyze.", required=True)
    parser.add_argument("--model_path", type=str, default="models/transformers/codet5/codet5-base_multilabel_finetuned", help="Path to the fine-tuned CodeT5 model.")
    args = parser.parse_args()

    with open(args.code, "r", encoding="utf-8") as f:
        code_content = f.read()

    detector = CodeSmellDetector(model_path=args.model_path)
    prediction = detector.predict_smells(code_content)
    print("\nPredicted code smells:", prediction)


if __name__ == "__main__":
    main()