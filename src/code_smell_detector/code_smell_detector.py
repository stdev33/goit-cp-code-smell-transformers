import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.cleaner import clean_code

logging.basicConfig(level=logging.WARN)
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["file", "directory"],
        default="file",
        help='Mode of operation: "file" to analyze a single file, "directory" to analyze all .java files recursively.',
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Path to the source code file or directory to analyze.",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/transformers/codet5/codet5-base_multilabel_finetuned",
        help="Path to the fine-tuned CodeT5 model.",
    )
    args = parser.parse_args()

    detector = CodeSmellDetector(model_path=args.model_path)

    def format_output(path: str, prediction: str) -> str:
        """
        Format prediction with an emoji for quick visual assessment.
        🟢 for 'Clean', 🟡 for any detected code smell.
        """
        label = prediction.strip().lower()
        emoji = "🟢" if label == "clean" else "🟡"
        return f"{emoji} {path}: {prediction}"

    if args.mode == "file":
        if not os.path.isfile(args.code):
            logger.error("Provided path is not a file: %s", args.code)
            print(f"Error: provided path is not a file: {args.code}", file=sys.stderr)
            sys.exit(1)

        with open(args.code, "r", encoding="utf-8") as f:
            code_content = f.read()

        prediction = detector.predict_smells(code_content)
        print("\nPredicted code smells:")
        print(format_output(args.code, prediction))

    elif args.mode == "directory":
        if not os.path.isdir(args.code):
            logger.error("Provided path is not a directory: %s", args.code)
            print(f"Error: provided path is not a directory: {args.code}", file=sys.stderr)
            sys.exit(1)

        java_files = []
        for root, _, files in os.walk(args.code):
            for name in files:
                if name.lower().endswith(".java"):
                    java_files.append(os.path.join(root, name))

        if not java_files:
            logger.info("No .java files found in directory: %s", args.code)
            print(f"No .java files found in directory (including subdirectories): {args.code}")
            return

        logger.info("Found %d .java files in directory %s", len(java_files), args.code)
        print(f"Found {len(java_files)} .java file(s) in '{args.code}'. Running predictions...\n")

        for path in java_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                prediction = detector.predict_smells(code_content)
                print(format_output(path, prediction))
            except Exception as e:
                logger.error("Failed to analyze file %s: %s", path, e)
                print(f"⚠️  Failed to analyze file {path}: {e}", file=sys.stderr)
