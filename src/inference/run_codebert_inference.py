from src.inference.transformer_models_inference import (
    load_sequence_classification_model,
    load_seq2seq_model,
    predict_multilabel_classification,
    predict_codet5,
    evaluate_predictions,
    plot_pr_curves,
    save_predictions_csv
)
from src.inference.load_evaluation_data import load_and_prepare_evaluation_data


texts, y_true, label_cols = load_and_prepare_evaluation_data("../../data/processed/merged_for_evaluation.csv")

model_path = "../../models/transformers/codebert/codebert-base_multilabel_finetuned"

tokenizer, model = load_sequence_classification_model(model_path)

y_pred, probs = predict_multilabel_classification(texts, tokenizer, model)

report, ham_loss, subset_acc = evaluate_predictions(y_true, y_pred, label_cols)

print(report)
print("Hamming Loss:", ham_loss)
print("Subset Accuracy:", subset_acc)

plot_pr_curves(y_true, probs, label_cols, "codebert", "../../data/images")
save_predictions_csv(texts, y_true, y_pred, probs, label_cols, "../../data/predictions/codebert_predictions.csv")