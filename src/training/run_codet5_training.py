from src.training.data_utils import load_and_prepare_dataset
from src.training.train_codet5 import train_codet5_multilabel

def main() -> None:
    df, label_cols = load_and_prepare_dataset("../../data/processed/merged_for_training.csv")

    # Display class distribution
    print("Number of samples:", len(df))
    print("\nClass distribution:")
    print(df[label_cols].sum())

    train_codet5_multilabel(
        model_name="Salesforce/codet5-base",
        df=df,
        label_cols=["Long Method", "God/Large Class", "Feature Envy", "Data Class", "Clean"],
        output_dir="../../models/transformers/codet5"
    )


if __name__ == "__main__":
    main()