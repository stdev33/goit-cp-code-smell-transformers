# Applied Computer Science Project

Applied Computer Science Project at GoIT Neoversity

---

# Code Smell Detection using Self-Supervised Learning: A Study on Transformer-based Representations

This repository contains the implementation and experimental results of the Master's Thesis project at GoIT Neoversity titled:

**"Code Smell Detection using Self-Supervised Learning: A Study on Transformer-based Representations"**  
(Original Ukrainian title: “Виявлення code smell за допомогою самонавчання: дослідження трансформерних представлень”)

---

## 📌 Project Overview

The goal of this project is to explore the effectiveness of Transformer-based models for automated detection of code smells in Java source code. The study focuses on comparing traditional machine learning models (e.g., Random Forest, XGBoost) with fine-tuned Transformer architectures (CodeBERT, GraphCodeBERT, CodeT5) trained on a merged dataset derived from the SmellyCode++[1] and CSQA[2] datasets.

---

## 🧪 Features

- **Code Smell Types**: Long Method, Large/God Class, Feature Envy, Data Class
- **Model Types**:
  - Classical: Random Forest, XGBoost
  - Transformer-based: CodeBERT, GraphCodeBERT, CodeT5
- **Evaluation Metrics**: F1-score, Hamming Loss, Precision-Recall curves
- **Data Merging**: Combines structural metrics (CSQA) with raw code and labels (SmellyCode++) for multi-label classification
- **Inference Pipelines**: For both classical and transformer-based models

---

## 📁 Project Structure

```
├── data/                   # Raw, processed, and prediction datasets
├── models/                 # Pretrained classical and Transformer models
├── notebooks/             # Jupyter notebooks for training, evaluation, and analysis
├── src/                    # Source code (data processing, training, inference modules)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📊 Results

All transformer-based models demonstrated superior F1-scores across most smell categories, with CodeT5 performing best overall. Classical models still offer faster inference and decent results with engineered features.

Graphs and CSVs of predictions are available in the `data/images/` and `data/predictions/` folders respectively.

---

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/goit-cp-code-smell-transformers.git
   cd goit-cp-code-smell-transformers
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Citation

If you use this project in your work, please cite it as:

> Tarasenko, Serhii. *Code Smell Detection using Self-Supervised Learning: A Study on Transformer-based Representations*. Master’s Thesis, GoIT Neoversity, 2025.
> 

---

## 📚 References

[1] Alomari, N., Alazba, A., Aljamaan, H., & Alshayeb, M. (2025). *SmellyCode++.csv* (Version 1). figshare. https://doi.org/10.6084/m9.figshare.28519385.v1

[2] Esmaili, E., Zakeri, M., & Parsa, S. (2023). *Code smells and quality attributes dataset* (Version 2). figshare. https://doi.org/10.6084/m9.figshare.24057336.v2

---

## 🔗 Repository Link

The source code and models are available at:  
👉 [https://github.com/stdev33/goit-cp-code-smell-transformers](https://github.com/stdev33/goit-cp-code-smell-transformers)

---