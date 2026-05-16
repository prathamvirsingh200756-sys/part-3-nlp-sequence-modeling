# Part 3 — Customer Support Text Classification: NLP & Sequence Modeling

## Overview
End-to-end NLP pipeline that classifies customer support messages into **positive**, **neutral**, or **negative** sentiment using classical ML baselines and a conceptual LSTM sequence model.

## Dataset
- **File**: `customer_support_text_classification.csv`
- **Records**: 1,500 customer support messages
- **Classes**: positive (479), neutral (524), negative (497)
- **Avg word count**: ~12.7 words per message

## Tasks Covered

| Task | Description |
|---|---|
| 1 | Dataset Understanding — class distribution, text length stats, sample records |
| 2 | Text Preprocessing — lowercasing, regex cleaning, tokenisation, stopword removal |
| 3 | Text Vectorisation — Bag of Words, TF-IDF (bigrams), integer sequence encoding |
| 4 | Baseline Models — Naive Bayes (BoW), Logistic Regression (TF-IDF), LinearSVC |
| 5 | Sequence Model — LSTM architecture definition + NumPy forward-pass simulation |
| 6 | Reflection — RNNs, LSTMs, Attention, Transformers |

## Results

All models achieved **very high accuracy** on this dataset, reflecting clear and distinct vocabulary patterns across sentiment classes. The best baseline model is compared using accuracy and macro/weighted F1 scores.

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Naive Bayes (BoW) | — | — |
| Logistic Regression (TF-IDF) | — | — |
| LinearSVC (TF-IDF) | — | — |

*(Run notebook to see live results)*

## Repository Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── eda_plots.png
    ├── top_words_per_class.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── lstm_architecture.png
    ├── model_evaluation.csv
    └── sample_predictions.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

For full LSTM training, install TensorFlow:
```bash
pip install tensorflow>=2.10.0
```
Then uncomment the Keras code block in Task 5.

## Key Takeaways

- **TF-IDF + bigrams** captures short-context sentiment well on concise customer messages
- **LSTMs** are better suited for longer text with sequential dependencies
- **Transformers** (e.g., DistilBERT) would be the production-grade upgrade for this task
