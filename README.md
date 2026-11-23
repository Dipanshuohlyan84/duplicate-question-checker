[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://duplicheck-nlp.streamlit.app/)
# 🔍 DupliCheck – Text Similarity & Duplicate Question Detection
NLP Capstone – Project 5. Compares two questions/sentences and decides if they are **Duplicate**, **Similar** or **Different**.

## Pipeline
1. **Problem**: detect repeated/semantically similar questions (forums, Quora, support desks, FAQ search).
2. **Dataset**: Quora Question Pairs (`train.csv`: question1, question2, is_duplicate) – https://www.kaggle.com/c/quora-question-pairs. A tiny `data/sample_pairs.csv` is included only for a smoke test.
3. **Cleaning**: drop nulls and duplicate rows.
4. **Preprocessing** (`similarity.py`): lowercase, remove URLs/punctuation, stopword removal (negations kept), stemming.
5. **EDA**: see `notebooks` section below (class balance, question length, word clouds).
6. **Features**: word TF-IDF (1-2 grams) cosine, char TF-IDF (2-4 grams) cosine, Jaccard overlap, length ratio.
7. **Model**: Logistic Regression on those features (baseline: plain cosine ≥ 0.5).
8. **Evaluation**: Accuracy, Precision, Recall, F1, confusion matrix.
9. **GUI**: Streamlit (`app.py`) – two text boxes, Compare Text button, score, label.
10. **Testing**: try paraphrases, unrelated pairs, negations, empty input.

## Setup & run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# put Quora's train.csv in data/
python train.py --data data/train.csv --sample 100000
streamlit run app.py
```

## Labels
Model score ≥ 0.75 → Duplicate, 0.45–0.75 → Similar, below → Different (adjustable in the sidebar).

## Limitations & future scope
TF-IDF captures word overlap, not deep meaning ("buy a car" vs "purchase an automobile" scores low). Upgrade: Sentence-BERT embeddings (`sentence-transformers`, model `all-MiniLM-L6-v2`) in place of TF-IDF, or fine-tune BERT.

## Team
Add names here. Add GUI screenshots in `screenshots/`.
