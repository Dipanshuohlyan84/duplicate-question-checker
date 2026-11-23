"""Train + evaluate the duplicate-question model.
Usage: python train.py --data data/train.csv --sample 50000
CSV needs columns: question1, question2, is_duplicate (Quora Question Pairs format)."""
import argparse, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from similarity import clean_text, pair_features, FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sample_pairs.csv")
ap.add_argument("--sample", type=int, default=0, help="use only N rows (faster)")
ap.add_argument("--embeddings", action="store_true", help="add Sentence-BERT semantic feature (needs sentence-transformers)")
args = ap.parse_args()

# 1-3. Load + clean data
df = pd.read_csv(args.data)[["question1", "question2", "is_duplicate"]]
print("Rows loaded:", len(df))
df = df.dropna().drop_duplicates()
df["question1"], df["question2"] = df["question1"].astype(str), df["question2"].astype(str)
if args.sample and len(df) > args.sample:
    df = df.sample(args.sample, random_state=42)
print("Rows after cleaning:", len(df), "| duplicate ratio:", round(df.is_duplicate.mean(), 3))

# split BEFORE fitting vectorizers (no leakage)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.is_duplicate)

# 4-6. Preprocess + TF-IDF vocabulary (word-level and char-level)
corpus = pd.concat([train.question1, train.question2]).map(clean_text)
word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(corpus)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(corpus)

print('Building features', '(with Sentence-BERT embeddings, this can take a while)...' if args.embeddings else '...')
Xtr = pair_features(train.question1, train.question2, word_vec, char_vec, use_emb=args.embeddings)
Xte = pair_features(test.question1, test.question2, word_vec, char_vec, use_emb=args.embeddings)

# 7. Model: logistic regression on similarity features
clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, train.is_duplicate)

# 8. Evaluation
pred = clf.predict(Xte)
print("\n--- Evaluation (hold-out test set) ---")
print("Accuracy :", round(accuracy_score(test.is_duplicate, pred), 3))
print("Precision:", round(precision_score(test.is_duplicate, pred, zero_division=0), 3))
print("Recall   :", round(recall_score(test.is_duplicate, pred, zero_division=0), 3))
print("F1-score :", round(f1_score(test.is_duplicate, pred, zero_division=0), 3))
print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(test.is_duplicate, pred))
print("Baseline (word TF-IDF cosine >= 0.5) F1:",
      round(f1_score(test.is_duplicate, (Xte[:, 0] >= 0.5).astype(int), zero_division=0), 3))

print("\nFeature weights (higher = more influence):")
for n, w in zip(FEATURE_NAMES, clf.coef_[0]):
    print(f"  {n:28s} {w:+.2f}")

joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf, "use_emb": args.embeddings}, "models/similarity_model.joblib")
print("\nSaved models/similarity_model.joblib")
