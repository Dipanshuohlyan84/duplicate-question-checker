import pandas as pd

BASE = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/"

def load(name, split):
    return pd.read_parquet(f"{BASE}{name}/{split}-00000-of-00001.parquet")

# Quora Question Pairs (~400k pairs)
df = pd.concat([load("qqp", "train"), load("qqp", "validation")])
df = df.rename(columns={"label": "is_duplicate"})[["question1", "question2", "is_duplicate"]]

# Optional: MRPC (~5.8k sentence pairs)
m = pd.concat([load("mrpc", "train"), load("mrpc", "validation")])
m = m.rename(columns={"sentence1": "question1", "sentence2": "question2", "label": "is_duplicate"})
m = m[["question1", "question2", "is_duplicate"]]

full = pd.concat([df, m]).dropna().drop_duplicates()
full.to_csv("data/train.csv", index=False)
print("Saved", len(full), "pairs")