"""Shared code: preprocessing + similarity features (used by train.py and app.py)."""
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negations: "is" vs "is not" changes meaning
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "never"}

try:  # optional: real lemmatization/stemming if nltk is installed
    from nltk.stem import PorterStemmer
    _stem = PorterStemmer().stem
except Exception:  # simple fallback stemmer
    def _stem(w):
        for suf in ("ingly", "edly", "ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/URLs, remove stopwords, stem."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [_stem(t) for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def cosine_rows(A, B) -> np.ndarray:
    """Row-wise cosine similarity of two L2-normalised sparse matrices."""
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


_ENCODER = None


def get_encoder(name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the pretrained Sentence-BERT model once (downloads ~90 MB the first time)."""
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(name)
    return _ENCODER


def embedding_cosine(q1, q2, batch_size=128):
    """Semantic similarity: cosine between Sentence-BERT embeddings of RAW text
    (no stopword removal/stemming, the model needs the full sentence)."""
    enc = get_encoder()
    show = len(q1) > 500
    a = enc.encode(list(q1), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show)
    b = enc.encode(list(q2), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show)
    return (a * b).sum(axis=1)


FEATURE_NAMES = ["word_tfidf_cosine", "char_tfidf_cosine", "jaccard", "length_ratio", "semantic_embedding_cosine"]


def pair_features(q1, q2, word_vec, char_vec, use_emb=False):
    """Feature matrix for lists of raw question pairs.
    Columns: word TF-IDF cosine, char TF-IDF cosine, Jaccard, length ratio,
    and (if use_emb) Sentence-BERT embedding cosine."""
    c1 = [clean_text(x) for x in q1]
    c2 = [clean_text(x) for x in q2]
    w = cosine_rows(word_vec.transform(c1), word_vec.transform(c2))
    c = cosine_rows(char_vec.transform(c1), char_vec.transform(c2))
    j = np.array([jaccard(a, b) for a, b in zip(c1, c2)])
    l = np.array([min(len(a), len(b)) / max(len(a), len(b), 1) for a, b in zip(c1, c2)])
    cols = [w, c, j, l]
    if use_emb:
        cols.append(embedding_cosine(list(q1), list(q2)))
    return np.column_stack(cols)


def label_from_score(score: float, dup=0.75, sim=0.45) -> str:
    if score >= dup:
        return "Duplicate"
    if score >= sim:
        return "Similar"
    return "Different"
