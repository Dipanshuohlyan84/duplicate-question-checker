"""DupliCheck - Streamlit GUI.  Run: python -m streamlit run app.py"""
import os
import random
import hashlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from similarity import clean_text, pair_features, label_from_score, get_encoder

st.set_page_config(page_title="DupliCheck", page_icon="🔍", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.hero {padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;
       background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%); color: white;}
.hero h1 {margin: 0; font-size: 2.4rem; color: white;}
.hero p  {margin: 6px 0 0 0; opacity: .92; font-size: 1.05rem;}
.card {padding: 22px 26px; border-radius: 16px; margin-top: 14px; color: #111;}
.card h2 {margin: 0 0 4px 0; color: #111;}
.card p  {margin: 0; color: #222;}
.dup  {background: #d1fae5; border-left: 8px solid #10b981;}
.sim  {background: #fef3c7; border-left: 8px solid #f59e0b;}
.diff {background: #fee2e2; border-left: 8px solid #ef4444;}
.bar-wrap {background: #e5e7eb; border-radius: 999px; height: 22px; margin-top: 12px; overflow: hidden;}
.bar {height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);}
.small {color: #6b7280; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔍 DupliCheck</h1>
  <p>Duplicate question &amp; text similarity detector &nbsp;•&nbsp; Sentence-BERT + TF-IDF + Cosine Similarity + Logistic Regression</p>
</div>
""", unsafe_allow_html=True)


# ---------- Load model ----------
@st.cache_resource
def load_model():
    path = "models/similarity_model.joblib"
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner="Building the search index (first time takes a few minutes)...")
def load_question_bank(_word_vec, use_emb, n=50000):
    """Question bank for the 'Find similar' tab: dataset questions + optional data/extra_questions.txt."""
    texts = []
    if os.path.exists("data/train.csv"):
        df = pd.read_csv("data/train.csv", usecols=["question1", "question2"]).dropna()
        qs = pd.unique(pd.concat([df.question1, df.question2]).astype(str))
        rng = np.random.RandomState(42)
        if len(qs) > n:
            qs = rng.choice(qs, n, replace=False)
        texts = list(qs)
    if os.path.exists("data/extra_questions.txt"):
        with open("data/extra_questions.txt", encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        return None, None, None
    tfidf = _word_vec.transform([clean_text(q) for q in texts])
    emb = None
    if use_emb:
        cache = "models/bank_embeddings.npz"
        key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
        if os.path.exists(cache):
            z = np.load(cache)
            if str(z["key"]) == key:
                emb = z["emb"]
        if emb is None:  # encode once, then reuse from disk
            emb = get_encoder().encode(texts, batch_size=128, normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
            np.savez(cache, emb=emb, key=np.array(key))
    return texts, tfidf, emb


@st.cache_resource
def warm_encoder():
    from similarity import get_encoder
    return get_encoder()


m = load_model()
USE_EMB = bool(m.get("use_emb", False)) if m else False
if m is not None and USE_EMB:
    with st.spinner("Loading semantic model (first time downloads ~90 MB)..."):
        warm_encoder()
if m is None:
    st.error("Model not found. Run `python train.py --data data/train.csv` first.")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    dup_t = st.slider("Duplicate if score ≥", 0.50, 1.00, 0.75, 0.01)
    sim_t = st.slider("Similar if score ≥", 0.10, dup_t, min(0.45, dup_t), 0.01)
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Clean text (lowercase, stopwords, stemming)\n"
                "2. Sentence-BERT embeddings (meaning) + TF-IDF (words)\n"
                "3. Cosine similarity, Jaccard, length ratio\n"
                "4. Logistic Regression combines them into the final score")
    if st.button("🗑️ Clear history"):
        st.session_state["history"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

EXAMPLES = {
    "✅ Duplicate": [
        ("How can I learn Python quickly?", "What is the fastest way to learn Python?"),
        ("What is natural language processing?", "Can you explain what NLP means?"),
        ("How do I reset my Gmail password?", "What are the steps to recover a forgotten Gmail password?"),
        ("Which is the best laptop for programming?", "What laptop should I buy for coding?"),
        ("How can I earn money online?", "What are some ways to make money on the internet?"),
        ("Why is the sky blue?", "What makes the sky look blue?"),
        ("How do I improve my English speaking?", "What are some tips to speak better English?"),
        ("How far is the moon from Earth?", "What is the distance between the Earth and the moon?"),
    ],
    "🟡 Similar": [
        ("How do I lose weight fast?", "What is the best diet for weight loss?"),
        ("How can I learn machine learning?", "What are the best books for machine learning?"),
        ("What is the best way to study for exams?", "How can I improve my concentration while studying?"),
        ("How do I start investing in stocks?", "What are the risks of investing in the stock market?"),
        ("Which programming language should I learn first?", "Is Python good for beginners?"),
        ("How do I prepare for a job interview?", "What questions are asked in a software engineer interview?"),
    ],
    "❌ Different": [
        ("What is the capital of France?", "How do I cook pasta at home?"),
        ("How do I learn Python?", "What is the population of India?"),
        ("Who wrote Hamlet?", "How do I change a flat tyre?"),
        ("What is machine learning?", "What are the symptoms of the common cold?"),
        ("Which is the best phone camera?", "How was the Roman Empire founded?"),
        ("How do I invest in mutual funds?", "Why do cats purr?"),
    ],
}


def set_example(category):
    """Fill both boxes with a random pair from the category (never the same pair twice in a row)."""
    last = (st.session_state.get("q1"), st.session_state.get("q2"))
    options = [p for p in EXAMPLES[category] if p != last] or EXAMPLES[category]
    st.session_state["q1"], st.session_state["q2"] = random.choice(options)


tab1, tab2, tab3 = st.tabs(["🔎 Compare two texts", "🧭 Find similar questions", "🕘 History"])

# ---------- Tab 1: compare ----------
with tab1:
    st.markdown("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=set_example, args=(name,), width="stretch")

    c1, c2 = st.columns(2)
    q1 = c1.text_area("📝 Text / Question 1", key="q1", height=140, placeholder="Type or paste the first question…")
    q2 = c2.text_area("📝 Text / Question 2", key="q2", height=140, placeholder="Type or paste the second question…")

    if st.button("🚀 Compare Text", type="primary", width="stretch"):
        if not q1.strip() or not q2.strip():
            st.warning("Please enter both texts.")
        else:
            X = pair_features([q1], [q2], m["word_vec"], m["char_vec"], use_emb=USE_EMB)
            cosine, char_cos, jac = float(X[0, 0]), float(X[0, 1]), float(X[0, 2])
            score = float(m["clf"].predict_proba(X)[0, 1])
            label = label_from_score(score, dup_t, sim_t)
            css, icon, msg = {
                "Duplicate": ("dup", "✅", "These two texts ask essentially the same thing."),
                "Similar": ("sim", "🟡", "These texts are related but not exact duplicates."),
                "Different": ("diff", "❌", "These texts are about different things."),
            }[label]
            st.markdown(f"""
            <div class="card {css}">
              <h2>{icon} {label} &nbsp;—&nbsp; {score:.1%} similarity</h2>
              <p>{msg}</p>
              <div class="bar-wrap"><div class="bar" style="width:{score*100:.1f}%"></div></div>
            </div>""", unsafe_allow_html=True)

            mc = st.columns(5 if USE_EMB else 4)
            mc[0].metric("Final score", f"{score:.2f}")
            if USE_EMB:
                mc[1].metric("Semantic (BERT) cosine", f"{float(X[0, 4]):.2f}")
            mc[-3].metric("Word TF-IDF cosine", f"{cosine:.2f}")
            mc[-2].metric("Char TF-IDF cosine", f"{char_cos:.2f}")
            mc[-1].metric("Jaccard overlap", f"{jac:.2f}")

            with st.expander("See the preprocessed text"):
                st.write("**Text 1 →**", clean_text(q1))
                st.write("**Text 2 →**", clean_text(q2))

            st.session_state["history"].insert(0, {
                "Text 1": q1[:60], "Text 2": q2[:60], "Score": round(score, 3), "Result": label})

# ---------- Tab 2: find similar ----------
with tab2:
    st.write("Type a question and see the most similar questions from the dataset "
             "(ranked by meaning using Sentence-BERT, plus keyword overlap).")
    bank, tfidf_mat, emb_mat = load_question_bank(m["word_vec"], USE_EMB)
    if bank is None:
        st.info("Put `train.csv` in the `data` folder to enable this tab.")
    else:
        query = st.text_input("Your question", placeholder="How do I start learning machine learning?")
        min_sim = st.slider("Minimum similarity to show", 0.0, 1.0, 0.50, 0.01,
                            help="Results below this score are hidden, so unrelated questions don't appear.")
        if st.button("Find similar questions") and query.strip():
            kw = (tfidf_mat @ m["word_vec"].transform([clean_text(query)]).T).toarray().ravel()
            if emb_mat is not None:
                sem = emb_mat @ get_encoder().encode([query], normalize_embeddings=True)[0]
                score = 0.8 * sem + 0.2 * kw          # meaning counts more than word overlap
            else:
                sem, score = None, kw
            rows = []
            for i in np.argsort(-score)[:20]:
                if score[i] < min_sim or len(rows) == 5:
                    break
                if bank[i].strip().lower() == query.strip().lower():
                    continue                           # skip the identical question
                row = {"Similar question": bank[i], "Score": round(float(score[i]), 3)}
                if sem is not None:
                    row["Semantic"] = round(float(sem[i]), 3)
                row["Keyword"] = round(float(kw[i]), 3)
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.warning("No sufficiently similar questions found. Try lowering the minimum "
                           "similarity, or add more questions to the dataset.")

# ---------- Tab 3: history ----------
with tab3:
    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]), width="stretch", hide_index=True)
    else:
        st.info("No comparisons yet. Run one in the first tab.")