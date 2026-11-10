import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from transformers import pipeline
from wordcloud import STOPWORDS, WordCloud

MODELS = {
    "English": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "Hinglish": "pascalrai/hinglish-twitter-roberta-base-sentiment",
}
MAX_ROWS = 500

HINGLISH_STOPWORDS = {
    "hai", "hain", "ho", "hu", "hun", "tha", "thi", "the", "ka", "ki", "ke",
    "ko", "se", "me", "mein", "par", "pe", "aur", "ya", "to", "toh", "bhi",
    "nahi", "nhi", "ye", "yeh", "wo", "woh", "mai", "main", "tum", "tu",
    "hum", "kya", "kuch", "bhai", "heh",
}

# Some models return LABEL_0/1/2 instead of names
LABEL_MAP = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}

LABELS = {
    "positive": ("Positive 😊", st.success),
    "neutral": ("Neutral 😐", st.info),
    "negative": ("Negative 😞", st.error),
}


@st.cache_resource
def load_model(model_name):
    """Load a model once and reuse it across reruns."""
    return pipeline("sentiment-analysis", model=model_name)


def normalize(label):
    label = label.lower()
    return LABEL_MAP.get(label, label)


def clean_for_cloud(text):
    """Remove URLs and mentions so they don't dominate the cloud."""
    text = re.sub(r"http\S+|www\S+", "", text)
    return re.sub(r"@\w+", "", text)


def show_wordcloud(texts, label, stopwords):
    combined = " ".join(texts)
    try:
        cloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stopwords,
            collocations=False,
        ).generate(combined)
    except ValueError:
        st.info(f"Not enough words to build a word cloud for: {label}.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


st.title("Twitter Sentiment Analysis")

language = st.radio("Language", ["English", "Hinglish"], horizontal=True)
classifier = load_model(MODELS[language])
stopwords = set(STOPWORDS) | HINGLISH_STOPWORDS if language == "Hinglish" else set(STOPWORDS)

tab_single, tab_csv = st.tabs(["Single tweet", "Upload CSV"])

with tab_single:
    placeholder = "tu mujhe pasandh heh" if language == "Hinglish" else ""
    tweet = st.text_area("Enter your tweet:", placeholder=placeholder)

    if st.button("Analyze"):
        if tweet.strip():
            result = classifier(tweet, truncation=True)[0]
            text, show = LABELS[normalize(result["label"])]
            show(f"{text} ({result['score']:.0%} confidence)")
        else:
            st.warning("Please enter some text first.")

with tab_csv:
    st.write(f"Upload a CSV file with a column named **text** (first {MAX_ROWS} rows are analysed).")
    file = st.file_uploader("Choose a CSV file", type="csv")

    if file is not None:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("The CSV must have a column named 'text'.")
        else:
            df = df.dropna(subset=["text"]).head(MAX_ROWS).copy()

            with st.spinner("Analysing tweets..."):
                results = classifier(
                    df["text"].astype(str).tolist(),
                    truncation=True,
                    batch_size=16,
                )

            df["sentiment"] = [normalize(r["label"]).capitalize() for r in results]
            df["confidence"] = [round(r["score"], 3) for r in results]

            st.subheader("Sentiment distribution")
            st.bar_chart(df["sentiment"].value_counts())

            st.subheader("Word cloud")
            choice = st.selectbox("Show words from", ["All", "Positive", "Neutral", "Negative"])
            subset = df if choice == "All" else df[df["sentiment"] == choice]
            cleaned = subset["text"].astype(str).apply(clean_for_cloud).tolist()
            show_wordcloud(cleaned, choice, stopwords)

            st.subheader("Results")
            st.dataframe(df)

            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )