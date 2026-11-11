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
LOW_CONFIDENCE = 0.6

EXAMPLES = {
    "English": [
        "I love this new phone!",
        "The meeting is at 3 pm tomorrow.",
        "Worst service ever, totally disappointed",
    ],
    "Hinglish": [
        "tu mujhe pasandh heh",
        "kal meeting hai 3 baje",
        "ye movie bahut bekar thi, paisa barbaad",
    ],
}

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


def set_example(text):
    st.session_state.tweet_text = text


st.title("Twitter Sentiment Analysis")

language = st.radio("Language", ["English", "Hinglish"], horizontal=True)
classifier = load_model(MODELS[language])
stopwords = set(STOPWORDS) | HINGLISH_STOPWORDS if language == "Hinglish" else set(STOPWORDS)

if "tweet_text" not in st.session_state:
    st.session_state.tweet_text = ""

tab_single, tab_csv = st.tabs(["Single tweet", "Upload CSV"])

with tab_single:
    st.write("Try an example:")
    columns = st.columns(len(EXAMPLES[language]))
    for column, example in zip(columns, EXAMPLES[language]):
        column.button(
            example,
            on_click=set_example,
            args=(example,),
            key=f"example_{language}_{example}",
        )

    tweet = st.text_area("Enter your tweet:", key="tweet_text")

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

            # Summary numbers
            total = len(df)
            positive_pct = (df["sentiment"] == "Positive").mean()
            low_conf = (df["confidence"] < LOW_CONFIDENCE).sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Tweets analysed", total)
            m2.metric("Positive", f"{positive_pct:.0%}")
            m3.metric(f"Low confidence (<{LOW_CONFIDENCE:.0%})", int(low_conf))

            # Bar chart and pie chart side by side
            counts = df["sentiment"].value_counts()
            left, right = st.columns(2)

            with left:
                st.subheader("Sentiment distribution")
                st.bar_chart(counts)

            with right:
                st.subheader("Sentiment share")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(
                    counts.values,
                    labels=counts.index,
                    autopct="%1.0f%%",
                    startangle=90,
                )
                ax.axis("equal")
                st.pyplot(fig)

            # Most confident tweets
            st.subheader("Most confident tweets")
            for sentiment in ["Positive", "Negative"]:
                top = (
                    df[df["sentiment"] == sentiment]
                    .nlargest(5, "confidence")[["text", "confidence"]]
                )
                st.write(f"**Top {sentiment.lower()} tweets**")
                if top.empty:
                    st.write("None found.")
                else:
                    st.dataframe(top, hide_index=True)

            # Hashtags
            hashtags = (
                df["text"].astype(str).str.findall(r"#\w+").explode().dropna().str.lower()
            )
            if not hashtags.empty:
                st.subheader("Top hashtags")
                st.bar_chart(hashtags.value_counts().head(10))

            # Word cloud
            st.subheader("Word cloud")
            choice = st.selectbox("Show words from", ["All", "Positive", "Neutral", "Negative"])
            subset = df if choice == "All" else df[df["sentiment"] == choice]
            cleaned = subset["text"].astype(str).apply(clean_for_cloud).tolist()
            show_wordcloud(cleaned, choice, stopwords)

            # Full results
            st.subheader("Results")
            st.dataframe(df)

            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )