import streamlit as st
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@st.cache_resource
def load_model():
    """Load the model once and reuse it across reruns."""
    return pipeline("sentiment-analysis", model=MODEL_NAME)


classifier = load_model()

LABELS = {
    "positive": ("Positive 😊", st.success),
    "neutral": ("Neutral 😐", st.info),
    "negative": ("Negative 😞", st.error),
}

st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to find out whether its sentiment is positive, negative or neutral.")

tweet = st.text_area("Enter your tweet:")

if st.button("Analyze"):
    if tweet.strip():
        result = classifier(tweet)[0]
        label = result["label"].lower()
        confidence = result["score"]

        text, show = LABELS[label]
        show(f"{text} ({confidence:.0%} confidence)")
    else:
        st.warning("Please enter some text first.")