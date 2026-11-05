import re
import joblib
from transformers import pipeline

lr_model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
roberta = pipeline("sentiment-analysis",
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")


def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


sentences = [
    "The meeting is at 3 pm tomorrow.",
    "I love this new phone!",
    "Worst service ever, totally disappointed",
    "I will submit the report by Friday.",
    "Not bad, not great either.",
]

print(f"{'Sentence':45} {'LogReg':10} {'RoBERTa':10}")
for s in sentences:
    lr = "Positive" if lr_model.predict(tfidf.transform([clean(s)]))[0] == 1 else "Negative"
    rb = roberta(s)[0]["label"].capitalize()
    print(f"{s:45} {lr:10} {rb:10}")