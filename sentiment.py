import pandas as pd, re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load
cols = ["target", "id", "date", "flag", "user", "text"]
import os
BASE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE, "data", "archive", "training.1600000.processed.noemoticon.csv")
print("Exists:", os.path.exists(path))

df = pd.read_csv(path, encoding="latin-1", names=cols)

df = df[["text", "target"]].sample(200000, random_state=42)
df["target"] = df["target"].replace(4, 1)

# Clean
def clean(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

df["clean"] = df["text"].apply(clean)

# Train
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["target"], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_v = tfidf.fit_transform(X_train)
X_test_v = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_v, y_train)

# Evaluate
pred = model.predict(X_test_v)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Test
for text in ["I love this new phone!", "Worst service ever, so disappointed"]:
    result = model.predict(tfidf.transform([clean(text)]))[0]
    print(text, "->", "Positive" if result == 1 else "Negative")
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("Model saved!")