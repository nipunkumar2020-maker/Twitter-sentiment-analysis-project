# Twitter Sentiment Analysis

A sentiment analysis project that classifies tweets as **Positive**, **Negative** or **Neutral**. It includes a classic machine learning baseline (TF-IDF + Logistic Regression) and a pretrained transformer model (RoBERTa), served through an interactive Streamlit web app.

![App screenshot](screenshot.png)

## Features

- Web app built with Streamlit: type a tweet and get its sentiment with a confidence score
- Three-class prediction (Positive / Neutral / Negative) using a Twitter-specific RoBERTa model
- Baseline model trained from scratch on 200,000 tweets for comparison
- Text cleaning pipeline (URLs, mentions, hashtags, punctuation)

## Dataset

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle: 1.6 million tweets labelled as negative (0) or positive (4).

The dataset is not included in this repository because of its size. To train the baseline model:

1. Download it from Kaggle and extract it.
2. Place `training.1600000.processed.noemoticon.csv` in `data/archive/`.

## Models

| Model | Classes | Training | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | Positive, Negative | Trained on 200k tweets from Sentiment140 | Fast and lightweight, but cannot detect neutral text |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Positive, Neutral, Negative | Pretrained on a large tweet corpus | Handles slang, context and neutral statements better |

**Baseline accuracy (Logistic Regression):** XX% on a held-out 20% test set *(replace with your result)*

### Example comparison

| Sentence | Logistic Regression | RoBERTa |
|---|---|---|
| The meeting is at 3 pm tomorrow. | Negative | Neutral |
| I love this new phone! | Positive | Positive |
| Worst service ever, totally disappointed | Negative | Negative |

*(Replace with the output of `python compare.py`.)*

The baseline model only knows two classes, so it is forced to label neutral sentences as positive or negative. This is the main reason the app uses RoBERTa.

## Project structure

```
twitter-sentiment/
├── app.py              # Streamlit app (RoBERTa)
├── sentiment.py        # Trains the Logistic Regression baseline
├── compare.py          # Compares both models on sample sentences
├── requirements.txt
├── data/               # Dataset (not tracked by git)
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/twitter-sentiment.git
cd twitter-sentiment
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The first run downloads the RoBERTa model (about 500 MB), so it may take a few minutes. After that, the app opens at `http://localhost:8501`.

### Optional: train the baseline model

After downloading the dataset (see above):

```bash
python sentiment.py
```

This trains the Logistic Regression model and saves `model.pkl` and `tfidf.pkl`. You can then run `python compare.py` to compare it with RoBERTa.

## Tech stack

Python, pandas, scikit-learn, Hugging Face Transformers, PyTorch, Streamlit

## Limitations

- Sarcasm and irony are still hard for both models.
- The baseline model was trained on English tweets only.
- Hinglish and other mixed-language text is not supported.

## Future improvements

- Upload a CSV and analyse many tweets at once
- Charts and word clouds of sentiment results
- Hinglish support
- Deploy on Streamlit Community Cloud

