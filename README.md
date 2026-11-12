# Twitter Sentiment Analysis

A sentiment analysis app that classifies tweets as **Positive**, **Negative** or **Neutral** in **English and Hinglish**. It includes a classic machine learning baseline (TF-IDF + Logistic Regression), pretrained RoBERTa models, and an interactive Streamlit web app.

**Live demo:** <add-your-streamlit-link-here>

![Single tweet analysis](screenshot.png)

## Features

- Analyse a single tweet and get its sentiment with a confidence score
- Three-class prediction (Positive / Neutral / Negative)
- English and Hinglish support through separate pretrained models
- CSV upload: analyse hundreds of tweets at once
- Sentiment distribution chart and word cloud (filterable by sentiment)
- Download the results as a CSV file
- Baseline model trained from scratch on 200,000 tweets for comparison
- Summary metrics, pie chart, top confident tweets and hashtag chart for CSV analysis
- One-click example tweets

![CSV analysis](screenshot-csv.png)

## Dataset

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle: 1.6 million tweets labelled as negative (0) or positive (4). It is used only to train the baseline model and is not included in this repository because of its size.

To train the baseline:

1. Download the dataset from Kaggle and extract it.
2. Place `training.1600000.processed.noemoticon.csv` in `data/archive/`.

## Models

| Model | Language | Classes | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | English | Positive, Negative | Trained on 200k tweets; fast, but cannot detect neutral text |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | English | Positive, Neutral, Negative | Pretrained on tweets; used in the app |
| `pascalrai/hinglish-twitter-roberta-base-sentiment` | Hinglish | Positive, Neutral, Negative | Fine-tuned on Hinglish tweets; used in the app |

**Baseline accuracy (Logistic Regression):** XX% on a held-out 20% test set *(replace with your result)*

### Example comparison

| Sentence | Logistic Regression | RoBERTa |
|---|---|---|
| The meeting is at 3 pm tomorrow. | Negative | Neutral |
| I love this new phone! | Positive | Positive |
| Worst service ever, totally disappointed | Negative | Negative |

*(Replace with the output of `python compare.py`.)*

The baseline model only knows two classes, so it must label neutral sentences as positive or negative. This is the main reason the app uses RoBERTa.

## Project structure

```
twitter-sentiment/
├── app.py              # Streamlit app
├── sentiment.py        # Trains the Logistic Regression baseline
├── compare.py          # Compares baseline and RoBERTa on sample sentences
├── make_sample.py      # Creates a sample CSV from Sentiment140
├── test.csv            # Small sample file for trying the CSV upload
├── requirements.txt
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

The first run downloads the models (about 500 MB each), so it may take a few minutes. The app opens at `http://localhost:8501`.

### Using the CSV upload

Upload a CSV file with a column named `text` (the first 500 rows are analysed). You can try it with the included `test.csv`.

### Optional: train the baseline model

Install the extra packages and run the training script (requires the dataset, see above):

```bash
pip install pandas scikit-learn joblib
python sentiment.py
```

This saves `model.pkl` and `tfidf.pkl`. Then `python compare.py` compares it with RoBERTa.

## Tech stack

Python, pandas, scikit-learn, Hugging Face Transformers, PyTorch, Streamlit, WordCloud, Matplotlib

## Limitations

- Sarcasm and irony are hard for all models.
- Hinglish accuracy is lower than English, and results depend on spelling and emoji use.
- The baseline model was trained on English tweets only and has no neutral class.
- The CSV analysis is limited to 500 rows to keep the app responsive.

## Future improvements

- Live tweet fetching
- More languages
- Model fine-tuning on domain-specific data



