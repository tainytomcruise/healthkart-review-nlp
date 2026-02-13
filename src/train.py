import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ---------- CLEANING FUNCTION ----------
def clean_text(text):
    if pd.isna(text):
        return ""
        
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ---------- LOAD DATA ----------
df = pd.read_csv("data/reviews.csv")

df["clean_text"] = df["reviews.text"].apply(clean_text)


# ---------- CREATE SENTIMENT ----------
def rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "average"
    else:
        return "negative"


df["sentiment"] = df["reviews.rating"].apply(rating_to_sentiment)


# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.85,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]


# ---------- TRAIN MODEL ----------
model = LinearSVC(class_weight="balanced")

model.fit(X, y)


# ---------- SAVE ----------
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")


print("Training complete. Model saved.")
