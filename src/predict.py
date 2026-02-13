import joblib

model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction


if __name__ == "__main__":
    sample = "This product is amazing. I love it."
    print("Prediction:", predict_sentiment(sample))
