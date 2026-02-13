import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load artifacts
vectorizer = joblib.load("models/vectorizer.pkl")

# Load dataset
df = pd.read_csv("data/reviews.csv")

df = df.drop_duplicates()


# Remove missing reviews
df = df.dropna(subset=["reviews.text"])

# Ensure text format
df["reviews.text"] = df["reviews.text"].astype(str)

# Combine reviews per product
product_reviews = df.groupby("name")["reviews.text"].apply(lambda x: " ".join(x))


# Vectorize products
tfidf_matrix = vectorizer.transform(product_reviews)

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
product_index = {product: idx for idx, product in enumerate(product_reviews.index)}


def recommend_products(product_name, top_n=5):
    if product_name not in product_index:
        return "Product not found."

    idx = product_index[product_name]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommended = [product_reviews.index[i[0]] for i in similarity_scores]

    return recommended


# Example usage
if __name__ == "__main__":
    product = product_reviews.index[0]
    print(f"\nProducts similar to: {product}\n")
    
    for rec in recommend_products(product):
        print(rec)
