FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/predict.py"]

Customer Sentiment Analysis & Product Intelligence System
📌 Overview
This project builds an end-to-end Natural Language Processing (NLP) pipeline to analyze customer product reviews, detect sentiment patterns, and generate actionable business recommendations.
The system helps identify high-performing products, detect potential risks, and support data-driven marketing and product decisions.
🚀 Key Features
✅ Automated sentiment classification using TF-IDF and Linear SVM
✅ Product-level promotion scoring based on customer feedback
✅ Confidence-aware recommendation system
✅ Business action framework (Promote / Monitor / Investigate)
✅ Model comparison (Logistic Regression vs Linear SVM)
✅ Portfolio visualization using sentiment vs review volume
✅ Dockerized pipeline for reproducible training and deployment
🧠 Business Impact
This solution enables organizations to:
Identify products suitable for aggressive promotion
Detect customer dissatisfaction early
Reduce brand risk
Support strategic marketing decisions
Monitor emerging high-potential products
📊 Model Selection
Two models were evaluated:
Model Purpose
Logistic Regression Baseline model
Linear SVM Final production model
Why Linear SVM?
Better handling of class imbalance
Improved minority sentiment detection
Higher overall accuracy (~88%)
Strong performance for high-dimensional text data
⚙️ Tech Stack
Python
Scikit-learn
Pandas
NumPy
Matplotlib / Seaborn
Docker
🏗️ Project Structure
healthkart-review-nlp/
│
├── data/ # Dataset
├── models/ # Saved model & vectorizer
├── notebooks/ # Analysis & experimentation
├── src/
│ ├── train.py # Training pipeline
│ └── predict.py # Inference script
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
▶️ Run with Docker
Build the container:
docker build -t healthkart-nlp .
Run the model:
docker run healthkart-nlp
🔬 End-to-End Pipeline
Data cleaning and preprocessing
Feature extraction using TF-IDF
Model training and evaluation
Product sentiment aggregation
Promotion score calculation
Business recommendation generation
Containerized deployment
📈 Key Business Recommendations
Promote Aggressively: Products with strong positive sentiment and high review volume.
Investigate Immediately: Products with elevated negative sentiment indicating potential quality gaps.
Monitor Emerging Products: Low-review but high-sentiment products that may represent future growth opportunities.
🔮 Future Improvements
Deploy as a real-time inference API
Experiment with transformer-based models (BERT)
Build an interactive dashboard
Implement real-time review monitoring
👤 Author
Mayank Sahu
