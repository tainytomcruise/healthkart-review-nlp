Customer Sentiment Analysis & Product Intelligence System

Overview

This project builds an end-to-end Natural Language Processing (NLP) pipeline to analyze customer product reviews, detect sentiment patterns, and generate actionable business recommendations.
The system enables organizations to identify high-performing products, detect potential risks, and support data-driven marketing and product decisions at scale.


Key Features

 Automated sentiment classification using TF-IDF and Linear SVM
 Hybrid use of structured data (ratings, product identifiers) and unstructured review text
 Product-level promotion scoring based on customer feedback
 Confidence-aware recommendation framework
 Business action strategy (Promote / Monitor / Investigate)
 Model comparison (Logistic Regression vs Linear SVM)
 Portfolio visualization using sentiment vs review volume
 Dockerized pipeline for reproducible training and deployment

Dataset Utilization

The dataset contains both structured attributes (ratings, product names, identifiers) and unstructured customer reviews.
Structured data was leveraged to:
Derive sentiment labels from ratings
Aggregate product-level sentiment
Calculate promotion scores
Improve decision reliability using review volume
Unstructured text was used to:
Train the NLP sentiment classifier
Capture customer perception signals
Identify root causes behind satisfaction or dissatisfaction
This hybrid approach improves both statistical robustness and contextual understanding.
Public dataset provided as part of the HealthKart assessment.

End-to-End Pipeline

Data cleaning and preprocessing
Feature extraction using TF-IDF
Train-test split to evaluate performance on unseen data
Model training and evaluation
Product sentiment aggregation
Promotion score calculation
Business recommendation generation
Containerized deployment

Model Selection & Evaluation

Two models were evaluated:
Model	Purpose
Logistic Regression	Baseline model
Linear SVM	Final production model
Why Linear SVM?
Better handling of high-dimensional sparse text data
Strong margin-based classification
Improved minority sentiment detection
Higher overall accuracy (~88%)
Evaluation Strategy
The dataset was split into training and testing subsets (80/20) to ensure unbiased evaluation on unseen reviews.

Key Observations:

Strong precision-recall balance.
Reliable detection of negative sentiment.
Reduced bias toward majority classes.
A confusion matrix was used to analyze misclassifications and validate model robustness.
The results indicate the model is suitable for large-scale automated sentiment monitoring.

Recommendation Framework

Instead of traditional collaborative filtering, this project implements a business-oriented recommendation strategy focused on product health.
Products are categorized into:

Promote Aggressively

High positive sentiment combined with strong review volume indicates customer trust and product reliability.

Investigate Immediately

Elevated negative sentiment suggests potential quality gaps, packaging issues, or expectation mismatches.

Monitor
Products with limited but promising feedback may represent emerging growth opportunities.
This approach aligns directly with real-world workflows used by product and marketing teams.

Business Impact

This solution enables organizations to:
Identify products suitable for aggressive promotion
Detect customer dissatisfaction early
Reduce brand and revenue risk
Support strategic marketing decisions
Monitor emerging high-potential products

Tech Stack

Python
Scikit-learn
Pandas
NumPy
Matplotlib / Seaborn
Docker


Project Structure

healthkart-review-nlp/
│
├── data/                # Dataset (ignored in repo)
├── models/              # Saved artifacts (ignored)
├── notebooks/           # Analysis & experimentation
├── src/
│     ├── train.py       # Training pipeline
│     └── predict.py     # Inference script
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md

Run with Docker
Build the container:
docker build -t healthkart-nlp .
Run the model:
docker run healthkart-nlp

Run Locally (Without Docker)
Train the model
python src/train.py
Run inference
python src/predict.py
Example output:
Prediction: positive

Future Improvements

Deploy as a real-time inference API
Experiment with transformer-based models (BERT)
Build an interactive analytics dashboard
Implement real-time review monitoring

👤 Author

Mayank Sahu
