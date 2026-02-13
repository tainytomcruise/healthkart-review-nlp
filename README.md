# Customer Sentiment Analysis & Product Intelligence System

## 📌 Overview

This project builds an end-to-end Natural Language Processing (NLP) pipeline to analyze customer product reviews, detect sentiment patterns, and generate actionable business recommendations. The system helps organizations identify high-performing products, detect potential risks early, and support data-driven marketing and product decisions.

**Built as part of a Data Science internship assessment to demonstrate end-to-end NLP, machine learning, and deployment capabilities.**

---

## 🚀 Key Features

✅ Automated sentiment classification using TF-IDF and Linear SVM
✅ Hybrid use of structured data (ratings, product identifiers) and unstructured review text
✅ Product-level promotion scoring based on customer feedback
✅ Confidence-aware recommendation framework
✅ Business action strategy (Promote / Monitor / Investigate)
✅ Model comparison (Logistic Regression vs Linear SVM)
✅ Portfolio visualization using sentiment vs review volume
✅ Dockerized pipeline for reproducible training and deployment

---

## 🧠 Business Impact

This solution enables organizations to:

* Identify products suitable for aggressive promotion
* Detect customer dissatisfaction before it escalates
* Reduce brand and revenue risk
* Support strategic marketing decisions
* Monitor emerging high-potential products

---

## 📦 Dataset Overview

The dataset consists of large-scale customer review data containing both structured and unstructured attributes.

**Key Characteristics:**

* ~70,000+ customer reviews
* Multi-class sentiment distribution (Positive / Neutral / Negative)
* Structured attributes such as ratings, product names, and identifiers
* Free-text customer reviews capturing detailed user perception

**How the Dataset Was Utilized:**

**Structured Data Used To:**

* Derive sentiment labels from ratings
* Aggregate product-level metrics
* Improve decision confidence using review volume

**Unstructured Text Used To:**

* Train the NLP sentiment classifier
* Capture nuanced customer opinions
* Identify satisfaction and dissatisfaction signals

This hybrid approach improves both statistical robustness and contextual understanding.

---

## 🔬 Methodology

### 1. Data Cleaning & Preprocessing

* Removed null and duplicate entries
* Standardized review text
* Applied basic text normalization

### 2. Feature Engineering

* TF-IDF vectorization to convert text into high-dimensional numerical features
* Sparse matrix representation optimized for linear models

### 3. Train-Test Strategy

A **stratified train-test split** was used to ensure that sentiment classes were proportionally represented in both datasets. This allows reliable evaluation on unseen data and reduces sampling bias.

### 4. Model Experimentation

Two supervised machine learning models were evaluated:

| Model               | Purpose                |
| ------------------- | ---------------------- |
| Logistic Regression | Baseline classifier    |
| Linear SVM          | Final production model |

---

## 📊 Model Performance

The models were evaluated on the held-out test dataset.

| Metric               | Logistic Regression | Linear SVM |
| -------------------- | ------------------- | ---------- |
| Accuracy             | 79%                 | **88%**    |
| Precision (weighted) | 0.89                | **0.89**   |
| Recall (weighted)    | 0.88                | **0.88**   |
| F1 Score             | 0.88                | **0.88**   |

### Confusion Matrix Insights

* Strong detection of positive sentiment (majority class)
* Improved minority-class detection due to class weighting
* Minimal signs of overfitting between training and test performance

The **Linear SVM** model was selected for production due to its superior performance on high-dimensional sparse text features and its robustness in handling class imbalance.

*(Optional: Add confusion matrix image inside an `/assets` folder for stronger visual validation.)*

---

## 🧠 Recommendation Strategy

Instead of traditional collaborative filtering, this project implements a **business-oriented recommendation framework** focused on operational decision-making.

### Promotion Score Formula

```
Promotion Score = Positive Ratio − Negative Ratio
```

### Decision Logic

| Promotion Score | Confidence (Review Volume) | Business Action            |
| --------------- | -------------------------- | -------------------------- |
| High            | High                       | ✅ Promote Aggressively     |
| Moderate        | Medium                     | 👀 Monitor                 |
| Low / Negative  | High                       | ⚠️ Investigate Immediately |
| Any             | Low                        | 📊 Collect More Reviews    |

### Why This Approach?

This strategy is particularly effective for:

* Marketing prioritization
* Product lifecycle management
* Early risk detection
* Inventory planning

Future iterations could incorporate embedding-based similarity search for personalized product recommendations.

---

## 📈 Visualization

A portfolio-style bubble chart was created to help stakeholders quickly interpret product performance:

**Axes Meaning:**

* X-axis → Review Volume (Market Adoption)
* Y-axis → Promotion Score (Customer Love)
* Bubble Size → Product popularity
* Color → Sentiment strength

This visualization allows decision-makers to instantly identify:

* Market leaders
* Hidden growth opportunities
* Risk-heavy products

---

## 🔎 Key Code Paths

| File                                  | Purpose                                                                           |
| ------------------------------------- | --------------------------------------------------------------------------------- |
| `notebooks/01_data_exploration.ipynb` | Data cleaning, EDA, feature engineering, visualization, and model experimentation |
| `src/train.py`                        | End-to-end training pipeline with TF-IDF + Linear SVM                             |
| `src/predict.py`                      | Loads trained artifacts and performs inference on new reviews                     |
| `models/`                             | Serialized vectorizer and trained model                                           |
| `Dockerfile`                          | Containerized runtime for reproducible execution                                  |

---

## 🔮 Example Inference

### Example 1

**Input Review:**

```
"This product stopped working within two days. Very disappointed."
```

**Output:**

```
Predicted Sentiment: Negative  
Business Action: Investigate Immediately
```

---

### Example 2

**Input Review:**

```
"I absolutely love this product. Highly recommended!"
```

**Output:**

```
Predicted Sentiment: Positive  
Business Action: Promote Aggressively
```

---

## 🏗️ Project Structure

```
healthkart-review-nlp/
│
├── data/                  # Dataset
├── models/                # Saved model & vectorizer
├── notebooks/             # Analysis & experimentation
├── src/
│   ├── train.py           # Training pipeline
│   └── predict.py         # Inference script
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally (Without Docker)

### Install dependencies

```
pip install -r requirements.txt
```

### Train the model

```
python src/train.py
```

### Run inference

```
python src/predict.py
```

---

## 🐳 Run With Docker

### Build the container

```
docker build -t healthkart-nlp .
```

### Run the model

```
docker run healthkart-nlp
```

This ensures a reproducible environment for training and inference across machines.

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib / Seaborn
* Docker

---

## 🔮 Future Improvements

* Deploy as a real-time inference API
* Experiment with transformer-based models (BERT/RoBERTa)
* Implement embedding-based product similarity
* Build an interactive analytics dashboard
* Enable real-time review monitoring

---

## 👤 Author

**Mayank Sahu**

---

## ⭐ Final Note

This project demonstrates the complete lifecycle of a production-style NLP system — from raw data processing and model training to business insight generation and containerized deployment.
