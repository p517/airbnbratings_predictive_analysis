# 🏡 Optimizing Airbnb Ratings: Predictive Analytics on Property Features

This project uses R to predict whether an Airbnb listing will receive a **high rating (≥ 4.9)** using various listing features. It covers the full pipeline — from data preprocessing and EDA to model training and evaluation.

---

## 🎯 Objective

To build a classification model that predicts whether an Airbnb listing will receive a **high rating** (≥ 4.9) based on listing features such as price, reviews, and property capacity.

---

## 📦 Dataset Overview

- **Source**: Aggregated Airbnb data (CSV)
- **Size**: 12,800 rows × 23 columns
- **Target Variable**: `high_rating` (Binary: `low` or `high`)
- **Key Features**: `price`, `reviews`, `beds`, `bathrooms`, `bedrooms`, `guests`

---

## 🔍 Exploratory Data Analysis (EDA)

The `perform_eda()` function automatically generates and saves plots in the `visualizations/` folder.  
**Key visualizations include:**

- Rating Distribution  
- Price Distribution by Rating Category  
- Feature Importance (based on correlation with high rating)  
- Scatter Plot: Reviews vs Price (colored by rating)  

![image](https://github.com/user-attachments/assets/cc9d1067-3b8d-4cf7-99fb-5ac26b91b411)


---

## 🧠 Machine Learning Models

Trained using an **80/20 train-test split** and **5-fold cross-validation**:

### Models Used:
- Logistic Regression
- Decision Tree
- Random Forest ✅ *(Best performer)*
- Support Vector Machine (SVM)
- Naive Bayes

**Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (Area Under Curve)


## 🌳 Decision Tree Visualization

The decision tree built using the `rpart` package provides interpretable paths.

![image](https://github.com/user-attachments/assets/1430b0e5-df79-4816-8b3e-da630aa38e3c)

---

## 🧪 Model Evaluation: ROC & Comparison

### ROC Curve Comparison  
![image](https://github.com/user-attachments/assets/c82c4afd-05b3-460a-909c-54ed51b3fc62)

### Metric Comparison (Bar Chart)  
![image](https://github.com/user-attachments/assets/4ff083dc-9ebc-4309-a48a-38610d585826)

📄 Tabular results saved at:  
`visualizations/model_comparison_results.csv`

---

## 📌 Business Insights

- **Higher price** and **fewer reviews** predict higher ratings.
- Random Forest consistently outperformed other models in both **AUC** and **F1 Score**.
- Airbnb can use such classification models for recommendations and pricing strategies.

---

## 💻 Project Structure

├── main.R # Main analysis script (calls everything)
├── data/ # Contains Airbnb dataset (e.g., airbnb-2.csv)
├── visualizations/ # Automatically generated plots + metrics
└── README.md # Project documentation

