# 📞 Telco Customer Churn Prediction

## 📌 Project Overview
This project predicts whether a telecom customer is likely to churn based on their service usage and account information.  
It uses multiple machine learning models and a final **Stacking Classifier**, and is deployed using **Streamlit** for real-time predictions.

---

## 🎯 Problem Statement
Telecom companies lose revenue due to customer churn.  
This project aims to build a machine learning system that can predict churn customers in advance so that retention strategies can be applied.

---

## 🧠 Machine Learning Workflow

### 🔹 Data Preprocessing
- Handled missing values in `TotalCharges`
- Converted categorical features using One-Hot Encoding
- Feature scaling using StandardScaler
- Feature selection using correlation + Chi-square test

---

### 🔹 Exploratory Data Analysis (EDA)
- Boxplots for numerical features vs churn
- Countplots for categorical features
- Correlation heatmap

---

### 🔹 Models Used
The following models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes
- XGBoost Classifier
- Stacking Classifier (Final Model)

---

### 🔹 Hyperparameter Tuning
Performed using **RandomizedSearchCV**:
- KNN
- Decision Tree
- Random Forest

---

## 🏆 Final Model
✔ Stacking Classifier (Best Model)

### Base Models:
- XGBoost
- KNN
- Logistic Regression
- Decision Tree

### Meta Model:
- Logistic Regression

---

## 📊 Model Performance

Best model achieved:
  Accuracy: ~82%
  Confusion Matrix:
    [[941, 95],
    [158, 215]]

---

## 🚀 Deployment

The model is deployed using **Streamlit**.

### Features:
- User-friendly input form
- Real-time churn prediction
- Probability score output
- Clean telecom-themed UI

---

## 🖥️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Streamlit
- Joblib

---

## 📁 Project Structure
  
  Telco-Churn-Prediction/
  │
  ├── app.py
  ├── churn_stacked_lr_model.pkl
  ├── churn_scale.pkl
  ├── Telco-Customer-Churn.csv
  ├── notebook.ipynb
  └── README.md


---

## 📈 Key Insights

- Month-to-month contracts have higher churn rates
- Electronic check users are more likely to churn
- Fiber optic internet users show higher churn probability
- Longer tenure reduces churn risk

---

## 💡 Business Impact

This project helps telecom companies to:
- Identify potential churn customers early
- Improve customer retention strategies
- Reduce revenue loss
- Optimize marketing campaigns

---

## ⚙️ How to Run This Project

```bash
pip install -r requirements.txt
streamlit run app.py
